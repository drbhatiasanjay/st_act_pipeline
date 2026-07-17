"""
Comprehensive tests for P0-5: Remove Double Sigmoid fix.

Tests verify:
A. Transformer returns logits (unbounded)
B. Correct loss chain (DivisionLoss with logits)
C. No double sigmoid
D. Inference conversion (logits -> one Sigmoid)
E. Threshold behavior
F. Gradient flow preservation
G. Detached comparison
H. Checkpoint compatibility
I. Training/inference consistency
J. Production call-site regression coverage (AST-based)
"""

import ast
import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.inference import greedy_edge_assignment
from src.model import SimpleNodeTransformer, UNet3D
from src.targets import DivisionLoss

logger = logging.getLogger(__name__)


class TestP05TransformerReturnsLogits:
    """Test A: Transformer returns logits, not probabilities."""

    def test_transformer_no_sigmoid_module(self):
        """Verify Sigmoid module is not in edge_scorer."""
        transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)

        for module in transformer.edge_scorer.modules():
            assert not isinstance(module, nn.Sigmoid), \
                "edge_scorer should not contain nn.Sigmoid() module"

        logger.info("Transformer edge_scorer has no Sigmoid module")

    def test_transformer_output_unbounded(self):
        """Deterministic proof the transformer returns unbounded logits.

        Zeroes edge_scorer's final Linear weight and controls only its
        bias -- this makes every edge's output exactly equal to the bias,
        regardless of random initialization elsewhere, so negative/zero/
        positive/outside-[0,1] are all REAL assertions, not values that
        merely happened to land there under a fixed random seed (the prior
        version of this test). Fails immediately if a final Sigmoid is ever
        reintroduced, since sigmoid(anything) is always in (0,1) and could
        never equal an exact bias of -3.0, 0.0, or 5.0.
        """
        transformer = SimpleNodeTransformer(hidden_dim=32, num_heads=4, num_blocks=1, feature_dim=16)
        transformer.eval()

        final_linear = transformer.edge_scorer[-1]
        assert isinstance(final_linear, nn.Linear), \
            "edge_scorer's last module must be the scoring Linear layer (no trailing Sigmoid)"

        nodes_t = torch.randn((2, 3))
        nodes_t1 = torch.randn((2, 3))
        features_t = torch.randn((2, 16))
        features_t1 = torch.randn((2, 16))

        def logits_for_bias(bias_value: float) -> torch.Tensor:
            with torch.no_grad():
                final_linear.weight.zero_()
                final_linear.bias.fill_(bias_value)
                return transformer(nodes_t, nodes_t1, features_t, features_t1)

        neg_logits = logits_for_bias(-3.0)
        assert (neg_logits < 0).all(), f"Expected all-negative logits, got {neg_logits.tolist()}"
        assert torch.allclose(neg_logits, torch.full_like(neg_logits, -3.0)), \
            "With final weight zeroed, output must equal the bias exactly"

        zero_logits = logits_for_bias(0.0)
        assert (zero_logits == 0).all(), f"Expected all-zero logits, got {zero_logits.tolist()}"

        pos_logits = logits_for_bias(5.0)
        assert (pos_logits > 0).all(), f"Expected all-positive logits, got {pos_logits.tolist()}"
        assert (pos_logits > 1.0).all(), (
            "Logits must be able to exceed 1.0 -- impossible if a final Sigmoid "
            "were reintroduced (sigmoid output is always < 1.0)"
        )

        logger.info("Transformer outputs proven unbounded: negative, zero, and >1.0 all reproduced deterministically")

    def test_transformer_empty_nodes(self):
        """Verify transformer handles empty node sets correctly."""
        transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)
        transformer.eval()

        device = torch.device("cpu")
        nodes_t = torch.zeros((0, 3), device=device)
        nodes_t1 = torch.randn((3, 3), device=device)
        features_t = torch.zeros((0, 128), device=device)
        features_t1 = torch.randn((3, 128), device=device)

        with torch.no_grad():
            logits = transformer(nodes_t, nodes_t1, features_t, features_t1)

        assert logits.shape == (0,), "Empty nodes should return empty logits"
        logger.info("Transformer correctly handles empty nodes")


class TestP05CorrectLossChain:
    """Test B: DivisionLoss correctly receives logits and applies BCEWithLogitsLoss."""

    def test_division_loss_expects_logits(self):
        """Verify DivisionLoss signature expects logits parameter."""
        loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)

        import inspect
        sig = inspect.signature(loss_fn.forward)
        param_names = list(sig.parameters.keys())
        assert 'logits' in param_names, \
            f"DivisionLoss.forward should have 'logits' param, got {param_names}"

        logger.info("DivisionLoss.forward has 'logits' parameter")

    def test_division_loss_uses_bce_with_logits(self):
        """Verify DivisionLoss uses BCEWithLogitsLoss internally."""
        loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)

        assert isinstance(loss_fn.bce_loss, nn.BCEWithLogitsLoss), \
            "DivisionLoss should use BCEWithLogitsLoss"

        logger.info("DivisionLoss uses BCEWithLogitsLoss")

    def test_production_loss_matches_independent_reference_formula(self):
        """Cross-check production DivisionLoss against an independently
        computed reference (raw torch.nn.functional.binary_cross_entropy_with_logits
        plus the documented class/division weighting) for the exact raw
        logits specified in the P0-5 spec, both target classes, and
        representative division-mask values. Also proves feeding
        sigmoid(logits) -- the double-sigmoid bug -- does NOT match the
        correct raw-logit result, a real regression guard rather than a
        finite/positive-only sanity check.
        """
        weight_division = 2.0
        pos_weight = 10.0
        loss_fn = DivisionLoss(weight_division=weight_division, pos_weight=pos_weight)

        raw_logits = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float32)

        scenarios = [
            ("all_negative_targets", [0.0, 0.0, 0.0, 0.0, 0.0], [False, True, False, True, False]),
            ("all_positive_targets", [1.0, 1.0, 1.0, 1.0, 1.0], [True, False, True, False, True]),
            ("mixed_targets", [0.0, 1.0, 0.0, 1.0, 0.0], [False, False, True, True, False]),
        ]

        for name, targets_list, division_list in scenarios:
            targets = torch.tensor(targets_list, dtype=torch.float32)
            division_mask = torch.tensor(division_list, dtype=torch.bool)

            actual = loss_fn(raw_logits, targets, division_mask)

            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                raw_logits, targets, reduction="none",
            )
            class_weight = pos_weight * targets + (1.0 - targets)
            division_weight = weight_division * division_mask.float() + (1.0 - division_mask.float())
            expected = (bce * class_weight * division_weight).mean()

            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6), (
                f"[{name}] DivisionLoss must match the independently-computed "
                f"reference exactly. targets={targets_list} division={division_list}: "
                f"got {actual.item()}, expected {expected.item()}"
            )

            # Regression guard: sigmoid(raw_logits) is the double-sigmoid bug's
            # input -- it must NOT match the correct raw-logit loss.
            wrong = loss_fn(torch.sigmoid(raw_logits), targets, division_mask)
            assert not torch.allclose(wrong, expected, atol=1e-3), (
                f"[{name}] DivisionLoss(sigmoid(logits), ...) must NOT equal the "
                f"correct raw-logit loss -- if it does, the double-sigmoid bug "
                f"has regressed and this test failed to catch it"
            )

        logger.info("Production DivisionLoss matches independent reference formula for all scenarios")

    def test_loss_gradient_magnitude_sanity(self):
        """Verify gradients have reasonable magnitude for logit inputs."""
        loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)

        edge_logits = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float32, requires_grad=True)
        edge_targets = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)

        loss = loss_fn(edge_logits, edge_targets)
        loss.backward()

        assert edge_logits.grad is not None, "Gradient should exist"
        max_grad = edge_logits.grad.abs().max().item()
        assert max_grad > 0.01, f"Gradient magnitude too small: {max_grad}"

        logger.info(f"Gradient magnitudes reasonable: max={max_grad:.6f}")


class TestP05NoDoubleSigmoid:
    """Test C: Verify double sigmoid does not occur."""

    def test_training_path_no_sigmoid_before_loss(self):
        """Verify training path passes logits directly to DivisionLoss."""
        transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)
        loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)

        nodes_t = torch.randn((5, 3))
        nodes_t1 = torch.randn((5, 3))
        features_t = torch.randn((5, 128))
        features_t1 = torch.randn((5, 128))
        targets = torch.randint(0, 2, (25,), dtype=torch.float32)

        transformer.eval()
        with torch.no_grad():
            edge_logits = transformer(nodes_t, nodes_t1, features_t, features_t1)
            loss = loss_fn(edge_logits, targets)

        assert loss.item() > 0, "Loss should be computed correctly"
        logger.info("Training path does not apply sigmoid before DivisionLoss")

    def test_regression_sigmoid_before_loss_would_fail(self):
        """Verify double sigmoid produces different loss trajectories."""
        loss_fn_bce_logits = nn.BCEWithLogitsLoss(reduction='none')

        logits = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)

        loss_correct = loss_fn_bce_logits(logits, targets).mean()
        loss_correct.backward()
        grad_correct = logits.grad.clone()

        logits2 = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float32, requires_grad=True)
        probs = torch.sigmoid(logits2)
        loss_wrong = loss_fn_bce_logits(probs, targets).mean()
        loss_wrong.backward()
        grad_wrong = logits2.grad.clone()

        grad_diff_ratio = (grad_correct.abs().max() / (grad_wrong.abs().max() + 1e-8)).item()
        assert grad_diff_ratio > 1.5, \
            f"Double sigmoid should suppress gradients (ratio: {grad_diff_ratio:.2f}x)"

        logger.info(f"Double sigmoid gradient suppression detected (ratio: {grad_diff_ratio:.2f}x)")


class TestP05InferenceConversion:
    """Test D: Inference paths apply exactly one Sigmoid."""

    def test_greedy_edge_assignment_receives_probabilities(self):
        """Verify greedy_edge_assignment expects probabilities [0,1]."""
        nodes_t = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=torch.float32)
        nodes_t1 = torch.tensor([[0.0, 0.0, 1.0], [10.0, 10.0, 11.0]], dtype=torch.float32)

        probs = torch.tensor([0.9, 0.1, 0.1, 0.9], dtype=torch.float32)
        threshold = 0.5

        assignment = greedy_edge_assignment(
            probs, nodes_t, nodes_t1, threshold=threshold,
            max_children=2, max_parents=1
        )

        assert isinstance(assignment, dict), "Should return a dict"
        assert 'edges' in assignment, "Dict should have 'edges' key"
        logger.info(f"greedy_edge_assignment correctly uses probability threshold "
                    f"({len(assignment['edges'])} edges selected)")

    def test_inference_logits_to_probs_conversion(self):
        """Verify inference paths convert logits to probs via sigmoid."""
        transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)
        transformer.eval()

        nodes_t = torch.randn((3, 3))
        nodes_t1 = torch.randn((3, 3))
        features_t = torch.randn((3, 128))
        features_t1 = torch.randn((3, 128))

        with torch.no_grad():
            edge_logits = transformer(nodes_t, nodes_t1, features_t, features_t1)
            edge_probs = torch.sigmoid(edge_logits)

        assert (edge_probs >= 0).all() and (edge_probs <= 1).all(), \
            "Sigmoid output should be in [0, 1]"

        logger.info(f"Inference conversion: logits -> sigmoid -> probs "
                    f"[{edge_probs.min():.4f}, {edge_probs.max():.4f}]")


class TestP05ThresholdBehavior:
    """Test E: Threshold operates on probabilities, not logits."""

    def test_threshold_at_probability_boundary(self):
        """Test edge_threshold=0.5 correctly accepts/rejects at probability 0.5."""
        logit_at_threshold = torch.tensor([0.0], dtype=torch.float32)
        prob_at_threshold = torch.sigmoid(logit_at_threshold)

        assert abs(prob_at_threshold.item() - 0.5) < 1e-6, \
            "Logit 0 should map to probability 0.5"

        logit_below = torch.tensor([-0.1], dtype=torch.float32)
        logit_above = torch.tensor([0.1], dtype=torch.float32)

        prob_below = torch.sigmoid(logit_below)
        prob_above = torch.sigmoid(logit_above)

        assert prob_below < 0.5, "Negative logit should map to prob < 0.5"
        assert prob_above > 0.5, "Positive logit should map to prob > 0.5"

        threshold = 0.5
        assert not (prob_below > threshold), "Probability below 0.5 should be rejected"
        assert (prob_above > threshold), "Probability above 0.5 should be accepted"

        logger.info("Threshold behavior correct at probability boundary")


def _make_small_unet_and_transformer():
    """Small-but-architecturally-real UNet3D + SimpleNodeTransformer pair for
    fast gradient-flow tests: same classes, same forward logic, smaller
    channel/hidden-dim counts and spatial extent for test speed (mirrors
    tests/test_model.py's make_small_unet() pattern, Y/X divisible by 16
    for a clean pool/upsample round-trip)."""
    unet = UNet3D(in_channels=2, channels=(4, 8, 16))
    transformer = SimpleNodeTransformer(hidden_dim=32, num_heads=4, num_blocks=1, feature_dim=16)
    return unet, transformer


class TestP05GradientFlow:
    """Test F: Gradient flow preserved for UNet and transformer.

    The prior version of this test sampled UNet features inside
    torch.no_grad(), which silently disconnected them from the UNet's
    autograd graph -- the "UNet gradient" claim was never actually tested.
    This version runs a real, grad-enabled UNet forward pass and samples
    features with ordinary tensor indexing (a view, not a detached copy),
    so gradient genuinely has to flow through the real UNet layers to
    reach the checked parameters.
    """

    def test_gradient_flow_unet_transformer(self):
        device = torch.device("cpu")
        unet, transformer = _make_small_unet_and_transformer()
        loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)

        unet.train()
        transformer.train()

        frame_t = torch.randn((1, 1, 4, 32, 32), device=device)
        frame_t1 = torch.randn((1, 1, 4, 32, 32), device=device)
        x = torch.cat([frame_t, frame_t1], dim=1)  # (1, 2, 4, 32, 32)

        # Real forward pass, autograd enabled (no torch.no_grad() anywhere
        # in this function).
        logits, features = unet(x)

        # Ordinary tensor indexing at fixed integer coordinates -- this is a
        # VIEW into `features`, not a detached copy, no .cpu(), no
        # reconstructed tensor.
        nodes_t = torch.tensor([[1, 8, 8], [2, 16, 16]], dtype=torch.long)
        nodes_t1 = torch.tensor([[1, 8, 8]], dtype=torch.long)

        features_t = features[0, :, nodes_t[:, 0], nodes_t[:, 1], nodes_t[:, 2]].t()    # (2, 16)
        features_t1 = features[0, :, nodes_t1[:, 0], nodes_t1[:, 1], nodes_t1[:, 2]].t()  # (1, 16)

        edge_logits = transformer(nodes_t.float(), nodes_t1.float(), features_t, features_t1)
        edge_targets = torch.randint(0, 2, (len(edge_logits),), dtype=torch.float32)

        loss = loss_fn(edge_logits, edge_targets)
        loss.backward()

        transformer_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in transformer.parameters() if p.requires_grad
        )
        assert transformer_grad, "Transformer should have nonzero gradients from edge loss"

        # Explicitly identified UNet parameters checked:
        # unet.bottleneck[0].weight -- unambiguously on the features_upsampled
        # path (features = up1(up2(bottleneck(...)))).
        bottleneck_weight = unet.bottleneck[0].weight
        assert bottleneck_weight.grad is not None, (
            "unet.bottleneck[0].weight must receive a gradient from edge loss "
            "(it is on the features_upsampled path the transformer consumes)"
        )
        assert bottleneck_weight.grad.abs().sum() > 0, \
            "unet.bottleneck[0].weight gradient must be nonzero"

        # unet.enc0[0].weight -- the earliest encoder layer, proving gradient
        # reaches the full depth of the feature path, not just the bottleneck.
        enc0_weight = unet.enc0[0].weight
        assert enc0_weight.grad is not None and enc0_weight.grad.abs().sum() > 0, (
            "unet.enc0[0].weight must also receive a nonzero gradient "
            "(proves gradient reaches the full feature-path depth)"
        )

        # unet.det_head[0].weight is NOT on the features_upsampled path (only
        # dec1/dec2/det_head feed `logits`, which this test never uses) --
        # per spec, document the actual result rather than forcing either
        # direction.
        det_head_weight = unet.det_head[0].weight
        if det_head_weight.grad is None:
            logger.info("det_head[0].weight.grad is None (expected: not on the features branch)")
        else:
            det_head_grad_sum = det_head_weight.grad.abs().sum().item()
            logger.info(f"det_head[0].weight.grad is not None, sum={det_head_grad_sum}")
            assert det_head_grad_sum == 0, (
                "det_head is architecturally not on the features_upsampled path; "
                f"a nonzero gradient ({det_head_grad_sum}) would contradict that"
            )

        logger.info(
            "Gradient flow confirmed: transformer parameters, "
            "unet.bottleneck[0].weight, and unet.enc0[0].weight all nonzero"
        )


class TestP05DetachedComparison:
    """Test G: Attached vs. detached features, both production-faithful."""

    def test_attached_vs_detached_features_gradient_comparison(self):
        """Two production-faithful passes, each starting from a real,
        grad-enabled UNet forward pass: one with attached (not detached)
        sampled features -- exactly as production does it -- and one with
        .detach() inserted ONLY in this test (never in production) to prove
        detaching really is what would sever the UNet gradient path, while
        the transformer's own gradient is unaffected either way.
        """
        device = torch.device("cpu")

        def run_pass(detach_features: bool):
            torch.manual_seed(0)
            unet, transformer = _make_small_unet_and_transformer()
            loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)
            unet.train()
            transformer.train()

            frame_t = torch.randn((1, 1, 4, 32, 32), device=device)
            frame_t1 = torch.randn((1, 1, 4, 32, 32), device=device)
            x = torch.cat([frame_t, frame_t1], dim=1)
            logits, features = unet(x)

            nodes_t = torch.tensor([[1, 8, 8], [2, 16, 16]], dtype=torch.long)
            nodes_t1 = torch.tensor([[1, 8, 8]], dtype=torch.long)
            features_t = features[0, :, nodes_t[:, 0], nodes_t[:, 1], nodes_t[:, 2]].t()
            features_t1 = features[0, :, nodes_t1[:, 0], nodes_t1[:, 1], nodes_t1[:, 2]].t()

            if detach_features:
                # Test-only: proves what detaching WOULD do. Never present
                # in production code.
                features_t = features_t.detach()
                features_t1 = features_t1.detach()

            edge_logits = transformer(nodes_t.float(), nodes_t1.float(), features_t, features_t1)
            edge_targets = torch.randint(0, 2, (len(edge_logits),), dtype=torch.float32)
            loss = loss_fn(edge_logits, edge_targets)
            loss.backward()

            transformer_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in transformer.parameters() if p.requires_grad
            )
            bottleneck_grad = unet.bottleneck[0].weight.grad
            return transformer_grad, bottleneck_grad

        attached_transformer_grad, attached_unet_grad = run_pass(detach_features=False)
        detached_transformer_grad, detached_unet_grad = run_pass(detach_features=True)

        assert attached_transformer_grad, "Attached pass: transformer must have nonzero gradient"
        assert attached_unet_grad is not None and attached_unet_grad.abs().sum() > 0, (
            "Attached pass: UNet bottleneck must have nonzero gradient "
            "(this is the real production joint-gradient path)"
        )

        assert detached_transformer_grad, "Detached pass: transformer gradient must still be nonzero"
        assert detached_unet_grad is None or detached_unet_grad.abs().sum() == 0, (
            "Detached pass: UNet bottleneck gradient must be absent or zero -- "
            "detach() must sever the path"
        )

        logger.info(
            "Detached comparison: attached pass has nonzero UNet gradient, "
            "detached pass does not -- transformer gradient present in both"
        )


class _LegacySimpleNodeTransformer(nn.Module):
    """Test-only reconstruction of the PRE-FIX SimpleNodeTransformer: same
    parameterized edge_scorer layers, with the former parameterless
    nn.Sigmoid() appended at the end (exactly what P0-5 removed from
    production). Used only to prove a real pre-fix checkpoint state_dict
    strict-loads into the fixed model -- NOT a production class, never
    imported outside this test file.
    """

    def __init__(self, hidden_dim=32, num_heads=4, num_blocks=1, dropout=0.3, feature_dim=16):
        super().__init__()
        node_input_dim = 3 + 24 + feature_dim
        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.encoder_t = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, activation='relu',
            ),
            num_layers=num_blocks,
        )
        self.encoder_t1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, activation='relu',
            ),
            num_layers=num_blocks,
        )
        # Same parameterized layers as the real production edge_scorer,
        # PLUS the former parameterless Sigmoid this test reconstructs.
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )


class TestP05CheckpointCompatibility:
    """Test H: Checkpoint state dict compatibility."""

    def test_legacy_checkpoint_with_sigmoid_strict_loads_into_fixed_model(self):
        """A genuine pre-fix checkpoint (edge_scorer + trailing Sigmoid)
        must strict-load into the fixed SimpleNodeTransformer with zero
        missing and zero unexpected keys, and every parameter tensor must
        match exactly -- proving Sigmoid removal did not touch any
        parameterized layer's identity. (The prior version of this test
        constructed the POST-fix class twice and called one of them
        "pre-fix", which never actually tested legacy compatibility.)
        """
        legacy = _LegacySimpleNodeTransformer(hidden_dim=32, num_heads=4, num_blocks=1, feature_dim=16)
        legacy_state = legacy.state_dict()

        fixed = SimpleNodeTransformer(hidden_dim=32, num_heads=4, num_blocks=1, feature_dim=16)

        result = fixed.load_state_dict(legacy_state, strict=True)

        assert result.missing_keys == [], f"Missing keys: {result.missing_keys}"
        assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys}"

        loaded_state = fixed.state_dict()
        for key, legacy_tensor in legacy_state.items():
            assert torch.equal(legacy_tensor, loaded_state[key]), \
                f"Parameter tensor mismatch after load for key '{key}'"

        logger.info(
            f"Legacy checkpoint ({len(legacy_state)} keys, including a "
            f"parameterless trailing Sigmoid) strict-loaded cleanly with "
            f"zero missing/unexpected keys and exact tensor equality"
        )

    def test_fixed_checkpoint_keys_stable(self):
        """Verify parameter keys don't change across instances."""
        t1 = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)
        t2 = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)

        state1 = t1.state_dict()
        state2 = t2.state_dict()

        assert set(state1.keys()) == set(state2.keys()), \
            "Parameter keys should be stable across instances"

        edge_scorer_keys = [k for k in state1.keys() if 'edge_scorer' in k]
        assert len(edge_scorer_keys) == 6, \
            f"Should have 6 parameter tensors in edge_scorer (3 Linear layers), got {len(edge_scorer_keys)}"

        logger.info("Parameter keys stable after Sigmoid removal")


class TestP05TrainingInferenceConsistency:
    """Test I: Training uses logits, inference applies Sigmoid."""

    def test_training_inference_consistency(self):
        device = torch.device("cpu")
        transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)
        loss_fn = DivisionLoss(weight_division=2.0, pos_weight=10.0)

        nodes_t = torch.randn((3, 3), device=device)
        nodes_t1 = torch.randn((3, 3), device=device)
        features_t = torch.randn((3, 128), device=device)
        features_t1 = torch.randn((3, 128), device=device)

        transformer.eval()

        with torch.no_grad():
            edge_logits = transformer(nodes_t, nodes_t1, features_t, features_t1)

            dummy_targets = torch.randint(0, 2, (len(edge_logits),), dtype=torch.float32, device=device)
            training_loss = loss_fn(edge_logits, dummy_targets)

            edge_probs = torch.sigmoid(edge_logits)

            assert (edge_probs >= 0).all() and (edge_probs <= 1).all(), \
                "Inference probabilities should be in [0, 1]"
            assert training_loss.item() > 0, "Training loss should be positive"

        logger.info("Training/inference consistency verified")


# ============================================================
# Test J: Production call-site regression coverage (AST-based)
# ============================================================
#
# AST-based specifically so a commented-out `torch.sigmoid(edge_logits)` or
# a docstring/string mentioning it cannot produce a false pass -- comments
# and string literals are not `ast.Call`/`ast.Assign` nodes, so none of the
# helpers below can be fooled by test-code or comment occurrences.

def _get_module_source(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def _get_class_method_source(file_path: str, class_name: str, method_name: str) -> str:
    source = _get_module_source(file_path)
    tree = ast.parse(source, filename=file_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    segment = ast.get_source_segment(source, item)
                    assert segment is not None
                    return segment
    raise AssertionError(f"{class_name}.{method_name} not found in {file_path}")


def _has_call(source: str, name_fragment: str) -> bool:
    """True if a genuine ast.Call node's function name/attribute contains
    name_fragment (e.g. 'division_loss_fn' or 'greedy_edge_assignment')."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and name_fragment in func.attr:
                return True
            if isinstance(func, ast.Name) and name_fragment in func.id:
                return True
    return False


def _sigmoid_call_count(source: str) -> int:
    """Count genuine ast.Call nodes invoking sigmoid (torch.sigmoid(...) or
    a bare sigmoid(...) import), anywhere in the given source."""
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "sigmoid") or \
               (isinstance(func, ast.Name) and func.id == "sigmoid"):
                count += 1
    return count


def _count_exact_sigmoid_assignment(source: str, target_name: str, source_name: str) -> int:
    """Count genuine `target_name = torch.sigmoid(source_name)` (or
    `= sigmoid(source_name)`) assignments as real ast.Assign nodes -- a
    comment or docstring containing this exact text is not an ast.Assign
    node and will not be counted."""
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == target_name:
                val = node.value
                if isinstance(val, ast.Call):
                    func = val.func
                    is_sigmoid = (isinstance(func, ast.Attribute) and func.attr == "sigmoid") or \
                                 (isinstance(func, ast.Name) and func.id == "sigmoid")
                    if is_sigmoid and len(val.args) == 1 and isinstance(val.args[0], ast.Name) \
                            and val.args[0].id == source_name:
                        count += 1
    return count


def _first_positional_arg_name(source: str, call_name_fragment: str) -> str | None:
    """Name of the first positional-argument Name passed to the first call
    whose function name/attribute contains call_name_fragment, or None."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            matched = (isinstance(func, ast.Attribute) and call_name_fragment in func.attr) or \
                      (isinstance(func, ast.Name) and call_name_fragment in func.id)
            if matched and node.args and isinstance(node.args[0], ast.Name):
                return node.args[0].id
    return None


class TestP05ProductionCallSiteRegression:
    """Structural/AST-based coverage of every real transformer call site:
    training must never sigmoid edge_logits before DivisionLoss; every
    inference/validation path must apply exactly one
    torch.sigmoid(edge_logits) and pass the resulting probabilities (not
    raw logits) into greedy_edge_assignment.
    """

    def test_train_py_training_path(self):
        source = _get_class_method_source("src/train.py", "TrainingLoop", "train_epoch")

        assert _has_call(source, "transformer"), "train_epoch must call the transformer"
        assert _has_call(source, "division_loss_fn"), "train_epoch must call DivisionLoss"
        assert _count_exact_sigmoid_assignment(source, "edge_probs", "edge_logits") == 0, (
            "train_epoch's training path must never sigmoid edge_logits -- "
            "raw logits must flow directly into DivisionLoss"
        )

    def test_train_py_validation_path(self):
        source = _get_class_method_source("src/train.py", "TrainingLoop", "validate_epoch")

        assert _has_call(source, "transformer"), "validate_epoch must call the transformer"
        assert _has_call(source, "greedy_edge_assignment"), \
            "validate_epoch must call greedy_edge_assignment"
        assert _count_exact_sigmoid_assignment(source, "edge_probs", "edge_logits") == 1, (
            "validate_epoch must apply exactly one torch.sigmoid(edge_logits) "
            "(assigned to edge_probs) before greedy_edge_assignment"
        )
        assert _first_positional_arg_name(source, "greedy_edge_assignment") == "edge_probs", (
            "greedy_edge_assignment must receive the sigmoided edge_probs, "
            "not raw edge_logits"
        )

    def _assert_inference_script_pattern(self, file_path: str):
        source = _get_module_source(file_path)

        assert _has_call(source, "transformer"), f"{file_path} must call the transformer"
        assert _has_call(source, "greedy_edge_assignment"), \
            f"{file_path} must call greedy_edge_assignment"
        assert _count_exact_sigmoid_assignment(source, "edge_probs", "edge_logits") == 1, (
            f"{file_path} must apply exactly one torch.sigmoid(edge_logits) "
            f"(assigned to edge_probs) before greedy_edge_assignment"
        )
        assert _first_positional_arg_name(source, "greedy_edge_assignment") == "edge_probs", (
            f"{file_path}: greedy_edge_assignment must receive edge_probs, not raw edge_logits"
        )

    def test_evaluate_checkpoint_py(self):
        self._assert_inference_script_pattern("evaluate_checkpoint.py")

    def test_verify_eval_fixed_py(self):
        self._assert_inference_script_pattern("verify_eval_fixed.py")

    def test_generate_submission_py(self):
        self._assert_inference_script_pattern("generate_submission.py")

    def test_inference_kernel_py(self):
        self._assert_inference_script_pattern("kaggle_kernel_inference/inference_kernel.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
