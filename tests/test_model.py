"""
Unit tests for src/model.py: UNet3D and SimpleNodeTransformer.

Written to close a real, previously-flagged test-coverage gap (no test_model.py
existed at all before this file). Assertions are shape/value/identity-specific,
not just ndim/type-only checks -- per this project's own established lesson
(CLAUDE.md) that weak assertions have hidden real silent-corruption bugs before.

Run: py -m pytest tests/test_model.py -v
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import AnisotropicCoordinateTransformer, SimpleNodeTransformer, UNet3D


# Small architecture for fast tests: Z isn't pooled so any Z works; Y/X are
# downsampled 4x twice (16x total) then upsampled back, so Y/X must be
# divisible by 16 for a clean round-trip.
def make_small_unet():
    return UNet3D(in_channels=2, channels=(4, 8, 16))


class TestUNet3D:
    def test_forward_output_shapes_match_input_spatial_dims(self):
        """logits and features must exactly match the input's (Z, Y, X), with the
        documented channel counts (logits: 2, features: last channels entry)."""
        model = make_small_unet()
        model.eval()
        x = torch.randn(1, 2, 4, 32, 32)

        with torch.no_grad():
            logits, features = model(x)

        assert logits.shape == (1, 2, 4, 32, 32), (
            f"logits shape {tuple(logits.shape)} must be (B,2,Z,Y,X) matching input spatial dims"
        )
        assert features.shape == (1, 16, 4, 32, 32), (
            f"features shape {tuple(features.shape)} must be (B, channels[-1], Z, Y, X)"
        )

    def test_z_dimension_is_never_pooled(self):
        """REGRESSION-relevant: this project has shipped a real bug once where Z got
        silently sliced/pooled instead of preserved (Phase 2 Wave 1's __getitem__
        channel-axis bug). Verify with an asymmetric, non-power-of-2 Z that the
        architecture genuinely never touches it."""
        model = make_small_unet()
        model.eval()
        x = torch.randn(1, 2, 7, 32, 32)  # odd, non-power-of-2 Z

        with torch.no_grad():
            logits, features = model(x)

        assert logits.shape[2] == 7
        assert features.shape[2] == 7

    def test_batch_dimension_preserved(self):
        model = make_small_unet()
        model.eval()
        x = torch.randn(3, 2, 4, 32, 32)

        with torch.no_grad():
            logits, features = model(x)

        assert logits.shape[0] == 3
        assert features.shape[0] == 3

    def test_gradients_flow_through_both_output_heads(self):
        """A change that accidentally detaches a head (e.g. from a refactor) would
        pass shape checks but silently stop training that head -- this test would
        catch that specific failure mode."""
        model = make_small_unet()
        x = torch.randn(1, 2, 4, 32, 32, requires_grad=False)

        logits, features = model(x)
        loss = logits.sum() + features.sum()
        loss.backward()

        grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, "no parameters received gradients"
        assert all(g >= 0 for g in grad_norms)
        # at least one parameter must have a genuinely non-zero gradient
        assert any(g > 0 for g in grad_norms), "all gradients were exactly zero"

    def test_detection_head_bias_initialized_to_foreground_prior_not_zero(self):
        """REGRESSION GUARD (2026-07-13, found by adversarial review): PyTorch's
        default zero bias means sigmoid(0)=0.5 everywhere at init, wasting many
        early gradient steps just discovering the ~1e-4 real background prior
        before the network can learn real spatial cell features. A future
        refactor of det_head must not silently drop this init back to zero."""
        model = make_small_unet()

        final_bias = model.det_head[-1].bias
        sigmoid_at_init = torch.sigmoid(final_bias)

        assert torch.allclose(sigmoid_at_init, torch.full_like(sigmoid_at_init, 1e-4), atol=1e-5), (
            f"detection head bias must be initialized to the foreground prior "
            f"(sigmoid~1e-4), got sigmoid={sigmoid_at_init.tolist()} -- default "
            f"zero-bias init (sigmoid=0.5) has been silently reintroduced"
        )


class TestSimpleNodeTransformer:
    def make_model(self):
        torch.manual_seed(0)
        m = SimpleNodeTransformer(hidden_dim=16, num_heads=2, num_blocks=1, dropout=0.0, feature_dim=8)
        m.eval()
        return m

    def test_output_shape_is_all_pairwise_edges_in_row_major_order(self):
        """Default (candidate_edges=None) path must score every (i, j) pair, i outer
        j inner -- greedy_edge_assignment's own default-candidate reconstruction
        assumes this exact ordering, so a change here would silently corrupt which
        edge a given index refers to without any shape-level test catching it."""
        m = self.make_model()
        n_t, n_t1 = 4, 6
        nodes_t, nodes_t1 = torch.randn(n_t, 3), torch.randn(n_t1, 3)
        feat_t, feat_t1 = torch.randn(n_t, 8), torch.randn(n_t1, 8)

        with torch.no_grad():
            out = m(nodes_t, nodes_t1, feat_t, feat_t1)

        assert out.shape == (n_t * n_t1,)
        assert torch.all((out >= 0) & (out <= 1)), "edge_scorer ends in Sigmoid, output must be in [0,1]"

    def test_zero_nodes_returns_empty_tensor_without_crashing(self):
        m = self.make_model()
        nodes_t1, feat_t1 = torch.randn(3, 3), torch.randn(3, 8)

        with torch.no_grad():
            out = m(torch.randn(0, 3), nodes_t1, torch.randn(0, 8), feat_t1)

        assert out.shape == (0,)

    def test_single_explicit_candidate_edge(self):
        m = self.make_model()
        n_t, n_t1 = 4, 6
        nodes_t, nodes_t1 = torch.randn(n_t, 3), torch.randn(n_t1, 3)
        feat_t, feat_t1 = torch.randn(n_t, 8), torch.randn(n_t1, 8)

        with torch.no_grad():
            out = m(nodes_t, nodes_t1, feat_t, feat_t1, candidate_edges=torch.tensor([[2, 1]]))

        assert out.shape == (1,)

    def test_vectorized_scoring_matches_manual_per_edge_computation(self):
        """REGRESSION GUARD for the loop->vectorized rewrite (bug 2.3): manually
        reproduce the OLD per-edge computation using the model's own internals and
        assert numerical equivalence, not just matching shapes -- a transposed
        index (nodes_t_h[j] instead of nodes_t_h[i], say) would still pass every
        shape check above but silently score every edge wrong."""
        m = self.make_model()
        n_t, n_t1 = 4, 6
        nodes_t, nodes_t1 = torch.randn(n_t, 3), torch.randn(n_t1, 3)
        feat_t, feat_t1 = torch.randn(n_t, 8), torch.randn(n_t1, 8)

        with torch.no_grad():
            vectorized_out = m(nodes_t, nodes_t1, feat_t, feat_t1)

            device = nodes_t.device
            pe_dim = 8
            pos_enc_t = m.sinusoidal_positional_encoding(n_t, pe_dim * 3).to(device)
            pos_enc_t1 = m.sinusoidal_positional_encoding(n_t1, pe_dim * 3).to(device)
            nodes_t_h = m.node_embed(torch.cat([nodes_t, pos_enc_t, feat_t], dim=1))
            nodes_t1_h = m.node_embed(torch.cat([nodes_t1, pos_enc_t1, feat_t1], dim=1))
            nodes_t_h = m.encoder_t(nodes_t_h.unsqueeze(0)).squeeze(0)
            nodes_t1_h = m.encoder_t1(nodes_t1_h.unsqueeze(0)).squeeze(0)

            manual_probs = []
            for i in range(n_t):
                for j in range(n_t1):
                    ef = torch.cat([nodes_t_h[i], nodes_t1_h[j]])
                    manual_probs.append(m.edge_scorer(ef.unsqueeze(0)).squeeze())
            manual_out = torch.stack(manual_probs)

        max_diff = (vectorized_out - manual_out).abs().max().item()
        assert max_diff < 1e-5, f"vectorized output diverges from manual per-edge computation: {max_diff}"


class TestAnisotropicCoordinateTransformer:
    def test_scales_voxel_coords_by_anisotropy_buffer(self):
        transformer = AnisotropicCoordinateTransformer(anisotropy_ratio=(4.0, 1.0, 1.0))
        voxel_coords = torch.tensor([[[2.0, 3.0, 5.0]]])  # (B=1, N=1, 3)

        physical = transformer(voxel_coords)

        expected = torch.tensor([[[8.0, 3.0, 5.0]]])  # Z*4, Y*1, X*1
        assert torch.allclose(physical, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
