"""
P0-6 tests: shared submission inference pipeline, verified checkpoint
manifest, and fail-closed submission deployment.

All tests are hermetic (no real Zarr/GPU/Kaggle/network dependency) and call
real production helpers -- no reimplemented validation logic, no vacuous
assertions.

Sections:
  F1 -- production submission-path tests (src/submission_pipeline.py)
  F2 -- sigmoid contract tests (functional + AST call-site checks)
  F4 -- manifest discovery and validation tests
  F5 -- checkpoint writing and eligibility tests
  F6 -- transaction and cleanup tests
  F7 -- hyperparameter validation tests
  F8 -- strict state-loading tests

All AST call-site regression coverage for the shared submission pipeline
lives entirely in this file (Part I correction: tests/test_p05_double_sigmoid_fix.py
is out of the frozen 13-path scope and must remain byte-identical to baseline).
"""

import ast
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import src.submission_pipeline as submission_pipeline_module
from src import checkpoint_manifest
from src.checkpoint_manifest import (
    MODEL_CONTRACT,
    deployment_eligibility_errors,
    find_single_manifest,
    load_verified_checkpoint,
    save_checkpoint_file,
    sha256_file,
    validate_inference_hyperparams,
    write_checkpoint_manifest,
)
from src.model import SimpleNodeTransformer, UNet3D
from src.prediction_graph import PredictionGraphAssembler
from src.submission_pipeline import (
    build_test_dataset,
    run_sample_loader_inference,
    run_submission_inference,
)

VALID_SHA40 = "a" * 40
VALID_SHA40_B = "c" * 40
VALID_SPLIT64 = "b" * 64
VALID_SPLIT64_B = "d" * 64


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_small_models():
    """Small-but-architecturally-real model configs (matches the project's
    established fast-test pattern, see tests/test_model.py's
    make_small_unet() / tests/test_p05_double_sigmoid_fix.py)."""
    unet3d = UNet3D(in_channels=2, channels=(4, 8, 16))
    transformer = SimpleNodeTransformer(hidden_dim=32, num_heads=4, num_blocks=1, feature_dim=16)
    unet3d.eval()
    transformer.eval()
    return unet3d, transformer


def _valid_hyperparams(**overrides) -> dict:
    hp = {
        "detection_threshold": 0.5,
        "edge_threshold": 0.5,
        "nms_radius_um": 5.0,
        "max_positive_voxel_fraction": 0.005,
    }
    hp.update(overrides)
    return hp


def _valid_val_metrics(**overrides) -> dict:
    vm = {
        "evaluation_completed_successfully": True,
        "validation_is_full_fold": True,
        "validation_samples_evaluated": 2,
        "validation_samples_total": 2,
        "num_datasets": 2,
        "predicted_nodes_total": 10,
        "predicted_edges_total": 5,
        "is_structural_zero": False,
        "adjusted_edge_jaccard": 0.75,
    }
    vm.update(overrides)
    return vm


def _make_eligible_checkpoint(unet3d, transformer, **overrides) -> dict:
    checkpoint = {
        "epoch": 3,
        "unet3d_state_dict": unet3d.state_dict(),
        "transformer_state_dict": transformer.state_dict(),
        "hyperparams": _valid_hyperparams(),
        "checkpoint_schema_version": checkpoint_manifest.CHECKPOINT_SCHEMA_VERSION,
        "training_code_sha": VALID_SHA40,
        "model_contract": MODEL_CONTRACT,
        "split_membership_sha256": VALID_SPLIT64,
        "val_metrics": _valid_val_metrics(),
    }
    checkpoint.update(overrides)
    return checkpoint


def _save_checkpoint(tmp_path, checkpoint, filename="epoch_3_val_score_0.7500.pt"):
    checkpoint_path = tmp_path / filename
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _valid_manifest_fields(checkpoint_path, checkpoint, **overrides) -> dict:
    """Hand-built manifest field dict matching checkpoint's REAL saved file
    hash -- used both for the write_checkpoint_manifest() happy path and
    (with overrides) for constructing deliberately forged/mismatched
    manifests independent of write_checkpoint_manifest()'s own validation."""
    val_metrics = checkpoint["val_metrics"]
    fields = {
        "schema_version": checkpoint_manifest.MANIFEST_SCHEMA_VERSION,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_code_sha": checkpoint["training_code_sha"],
        "split_membership_sha256": checkpoint["split_membership_sha256"],
        "model_contract": MODEL_CONTRACT,
        "epoch": checkpoint["epoch"],
        "validation_is_full_fold": val_metrics["validation_is_full_fold"],
        "validation_samples_evaluated": val_metrics["validation_samples_evaluated"],
        "validation_samples_total": val_metrics["validation_samples_total"],
        "num_datasets": val_metrics["num_datasets"],
        "predicted_nodes_total": val_metrics["predicted_nodes_total"],
        "predicted_edges_total": val_metrics["predicted_edges_total"],
        "is_structural_zero": val_metrics["is_structural_zero"],
        "adjusted_edge_jaccard": float(val_metrics["adjusted_edge_jaccard"]),
    }
    fields.update(overrides)
    return fields


def _write_raw_manifest(path, fields: dict) -> None:
    path.write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_valid_manifest(tmp_path, checkpoint_path, checkpoint, manifest_name="checkpoint_manifest.json"):
    manifest_path = tmp_path / manifest_name
    _write_raw_manifest(manifest_path, _valid_manifest_fields(checkpoint_path, checkpoint))
    return manifest_path


# ---------------------------------------------------------------------------
# F1 -- production submission-path tests
# ---------------------------------------------------------------------------

class _FakeFrameDataset(Dataset):
    """Deterministic, hermetic stand-in for CompetitionDataset's test-mode
    behavior -- produces the exact same batch dict shape (frame_t, frame_t1,
    sample_id, t_idx) without touching real Zarr data."""

    def __init__(self, sample_id, t_indices, shape=(1, 4, 32, 32)):
        self.sample_id = sample_id
        self.t_indices = list(t_indices)
        self.shape = shape

    def __len__(self):
        return len(self.t_indices)

    def __getitem__(self, idx):
        t_idx = self.t_indices[idx]
        gen = torch.Generator().manual_seed(abs(hash((self.sample_id, t_idx))) % (2**31))
        frame_t = torch.rand(self.shape, generator=gen)
        frame_t1 = torch.rand(self.shape, generator=gen)
        return {
            "frame_t": frame_t,
            "frame_t1": frame_t1,
            "sample_id": self.sample_id,
            "t_idx": t_idx,
            "metadata": {},
        }


class _FakePeakUNet3D(nn.Module):
    """Deterministic detection model: on call number `call_idx` (0-indexed,
    incrementing per forward()), places an isolated strong logit spike at
    each given (channel -> (z, y, x)) coordinate and leaves everything else
    far below threshold. Produces exactly one real NMS peak per specified
    channel per call -- makes canonical-graph-identity tests
    (F1.1/F1.2/F1.13/F1.14) fully deterministic instead of depending on a
    randomly-initialized model's real output."""

    def __init__(self, peak_coords_by_call: dict, shape=(4, 32, 32), feature_dim=16):
        super().__init__()
        self.peak_coords_by_call = peak_coords_by_call
        self.shape = shape
        self.feature_dim = feature_dim
        self.call_idx = 0
        self._dummy = nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def forward(self, x):
        batch = x.shape[0]
        z, y, xx = self.shape
        logits = torch.full((batch, 2, z, y, xx), -10.0)
        coords = self.peak_coords_by_call.get(self.call_idx, {})
        for ch, (zc, yc, xc) in coords.items():
            logits[0, ch, zc, yc, xc] = 10.0
        self.call_idx += 1
        features = torch.zeros((batch, self.feature_dim, z, y, xx))
        return logits, features


class _FakeAcceptAllEdgeTransformer(nn.Module):
    """Deterministic edge transformer: returns a fixed positive logit for
    every candidate (all-pairs) edge, so every real candidate is accepted by
    greedy_edge_assignment (subject to cardinality limits)."""

    def __init__(self, logit_value=10.0):
        super().__init__()
        self.logit_value = logit_value
        self._dummy = nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def forward(self, nodes_t, nodes_t1, features_t, features_t1, candidate_edges=None):
        n_t = nodes_t.shape[0]
        n_t1 = nodes_t1.shape[0]
        return torch.full((n_t * n_t1,), self.logit_value)


def _fake_loader(sample_id, t_indices, shape=(1, 4, 32, 32)):
    dataset = _FakeFrameDataset(sample_id, t_indices, shape=shape)
    return DataLoader(dataset, batch_size=1, shuffle=False)


class TestF1ProductionSubmissionPath:
    """Part F1: production submission-path tests against real
    run_sample_loader_inference()/run_submission_inference()."""

    def test_three_frame_two_window_continuity(self):
        """F1.1: 3 frames / 2 windows -> exactly 3 nodes, 2 edges, per-frame
        counts {0:1, 1:1, 2:1} -- proves PredictionGraphAssembler's
        canonical-identity rule is correctly wired through the shared
        pipeline (not re-verified here in isolation -- see
        tests/test_prediction_graph.py for that)."""
        unet3d = _FakePeakUNet3D({
            0: {0: (0, 1, 1), 1: (0, 1, 2)},  # window t=0: frame0 peak, frame1 peak
            1: {0: (0, 1, 5), 1: (0, 1, 3)},  # window t=1: frame1 peak (ignored), frame2 peak
        })
        transformer = _FakeAcceptAllEdgeTransformer()
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("sample_a", [0, 1])

        diag = run_sample_loader_inference(
            sample_id="sample_a", loader=loader, expected_pair_count=2,
            assembler=assembler, unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
        )
        assert diag["processed_pair_count"] == 2

        graphs = assembler.pred_graphs()
        graph = graphs["sample_a"]
        assert len(graph.node_ids()) == 3
        assert len(graph.edge_list()) == 2

        counts = {0: 0, 1: 0, 2: 0}
        for node_id in graph.node_ids():
            t = graph.nodes[node_id]["t"]
            counts[t] += 1
        assert counts == {0: 1, 1: 1, 2: 1}

    def test_non_identical_overlap_still_reuses_canonical_frame1_nodes(self):
        """F1.2: window B's channel-0 peak for frame 1 is at a DIFFERENT
        coordinate than window A's channel-1 peak for frame 1 -- the
        canonical node must still be window A's, never re-created."""
        unet3d = _FakePeakUNet3D({
            0: {0: (0, 1, 1), 1: (0, 5, 5)},   # window t=0: frame1 peak at (0,5,5)
            1: {0: (0, 9, 9), 1: (0, 2, 2)},   # window t=1: frame1 "peak" at (0,9,9) -- must be ignored
        })
        transformer = _FakeAcceptAllEdgeTransformer()
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("sample_a", [0, 1])

        run_sample_loader_inference(
            sample_id="sample_a", loader=loader, expected_pair_count=2,
            assembler=assembler, unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
        )
        graph = assembler.pred_graphs()["sample_a"]
        assert len(graph.node_ids()) == 3
        frame1_nodes = [n for n in graph.node_ids() if graph.nodes[n]["t"] == 1]
        assert len(frame1_nodes) == 1
        node = graph.nodes[frame1_nodes[0]]
        assert (node["y"], node["x"]) == (5, 5), "frame 1's canonical node must be window A's peak, not window B's"

    def test_empty_test_zarrs_raises(self, tmp_path):
        """F1.3."""
        unet3d, transformer = _make_small_models()
        with pytest.raises(RuntimeError, match="empty"):
            run_submission_inference(
                test_dir=tmp_path, test_zarrs=[], unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_duplicate_sample_ids_raise(self, tmp_path, monkeypatch):
        """F1.4."""
        unet3d, transformer = _make_small_models()

        def fake_build(_test_dir, sample_id):
            return _FakeFrameDataset(sample_id, [0])

        monkeypatch.setattr(submission_pipeline_module, "build_test_dataset", fake_build)
        with pytest.raises(RuntimeError, match="duplicate"):
            run_submission_inference(
                test_dir=tmp_path,
                test_zarrs=[tmp_path / "dsA" / "same.zarr", tmp_path / "dsB" / "same.zarr"],
                unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_zero_frame_pairs_raises(self, tmp_path, monkeypatch):
        """F1.5."""
        unet3d, transformer = _make_small_models()

        def fake_build(_test_dir, sample_id):
            return _FakeFrameDataset(sample_id, [])  # zero pairs

        monkeypatch.setattr(submission_pipeline_module, "build_test_dataset", fake_build)
        with pytest.raises(RuntimeError, match="ZERO frame pairs"):
            run_submission_inference(
                test_dir=tmp_path, test_zarrs=[tmp_path / "dsA.zarr"],
                unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_mismatched_sample_id_raises(self):
        """F1.6."""
        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer()
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("wrong_sample", [0])

        with pytest.raises(RuntimeError, match="mismatched or mixed-sample"):
            run_sample_loader_inference(
                sample_id="expected_sample", loader=loader, expected_pair_count=1,
                assembler=assembler, unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_mixed_sample_loader_raises(self):
        """F1.7: a loader whose batches span more than one sample_id."""
        class _MixedDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                sample_id = "sample_a" if idx == 0 else "sample_b"
                return {
                    "frame_t": torch.rand(1, 4, 32, 32), "frame_t1": torch.rand(1, 4, 32, 32),
                    "sample_id": sample_id, "t_idx": 0, "metadata": {},
                }

        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}, 1: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer()
        assembler = PredictionGraphAssembler()
        loader = DataLoader(_MixedDataset(), batch_size=1, shuffle=False)

        with pytest.raises(RuntimeError, match="mismatched or mixed-sample"):
            run_sample_loader_inference(
                sample_id="sample_a", loader=loader, expected_pair_count=2,
                assembler=assembler, unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_batch_size_greater_than_one_raises(self):
        """F1.8."""
        dataset = _FakeFrameDataset("sample_a", [0, 1])
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        unet3d, transformer = _make_small_models()
        assembler = PredictionGraphAssembler()

        with pytest.raises(RuntimeError, match="batch size 1"):
            run_sample_loader_inference(
                sample_id="sample_a", loader=loader, expected_pair_count=1,
                assembler=assembler, unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_expected_processed_pair_mismatch_raises(self):
        """F1.9: expected_pair_count deliberately wrong vs. the real loader length."""
        unet3d, transformer = _make_small_models()
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("sample_a", [0, 1])

        with pytest.raises(RuntimeError, match="does not match expected_pair_count"):
            run_sample_loader_inference(
                sample_id="sample_a", loader=loader, expected_pair_count=5,
                assembler=assembler, unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_required_sample_zero_nodes_raises(self, tmp_path, monkeypatch):
        """F1.10: a model producing no detections anywhere for a required
        sample must raise, not silently omit it."""
        class _AllBackgroundUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                batch = x.shape[0]
                # Perfectly uniform volume -- no peak, adaptive-threshold
                # branch has nothing to select either (all voxels tie).
                logits = torch.zeros((batch, 2, 4, 32, 32))
                features = torch.zeros((batch, 16, 4, 32, 32))
                return logits, features

        unet3d = _AllBackgroundUNet()
        transformer = _FakeAcceptAllEdgeTransformer()

        def fake_build(_test_dir, sample_id):
            return _FakeFrameDataset(sample_id, [0])

        monkeypatch.setattr(submission_pipeline_module, "build_test_dataset", fake_build)
        with pytest.raises(RuntimeError, match="ZERO nodes"):
            run_submission_inference(
                test_dir=tmp_path, test_zarrs=[tmp_path / "dsA.zarr"],
                unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_zero_total_edges_raises(self, tmp_path, monkeypatch):
        """F1.11: nodes exist but the transformer rejects every candidate edge."""
        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})

        class _RejectAllEdgeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, nodes_t, nodes_t1, features_t, features_t1, candidate_edges=None):
                return torch.full((nodes_t.shape[0] * nodes_t1.shape[0],), -10.0)

        transformer = _RejectAllEdgeTransformer()

        def fake_build(_test_dir, sample_id):
            return _FakeFrameDataset(sample_id, [0])

        monkeypatch.setattr(submission_pipeline_module, "build_test_dataset", fake_build)
        with pytest.raises(RuntimeError, match="ZERO"):
            run_submission_inference(
                test_dir=tmp_path, test_zarrs=[tmp_path / "dsA.zarr"],
                unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_multiple_samples_remain_isolated(self, tmp_path, monkeypatch):
        """F1.12: two independent samples' node/edge counts and identities
        never bleed into each other."""
        call_state = {"n": 0}

        def fake_build(_test_dir, sample_id):
            return _FakeFrameDataset(sample_id, [0])

        peak_maps = {
            "dsA": (0, 1, 1),
            "dsB": (0, 20, 20),
        }

        class _PerSampleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))
                self.calls = []

            def forward(self, x):
                # Determine which sample via call order (dsA then dsB, sorted).
                sample_order = ["dsA", "dsB"]
                sample_id = sample_order[call_state["n"]]
                call_state["n"] += 1
                logits = torch.full((1, 2, 4, 32, 32), -10.0)
                z, y, xc = peak_maps[sample_id]
                logits[0, 0, z, y, xc] = 10.0
                logits[0, 1, z, y, xc] = 10.0
                features = torch.zeros((1, 16, 4, 32, 32))
                return logits, features

        unet3d = _PerSampleUNet()
        transformer = _FakeAcceptAllEdgeTransformer()
        monkeypatch.setattr(submission_pipeline_module, "build_test_dataset", fake_build)

        pred_graphs, diagnostics = run_submission_inference(
            test_dir=tmp_path, test_zarrs=[tmp_path / "dsA.zarr", tmp_path / "dsB.zarr"],
            unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
        )
        assert set(pred_graphs.keys()) == {"dsA", "dsB"}
        assert len(pred_graphs["dsA"].node_ids()) == 2
        assert len(pred_graphs["dsB"].node_ids()) == 2
        node = pred_graphs["dsA"].nodes[pred_graphs["dsA"].node_ids()[0]]
        assert (node["y"], node["x"]) == (1, 1)
        node_b = pred_graphs["dsB"].nodes[pred_graphs["dsB"].node_ids()[0]]
        assert (node_b["y"], node_b["x"]) == (20, 20)

    def test_out_of_order_windows_raise(self):
        """F1.13: t_idx sequence 0 then 2 (skipping 1) must raise."""
        class _SkippingDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                t_idx = [0, 2][idx]
                return {
                    "frame_t": torch.rand(1, 4, 32, 32), "frame_t1": torch.rand(1, 4, 32, 32),
                    "sample_id": "sample_a", "t_idx": t_idx, "metadata": {},
                }

        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}, 1: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer()
        assembler = PredictionGraphAssembler()
        loader = DataLoader(_SkippingDataset(), batch_size=1, shuffle=False)

        with pytest.raises(RuntimeError, match="chronological window-order"):
            run_sample_loader_inference(
                sample_id="sample_a", loader=loader, expected_pair_count=2,
                assembler=assembler, unet3d=unet3d, transformer=transformer,
                device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
            )

    def test_complete_valid_coverage_passes(self, tmp_path, monkeypatch):
        """F1.14: full happy path across two required samples, structural
        circuit breakers and diagnostics all satisfied."""
        def fake_build(_test_dir, sample_id):
            return _FakeFrameDataset(sample_id, [0, 1])

        unet3d = _FakePeakUNet3D({
            0: {0: (0, 1, 1), 1: (0, 1, 2)}, 1: {0: (0, 1, 5), 1: (0, 1, 3)},
            2: {0: (0, 1, 1), 1: (0, 1, 2)}, 3: {0: (0, 1, 5), 1: (0, 1, 3)},
        })
        transformer = _FakeAcceptAllEdgeTransformer()
        monkeypatch.setattr(submission_pipeline_module, "build_test_dataset", fake_build)

        pred_graphs, diagnostics = run_submission_inference(
            test_dir=tmp_path, test_zarrs=[tmp_path / "dsA.zarr", tmp_path / "dsB.zarr"],
            unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
        )
        assert diagnostics["required_dataset_ids"] == ["dsA", "dsB"]
        assert diagnostics["total_unique_nodes"] == 6
        assert diagnostics["total_unique_edges"] == 4
        assert diagnostics["per_sample"]["dsA"]["processed_pair_count"] == 2
        assert diagnostics["per_sample"]["dsB"]["processed_pair_count"] == 2
        assert diagnostics["per_sample"]["dsA"]["unique_node_count"] == 3
        assert diagnostics["per_sample"]["dsB"]["unique_node_count"] == 3


class TestF1BuildTestDataset:
    """Part A1: build_test_dataset()'s exact required CompetitionDataset construction."""

    def test_build_test_dataset_construction(self, tmp_path):
        (tmp_path / "dsA.zarr").mkdir()
        dataset = build_test_dataset(tmp_path, "dsA")
        assert dataset.data_dir == tmp_path
        assert dataset.split_type == "test"
        assert dataset.normalize is True
        assert dataset.anisotropy == (4.0, 1.0, 1.0)
        assert dataset.physical_voxel_size == (1.625, 0.40625, 0.40625)
        assert dataset.sample_ids == ["dsA"]
        assert dataset.filter_unannotated_pairs is False
        assert dataset.annotation_pair_stats is None


# ---------------------------------------------------------------------------
# F2 -- sigmoid contract tests (functional; AST checks live in
# tests/test_p05_double_sigmoid_fix.py)
# ---------------------------------------------------------------------------

class TestF2SigmoidContract:
    def test_production_shared_submission_converts_logits_exactly_once(self, monkeypatch):
        """F2.1: capture the actual edge_probs tensor passed into
        greedy_edge_assignment and assert it equals sigmoid(logit) applied
        EXACTLY once -- not sigmoid(sigmoid(logit))."""
        fixed_logit = -2.0
        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer(logit_value=fixed_logit)
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("sample_a", [0])

        captured = {}
        real_assign = submission_pipeline_module.greedy_edge_assignment

        def spy_assign(edge_probs, *args, **kwargs):
            captured["edge_probs"] = edge_probs.clone()
            return real_assign(edge_probs, *args, **kwargs)

        monkeypatch.setattr(submission_pipeline_module, "greedy_edge_assignment", spy_assign)

        run_sample_loader_inference(
            sample_id="sample_a", loader=loader, expected_pair_count=1,
            assembler=assembler, unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(),
        )

        expected_once = torch.sigmoid(torch.tensor(fixed_logit))
        expected_twice = torch.sigmoid(torch.sigmoid(torch.tensor(fixed_logit)))
        assert torch.allclose(captured["edge_probs"], expected_once, atol=1e-6)
        assert not torch.allclose(captured["edge_probs"], expected_twice, atol=1e-3)

    def test_negative_logit_remains_below_threshold_after_one_sigmoid(self):
        """F2.2: sigmoid(-3.0) ~= 0.047 < 0.5 -- edge correctly rejected."""
        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer(logit_value=-3.0)
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("sample_a", [0])

        diag = run_sample_loader_inference(
            sample_id="sample_a", loader=loader, expected_pair_count=1,
            assembler=assembler, unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(edge_threshold=0.5),
        )
        assert diag["total_accepted_edges"] == 0

    def test_double_sigmoid_would_flip_negative_logit_above_half(self):
        """F2.3/F2.4: mathematical proof any negative logit's DOUBLE
        sigmoid always exceeds 0.5 (sigmoid is monotonic increasing and
        sigmoid(0)=0.5; sigmoid(x) for x<0 lies in (0, 0.5), and sigmoid of
        anything in (0, 0.5) lies in (0.5, sigmoid(0.5))) -- this is exactly
        why F2.1's single-application proof matters: a reintroduced double
        Sigmoid would silently flip every negative-logit edge to
        'accepted'."""
        for logit in (-0.1, -1.0, -3.0, -10.0):
            once = torch.sigmoid(torch.tensor(float(logit)))
            twice = torch.sigmoid(once)
            assert once < 0.5
            assert twice > 0.5, f"double-sigmoid of negative logit {logit} must exceed 0.5"

    def test_positive_logit_accepted(self):
        """F2.5."""
        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer(logit_value=3.0)
        assembler = PredictionGraphAssembler()
        loader = _fake_loader("sample_a", [0])

        diag = run_sample_loader_inference(
            sample_id="sample_a", loader=loader, expected_pair_count=1,
            assembler=assembler, unet3d=unet3d, transformer=transformer,
            device=torch.device("cpu"), hyperparams=_valid_hyperparams(edge_threshold=0.5),
        )
        assert diag["total_accepted_edges"] == 1

    def test_no_sigmoid_module_in_edge_scorer(self):
        """F2.6 (duplicate-but-independent of test_p05's own coverage)."""
        transformer = SimpleNodeTransformer(hidden_dim=32, num_heads=4, num_blocks=1, feature_dim=16)
        for module in transformer.edge_scorer.modules():
            assert not isinstance(module, nn.Sigmoid)


# ---------------------------------------------------------------------------
# F2 (AST) -- production call-site regression coverage for the shared
# submission pipeline. This lives here (not in tests/test_p05_double_sigmoid_fix.py,
# which is out of the frozen 13-path scope and must stay byte-identical to
# baseline) -- self-contained AST helpers, independent of any helper defined
# in that other file.
# ---------------------------------------------------------------------------

def _get_module_source(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def _get_function_source(file_path: str, function_name: str) -> str:
    """Top-level (module-scope) function source -- src/submission_pipeline.py
    owns the shared inference call site as a plain function, not a class method."""
    source = _get_module_source(file_path)
    tree = ast.parse(source, filename=file_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"{function_name} not found in {file_path}")


def _has_call(source: str, name_fragment: str) -> bool:
    """True if a genuine ast.Call node's function name/attribute contains
    name_fragment (e.g. 'run_submission_inference' or 'greedy_edge_assignment')."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and name_fragment in func.attr:
                return True
            if isinstance(func, ast.Name) and name_fragment in func.id:
                return True
    return False


def _is_source_or_source_dot_float(node: ast.AST, source_name: str) -> bool:
    """True if node is the bare Name source_name, OR the Part A4 dtype-safe
    call pattern `source_name.float()` (a zero-arg .float() call on that
    exact Name) -- e.g. torch.sigmoid(edge_logits.float())."""
    if isinstance(node, ast.Name) and node.id == source_name:
        return True
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        func = node.func
        if (
            isinstance(func, ast.Attribute) and func.attr == "float"
            and isinstance(func.value, ast.Name) and func.value.id == source_name
        ):
            return True
    return False


def _count_exact_sigmoid_assignment(source: str, target_name: str, source_name: str) -> int:
    """Count genuine `target_name = torch.sigmoid(source_name)` (or the
    dtype-safe `source_name.float()` variant) assignments as real ast.Assign
    nodes -- a comment or docstring containing this exact text is not an
    ast.Assign node and will not be counted."""
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
                    if is_sigmoid and len(val.args) == 1 and _is_source_or_source_dot_float(val.args[0], source_name):
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


class TestF2SubmissionPipelineCallSite:
    """The real transformer/sigmoid/greedy_edge_assignment call site --
    src/submission_pipeline.py's run_sample_loader_inference() -- must
    follow the exact one-Sigmoid-before-assignment contract."""

    def test_run_sample_loader_inference(self):
        source = _get_function_source("src/submission_pipeline.py", "run_sample_loader_inference")

        assert _has_call(source, "transformer"), "run_sample_loader_inference must call the transformer"
        assert _has_call(source, "greedy_edge_assignment"), \
            "run_sample_loader_inference must call greedy_edge_assignment"
        assert _count_exact_sigmoid_assignment(source, "edge_probs", "edge_logits") == 1, (
            "run_sample_loader_inference must apply exactly one torch.sigmoid(edge_logits) "
            "(assigned to edge_probs) before greedy_edge_assignment"
        )
        assert _first_positional_arg_name(source, "greedy_edge_assignment") == "edge_probs", (
            "greedy_edge_assignment must receive the sigmoided edge_probs, not raw edge_logits"
        )

    def test_no_double_sigmoid_in_detection_path(self):
        source = _get_function_source("src/submission_pipeline.py", "run_sample_loader_inference")
        assert _count_exact_sigmoid_assignment(source, "detection_probs", "detection_logits") == 1, (
            "run_sample_loader_inference must apply exactly one torch.sigmoid(detection_logits) "
            "(assigned to detection_probs)"
        )


class TestF2ProductionCallersUseSharedPipeline:
    """generate_submission.py and inference_kernel.py must delegate all
    transformer/sigmoid/greedy_edge_assignment/graph-construction logic to
    src/submission_pipeline.py -- neither may duplicate it directly."""

    def _assert_shared_pipeline_caller_pattern(self, file_path: str):
        source = _get_module_source(file_path)

        assert _has_call(source, "run_submission_inference"), (
            f"{file_path} must call the shared src.submission_pipeline.run_submission_inference()"
        )
        assert not _has_call(source, "transformer"), (
            f"{file_path} must not call the transformer directly -- that belongs in "
            f"src/submission_pipeline.py only"
        )
        assert not _has_call(source, "greedy_edge_assignment"), (
            f"{file_path} must not call greedy_edge_assignment directly -- that belongs in "
            f"src/submission_pipeline.py only"
        )
        assert _count_exact_sigmoid_assignment(source, "edge_probs", "edge_logits") == 0, (
            f"{file_path} must not apply edge-logit Sigmoid directly -- that belongs in "
            f"src/submission_pipeline.py only"
        )
        assert not _has_call(source, "add_node"), f"{file_path} must not construct graphs directly"
        assert not _has_call(source, "add_edge"), f"{file_path} must not construct graphs directly"
        assert not _has_call(source, "IndexedRXGraph"), f"{file_path} must not construct graphs directly"

    def test_generate_submission_py(self):
        self._assert_shared_pipeline_caller_pattern("generate_submission.py")

    def test_inference_kernel_py(self):
        self._assert_shared_pipeline_caller_pattern("kaggle_kernel_inference/inference_kernel.py")


# ---------------------------------------------------------------------------
# F4 -- manifest discovery and validation tests
# ---------------------------------------------------------------------------

class TestF4ManifestDiscovery:
    def test_no_manifest_raises(self, tmp_path):
        """F4.1."""
        with pytest.raises(RuntimeError, match="No checkpoint_manifest.json"):
            find_single_manifest(tmp_path)

    def test_multiple_manifests_lists_all(self, tmp_path):
        """F4.2."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
        (tmp_path / "b" / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Multiple") as exc_info:
            find_single_manifest(tmp_path)
        assert "a" in str(exc_info.value) and "b" in str(exc_info.value)

    def test_find_single_manifest_success(self, tmp_path):
        (tmp_path / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
        found = find_single_manifest(tmp_path)
        assert found == tmp_path / "checkpoint_manifest.json"


class TestF4ManifestValidation:
    """F4.3-31: load_verified_checkpoint()'s full fail-closed verification order."""

    @pytest.fixture
    def eligible_setup(self, tmp_path):
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        return tmp_path, checkpoint, checkpoint_path

    def test_valid_pair_loads(self, eligible_setup):
        """F4.14/F4.28: valid regular sibling checkpoint loads successfully."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        loaded_checkpoint, manifest, loaded_path = load_verified_checkpoint(
            manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu",
        )
        assert loaded_path == checkpoint_path
        assert manifest["checkpoint_sha256"] == sha256_file(checkpoint_path)
        assert loaded_checkpoint["epoch"] == 3

    def test_exact_expected_manifest_field_set_passes(self, eligible_setup):
        """F4.31."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        fields = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert set(fields.keys()) == checkpoint_manifest._REQUIRED_MANIFEST_FIELDS

    def test_malformed_json_raises(self, tmp_path, eligible_setup):
        """F4.3."""
        _, checkpoint, checkpoint_path = eligible_setup
        manifest_path = tmp_path / "checkpoint_manifest.json"
        manifest_path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_duplicate_checkpoint_sha_key_raises(self, eligible_setup):
        """F4.4."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint)
        raw = json.dumps(fields, indent=2, sort_keys=True)
        # Inject a literal duplicate key by hand -- json.dumps never
        # produces this itself, so we splice it directly into the text.
        raw = raw.replace(
            '"checkpoint_sha256":', '"checkpoint_sha256_DUP_MARKER":', 1
        )
        raw = raw.replace('"checkpoint_sha256_DUP_MARKER":', '"checkpoint_sha256": "0" , "checkpoint_sha256":', 1)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        manifest_path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate JSON key"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_duplicate_training_sha_key_raises(self, eligible_setup):
        """F4.5."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        raw = f'{{"training_code_sha": "{VALID_SHA40}", "training_code_sha": "{VALID_SHA40_B}"}}'
        manifest_path = tmp_path / "checkpoint_manifest.json"
        manifest_path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate JSON key"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_missing_field_raises(self, eligible_setup):
        """F4.6."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint)
        del fields["epoch"]
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="missing required field"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_unexpected_field_raises(self, eligible_setup):
        """F4.7."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, extra_field="surprise")
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="unexpected field"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_wrong_type_raises(self, eligible_setup):
        """F4.8."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, epoch="three")
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="epoch"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_bool_as_int_rejected(self, eligible_setup):
        """F4.9."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, epoch=True)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="non-bool"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_invalid_uppercase_hash_rejected(self, eligible_setup):
        """F4.10."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, checkpoint_sha256=sha256_file(checkpoint_path).upper())
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="checkpoint_sha256"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    @pytest.mark.parametrize("bad_name", ["/abs/path.pt", "sub/dir.pt", "sub\\dir.pt", "..\\escape.pt", "no_pt_suffix"])
    def test_unsafe_checkpoint_paths_rejected(self, eligible_setup, bad_name):
        """F4.11."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, checkpoint_file=bad_name)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="checkpoint_file"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_missing_checkpoint_file_raises(self, eligible_setup):
        """F4.12."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, checkpoint_file="does_not_exist.pt")
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(FileNotFoundError):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_symlink_checkpoint_rejected(self, eligible_setup, tmp_path):
        """F4.13: symlink escape/redirect rejection. Skip only on genuine
        OS symlink-creation denial (e.g. unprivileged Windows)."""
        _, checkpoint, checkpoint_path = eligible_setup
        link_path = tmp_path / "linked.pt"
        try:
            link_path.symlink_to(checkpoint_path)
        except OSError as e:
            pytest.skip(f"OS denied symlink creation: {e}")
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, checkpoint_file="linked.pt")
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="symlink"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_hash_mismatch_raises(self, eligible_setup):
        """F4.15."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, checkpoint_sha256="0" * 64)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="hash mismatch"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_source_mismatch_raises(self, eligible_setup):
        """F4.16."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        with pytest.raises(ValueError, match="does not match expected_source_sha"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40_B, map_location="cpu")

    def test_contract_mismatch_raises(self, tmp_path):
        """F4.17: checkpoint's OWN model_contract field disagrees with the
        (schema-required, therefore always-valid) manifest value -- caught
        at eligibility (step 13), since a manifest can never itself declare
        an invalid model_contract literal."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, model_contract="wrong_contract_v0")
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        with pytest.raises(ValueError, match="model_contract"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_legacy_missing_contract_raises(self, tmp_path):
        """F4.18."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        del checkpoint["model_contract"]
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        # _valid_manifest_fields() builds a manifest matching the exact
        # 15-field schema (using the MODEL_CONTRACT constant directly, not
        # checkpoint["model_contract"]) -- the manifest itself is schema-valid,
        # but the referenced checkpoint's own dict is missing the key, which
        # load_verified_checkpoint() must catch at the "required checkpoint
        # keys present" step.
        fields = _valid_manifest_fields(checkpoint_path, checkpoint)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="missing required key"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_split_mismatch_raises(self, tmp_path):
        """F4.19: checkpoint's real split differs from manifest's declared
        split -- caught at step 14 (manifest/checkpoint exact-match)."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, split_membership_sha256=VALID_SPLIT64_B)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, split_membership_sha256=VALID_SPLIT64)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="split_membership_sha256"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_epoch_mismatch_raises(self, tmp_path):
        """F4.20."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=7)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, epoch=3)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="epoch"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_invalid_validation_coverage_raises(self, eligible_setup):
        """F4.21: evaluated != total in the manifest itself."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, validation_samples_evaluated=1, validation_samples_total=2)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="validation_samples_evaluated"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_mismatched_num_datasets_raises(self, eligible_setup):
        """F4.22."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, num_datasets=99)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="num_datasets"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_zero_nodes_edges_raises(self, eligible_setup):
        """F4.23."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, predicted_nodes_total=0)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="predicted_nodes_total"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_structural_zero_raises(self, eligible_setup):
        """F4.24."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, is_structural_zero=True)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="is_structural_zero"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.5])
    def test_nan_inf_negative_metric_raises(self, eligible_setup, bad_value):
        """F4.25."""
        tmp_path, checkpoint, checkpoint_path = eligible_setup
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, adjusted_edge_jaccard=bad_value)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="adjusted_edge_jaccard"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_missing_state_dicts_raises(self, tmp_path):
        """F4.26."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        del checkpoint["unet3d_state_dict"]
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        with pytest.raises(ValueError, match="missing required key"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_invalid_hyperparams_raises(self, tmp_path):
        """F4.27."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, hyperparams={"detection_threshold": 0.5})
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        with pytest.raises(ValueError, match="invalid inference hyperparameters"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_torch_load_only_after_hash_verification(self, tmp_path):
        """F4.29: a checkpoint 'file' that isn't even a real torch archive
        must fail with a HASH mismatch (not a torch.load parse crash) when
        the manifest declares a wrong hash for it -- proving hash
        verification runs strictly before torch.load()."""
        garbage_path = tmp_path / "garbage.pt"
        garbage_path.write_bytes(b"not a real torch checkpoint at all")
        fields = {
            "schema_version": checkpoint_manifest.MANIFEST_SCHEMA_VERSION,
            "checkpoint_file": "garbage.pt",
            "checkpoint_sha256": "0" * 64,  # deliberately wrong
            "training_code_sha": VALID_SHA40,
            "split_membership_sha256": VALID_SPLIT64,
            "model_contract": MODEL_CONTRACT,
            "epoch": 1,
            "validation_is_full_fold": True,
            "validation_samples_evaluated": 1,
            "validation_samples_total": 1,
            "num_datasets": 1,
            "predicted_nodes_total": 1,
            "predicted_edges_total": 1,
            "is_structural_zero": False,
            "adjusted_edge_jaccard": 0.5,
        }
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="hash mismatch"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_forged_ineligible_checkpoint_rejected(self, tmp_path):
        """F4.30: a checkpoint whose file hash genuinely matches the
        manifest (not forged at the hash level) but whose OWN content is
        deployment-ineligible (structural-zero val_metrics) must still be
        rejected at load time."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(
            unet3d, transformer, val_metrics=_valid_val_metrics(is_structural_zero=True),
        )
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, is_structural_zero=True)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="is_structural_zero"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")


# ---------------------------------------------------------------------------
# F5 -- checkpoint writing and eligibility tests
# ---------------------------------------------------------------------------

class TestF5CheckpointWritingAndEligibility:
    def test_checkpoint_metadata_embedded(self, tmp_path):
        """F5.1."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        assert checkpoint["checkpoint_schema_version"] == checkpoint_manifest.CHECKPOINT_SCHEMA_VERSION
        assert checkpoint["training_code_sha"] == VALID_SHA40
        assert checkpoint["model_contract"] == MODEL_CONTRACT

    def test_healthy_full_fold_writes_manifest(self, tmp_path):
        """F5.2."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        assert deployment_eligibility_errors(checkpoint) == []
        manifest_path = write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
        assert manifest_path.exists()

    @pytest.mark.parametrize("overrides", [
        {"val_metrics": _valid_val_metrics(validation_is_full_fold=False)},
        {"training_code_sha": "not-a-sha"},
        {"split_membership_sha256": "not-a-hash"},
        {"val_metrics": _valid_val_metrics(predicted_nodes_total=0)},
        {"val_metrics": _valid_val_metrics(predicted_edges_total=0)},
        {"val_metrics": _valid_val_metrics(is_structural_zero=True)},
        {"val_metrics": _valid_val_metrics(validation_samples_evaluated=1, validation_samples_total=2)},
        {"hyperparams": {"detection_threshold": 0.5}},
    ])
    def test_ineligible_checkpoints_write_no_manifest(self, tmp_path, overrides):
        """F5.3."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, **overrides)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        errors = deployment_eligibility_errors(checkpoint)
        assert errors != []
        with pytest.raises(ValueError, match="not eligible"):
            write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
        assert not (tmp_path / checkpoint_manifest.MANIFEST_FILENAME).exists()

    def test_eligible_newer_checkpoint_replaces_manifest(self, tmp_path):
        """F5.4."""
        unet3d, transformer = _make_small_models()
        checkpoint_v1 = _make_eligible_checkpoint(unet3d, transformer, epoch=1)
        path_v1 = _save_checkpoint(tmp_path, checkpoint_v1, "epoch_1_val_score_0.5000.pt")
        write_checkpoint_manifest(path_v1, checkpoint=checkpoint_v1)

        checkpoint_v2 = _make_eligible_checkpoint(unet3d, transformer, epoch=2)
        path_v2 = _save_checkpoint(tmp_path, checkpoint_v2, "epoch_2_val_score_0.8000.pt")
        manifest_path = write_checkpoint_manifest(path_v2, checkpoint=checkpoint_v2)

        active = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert active["checkpoint_file"] == "epoch_2_val_score_0.8000.pt"
        assert active["epoch"] == 2

    def test_last_checkpoint_never_manifested(self, tmp_path):
        """F5.5/F6.7: last_checkpoint.pt (train_loss shape, no val_metrics)
        is structurally ineligible for a manifest -- deployment_eligibility_errors()
        catches it via a missing/non-dict val_metrics."""
        checkpoint = {
            "epoch": 1, "train_loss": 0.5,
            "checkpoint_schema_version": checkpoint_manifest.CHECKPOINT_SCHEMA_VERSION,
            "training_code_sha": VALID_SHA40, "model_contract": MODEL_CONTRACT,
            "hyperparams": _valid_hyperparams(),
        }
        errors = deployment_eligibility_errors(checkpoint)
        assert any("val_metrics" in e for e in errors)

    def test_forged_ineligible_checkpoint_rejected_during_load(self, tmp_path):
        """F5.6 (mirrors F4.30)."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, model_contract="forged")
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        fields = _valid_manifest_fields(checkpoint_path, checkpoint)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="model_contract"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")


# ---------------------------------------------------------------------------
# Explicit evaluation-success provenance (correction of adversarial finding
# 23): a checkpoint whose validation evaluation crashed or had no usable GT
# must be rejected on that basis alone, not incidentally via some other
# missing field -- val_metrics['evaluation_completed_successfully'] must be
# literal True, checked independently of adjusted_edge_jaccard's value.
# ---------------------------------------------------------------------------

class TestEvaluationSuccessProvenance:
    def test_missing_provenance_is_ineligible(self, tmp_path):
        unet3d, transformer = _make_small_models()
        val_metrics = _valid_val_metrics()
        del val_metrics["evaluation_completed_successfully"]
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, val_metrics=val_metrics)
        errors = deployment_eligibility_errors(checkpoint)
        assert any("evaluation_completed_successfully" in e for e in errors)

    @pytest.mark.parametrize("bad_value", [False, 1, "true", 0, "True", None])
    def test_non_true_provenance_is_ineligible(self, bad_value):
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(
            unet3d, transformer,
            val_metrics=_valid_val_metrics(evaluation_completed_successfully=bad_value),
        )
        errors = deployment_eligibility_errors(checkpoint)
        assert any("evaluation_completed_successfully" in e for e in errors)

    def test_fully_populated_fallback_metrics_still_ineligible(self, tmp_path):
        """Even when every OTHER metric field is a real, individually-valid,
        positive-looking value (i.e. exactly what a genuine 'fully populated'
        fallback dict could look like), evaluation_completed_successfully=False
        alone must be sufficient to reject it -- proves the check is not
        bypassable by having a rich-looking but fabricated fallback dict."""
        unet3d, transformer = _make_small_models()
        rich_but_fallback_metrics = _valid_val_metrics(
            evaluation_completed_successfully=False,
            validation_is_full_fold=True,
            validation_samples_evaluated=10,
            validation_samples_total=10,
            num_datasets=10,
            predicted_nodes_total=500,
            predicted_edges_total=300,
            is_structural_zero=False,
            adjusted_edge_jaccard=0.85,
        )
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, val_metrics=rich_but_fallback_metrics)
        errors = deployment_eligibility_errors(checkpoint)
        assert any("evaluation_completed_successfully" in e for e in errors)

    def test_legitimate_zero_score_full_fold_remains_eligible(self):
        """A REAL, successful, full-fold evaluation that happens to score
        exactly adjusted_edge_jaccard=0.0 (a genuinely terrible but honestly-
        measured model) must NOT be confused with a fallback -- fallback
        status must never be inferred from the score value."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(
            unet3d, transformer,
            val_metrics=_valid_val_metrics(evaluation_completed_successfully=True, adjusted_edge_jaccard=0.0),
        )
        errors = deployment_eligibility_errors(checkpoint)
        assert errors == []

    def test_eligibility_error_names_evaluation_failure(self):
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(
            unet3d, transformer,
            val_metrics=_valid_val_metrics(evaluation_completed_successfully=False),
        )
        errors = deployment_eligibility_errors(checkpoint)
        matching = [e for e in errors if "evaluation_completed_successfully" in e]
        assert len(matching) == 1
        assert "fallback" in matching[0].lower() or "fail" in matching[0].lower()

    def test_save_checkpoint_cannot_manifest_fallback_metrics(self, tmp_path):
        """Real production-path proof: TrainingLoop.save_checkpoint() with
        genuinely fallback-shaped val_metrics (evaluation_completed_successfully=False)
        creates no manifest at all, even though every other field looks
        individually valid."""
        loop = _make_training_loop_for_checkpoint_save(tmp_path)
        fallback_metrics = _valid_val_metrics(evaluation_completed_successfully=False)
        loop.save_checkpoint(epoch=1, metrics=fallback_metrics)

        manifest_path = tmp_path / checkpoint_manifest.MANIFEST_FILENAME
        assert not manifest_path.exists()
        checkpoint_path = tmp_path / "epoch_1_val_score_0.7500.pt"
        assert checkpoint_path.exists()  # checkpoint itself still saved, just unmanifested

    def test_validate_epoch_success_path_sets_provenance_true(self, tmp_path, monkeypatch):
        """Real end-to-end validate_epoch() call (hermetic: fake val_loader,
        deterministic detection/edge models, monkeypatched GT load/evaluate)
        proving the SUCCESS path's returned dict has
        evaluation_completed_successfully=True."""
        import src.train as train_module

        (tmp_path / "sample_a.geff").write_bytes(b"fake geff placeholder")
        monkeypatch.setattr(
            train_module, "load_geff_ground_truth",
            lambda path: (object(), object()),
        )
        monkeypatch.setattr(
            train_module, "evaluate_submission",
            lambda pred_graphs, gt_graphs, gt_metadata=None: {
                'edge_jaccard': 0.5, 'adjusted_edge_jaccard': 0.5, 'division_jaccard': 0.0,
                'score': 0.5, 'num_pred_nodes_total': 1, 'num_gt_nodes_total': 1, 'num_datasets': 1,
            },
        )

        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer()
        loop = train_module.TrainingLoop.__new__(train_module.TrainingLoop)
        loop.unet3d = unet3d
        loop.transformer = transformer
        loop.device = torch.device("cpu")
        loop.data_dir = tmp_path
        loop.hyperparams = _valid_hyperparams()
        loop.val_loader = _fake_loader("sample_a", [0])
        loop.epoch_fallback_counts = {'evaluation_failure': 0}
        loop._amp_enabled = False

        result = loop.validate_epoch()
        assert result['evaluation_completed_successfully'] is True

    def test_validate_epoch_exception_fallback_sets_provenance_false(self, tmp_path, monkeypatch):
        """Real end-to-end validate_epoch() call where evaluate_submission()
        raises -- the exception-fallback branch's returned dict must have
        evaluation_completed_successfully=False."""
        import src.train as train_module

        (tmp_path / "sample_a.geff").write_bytes(b"fake geff placeholder")
        monkeypatch.setattr(
            train_module, "load_geff_ground_truth",
            lambda path: (object(), object()),
        )

        def raising_evaluate(*args, **kwargs):
            raise RuntimeError("simulated evaluation crash")

        monkeypatch.setattr(train_module, "evaluate_submission", raising_evaluate)

        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer()
        loop = train_module.TrainingLoop.__new__(train_module.TrainingLoop)
        loop.unet3d = unet3d
        loop.transformer = transformer
        loop.device = torch.device("cpu")
        loop.data_dir = tmp_path
        loop.hyperparams = _valid_hyperparams()
        loop.val_loader = _fake_loader("sample_a", [0])
        loop.epoch_fallback_counts = {'evaluation_failure': 0}
        loop._amp_enabled = False

        result = loop.validate_epoch()
        assert result['evaluation_completed_successfully'] is False

    def test_validate_epoch_no_gt_fallback_sets_provenance_false(self, tmp_path):
        """Real end-to-end validate_epoch() call where no .geff file exists
        for the sample at all (the 'no usable GT' branch) -- the returned
        dict must have evaluation_completed_successfully=False."""
        import src.train as train_module

        unet3d = _FakePeakUNet3D({0: {0: (0, 1, 1), 1: (0, 1, 2)}})
        transformer = _FakeAcceptAllEdgeTransformer()
        loop = train_module.TrainingLoop.__new__(train_module.TrainingLoop)
        loop.unet3d = unet3d
        loop.transformer = transformer
        loop.device = torch.device("cpu")
        loop.data_dir = tmp_path  # no .geff files written here
        loop.hyperparams = _valid_hyperparams()
        loop.val_loader = _fake_loader("sample_a", [0])
        loop.epoch_fallback_counts = {'evaluation_failure': 0}
        loop._amp_enabled = False

        result = loop.validate_epoch()
        assert result['evaluation_completed_successfully'] is False


# ---------------------------------------------------------------------------
# F6 -- transaction and cleanup tests
# ---------------------------------------------------------------------------

class TestF6TransactionAndCleanup:
    def test_ineligible_new_checkpoint_cannot_delete_active_checkpoint(self, tmp_path):
        """F6.1: write a healthy manifested checkpoint, then attempt to
        write an INELIGIBLE one -- the original manifest/checkpoint must be
        completely untouched."""
        unet3d, transformer = _make_small_models()
        good_checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        good_path = _save_checkpoint(tmp_path, good_checkpoint, "epoch_1_val_score_0.5000.pt")
        manifest_path = write_checkpoint_manifest(good_path, checkpoint=good_checkpoint)
        original_manifest_bytes = manifest_path.read_bytes()

        bad_checkpoint = _make_eligible_checkpoint(unet3d, transformer, training_code_sha="bad")
        bad_path = _save_checkpoint(tmp_path, bad_checkpoint, "epoch_2_val_score_0.1000.pt")
        with pytest.raises(ValueError):
            write_checkpoint_manifest(bad_path, checkpoint=bad_checkpoint)

        assert manifest_path.read_bytes() == original_manifest_bytes
        assert good_path.exists()

    def test_replacement_activates_before_old_becomes_cleanup_eligible(self, tmp_path):
        """F6.2: after a valid replacement, the manifest atomically points
        at the NEW checkpoint (never an intermediate/missing state)."""
        unet3d, transformer = _make_small_models()
        checkpoint_v1 = _make_eligible_checkpoint(unet3d, transformer, epoch=1)
        path_v1 = _save_checkpoint(tmp_path, checkpoint_v1, "epoch_1_val_score_0.5000.pt")
        manifest_path = write_checkpoint_manifest(path_v1, checkpoint=checkpoint_v1)
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_file"] == "epoch_1_val_score_0.5000.pt"

        checkpoint_v2 = _make_eligible_checkpoint(unet3d, transformer, epoch=2)
        path_v2 = _save_checkpoint(tmp_path, checkpoint_v2, "epoch_2_val_score_0.9000.pt")
        write_checkpoint_manifest(path_v2, checkpoint=checkpoint_v2)
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_file"] == "epoch_2_val_score_0.9000.pt"

    def test_temp_manifest_validation_failure_preserves_old_state(self, tmp_path, monkeypatch):
        """F6.3: if the temp-manifest self-consistency check fails
        mid-write, the old manifest/checkpoint must be untouched and no
        temp file left behind."""
        unet3d, transformer = _make_small_models()
        good_checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        good_path = _save_checkpoint(tmp_path, good_checkpoint, "epoch_1_val_score_0.5000.pt")
        manifest_path = write_checkpoint_manifest(good_path, checkpoint=good_checkpoint)
        original_bytes = manifest_path.read_bytes()

        new_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=2)
        new_path = _save_checkpoint(tmp_path, new_checkpoint, "epoch_2_val_score_0.6000.pt")

        # Force _validate_manifest_semantics to explode on the SECOND call
        # (the temp-file self-consistency validation) to simulate a
        # corrupted intermediate write, without touching real disk state.
        call_count = {"n": 0}
        real_validate = checkpoint_manifest._validate_manifest_semantics

        def flaky_validate(manifest):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated corruption")
            return real_validate(manifest)

        monkeypatch.setattr(checkpoint_manifest, "_validate_manifest_semantics", flaky_validate)
        with pytest.raises(RuntimeError, match="simulated corruption"):
            write_checkpoint_manifest(new_path, checkpoint=new_checkpoint)

        assert manifest_path.read_bytes() == original_bytes
        leftover_tmp = list(tmp_path.glob(".checkpoint_manifest.*.tmp"))
        assert leftover_tmp == []

    def test_post_replacement_verification_failure_restores_old_manifest(self, tmp_path, monkeypatch):
        """F6.4: if post-replacement re-verification fails, the OLD manifest
        content is atomically restored and both checkpoints are preserved."""
        unet3d, transformer = _make_small_models()
        good_checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        good_path = _save_checkpoint(tmp_path, good_checkpoint, "epoch_1_val_score_0.5000.pt")
        manifest_path = write_checkpoint_manifest(good_path, checkpoint=good_checkpoint)
        original_bytes = manifest_path.read_bytes()

        new_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=2)
        new_path = _save_checkpoint(tmp_path, new_checkpoint, "epoch_2_val_score_0.6000.pt")

        call_count = {"n": 0}
        real_validate = checkpoint_manifest._validate_manifest_semantics

        def flaky_validate(manifest):
            call_count["n"] += 1
            # 1st call: temp-file validation (must succeed so replace() runs).
            # 2nd call: post-replacement active-manifest validation (fails).
            if call_count["n"] == 2:
                raise RuntimeError("simulated post-replacement corruption")
            return real_validate(manifest)

        monkeypatch.setattr(checkpoint_manifest, "_validate_manifest_semantics", flaky_validate)
        with pytest.raises(RuntimeError, match="simulated post-replacement corruption"):
            write_checkpoint_manifest(new_path, checkpoint=new_checkpoint)

        assert manifest_path.read_bytes() == original_bytes
        assert good_path.exists()
        assert new_path.exists()

    def test_no_cleanup_after_failure(self, tmp_path, monkeypatch):
        """F6.5: a failed write_checkpoint_manifest() call must not delete
        the new (failed) checkpoint file either -- it's preserved for
        diagnosis."""
        unet3d, transformer = _make_small_models()
        new_checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        new_path = _save_checkpoint(tmp_path, new_checkpoint)

        def raising_parse(_raw_bytes):
            raise ValueError("simulated parse failure")

        monkeypatch.setattr(checkpoint_manifest, "_parse_manifest_bytes", raising_parse)
        with pytest.raises(ValueError, match="simulated parse failure"):
            write_checkpoint_manifest(new_path, checkpoint=new_checkpoint)
        assert new_path.exists()

    def test_missing_active_checkpoint_is_loud(self, tmp_path):
        """F6.6: a manifest referencing a checkpoint file that has since
        been deleted must raise FileNotFoundError, not silently proceed."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        manifest_path = write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
        checkpoint_path.unlink()
        with pytest.raises(FileNotFoundError):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_unrelated_old_checkpoints_deleted_only_after_preserving_required(self, tmp_path):
        """F6.8: write_checkpoint_manifest() itself never deletes ANY
        checkpoint file (required or not) -- checkpoint retention/cleanup
        is exclusively TrainingLoop.save_checkpoint()'s responsibility (out
        of this file's allowed scope to re-test at that call-site level).
        Verified behaviorally here: an unrelated, older checkpoint file
        sitting in the same directory survives untouched through both a
        successful and a failed write_checkpoint_manifest() call."""
        unet3d, transformer = _make_small_models()
        unrelated_path = tmp_path / "epoch_0_val_score_0.1000.pt"
        unrelated_path.write_bytes(b"an old, unrelated checkpoint file")

        good_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=1)
        good_path = _save_checkpoint(tmp_path, good_checkpoint, "epoch_1_val_score_0.5000.pt")
        write_checkpoint_manifest(good_path, checkpoint=good_checkpoint)
        assert unrelated_path.exists()
        assert unrelated_path.read_bytes() == b"an old, unrelated checkpoint file"

        bad_checkpoint = _make_eligible_checkpoint(unet3d, transformer, training_code_sha="bad")
        bad_path = _save_checkpoint(tmp_path, bad_checkpoint, "epoch_2_val_score_0.1000.pt")
        with pytest.raises(ValueError):
            write_checkpoint_manifest(bad_path, checkpoint=bad_checkpoint)
        assert unrelated_path.exists()
        assert unrelated_path.read_bytes() == b"an old, unrelated checkpoint file"


# ---------------------------------------------------------------------------
# Active-manifest checkpoint protection must fail closed (correction of
# adversarial finding 24): TrainingLoop.save_checkpoint() must use the
# shared, strict production manifest parser
# (checkpoint_manifest.read_active_manifest_checkpoint_path()) to determine
# what checkpoint the CURRENTLY active manifest protects, and must raise --
# never warn-and-proceed -- if that manifest is malformed, ambiguous, or
# unsafe in any way, since proceeding to cleanup in that state risks
# deleting a checkpoint a still-active (just unparseable-by-us) manifest
# references.
# ---------------------------------------------------------------------------

def _make_training_loop_for_checkpoint_save(checkpoint_dir, **extra_attrs):
    """Bare TrainingLoop instantiation (bypassing __init__, which needs real
    DataLoaders) exercising the REAL, unmodified save_checkpoint() method
    against real small models -- exactly the same pattern already
    established for TrainingLoop hermetic tests, kept self-contained here."""
    from src.train import TrainingLoop

    unet3d, transformer = _make_small_models()
    optimizer = torch.optim.SGD(
        list(unet3d.parameters()) + list(transformer.parameters()), lr=0.01
    )
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.unet3d = unet3d
    loop.transformer = transformer
    loop.optimizer = optimizer
    loop.checkpoint_dir = Path(checkpoint_dir)
    loop.hyperparams = _valid_hyperparams()
    loop.split_identity = VALID_SPLIT64
    loop.deployed_sha = VALID_SHA40
    loop.best_checkpoint_path = None
    for key, value in extra_attrs.items():
        setattr(loop, key, value)
    return loop


class TestActiveManifestProtectionFailsClosed:
    """Real production-path (TrainingLoop.save_checkpoint()) regressions:
    a malformed/unsafe/ambiguous active manifest must cause save_checkpoint()
    to raise BEFORE any cleanup runs, leaving every checkpoint file and the
    (corrupted) manifest bytes completely untouched."""

    def _seed_first_eligible_checkpoint(self, tmp_path):
        """Establish a real, valid manifest + checkpoint via the actual
        production save_checkpoint() path, then return (loop, first
        checkpoint path, manifest path, manifest bytes) for the test to
        corrupt."""
        loop = _make_training_loop_for_checkpoint_save(tmp_path)
        loop.save_checkpoint(epoch=1, metrics=_valid_val_metrics(adjusted_edge_jaccard=0.5))
        first_checkpoint_path = tmp_path / "epoch_1_val_score_0.5000.pt"
        manifest_path = tmp_path / checkpoint_manifest.MANIFEST_FILENAME
        assert first_checkpoint_path.exists()
        assert manifest_path.exists()
        return loop, first_checkpoint_path, manifest_path

    def _assert_untouched_and_raises(self, loop, first_checkpoint_path, manifest_path, corrupted_bytes):
        with pytest.raises((ValueError, FileNotFoundError)):
            loop.save_checkpoint(epoch=2, metrics=_valid_val_metrics(adjusted_edge_jaccard=0.9))

        # Manifest bytes exactly as corrupted -- no write, no restore attempt.
        assert manifest_path.read_bytes() == corrupted_bytes
        # The original checkpoint the (now-unreadable) manifest referenced
        # must survive -- cleanup never ran.
        assert first_checkpoint_path.exists()
        # Collision-safety correction: the active manifest is read and
        # validated BEFORE any torch.save() call, so when it is malformed/
        # unsafe, save_checkpoint() now raises before writing ANYTHING to
        # disk -- the new (epoch=2) checkpoint file must not exist either.
        second_checkpoint_path = loop.checkpoint_dir / "epoch_2_val_score_0.9000.pt"
        assert not second_checkpoint_path.exists()

    def test_malformed_json_active_manifest_raises(self, tmp_path):
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        manifest_path.write_text("{not valid json at all", encoding="utf-8")
        corrupted_bytes = manifest_path.read_bytes()
        self._assert_untouched_and_raises(loop, first_checkpoint_path, manifest_path, corrupted_bytes)

    def test_duplicate_key_active_manifest_raises(self, tmp_path):
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        manifest_path.write_text(
            '{"training_code_sha": "' + VALID_SHA40 + '", "training_code_sha": "' + VALID_SHA40_B + '"}',
            encoding="utf-8",
        )
        corrupted_bytes = manifest_path.read_bytes()
        self._assert_untouched_and_raises(loop, first_checkpoint_path, manifest_path, corrupted_bytes)

    def test_unknown_field_active_manifest_raises(self, tmp_path):
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        fields = json.loads(manifest_path.read_text(encoding="utf-8"))
        fields["surprise_field"] = "unexpected"
        _write_raw_manifest(manifest_path, fields)
        corrupted_bytes = manifest_path.read_bytes()
        self._assert_untouched_and_raises(loop, first_checkpoint_path, manifest_path, corrupted_bytes)

    def test_missing_checkpoint_active_manifest_raises(self, tmp_path):
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        first_checkpoint_path.unlink()  # manifest now references a nonexistent file
        corrupted_bytes = manifest_path.read_bytes()

        with pytest.raises((ValueError, FileNotFoundError)):
            loop.save_checkpoint(epoch=2, metrics=_valid_val_metrics(adjusted_edge_jaccard=0.9))

        assert manifest_path.read_bytes() == corrupted_bytes
        assert not first_checkpoint_path.exists()  # we deleted it ourselves, not save_checkpoint()
        # Collision-safety correction: raises before any write.
        second_checkpoint_path = loop.checkpoint_dir / "epoch_2_val_score_0.9000.pt"
        assert not second_checkpoint_path.exists()

    def test_symlink_active_manifest_checkpoint_raises(self, tmp_path):
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        fields = json.loads(manifest_path.read_text(encoding="utf-8"))

        real_target = tmp_path / "real_target.pt"
        real_target.write_bytes(first_checkpoint_path.read_bytes())
        link_path = tmp_path / "linked.pt"
        try:
            link_path.symlink_to(real_target)
        except OSError as e:
            pytest.skip(f"OS denied symlink creation: {e}")

        fields["checkpoint_file"] = "linked.pt"
        fields["checkpoint_sha256"] = sha256_file(real_target)
        _write_raw_manifest(manifest_path, fields)
        corrupted_bytes = manifest_path.read_bytes()

        self._assert_untouched_and_raises(loop, first_checkpoint_path, manifest_path, corrupted_bytes)

    def test_unsafe_path_active_manifest_raises(self, tmp_path):
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        fields = json.loads(manifest_path.read_text(encoding="utf-8"))
        fields["checkpoint_file"] = "../escape.pt"
        _write_raw_manifest(manifest_path, fields)
        corrupted_bytes = manifest_path.read_bytes()
        self._assert_untouched_and_raises(loop, first_checkpoint_path, manifest_path, corrupted_bytes)

    def test_healthy_active_manifest_allows_normal_replacement(self, tmp_path):
        """Positive control: an uncorrupted active manifest lets
        save_checkpoint() proceed normally, replacing the manifest and
        protecting the new checkpoint."""
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        loop.save_checkpoint(epoch=2, metrics=_valid_val_metrics(adjusted_edge_jaccard=0.9))
        active = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert active["checkpoint_file"] == "epoch_2_val_score_0.9000.pt"


class TestSaveCheckpointFileFsync:
    """Direct unit coverage of checkpoint_manifest.save_checkpoint_file() --
    the shared helper save_checkpoint() and correctness both depend on."""

    def test_saves_a_loadable_checkpoint(self, tmp_path):
        checkpoint = {"epoch": 1, "data": [1, 2, 3]}
        path = tmp_path / "test.pt"
        save_checkpoint_file(checkpoint, path)
        assert path.exists()
        loaded = torch.load(path, weights_only=False)
        assert loaded == checkpoint

    def test_fsync_is_actually_invoked(self, tmp_path, monkeypatch):
        import src.checkpoint_manifest as checkpoint_manifest_module

        calls = []
        real_fsync = os.fsync

        def spy_fsync(fd):
            calls.append(fd)
            return real_fsync(fd)

        monkeypatch.setattr(checkpoint_manifest_module.os, "fsync", spy_fsync)
        path = tmp_path / "test.pt"
        save_checkpoint_file({"epoch": 1}, path)
        assert len(calls) == 1

    def test_unsupported_fsync_errno_is_logged_not_raised(self, tmp_path, monkeypatch):
        import errno as errno_module

        import src.checkpoint_manifest as checkpoint_manifest_module

        def raising_fsync(fd):
            raise OSError(errno_module.ENOTSUP, "fsync not supported")

        monkeypatch.setattr(checkpoint_manifest_module.os, "fsync", raising_fsync)
        path = tmp_path / "test.pt"
        save_checkpoint_file({"epoch": 1}, path)  # must not raise
        assert path.exists()
        loaded = torch.load(path, weights_only=False)
        assert loaded == {"epoch": 1}

    def test_genuine_fsync_failure_propagates(self, tmp_path, monkeypatch):
        import errno as errno_module

        import src.checkpoint_manifest as checkpoint_manifest_module

        def raising_fsync(fd):
            raise OSError(errno_module.EIO, "simulated real I/O error")

        monkeypatch.setattr(checkpoint_manifest_module.os, "fsync", raising_fsync)
        path = tmp_path / "test.pt"
        with pytest.raises(OSError):
            save_checkpoint_file({"epoch": 1}, path)


class TestCheckpointFilenameCollisionSafety:
    """Corrects a confirmed transactional defect: the derived checkpoint
    filename (epoch + rounded val_score) can collide with the checkpoint
    currently referenced by the active manifest -- e.g. a repeated/resumed
    epoch landing on the identical rounded score. save_checkpoint() must
    never let torch.save() destroy the active checkpoint's bytes before its
    protected path and this new checkpoint's eligibility are known."""

    def _seed_first_eligible_checkpoint(self, tmp_path):
        loop = _make_training_loop_for_checkpoint_save(tmp_path)
        loop.save_checkpoint(epoch=1, metrics=_valid_val_metrics(adjusted_edge_jaccard=0.5))
        first_checkpoint_path = tmp_path / "epoch_1_val_score_0.5000.pt"
        manifest_path = tmp_path / checkpoint_manifest.MANIFEST_FILENAME
        assert first_checkpoint_path.exists()
        assert manifest_path.exists()
        return loop, first_checkpoint_path, manifest_path

    def test_ineligible_collision_raises_before_overwrite(self, tmp_path):
        """Required regression 1: same derived filename, INELIGIBLE new
        metrics -- must raise before overwrite; active checkpoint bytes/
        hash and manifest bytes unchanged; production verification still
        succeeds for the OLD checkpoint afterward."""
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        original_checkpoint_bytes = first_checkpoint_path.read_bytes()
        original_checkpoint_hash = sha256_file(first_checkpoint_path)
        original_manifest_bytes = manifest_path.read_bytes()

        # Identical epoch + rounded val_score -> identical derived filename
        # ("epoch_1_val_score_0.5000.pt"), but this new checkpoint is NOT
        # deployment-eligible (evaluation did not complete successfully).
        ineligible_metrics = _valid_val_metrics(
            adjusted_edge_jaccard=0.5, evaluation_completed_successfully=False,
        )
        with pytest.raises(RuntimeError, match="collides"):
            loop.save_checkpoint(epoch=1, metrics=ineligible_metrics)

        # Active checkpoint bytes/hash completely unchanged.
        assert first_checkpoint_path.read_bytes() == original_checkpoint_bytes
        assert sha256_file(first_checkpoint_path) == original_checkpoint_hash
        # Manifest bytes completely unchanged.
        assert manifest_path.read_bytes() == original_manifest_bytes
        # No disambiguated file was written either -- nothing written at all.
        assert list(tmp_path.glob("epoch_1_val_score_0.5000_r*.pt")) == []

        # Production verification still succeeds for the OLD checkpoint.
        checkpoint, manifest, verified_path = load_verified_checkpoint(
            manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu",
        )
        assert verified_path == first_checkpoint_path
        assert manifest["checkpoint_sha256"] == original_checkpoint_hash

    def test_eligible_collision_disambiguates_old_checkpoint_survives(self, tmp_path):
        """Required regression 2: same derived filename, ELIGIBLE new
        metrics -- the active checkpoint must remain available, untouched,
        for the entire operation; the new checkpoint is saved under a
        disambiguated filename and only becomes active once its manifest
        replacement is written and independently re-verified."""
        loop, first_checkpoint_path, manifest_path = self._seed_first_eligible_checkpoint(tmp_path)
        original_checkpoint_bytes = first_checkpoint_path.read_bytes()
        original_checkpoint_hash = sha256_file(first_checkpoint_path)

        eligible_metrics = _valid_val_metrics(adjusted_edge_jaccard=0.5)
        loop.save_checkpoint(epoch=1, metrics=eligible_metrics)

        # The OLD checkpoint file, at its ORIGINAL path, is byte-for-byte
        # untouched -- it was never opened for writing.
        assert first_checkpoint_path.exists()
        assert first_checkpoint_path.read_bytes() == original_checkpoint_bytes
        assert sha256_file(first_checkpoint_path) == original_checkpoint_hash

        # The new checkpoint landed at a disambiguated, non-colliding path.
        disambiguated_path = tmp_path / "epoch_1_val_score_0.5000_r1.pt"
        assert disambiguated_path.exists()
        assert disambiguated_path.resolve() != first_checkpoint_path.resolve()

        # The manifest is now active, verified, and references the NEW file
        # -- the replacement completed and is independently loadable.
        active_fields = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert active_fields["checkpoint_file"] == "epoch_1_val_score_0.5000_r1.pt"
        checkpoint, manifest, verified_path = load_verified_checkpoint(
            manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu",
        )
        assert verified_path == disambiguated_path
        # (Model weights/metrics are identical between the two saves in this
        # test, so the checkpoint BYTES may legitimately hash the same --
        # the real proof this fix works is the distinct PATH, verified above.)


# ---------------------------------------------------------------------------
# F7 -- hyperparameter validation tests
# ---------------------------------------------------------------------------

class TestF7HyperparameterValidation:
    def test_missing_hyperparams_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_inference_hyperparams(None)

    def test_non_dict_hyperparams_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_inference_hyperparams(["not", "a", "dict"])

    @pytest.mark.parametrize("missing_key", ["detection_threshold", "edge_threshold", "nms_radius_um"])
    def test_each_missing_required_key_raises(self, missing_key):
        hp = _valid_hyperparams()
        del hp[missing_key]
        with pytest.raises(ValueError, match=f"missing required field '{missing_key}'"):
            validate_inference_hyperparams(hp)

    @pytest.mark.parametrize("key", ["detection_threshold", "edge_threshold", "nms_radius_um"])
    def test_bool_value_rejected(self, key):
        hp = _valid_hyperparams(**{key: True})
        with pytest.raises(ValueError, match="not bool"):
            validate_inference_hyperparams(hp)

    @pytest.mark.parametrize("key", ["detection_threshold", "edge_threshold", "nms_radius_um"])
    def test_nan_value_rejected(self, key):
        hp = _valid_hyperparams(**{key: float("nan")})
        with pytest.raises(ValueError, match="finite"):
            validate_inference_hyperparams(hp)

    @pytest.mark.parametrize("key", ["detection_threshold", "edge_threshold", "nms_radius_um"])
    def test_inf_value_rejected(self, key):
        hp = _valid_hyperparams(**{key: float("inf")})
        with pytest.raises(ValueError, match="finite"):
            validate_inference_hyperparams(hp)

    @pytest.mark.parametrize("key,bad_value", [
        ("detection_threshold", -0.1), ("detection_threshold", 1.1),
        ("edge_threshold", -0.1), ("edge_threshold", 1.1),
    ])
    def test_range_violations_rejected(self, key, bad_value):
        hp = _valid_hyperparams(**{key: bad_value})
        with pytest.raises(ValueError, match="out of the required range"):
            validate_inference_hyperparams(hp)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0])
    def test_non_positive_nms_radius_rejected(self, bad_value):
        hp = _valid_hyperparams(nms_radius_um=bad_value)
        with pytest.raises(ValueError, match="out of the required range"):
            validate_inference_hyperparams(hp)

    def test_nan_nms_radius_rejected(self):
        hp = _valid_hyperparams(nms_radius_um=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            validate_inference_hyperparams(hp)

    @pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.1])
    def test_invalid_positive_voxel_fraction_rejected(self, bad_value):
        hp = _valid_hyperparams(max_positive_voxel_fraction=bad_value)
        with pytest.raises(ValueError, match="max_positive_voxel_fraction"):
            validate_inference_hyperparams(hp)

    def test_valid_dictionary_passes_unchanged(self):
        hp = _valid_hyperparams()
        result = validate_inference_hyperparams(hp)
        assert result == hp
        assert result is hp

    def test_valid_dictionary_without_optional_key_passes(self):
        hp = {"detection_threshold": 0.5, "edge_threshold": 0.5, "nms_radius_um": 5.0}
        result = validate_inference_hyperparams(hp)
        assert result == hp


# ---------------------------------------------------------------------------
# F8 -- strict state-loading tests
# ---------------------------------------------------------------------------

class TestF8StrictStateLoading:
    def test_missing_unet_parameter_fails(self):
        """F8.1."""
        unet3d, _ = _make_small_models()
        state = unet3d.state_dict()
        del state[next(iter(state.keys()))]
        fresh_unet, _ = _make_small_models()
        with pytest.raises(RuntimeError):
            fresh_unet.load_state_dict(state, strict=True)

    def test_unexpected_unet_parameter_fails(self):
        """F8.2."""
        unet3d, _ = _make_small_models()
        state = dict(unet3d.state_dict())
        state["totally_unexpected_param"] = torch.zeros(1)
        fresh_unet, _ = _make_small_models()
        with pytest.raises(RuntimeError):
            fresh_unet.load_state_dict(state, strict=True)

    def test_missing_transformer_parameter_fails(self):
        """F8.3."""
        _, transformer = _make_small_models()
        state = transformer.state_dict()
        del state[next(iter(state.keys()))]
        _, fresh_transformer = _make_small_models()
        with pytest.raises(RuntimeError):
            fresh_transformer.load_state_dict(state, strict=True)

    def test_unexpected_transformer_parameter_fails(self):
        """F8.4."""
        _, transformer = _make_small_models()
        state = dict(transformer.state_dict())
        state["totally_unexpected_param"] = torch.zeros(1)
        _, fresh_transformer = _make_small_models()
        with pytest.raises(RuntimeError):
            fresh_transformer.load_state_dict(state, strict=True)

    def test_valid_state_dicts_load_strictly(self):
        """F8.5."""
        unet3d, transformer = _make_small_models()
        fresh_unet, fresh_transformer = _make_small_models()
        fresh_unet.load_state_dict(unet3d.state_dict(), strict=True)
        fresh_transformer.load_state_dict(transformer.state_dict(), strict=True)

    @pytest.mark.parametrize("file_path", ["generate_submission.py", "kaggle_kernel_inference/inference_kernel.py"])
    def test_no_production_caller_downgrades_incompatibility(self, file_path):
        """F8.6: every load_state_dict(...) call in the two production
        submission callers must pass strict=True as a literal keyword --
        never omitted, never strict=False."""
        import ast

        source = open(file_path, encoding="utf-8").read()
        tree = ast.parse(source, filename=file_path)
        found_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict":
                found_calls += 1
                strict_kwargs = [kw for kw in node.keywords if kw.arg == "strict"]
                assert len(strict_kwargs) == 1, f"{file_path}: load_state_dict call must pass strict= explicitly"
                strict_value = strict_kwargs[0].value
                assert isinstance(strict_value, ast.Constant) and strict_value.value is True, (
                    f"{file_path}: load_state_dict must be called with strict=True, never omitted or False"
                )
        assert found_calls == 2, f"{file_path}: expected exactly 2 load_state_dict calls (unet3d, transformer), found {found_calls}"


# ---------------------------------------------------------------------------
# V4 remediation (reviewer round v3 -> v4): 4 narrow blockers, all scoped to
# src/checkpoint_manifest.py and src/submission_pipeline.py only.
# ---------------------------------------------------------------------------

class TestV4SchemaVersionStrictTypeChecks:
    """BLOCKER 1: schema_version (manifest) and checkpoint_schema_version
    (checkpoint) must reject bool/float/str lookalikes, not just fail a bare
    `!= 1` equality check -- Python treats True == 1 and 1.0 == 1, so a naive
    equality check would wrongly accept a JSON `"schema_version": true` (or
    a checkpoint dict with `checkpoint_schema_version: True`) as valid."""

    @pytest.mark.parametrize("bad_value", [True, False, 1.0, "1"])
    def test_manifest_schema_version_lookalike_rejected(self, tmp_path, bad_value):
        """V1.1-4: manifest schema_version=True/False/1.0/"1" all rejected."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        fields = _valid_manifest_fields(checkpoint_path, checkpoint, schema_version=bad_value)
        manifest_path = tmp_path / "checkpoint_manifest.json"
        _write_raw_manifest(manifest_path, fields)
        with pytest.raises(ValueError, match="schema_version"):
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_manifest_schema_version_exact_int_passes(self, tmp_path):
        """V1.5: schema_version=1 (a real int) passes."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        _loaded_checkpoint, manifest, _loaded_path = load_verified_checkpoint(
            manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu",
        )
        assert manifest["schema_version"] == checkpoint_manifest.MANIFEST_SCHEMA_VERSION

    @pytest.mark.parametrize("bad_value", [True, False, 1.0, "1"])
    def test_checkpoint_schema_version_lookalike_ineligible(self, bad_value):
        """V1.6-9: checkpoint_schema_version=True/False/1.0/"1" all make the
        checkpoint deployment-ineligible."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, checkpoint_schema_version=bad_value)
        errors = deployment_eligibility_errors(checkpoint)
        assert any("checkpoint_schema_version" in e for e in errors), errors

    def test_checkpoint_schema_version_exact_int_passes(self):
        """V1.10: checkpoint_schema_version=1 (a real int) passes this
        specific eligibility condition (checkpoint is otherwise eligible)."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(
            unet3d, transformer, checkpoint_schema_version=checkpoint_manifest.CHECKPOINT_SCHEMA_VERSION,
        )
        assert deployment_eligibility_errors(checkpoint) == []

    def test_forged_manifest_around_bool_schema_version_checkpoint_rejected(self, tmp_path):
        """V1.11/V1.12: an otherwise schema/hash-valid manifest around a
        checkpoint whose OWN checkpoint_schema_version is True (not int 1)
        must be rejected by load_verified_checkpoint() at the eligibility
        step, not silently accepted because True == 1 -- and the error
        message must explicitly name checkpoint_schema_version."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer, checkpoint_schema_version=True)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        manifest_path = _write_valid_manifest(tmp_path, checkpoint_path, checkpoint)
        with pytest.raises(ValueError, match="checkpoint_schema_version") as exc_info:
            load_verified_checkpoint(manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu")
        assert "checkpoint_schema_version" in str(exc_info.value)


# ---------------------------------------------------------------------------
# BLOCKER 2
# ---------------------------------------------------------------------------

class TestV4DeploymentEligibilityRequiresDictAndStateDicts:
    """BLOCKER 2: deployment_eligibility_errors() must reject (a) a checkpoint
    argument that isn't even a dict, without raising AttributeError from
    checkpoint.get(), and (b) a checkpoint dict missing unet3d_state_dict/
    transformer_state_dict -- an obviously incomplete checkpoint (no model
    weights) must never become the actively manifested checkpoint, even
    though load_verified_checkpoint() would separately reject it later at
    load time. Fail-closed means caught before write_checkpoint_manifest()
    ever touches disk, not after."""

    @pytest.mark.parametrize("bad_checkpoint", [None, [], "not a dict", 42, ("a", "b")])
    def test_non_dict_checkpoint_is_ineligible_no_raise(self, bad_checkpoint):
        """V2.1: a non-dict checkpoint argument does not raise
        AttributeError -- it returns a named eligibility error instead."""
        errors = deployment_eligibility_errors(bad_checkpoint)
        assert errors != []
        assert any("dict" in e for e in errors), errors

    def test_missing_unet3d_state_dict_ineligible(self):
        """V2.2/V2.4: missing unet3d_state_dict is ineligible and named."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        del checkpoint["unet3d_state_dict"]
        errors = deployment_eligibility_errors(checkpoint)
        assert any("unet3d_state_dict" in e for e in errors), errors

    def test_missing_transformer_state_dict_ineligible(self):
        """V2.3/V2.4: missing transformer_state_dict is ineligible and named."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        del checkpoint["transformer_state_dict"]
        errors = deployment_eligibility_errors(checkpoint)
        assert any("transformer_state_dict" in e for e in errors), errors

    def test_write_checkpoint_manifest_refuses_missing_unet3d_state_dict(self, tmp_path):
        """V2.5."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        del checkpoint["unet3d_state_dict"]
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        with pytest.raises(ValueError, match="unet3d_state_dict"):
            write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
        assert not (tmp_path / checkpoint_manifest.MANIFEST_FILENAME).exists()

    def test_write_checkpoint_manifest_refuses_missing_transformer_state_dict(self, tmp_path):
        """V2.6."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        del checkpoint["transformer_state_dict"]
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        with pytest.raises(ValueError, match="transformer_state_dict"):
            write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
        assert not (tmp_path / checkpoint_manifest.MANIFEST_FILENAME).exists()

    @pytest.mark.parametrize("missing_key", ["unet3d_state_dict", "transformer_state_dict"])
    def test_missing_state_dict_cannot_replace_existing_active_manifest(self, tmp_path, missing_key):
        """V2.7: with an old valid active manifest already in place, an
        attempt to manifest a checkpoint missing either state dict raises,
        leaves the old manifest byte-identical, leaves the old checkpoint
        intact, and never reaches (let alone runs) any destructive cleanup
        stage."""
        unet3d, transformer = _make_small_models()
        good_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=1)
        good_path = _save_checkpoint(tmp_path, good_checkpoint, "epoch_1_val_score_0.5000.pt")
        manifest_path = write_checkpoint_manifest(good_path, checkpoint=good_checkpoint)
        original_manifest_bytes = manifest_path.read_bytes()

        bad_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=2)
        del bad_checkpoint[missing_key]
        bad_path = _save_checkpoint(tmp_path, bad_checkpoint, "epoch_2_val_score_0.8000.pt")

        with pytest.raises(ValueError, match=missing_key):
            write_checkpoint_manifest(bad_path, checkpoint=bad_checkpoint)

        assert manifest_path.read_bytes() == original_manifest_bytes
        assert good_path.exists()
        assert good_path.stat().st_size > 0
        assert bad_path.exists()  # write_checkpoint_manifest() never deletes checkpoints itself

    def test_healthy_checkpoint_with_both_state_dicts_still_writes_manifest(self, tmp_path):
        """V2.8: a healthy checkpoint containing both required state
        dictionaries still writes a valid manifest -- the new checks are
        additive, not a regression on the existing happy path."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        assert "unet3d_state_dict" in checkpoint and "transformer_state_dict" in checkpoint
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        assert deployment_eligibility_errors(checkpoint) == []
        manifest_path = write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
        assert manifest_path.exists()


# ---------------------------------------------------------------------------
# BLOCKER 3
# ---------------------------------------------------------------------------

class TestV4ManifestFilenameContract:
    """BLOCKER 3: production deployment identity has exactly one canonical
    manifest filename (checkpoint_manifest.json) -- find_single_manifest()
    already enforces this via discovery, but the DIRECT load/write helpers
    previously accepted any filename as long as the contents were otherwise
    valid, creating two inconsistent manifest contracts. Both direct paths
    must now enforce the same canonical name."""

    def test_load_rejects_manifest_named_deployment_json(self, tmp_path):
        """V3.1."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        wrong_path = tmp_path / "deployment.json"
        _write_raw_manifest(wrong_path, _valid_manifest_fields(checkpoint_path, checkpoint))
        with pytest.raises(ValueError, match="checkpoint_manifest.json"):
            load_verified_checkpoint(wrong_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_load_rejects_manifest_named_manifest_json(self, tmp_path):
        """V3.2."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        wrong_path = tmp_path / "manifest.json"
        _write_raw_manifest(wrong_path, _valid_manifest_fields(checkpoint_path, checkpoint))
        with pytest.raises(ValueError, match="checkpoint_manifest.json"):
            load_verified_checkpoint(wrong_path, expected_source_sha=VALID_SHA40, map_location="cpu")

    def test_load_rejection_error_names_canonical_filename(self, tmp_path):
        """V3.3."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        wrong_path = tmp_path / "deployment.json"
        _write_raw_manifest(wrong_path, _valid_manifest_fields(checkpoint_path, checkpoint))
        with pytest.raises(ValueError) as exc_info:
            load_verified_checkpoint(wrong_path, expected_source_sha=VALID_SHA40, map_location="cpu")
        assert checkpoint_manifest.MANIFEST_FILENAME in str(exc_info.value)

    def test_write_with_wrong_output_filename_rejects_before_replacement(self, tmp_path):
        """V3.4: write_checkpoint_manifest(..., output_path=<dir>/deployment.json)
        rejects before any active-manifest replacement -- no file is ever
        created at the wrong path (see
        test_wrong_output_filename_cannot_touch_existing_canonical_manifest
        for the direct proof that an existing canonical manifest survives)."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        wrong_output = tmp_path / "deployment.json"
        with pytest.raises(ValueError, match="checkpoint_manifest.json"):
            write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint, output_path=wrong_output)
        assert not wrong_output.exists()

    def test_canonical_filename_still_passes(self, tmp_path):
        """V3.5: the canonical checkpoint_manifest.json path (explicitly
        passed as output_path, not just relying on the default) still
        works end to end (write, then load_verified_checkpoint)."""
        unet3d, transformer = _make_small_models()
        checkpoint = _make_eligible_checkpoint(unet3d, transformer)
        checkpoint_path = _save_checkpoint(tmp_path, checkpoint)
        canonical_output = tmp_path / checkpoint_manifest.MANIFEST_FILENAME
        manifest_path = write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint, output_path=canonical_output)
        assert manifest_path == canonical_output
        assert manifest_path.exists()
        _loaded_checkpoint, _manifest, loaded_path = load_verified_checkpoint(
            manifest_path, expected_source_sha=VALID_SHA40, map_location="cpu",
        )
        assert loaded_path == checkpoint_path

    def test_wrong_output_filename_cannot_touch_existing_canonical_manifest(self, tmp_path):
        """V3.6: rejecting a wrong output filename must not replace or
        delete an existing valid canonical manifest."""
        unet3d, transformer = _make_small_models()
        good_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=1)
        good_path = _save_checkpoint(tmp_path, good_checkpoint, "epoch_1_val_score_0.5000.pt")
        manifest_path = write_checkpoint_manifest(good_path, checkpoint=good_checkpoint)
        original_bytes = manifest_path.read_bytes()

        new_checkpoint = _make_eligible_checkpoint(unet3d, transformer, epoch=2)
        new_path = _save_checkpoint(tmp_path, new_checkpoint, "epoch_2_val_score_0.8000.pt")
        wrong_output = tmp_path / "deployment.json"
        with pytest.raises(ValueError, match="checkpoint_manifest.json"):
            write_checkpoint_manifest(new_path, checkpoint=new_checkpoint, output_path=wrong_output)

        assert manifest_path.read_bytes() == original_bytes
        assert not wrong_output.exists()
        assert good_path.exists()


# ---------------------------------------------------------------------------
# BLOCKER 4
# ---------------------------------------------------------------------------

class TestV4PerSampleCudaSynchronization:
    """BLOCKER 4: run_submission_inference()'s per-sample elapsed_seconds
    must be bounded by real CUDA synchronization -- GPU ops are async, so
    without an explicit torch.cuda.synchronize() immediately before starting
    and immediately after stopping each per-sample timer, elapsed_seconds
    would not reliably represent completed GPU execution, even though the
    already-present run-wide total_elapsed_seconds sync is unaffected.
    Proven behaviorally against the real production orchestration function
    (not a source/AST check) by spying on torch.cuda.synchronize and on
    run_sample_loader_inference / build_test_dataset / PredictionGraphAssembler
    -- no real CUDA device is required or touched."""

    def test_per_sample_timing_synchronizes_before_and_after(self, monkeypatch):
        import types

        calls: list[str] = []

        def fake_synchronize(device=None):
            calls.append("sync")

        def fake_run_sample_loader_inference(**kwargs):
            calls.append("run")
            return {
                "sample_id": kwargs["sample_id"],
                "expected_pair_count": kwargs["expected_pair_count"],
                "loader_length": kwargs["expected_pair_count"],
                "processed_pair_count": kwargs["expected_pair_count"],
                "total_candidate_edges": 1,
                "total_accepted_edges": 1,
            }

        class _FakeLenOnlyDataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                raise NotImplementedError

        class _FakeGraph:
            def node_ids(self):
                return [0, 1]

            def edge_list(self):
                return [(0, 1)]

        class _FakeAssembler:
            def diagnostics(self):
                return {"predicted_nodes_total": 2, "predicted_edges_total": 1}

            def pred_graphs(self):
                return {"sampleA": _FakeGraph()}

        monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
        monkeypatch.setattr(
            submission_pipeline_module, "run_sample_loader_inference", fake_run_sample_loader_inference,
        )
        monkeypatch.setattr(
            submission_pipeline_module, "build_test_dataset", lambda test_dir, sample_id: _FakeLenOnlyDataset(),
        )
        monkeypatch.setattr(submission_pipeline_module, "PredictionGraphAssembler", _FakeAssembler)

        unet3d, transformer = _make_small_models()
        fake_device = types.SimpleNamespace(type="cuda", index=0)

        _pred_graphs, diagnostics = submission_pipeline_module.run_submission_inference(
            test_dir="unused",
            test_zarrs=[Path("sampleA.zarr")],
            unet3d=unet3d,
            transformer=transformer,
            device=fake_device,
            hyperparams=_valid_hyperparams(),
        )

        assert calls.count("run") == 1, f"expected exactly one sample run, got {calls}"
        run_idx = calls.index("run")
        assert calls[run_idx - 1] == "sync", (
            f"torch.cuda.synchronize() must be called immediately before the "
            f"per-sample timer starts: {calls}"
        )
        assert calls[run_idx + 1] == "sync", (
            f"torch.cuda.synchronize() must be called immediately after the "
            f"per-sample timer stops: {calls}"
        )
        assert diagnostics["per_sample"]["sampleA"]["elapsed_seconds"] >= 0
