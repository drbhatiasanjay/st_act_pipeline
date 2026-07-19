"""
P0-7 (2026-07-19) training-integrity regression tests.

Dedicated test file for P0-7's fail-closed training/validation semantics --
per the frozen spec, does NOT broadly edit tests/test_train.py or
tests/test_dataset.py. No test in this file depends on git tracked/staged/
untracked state.

Uses this codebase's established __new__-bypass TrainingLoop/CompetitionDataset
construction pattern (see tests/test_train.py, tests/test_dataset.py) plus
minimal fake nn.Module stand-ins, so these tests run fast, deterministically,
and without needing real staged data, a real GPU, or a live Kaggle environment.

kaggle_kernel/train_kernel.py's provenance functions (exact-one discovery,
strict GIT_SHA.txt validation, import-origin verification) are exercised
against their REAL source bodies, extracted via AST and exec'd in an isolated
namespace -- this runs the actual production logic without executing the rest
of the script (which does real pip installs / data loading / model
construction at module import time).

Run: py -m pytest tests/test_p07_training_integrity.py -v
"""
import ast
import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.train as train_module
from src.dataset import CompetitionDataset
from src.split_utils import load_and_validate_split
from src.train import TrainingLoop

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_KERNEL_PATH = REPO_ROOT / "kaggle_kernel" / "train_kernel.py"


# ---------------------------------------------------------------------------
# Section A -- CompetitionDataset strict_sample_coverage (P0-7 DATASET
# COVERAGE CONTRACT). Self-contained helpers (not imported from
# tests/test_dataset.py), matching this codebase's existing per-file
# self-containment convention.
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Reports a controlled frame count, or raises on get_shape() to simulate
    an unreadable Zarr store, without touching a real Zarr store."""

    def __init__(self, num_frames=None, raise_on_get_shape=False):
        self._num_frames = num_frames
        self._raise_on_get_shape = raise_on_get_shape

    def get_shape(self):
        if self._raise_on_get_shape:
            raise OSError("simulated unreadable Zarr store")
        return (self._num_frames, 64, 256, 256)


def _new_bare_dataset(
    data_dir, sample_ids, split_type="train",
    filter_unannotated_pairs=False, strict_sample_coverage=False,
):
    dataset = CompetitionDataset.__new__(CompetitionDataset)
    dataset.data_dir = data_dir
    dataset.split_type = split_type
    dataset.normalize = True
    dataset.anisotropy = (4.0, 1.0, 1.0)
    dataset.physical_voxel_size = (1.625, 0.40625, 0.40625)
    dataset.zip_path = None
    dataset.filter_unannotated_pairs = filter_unannotated_pairs
    dataset.strict_sample_coverage = strict_sample_coverage
    dataset.sample_ids = sample_ids
    dataset.pairs = []
    dataset._loader_cache = {}
    dataset._gt_counts_by_time_cache = {}
    dataset.annotation_pair_stats = None
    return dataset


class TestDatasetCoverageContract:
    """P0-7 DATASET COVERAGE CONTRACT: strict_sample_coverage semantics."""

    def test_missing_zarr_strict_raises_and_records_failed(self, tmp_path):
        dataset = _new_bare_dataset(tmp_path, ["missing_sample"], strict_sample_coverage=True)
        with pytest.raises(RuntimeError, match="expected Zarr not found"):
            dataset._build_pair_index()
        assert dataset.failed_sample_ids == ["missing_sample"]

    def test_missing_zarr_non_strict_soft_skips_but_records_coverage(self, tmp_path):
        dataset = _new_bare_dataset(tmp_path, ["missing_sample"], strict_sample_coverage=False)
        dataset._build_pair_index()  # must NOT raise
        assert dataset.expected_sample_ids == ["missing_sample"]
        assert dataset.failed_sample_ids == ["missing_sample"]
        assert dataset.successfully_opened_sample_ids == []
        assert dataset.zero_pairs_sample_ids == []

    def test_unreadable_zarr_strict_raises_and_records_failed(self, tmp_path, monkeypatch):
        sample_id = "bad_sample"
        (tmp_path / f"{sample_id}.zarr").mkdir()
        dataset = _new_bare_dataset(tmp_path, [sample_id], strict_sample_coverage=True)
        monkeypatch.setattr(
            CompetitionDataset, "_get_loader",
            lambda self, sid: _FakeLoader(raise_on_get_shape=True),
        )
        with pytest.raises(RuntimeError, match="failed to open expected Zarr"):
            dataset._build_pair_index()
        assert dataset.failed_sample_ids == [sample_id]

    def test_unreadable_zarr_non_strict_soft_skips_but_records(self, tmp_path, monkeypatch):
        sample_id = "bad_sample"
        (tmp_path / f"{sample_id}.zarr").mkdir()
        dataset = _new_bare_dataset(tmp_path, [sample_id], strict_sample_coverage=False)
        monkeypatch.setattr(
            CompetitionDataset, "_get_loader",
            lambda self, sid: _FakeLoader(raise_on_get_shape=True),
        )
        dataset._build_pair_index()  # must NOT raise
        assert dataset.failed_sample_ids == [sample_id]

    def test_zero_usable_pairs_strict_raises_and_records_zero_pairs_not_failed(
        self, tmp_path, monkeypatch,
    ):
        sample_id = "empty_sample"
        (tmp_path / f"{sample_id}.zarr").mkdir()
        dataset = _new_bare_dataset(tmp_path, [sample_id], strict_sample_coverage=True)
        # num_frames=1 -> range(num_frames - 1) == range(0) -> zero usable pairs
        monkeypatch.setattr(CompetitionDataset, "_get_loader", lambda self, sid: _FakeLoader(num_frames=1))
        with pytest.raises(RuntimeError, match="zero usable"):
            dataset._build_pair_index()
        assert dataset.zero_pairs_sample_ids == [sample_id]
        assert dataset.failed_sample_ids == [], "zero-pairs samples must NOT be placed in failed_sample_ids"

    def test_zero_usable_pairs_non_strict_warns_and_continues(self, tmp_path, monkeypatch):
        sample_id = "empty_sample"
        (tmp_path / f"{sample_id}.zarr").mkdir()
        dataset = _new_bare_dataset(tmp_path, [sample_id], strict_sample_coverage=False)
        monkeypatch.setattr(CompetitionDataset, "_get_loader", lambda self, sid: _FakeLoader(num_frames=1))
        dataset._build_pair_index()  # must NOT raise
        assert dataset.zero_pairs_sample_ids == [sample_id]
        assert dataset.failed_sample_ids == []
        assert dataset.successfully_opened_sample_ids == []

    def test_coverage_categories_disjoint_and_cover_expected_exactly_once(self, tmp_path, monkeypatch):
        good, empty, missing = "good_sample", "empty_sample", "missing_sample"
        for sid in (good, empty):
            (tmp_path / f"{sid}.zarr").mkdir()
        # missing_sample: deliberately no .zarr dir created at all.

        dataset = _new_bare_dataset(tmp_path, [good, empty, missing], strict_sample_coverage=False)

        def fake_get_loader(self, sample_id):
            return _FakeLoader(num_frames=5) if sample_id == good else _FakeLoader(num_frames=1)

        monkeypatch.setattr(CompetitionDataset, "_get_loader", fake_get_loader)
        dataset._build_pair_index()

        assert dataset.expected_sample_ids == [good, empty, missing]
        assert dataset.successfully_opened_sample_ids == [good]
        assert dataset.zero_pairs_sample_ids == [empty]
        assert dataset.failed_sample_ids == [missing]

        union = (
            set(dataset.successfully_opened_sample_ids)
            | set(dataset.zero_pairs_sample_ids)
            | set(dataset.failed_sample_ids)
        )
        assert union == set(dataset.expected_sample_ids)
        assert not (set(dataset.successfully_opened_sample_ids) & set(dataset.zero_pairs_sample_ids))
        assert not (set(dataset.successfully_opened_sample_ids) & set(dataset.failed_sample_ids))
        assert not (set(dataset.zero_pairs_sample_ids) & set(dataset.failed_sample_ids))


# ---------------------------------------------------------------------------
# Section B -- TrainingLoop.train_epoch() fail-closed rules A-E.
# ---------------------------------------------------------------------------

def _make_edge_loss_harness(
    monkeypatch, *, nodes_t_coords=None, nodes_t1_coords=None,
    gt_nodes_side_effect=None,
    generate_edge_targets_side_effect=None,
    transformer_forward=None,
    division_loss_side_effect=None,
    use_real_get_gt_nodes=False,
    data_dir=None,
):
    """Build a minimal TrainingLoop (via __new__ bypass) whose train_epoch()
    exercises P0-7's edge-loss block (Rules A-E) against ONE fake batch, with
    every upstream/downstream dependency monkeypatched to a controlled fake --
    avoids needing a real UNet3D/SimpleNodeTransformer/staged data."""
    device = torch.device("cpu")
    z, y, x = 4, 4, 4

    class _FakeUNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x_in):
            return torch.zeros(1, 2, z, y, x) + 0.0 * self.p, torch.zeros(1, 8, z, y, x)

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, nodes_t, nodes_t1, features_t, features_t1):
            if transformer_forward is not None:
                return transformer_forward(nodes_t, nodes_t1, features_t, features_t1) + 0.0 * self.p
            n = nodes_t.shape[0] * nodes_t1.shape[0]
            return torch.zeros(n) + 0.0 * self.p

    unet3d = _FakeUNet3D()
    transformer = _FakeTransformer()

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.unet3d = unet3d
    loop.transformer = transformer
    loop.device = device
    loop.data_dir = data_dir if data_dir is not None else Path("unused")
    loop.hyperparams = {
        'heatmap_loss_weight': 1.0, 'grad_clip': 1.0, 'warmup_steps': 0,
        'max_batches_per_epoch': None,
    }
    loop.optimizer = torch.optim.AdamW(
        list(unet3d.parameters()) + list(transformer.parameters()), lr=1e-4,
    )
    loop._amp_enabled = False
    loop.scaler = torch.amp.GradScaler('cpu', enabled=False)
    loop.detection_loss_fn = lambda logits, target: torch.tensor(0.0, requires_grad=True)
    loop.division_loss_fn = (
        division_loss_side_effect if division_loss_side_effect is not None
        else (lambda logits, targets, mask: torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))
    )
    loop.epoch_fallback_counts = {
        'heatmap_generation_failure': 0,
        'edge_target_generation_failure': 0,
        'edge_loss_computation_failure': 0,
        'evaluation_failure': 0,
        'gt_node_load_failure': 0,
        'retained_pair_zero_gt_nodes_failure': 0,
    }
    loop.epoch_biological_zero_counts = {'legitimate_zero_positive_edge_batches': 0}
    loop._global_step = 0
    loop._geff_cache = {}
    loop.last_epoch_wall_clock_seconds = 0.0
    loop.last_epoch_num_batches = 0
    loop.progress_file = None

    batch = {
        "frame_t": torch.zeros(1, 1, z, y, x),
        "frame_t1": torch.zeros(1, 1, z, y, x),
        "sample_id": ["fake_sample"],
        "t_idx": torch.tensor([0]),
    }
    loop.train_loader = [batch]

    monkeypatch.setattr(
        loop, "_generate_and_validate_heatmap_target",
        lambda sample_id, t_idx, volume_shape, z_, y_, x_: torch.zeros(1, 2, z_, y_, x_),
    )

    if not use_real_get_gt_nodes:
        if gt_nodes_side_effect is not None:
            monkeypatch.setattr(loop, "_get_gt_nodes", gt_nodes_side_effect)
        else:
            coords = {0: nodes_t_coords, 1: nodes_t1_coords}

            def default_get_gt_nodes(sample_id, t_idx):
                return coords[int(t_idx)]

            monkeypatch.setattr(loop, "_get_gt_nodes", default_get_gt_nodes)

    if generate_edge_targets_side_effect is not None:
        monkeypatch.setattr(train_module, "generate_edge_targets", generate_edge_targets_side_effect)

    return loop


class TestRuleBRetainedPairEmptyGtNodesFatal:
    def test_empty_nodes_t_raises_and_counts(self, monkeypatch):
        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((0, 3)), nodes_t1_coords=torch.zeros((5, 3)),
        )
        with pytest.raises(RuntimeError, match="zero GT nodes"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['retained_pair_zero_gt_nodes_failure'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0
        # A successfully-parsed-but-empty result is Rule B, never Rule A/
        # gt_node_load_failure -- the two counters must never be conflated.
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 0

    def test_empty_nodes_t1_raises_and_counts(self, monkeypatch):
        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((5, 3)), nodes_t1_coords=torch.zeros((0, 3)),
        )
        with pytest.raises(RuntimeError, match="zero GT nodes"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['retained_pair_zero_gt_nodes_failure'] == 1
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 0


class TestRuleATechnicalGtLoadFailurePropagates:
    def test_get_gt_nodes_raise_propagates_uncaught(self, monkeypatch):
        def raising_get_gt_nodes(sample_id, t_idx):
            raise RuntimeError("Technical GT-load failure: simulated missing .geff")

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((1, 3)), nodes_t1_coords=torch.zeros((1, 3)),
            gt_nodes_side_effect=raising_get_gt_nodes,
        )
        with pytest.raises(RuntimeError, match="simulated missing .geff"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 1


class TestDefect2GtNodeLoadFailureCounted:
    """Reviewer-required (P0-7 v1 review, Defect 2): a technical exception from
    _get_gt_nodes() itself (missing GEFF, parse exception, malformed node
    attributes) must increment gt_node_load_failure exactly once -- not
    retained_pair_zero_gt_nodes_failure, a DIFFERENT case -- then immediately
    propagate, with no edge-target or edge-loss work occurring afterward.
    Uses the REAL (unpatched) _get_gt_nodes against a real tmp_path data_dir,
    not an injected side effect, so this exercises the production method
    itself, not a test double standing in for it."""

    def test_missing_geff_counted_and_raises(self, monkeypatch, tmp_path):
        loop = _make_edge_loss_harness(monkeypatch, use_real_get_gt_nodes=True, data_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Technical GT node load failure"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0
        assert loop.epoch_fallback_counts['edge_target_generation_failure'] == 0
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 0

    def test_geff_parse_exception_counted_and_raises(self, monkeypatch, tmp_path):
        (tmp_path / "fake_sample.geff").mkdir()  # geff_path.exists() must be True

        def raising_load_geff_cached(path, cache):
            raise RuntimeError("simulated GEFF parse failure")

        monkeypatch.setattr(train_module, "load_geff_cached", raising_load_geff_cached)
        loop = _make_edge_loss_harness(monkeypatch, use_real_get_gt_nodes=True, data_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Technical GT node load failure"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0
        assert loop.epoch_fallback_counts['edge_target_generation_failure'] == 0
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 0

    def test_malformed_node_attrs_counted_and_raises(self, monkeypatch, tmp_path):
        (tmp_path / "fake_sample.geff").mkdir()

        class _MalformedGraph:
            def node_attrs(self, attr_keys=None):
                import polars as pl
                return pl.DataFrame({"t": [0]})  # missing required z/y/x columns

        def fake_load_geff_cached(path, cache):
            return _MalformedGraph(), {}

        monkeypatch.setattr(train_module, "load_geff_cached", fake_load_geff_cached)
        loop = _make_edge_loss_harness(monkeypatch, use_real_get_gt_nodes=True, data_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Technical GT node load failure"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 1

    def test_second_call_failure_not_double_counted(self, monkeypatch, tmp_path):
        """The FIRST _get_gt_nodes call (t_idx) succeeds; the SECOND
        (t_idx+1) fails -- gt_node_load_failure must increment exactly once
        for the whole batch, not once per failed sub-call."""
        (tmp_path / "fake_sample.geff").mkdir()
        calls = {"n": 0}
        import polars as pl

        class _OkThenFailGraph:
            def node_attrs(self, attr_keys=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    return pl.DataFrame({"t": [0], "z": [1.0], "y": [1.0], "x": [1.0]})
                raise RuntimeError("simulated failure on second call")

        def fake_load_geff_cached(path, cache):
            return _OkThenFailGraph(), {}

        monkeypatch.setattr(train_module, "load_geff_cached", fake_load_geff_cached)
        loop = _make_edge_loss_harness(monkeypatch, use_real_get_gt_nodes=True, data_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Technical GT node load failure"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 1


class TestRuleCEdgeTargetGenerationFailure:
    def test_generate_edge_targets_exception_counted_then_raised(self, monkeypatch):
        def raising_generate_edge_targets(*a, **kw):
            raise ValueError("simulated edge target generation failure")

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=raising_generate_edge_targets,
        )
        with pytest.raises(RuntimeError, match="Edge target generation failed"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['edge_target_generation_failure'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0


class TestRuleDLegitimateZeroPositiveEdge:
    def test_all_zero_edge_targets_counted_as_biological_not_technical(self, monkeypatch):
        def fake_generate_edge_targets(sample_id, geff_path, nodes_t, nodes_t1, **kw):
            n = nodes_t.shape[0] * nodes_t1.shape[0]
            return torch.zeros(n), {}

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=fake_generate_edge_targets,
        )
        loop.train_epoch()  # must NOT raise
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 1
        assert loop.epoch_fallback_counts['edge_target_generation_failure'] == 0
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 0
        assert loop.epoch_fallback_counts['retained_pair_zero_gt_nodes_failure'] == 0

    def test_all_negative_target_still_runs_transformer_and_loss_with_real_gradient(self, monkeypatch):
        """Reviewer-required (P0-7 v1 review, Defect 1): a non-empty
        all-negative edge target is real, valid training data -- the
        transformer's false-edge rejection signal -- and must NOT be replaced
        with a disconnected edge_loss=0.0. Proves: legitimate_zero_positive_
        edge_batches increments exactly once; transformer.forward is called;
        DivisionLoss is called; the resulting loss is finite; the transformer
        receives a real, nonzero gradient; no technical-failure counter
        increments."""
        transformer_calls = {"n": 0}
        division_loss_calls = {"n": 0}
        captured_logits = {}

        def counting_transformer_forward(nodes_t, nodes_t1, features_t, features_t1):
            transformer_calls["n"] += 1
            n = nodes_t.shape[0] * nodes_t1.shape[0]
            # Nonzero, real-valued logits (not all-zero) so BCE produces a real,
            # nonzero gradient w.r.t. this tensor, not a degenerate zero one.
            # retain_grad(): this leaf tensor is added to the harness's
            # 0.0 * self.p wiring below (see _make_edge_loss_harness) purely to
            # connect it to the fake transformer's own param graph -- that
            # multiply-by-zero deliberately makes self.p.grad always exactly 0,
            # so the real gradient signal must be read off THIS tensor instead.
            logits = torch.full((n,), 0.7, requires_grad=True)
            captured_logits["t"] = logits
            return logits

        def counting_division_loss(logits, targets, mask):
            division_loss_calls["n"] += 1
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

        def fake_generate_edge_targets(sample_id, geff_path, nodes_t, nodes_t1, **kw):
            n = nodes_t.shape[0] * nodes_t1.shape[0]
            return torch.zeros(n), {}  # non-empty, all-negative

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=fake_generate_edge_targets,
            transformer_forward=counting_transformer_forward,
            division_loss_side_effect=counting_division_loss,
        )
        loop.train_epoch()  # must NOT raise

        assert transformer_calls["n"] == 1, "transformer.forward must run for a non-empty all-negative target"
        assert division_loss_calls["n"] == 1, "DivisionLoss must run for a non-empty all-negative target"
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 1
        assert loop.epoch_fallback_counts['edge_target_generation_failure'] == 0
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 0
        assert loop.epoch_fallback_counts['gt_node_load_failure'] == 0
        assert loop.epoch_fallback_counts['retained_pair_zero_gt_nodes_failure'] == 0
        logits_grad = captured_logits["t"].grad
        assert logits_grad is not None, "the transformer's raw logits must receive a real gradient"
        assert bool((logits_grad != 0.0).any()), "gradient must be a real, nonzero learning signal"

    def test_zero_candidate_edges_with_nonempty_nodes_is_counted_technical_failure(self, monkeypatch):
        """Defensive regression: generate_edge_targets() (src/targets.py) is
        verified to ONLY return numel()==0 when n_t==0 or n_t1==0, which Rule B
        already rules out upstream -- if this invariant is ever violated
        anyway, it must be treated as a counted technical integrity failure and
        raise, never silently classified as biological zero."""
        def fake_generate_edge_targets_empty(sample_id, geff_path, nodes_t, nodes_t1, **kw):
            return torch.zeros(0), {}  # contract violation: nodes_t/nodes_t1 both non-empty

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=fake_generate_edge_targets_empty,
        )
        with pytest.raises(RuntimeError, match="zero candidate edges"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['edge_target_generation_failure'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0


class TestRuleETechnicalEdgeLossFailure:
    def _fake_generate_edge_targets_positive(self, sample_id, geff_path, nodes_t, nodes_t1, **kw):
        n = nodes_t.shape[0] * nodes_t1.shape[0]
        return torch.ones(n), {}

    def test_transformer_exception_counted_then_raised(self, monkeypatch):
        def raising_transformer_forward(nodes_t, nodes_t1, features_t, features_t1):
            raise RuntimeError("simulated transformer failure")

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=self._fake_generate_edge_targets_positive,
            transformer_forward=raising_transformer_forward,
        )
        with pytest.raises(RuntimeError, match="Edge loss computation failed"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0

    def test_division_loss_exception_counted_then_raised(self, monkeypatch):
        def raising_division_loss(logits, targets, mask):
            raise RuntimeError("simulated DivisionLoss failure")

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=self._fake_generate_edge_targets_positive,
            division_loss_side_effect=raising_division_loss,
        )
        with pytest.raises(RuntimeError, match="Edge loss computation failed"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 1

    def test_nan_loss_raises_and_counts(self, monkeypatch):
        def nan_division_loss(logits, targets, mask):
            return torch.tensor(float('nan'), requires_grad=True)

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=self._fake_generate_edge_targets_positive,
            division_loss_side_effect=nan_division_loss,
        )
        with pytest.raises(RuntimeError, match="non-finite"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 1

    def test_inf_loss_raises_and_counts(self, monkeypatch):
        def inf_division_loss(logits, targets, mask):
            return torch.tensor(float('inf'), requires_grad=True)

        loop = _make_edge_loss_harness(
            monkeypatch, nodes_t_coords=torch.zeros((2, 3)), nodes_t1_coords=torch.zeros((2, 3)),
            generate_edge_targets_side_effect=self._fake_generate_edge_targets_positive,
            division_loss_side_effect=inf_division_loss,
        )
        with pytest.raises(RuntimeError, match="non-finite"):
            loop.train_epoch()
        assert loop.epoch_fallback_counts['edge_loss_computation_failure'] == 1


# ---------------------------------------------------------------------------
# Section C -- TrainingLoop.validate_epoch() accounting + strict_integrity_mode.
# ---------------------------------------------------------------------------

def _make_validate_harness(
    monkeypatch, *, expected_sample_ids, sample_ids_with_batches,
    strict_integrity_mode=False, max_validation_samples=None,
    load_geff_ground_truth_side_effect=None, evaluate_submission_side_effect=None,
):
    device = torch.device("cpu")
    z, y, x = 4, 4, 4

    class _FakeUNet3D(nn.Module):
        def forward(self, x_in):
            return torch.zeros(1, 2, z, y, x), torch.zeros(1, 8, z, y, x)

    class _FakeTransformer(nn.Module):
        def forward(self, nodes_t, nodes_t1, features_t, features_t1):
            return torch.zeros(nodes_t.shape[0] * nodes_t1.shape[0])

    class _FakeDataset:
        def __init__(self, ids):
            self.expected_sample_ids = ids

    class _FakeValLoader:
        def __init__(self, batches, dataset):
            self._batches = batches
            self.dataset = dataset

        def __iter__(self):
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    batches = [
        {
            "sample_id": [sid],
            "frame_t": torch.zeros(1, 1, z, y, x),
            "frame_t1": torch.zeros(1, 1, z, y, x),
            "t_idx": [0],
        }
        for sid in sample_ids_with_batches
    ]

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.unet3d = _FakeUNet3D()
    loop.unet3d.eval()
    loop.transformer = _FakeTransformer()
    loop.transformer.eval()
    loop.device = device
    loop.data_dir = Path("unused")
    loop.hyperparams = {
        'detection_threshold': 0.5, 'edge_threshold': 0.5,
        'max_batches_per_epoch': None, 'max_validation_samples': max_validation_samples,
    }
    loop._amp_enabled = False
    loop.epoch_fallback_counts = {
        'heatmap_generation_failure': 0,
        'edge_target_generation_failure': 0,
        'edge_loss_computation_failure': 0,
        'evaluation_failure': 0,
        'retained_pair_zero_gt_nodes_failure': 0,
    }
    loop.strict_integrity_mode = strict_integrity_mode
    loop.val_loader = _FakeValLoader(batches, _FakeDataset(expected_sample_ids))

    monkeypatch.setattr(loop, "_peaks_for_channel", lambda detection_probs, channel, t_idx: [(0, 0, 0)])
    monkeypatch.setattr(
        loop, "_nodes_and_features_at_peaks",
        lambda features, peaks: (torch.zeros(len(peaks), 3), torch.zeros(len(peaks), 8)),
    )

    if load_geff_ground_truth_side_effect is not None:
        monkeypatch.setattr(train_module, "load_geff_ground_truth", load_geff_ground_truth_side_effect)
    if evaluate_submission_side_effect is not None:
        monkeypatch.setattr(train_module, "evaluate_submission", evaluate_submission_side_effect)

    return loop


def _fake_evaluate_submission(pred_graphs, gt_graphs, gt_metadata=None):
    return {'edge_jaccard': 1.0, 'adjusted_edge_jaccard': 1.0, 'division_jaccard': 0.0, 'score': 1.0}


class TestValidationAccounting:
    def test_validation_samples_total_uses_full_expected_fold_uncapped(self, monkeypatch):
        loop = _make_validate_harness(
            monkeypatch, expected_sample_ids=["s1", "s2"], sample_ids_with_batches=["s1", "s2"],
            load_geff_ground_truth_side_effect=lambda path: (object(), {}),
            evaluate_submission_side_effect=_fake_evaluate_submission,
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        result = loop.validate_epoch()
        assert result['validation_samples_total'] == 2
        assert result['validation_samples_evaluated'] == 2
        assert result['validation_is_full_fold'] is True

    def test_capped_validation_reports_full_fold_false(self, monkeypatch):
        loop = _make_validate_harness(
            monkeypatch, expected_sample_ids=["s1", "s2", "s3"], sample_ids_with_batches=["s1"],
            max_validation_samples=1,
            load_geff_ground_truth_side_effect=lambda path: (object(), {}),
            evaluate_submission_side_effect=_fake_evaluate_submission,
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        result = loop.validate_epoch()
        assert result['validation_samples_total'] == 3
        assert result['validation_samples_evaluated'] == 1
        assert result['validation_is_full_fold'] is False


class TestStrictValidationIntegrity:
    def test_strict_mode_raises_immediately_on_gt_load_failure(self, monkeypatch):
        def raising_load_geff(path):
            raise RuntimeError("simulated GT parse failure")

        loop = _make_validate_harness(
            monkeypatch, expected_sample_ids=["s1"], sample_ids_with_batches=["s1"],
            strict_integrity_mode=True,
            load_geff_ground_truth_side_effect=raising_load_geff,
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        with pytest.raises(RuntimeError, match="strict_integrity_mode=True"):
            loop.validate_epoch()
        assert loop.epoch_fallback_counts['evaluation_failure'] == 1

    def test_non_strict_mode_tolerates_single_gt_load_failure_below_threshold(self, monkeypatch):
        calls = {"n": 0}

        def sometimes_raising_load_geff(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated GT parse failure for s1")
            return object(), {}

        loop = _make_validate_harness(
            monkeypatch, expected_sample_ids=["s1", "s2", "s3"],
            sample_ids_with_batches=["s1", "s2", "s3"],
            strict_integrity_mode=False,
            load_geff_ground_truth_side_effect=sometimes_raising_load_geff,
            evaluate_submission_side_effect=_fake_evaluate_submission,
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)
        result = loop.validate_epoch()  # must NOT raise -- 1/3 failure rate < 50% threshold
        assert loop.epoch_fallback_counts['evaluation_failure'] == 1
        # validation_samples_evaluated tracks PREDICTION coverage (all 3
        # samples had complete predicted windows), independent of the
        # separately-tracked GT-load failure counted above -- matches the
        # pre-existing P0-4 convention (see src/train.py's evaluated_sample_ids
        # comment).
        assert result['validation_samples_evaluated'] == 3


class TestMissingGeffHandling:
    """P0-7 COUNTED_THEN_FATAL: missing GEFF must be counted then raised in
    strict mode, not silently skipped (the pre-fix bug was that the
    `if geff_path.exists():` branch was simply not entered, so the except
    block was never reached and evaluation_failure stayed 0)."""

    def test_strict_mode_missing_geff_raises_immediately(self, monkeypatch):
        # evaluate_submission must NOT be called after a missing-GEFF failure.
        evaluate_submission_called = {"called": False}

        def failing_evaluate(*args, **kwargs):
            evaluate_submission_called["called"] = True
            return {}

        loop = _make_validate_harness(
            monkeypatch,
            expected_sample_ids=["s1"],
            sample_ids_with_batches=["s1"],
            strict_integrity_mode=True,
            evaluate_submission_side_effect=failing_evaluate,
        )
        # GEFF does not exist -- no load_geff_ground_truth mock needed because
        # the FileNotFoundError fires before that call.
        monkeypatch.setattr(Path, "exists", lambda self: False)

        with pytest.raises(RuntimeError, match="strict_integrity_mode=True"):
            loop.validate_epoch()

        assert loop.epoch_fallback_counts['evaluation_failure'] == 1
        assert not evaluate_submission_called["called"], (
            "evaluate_submission must not be called after a missing-GEFF failure"
        )

    def test_strict_mode_missing_geff_error_includes_sample_and_path(self, monkeypatch):
        loop = _make_validate_harness(
            monkeypatch,
            expected_sample_ids=["my_sample"],
            sample_ids_with_batches=["my_sample"],
            strict_integrity_mode=True,
        )
        monkeypatch.setattr(Path, "exists", lambda self: False)

        with pytest.raises(RuntimeError, match="my_sample"):
            loop.validate_epoch()

    def test_non_strict_missing_geff_counted_circuit_breaker_not_triggered(self, monkeypatch):
        # 1 of 3 samples has a missing GEFF -- well below the 50% circuit breaker.
        call_count = {"n": 0}

        def selective_load_geff(path):
            call_count["n"] += 1
            return object(), {}

        loop = _make_validate_harness(
            monkeypatch,
            expected_sample_ids=["s1", "s2", "s3"],
            sample_ids_with_batches=["s1", "s2", "s3"],
            strict_integrity_mode=False,
            load_geff_ground_truth_side_effect=selective_load_geff,
            evaluate_submission_side_effect=_fake_evaluate_submission,
        )
        existing = {"s2", "s3"}
        monkeypatch.setattr(Path, "exists", lambda self: any(s in str(self) for s in existing))

        result = loop.validate_epoch()  # must not raise

        assert loop.epoch_fallback_counts['evaluation_failure'] == 1
        assert call_count["n"] == 2  # load_geff_ground_truth called for s2 and s3 only
        assert result is not None

    def test_missing_geff_not_double_counted(self, monkeypatch):
        # A missing GEFF must increment evaluation_failure exactly once.
        # With 1/1 samples failing, the circuit-breaker fires in non-strict mode
        # (100% > 50% threshold), so we expect that raise -- the important
        # invariant is that the counter is 1, not 0 or 2.
        loop = _make_validate_harness(
            monkeypatch,
            expected_sample_ids=["s1"],
            sample_ids_with_batches=["s1"],
            strict_integrity_mode=False,
            evaluate_submission_side_effect=_fake_evaluate_submission,
        )
        monkeypatch.setattr(Path, "exists", lambda self: False)

        with pytest.raises(RuntimeError, match="GT loading failed"):
            loop.validate_epoch()

        assert loop.epoch_fallback_counts['evaluation_failure'] == 1


# ---------------------------------------------------------------------------
# Section D -- kaggle_kernel/train_kernel.py provenance: exact-one discovery,
# strict GIT_SHA.txt validation, import-origin verification. Exercised against
# the REAL function bodies, extracted via AST and exec'd in isolation.
# ---------------------------------------------------------------------------

def _extract_function_source(name: str) -> str:
    source = TRAIN_KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"{name} not found at module level in {TRAIN_KERNEL_PATH}")


def _extract_discovery_functions():
    namespace = {"os": os, "MAX_SEARCH_DEPTH": 5}
    exec(_extract_function_source("find_all_kaggle_input_dirs"), namespace)
    exec(_extract_function_source("find_exactly_one_kaggle_input_dir"), namespace)
    return namespace["find_all_kaggle_input_dirs"], namespace["find_exactly_one_kaggle_input_dir"]


def _patch_fake_kaggle_input(monkeypatch, real_root: Path):
    """Redirect the literal '/kaggle/input' path (hardcoded in production, by
    design -- matches kaggle_kernel_inference/inference_kernel.py's identical
    pattern) to a real temp directory tree, for the duration of one test."""
    real_exists = os.path.exists
    real_walk = os.walk
    real_isfile = os.path.isfile
    real_root_str = str(real_root)

    def fake_exists(path):
        if str(path) == "/kaggle/input":
            return real_root.exists()
        return real_exists(path)

    def fake_walk(path, *a, **kw):
        if str(path) == "/kaggle/input":
            for dirpath, dirnames, filenames in real_walk(real_root_str, *a, **kw):
                yield "/kaggle/input" + dirpath[len(real_root_str):], dirnames, filenames
        else:
            yield from real_walk(path, *a, **kw)

    def fake_isfile(path):
        spath = str(path)
        if spath.startswith("/kaggle/input"):
            rel = spath[len("/kaggle/input"):].lstrip("/\\")
            return real_isfile(str(real_root / rel)) if rel else real_isfile(real_root_str)
        return real_isfile(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os, "walk", fake_walk)
    monkeypatch.setattr(os.path, "isfile", fake_isfile)


class TestExactOneSourceDiscovery:
    def test_zero_matches_raises(self, monkeypatch, tmp_path):
        _find_all, find_one = _extract_discovery_functions()
        _patch_fake_kaggle_input(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match="No directory"):
            find_one(os.path.join("src", "dataset.py"))

    def test_exactly_one_match_succeeds(self, monkeypatch, tmp_path):
        _find_all, find_one = _extract_discovery_functions()
        src_dir = tmp_path / "st-act-src" / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "dataset.py").touch()
        _patch_fake_kaggle_input(monkeypatch, tmp_path)
        result = find_one(os.path.join("src", "dataset.py"))
        assert result.replace("\\", "/").rstrip("/").endswith("st-act-src")

    def test_multiple_matches_raises_listing_candidates(self, monkeypatch, tmp_path):
        _find_all, find_one = _extract_discovery_functions()
        for name in ("dataset-a", "dataset-b"):
            d = tmp_path / name / "src"
            d.mkdir(parents=True)
            (d / "dataset.py").touch()
        _patch_fake_kaggle_input(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match="Multiple directories"):
            find_one(os.path.join("src", "dataset.py"))


def _extract_sha_validation_block() -> str:
    source = TRAIN_KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            segment = ast.get_source_segment(source, node)
            if segment and "_sha_file" in segment and "GIT_SHA.txt" in segment:
                return segment
    raise AssertionError("Could not locate the GIT_SHA.txt validation if-block in train_kernel.py")


def _run_sha_validation(kaggle_src_dataset_dir: str) -> str:
    segment = _extract_sha_validation_block()
    namespace = {
        "KAGGLE_MODE": True,
        "KAGGLE_SRC_DATASET_DIR": kaggle_src_dataset_dir,
        "Path": Path,
        "RuntimeError": RuntimeError,
    }
    exec(segment, namespace)
    return namespace["DEPLOYED_SHA"]


class TestStrictShaValidation:
    VALID_SHA = "a" * 40

    def test_missing_sha_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="GIT_SHA.txt not found"):
            _run_sha_validation(str(tmp_path))

    def test_empty_sha_file_raises(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text("")
        with pytest.raises(RuntimeError, match="empty or whitespace"):
            _run_sha_validation(str(tmp_path))

    def test_whitespace_only_sha_file_raises(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text("   \n")
        with pytest.raises(RuntimeError, match="empty or whitespace"):
            _run_sha_validation(str(tmp_path))

    def test_too_short_sha_raises(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text("abc123")
        with pytest.raises(RuntimeError, match="40-character"):
            _run_sha_validation(str(tmp_path))

    def test_uppercase_sha_raises(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text("A" * 40)
        with pytest.raises(RuntimeError, match="40-character"):
            _run_sha_validation(str(tmp_path))

    def test_non_hex_sha_raises(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text("g" * 40)
        with pytest.raises(RuntimeError, match="40-character"):
            _run_sha_validation(str(tmp_path))

    def test_valid_sha_accepted(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text(self.VALID_SHA)
        assert _run_sha_validation(str(tmp_path)) == self.VALID_SHA

    def test_valid_sha_with_trailing_whitespace_stripped(self, tmp_path):
        (tmp_path / "GIT_SHA.txt").write_text(self.VALID_SHA + "\n")
        assert _run_sha_validation(str(tmp_path)) == self.VALID_SHA


class _FakeModule:
    def __init__(self, file_path: str):
        self.__file__ = file_path
        self.__name__ = "fake.module"


class TestImportOriginVerification:
    def _run(self, expected_root: str, modules: list):
        segment = _extract_function_source("verify_import_origins")
        namespace = {
            "Path": Path,
            "RuntimeError": RuntimeError,
            "getattr": getattr,
            "logger": logging.getLogger("test_p07_import_origin"),
            "src": types.SimpleNamespace(
                dataset=modules[0], model=modules[1], split_utils=modules[2], train=modules[3],
            ),
        }
        exec(segment, namespace)
        namespace["verify_import_origins"](expected_root)

    def test_all_modules_under_expected_root_passes(self, tmp_path):
        root = tmp_path / "st-act-src"
        (root / "src").mkdir(parents=True)
        modules = [
            _FakeModule(str(root / "src" / n))
            for n in ("dataset.py", "model.py", "split_utils.py", "train.py")
        ]
        self._run(str(root), modules)  # must not raise

    def test_module_outside_expected_root_raises(self, tmp_path):
        root = tmp_path / "st-act-src"
        other = tmp_path / "stray-checkout"
        (root / "src").mkdir(parents=True)
        (other / "src").mkdir(parents=True)
        modules = [
            _FakeModule(str(root / "src" / "dataset.py")),
            _FakeModule(str(root / "src" / "model.py")),
            _FakeModule(str(root / "src" / "split_utils.py")),
            _FakeModule(str(other / "src" / "train.py")),  # escaped the selected source mount
        ]
        with pytest.raises(RuntimeError, match="is NOT beneath"):
            self._run(str(root), modules)


# ---------------------------------------------------------------------------
# Section E -- ORD-4: duplicate sample IDs already rejected by
# src/split_utils.py::load_and_validate_split() (out of P0-7 scope, not
# modified). Lightweight regression assertion only.
# ---------------------------------------------------------------------------

class TestOrd4DuplicateSampleIdsAlreadyRejected:
    def test_duplicate_train_sample_id_rejected(self, tmp_path):
        split_path = tmp_path / "data_split.json"
        split_path.write_text(json.dumps({
            "train": ["sample_a", "sample_b", "sample_a"],
            "validation": ["sample_c"],
        }))
        with pytest.raises(RuntimeError, match="duplicate sample IDs"):
            load_and_validate_split(split_path)
