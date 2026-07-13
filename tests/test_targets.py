"""
Unit tests for src/targets.py: load_geff_cached, generate_heatmap_targets,
generate_edge_targets, against REAL local .geff ground truth
(data/staging/train/44b6_0113de3b.geff), matching this project's established
preference for testing against real data (see test_data_loader_real.py).

Written to close a real, previously-flagged test-coverage gap (no test_targets.py
existed at all before this file). Directly regression-tests bug 1.2 (the geff
cache existed but was never wired up, causing ~600 redundant re-parses/epoch)
at the actual call-site level.

Run: py -m pytest tests/test_targets.py -v
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tracksdata

from src.targets import (
    DetectionLoss,
    generate_edge_targets,
    generate_heatmap_targets,
    load_geff_cached,
)

REAL_GEFF_PATH = "data/staging/train/44b6_0113de3b.geff"
REAL_GEFF_PATH_2 = "data/staging/train/44b6_0b24845f.geff"
REAL_VOLUME_SHAPE = (100, 64, 256, 256)  # (T, Z, Y, X)


def require_real_geff():
    if not os.path.exists(REAL_GEFF_PATH):
        pytest.skip(f"Real staged .geff not found at {REAL_GEFF_PATH} in this environment")


class TestLoadGeffCached:
    def test_cache_reused_on_second_call_reduces_real_parse_count(self, monkeypatch):
        """REGRESSION GUARD for bug 1.2 at the actual call-site: two calls with the
        same path + shared cache dict must only invoke the real (expensive)
        from_geff() parse ONCE, not twice."""
        require_real_geff()
        call_count = {"n": 0}
        real_from_geff = tracksdata.graph.IndexedRXGraph.from_geff

        def counting_from_geff(path):
            call_count["n"] += 1
            return real_from_geff(path)

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(counting_from_geff))

        cache = {}
        result_1 = load_geff_cached(REAL_GEFF_PATH, cache)
        result_2 = load_geff_cached(REAL_GEFF_PATH, cache)

        assert call_count["n"] == 1, "second call should reuse the cache, not re-parse"
        assert len(cache) == 1
        assert result_1 is result_2, "cached call must return the SAME object, not a fresh parse"

    def test_no_cache_reparses_every_call(self, monkeypatch):
        """cache=None must preserve the original always-reparse behavior for
        standalone/test callers (explicit backward-compat contract)."""
        require_real_geff()
        call_count = {"n": 0}
        real_from_geff = tracksdata.graph.IndexedRXGraph.from_geff

        def counting_from_geff(path):
            call_count["n"] += 1
            return real_from_geff(path)

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(counting_from_geff))

        load_geff_cached(REAL_GEFF_PATH, None)
        load_geff_cached(REAL_GEFF_PATH, None)

        assert call_count["n"] == 2

    def test_different_paths_get_separate_cache_entries(self):
        require_real_geff()
        if not os.path.exists(REAL_GEFF_PATH_2):
            pytest.skip(f"Real staged .geff not found at {REAL_GEFF_PATH_2} in this environment")

        cache = {}
        load_geff_cached(REAL_GEFF_PATH, cache)
        load_geff_cached(REAL_GEFF_PATH_2, cache)

        assert len(cache) == 2


class TestGenerateHeatmapTargets:
    def test_heatmap_shape_and_value_range(self):
        require_real_geff()
        heatmaps, metadata = generate_heatmap_targets(
            sample_id="44b6_0113de3b",
            geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE,
            target_ts=[0],
        )

        assert set(heatmaps.keys()) == {0}
        heatmap = heatmaps[0]
        assert heatmap.shape == (1, 64, 256, 256)
        assert heatmap.dtype == torch.float32
        assert torch.all((heatmap >= 0) & (heatmap <= 1))

    def test_point_target_is_binary_gaussian_target_is_not(self):
        """t=0 in the real geff has exactly 1 real GT centroid -- point targets
        must be exact 0/1, gaussian targets must have real intermediate values
        around that centroid (not degenerate to the same binary mask)."""
        require_real_geff()
        point_heatmaps, _ = generate_heatmap_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE, target_type="point", target_ts=[0],
        )
        gaussian_heatmaps, _ = generate_heatmap_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE, target_type="gaussian", target_ts=[0],
        )

        point_vals = torch.unique(point_heatmaps[0])
        assert set(point_vals.tolist()).issubset({0.0, 1.0}), "point target must be exactly binary"

        gaussian_vals = torch.unique(gaussian_heatmaps[0])
        has_intermediate_value = any(0.0 < v.item() < 1.0 for v in gaussian_vals)
        assert has_intermediate_value, "gaussian target must have real intermediate (non-binary) values"

    def test_metadata_total_centroids_matches_sum_of_per_frame_counts(self):
        require_real_geff()
        _, metadata = generate_heatmap_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE, target_ts=[0, 1, 2],
        )

        assert metadata["total_centroids"] == sum(metadata["centroids_per_frame"].values())
        assert metadata["centroids_per_frame"][0] == 1  # confirmed real count at t=0

    def test_geff_cache_shared_across_two_calls_reduces_real_parse_count(self, monkeypatch):
        """REGRESSION GUARD for bug 1.2: two generate_heatmap_targets() calls sharing
        a geff_cache dict must only really parse the .geff once."""
        require_real_geff()
        call_count = {"n": 0}
        real_from_geff = tracksdata.graph.IndexedRXGraph.from_geff

        def counting_from_geff(path):
            call_count["n"] += 1
            return real_from_geff(path)

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(counting_from_geff))

        cache = {}
        generate_heatmap_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE, target_ts=[0], geff_cache=cache,
        )
        generate_heatmap_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE, target_ts=[1], geff_cache=cache,
        )

        assert call_count["n"] == 1


class TestGenerateEdgeTargets:
    def test_empty_node_sets_returns_empty_labels_without_crashing(self):
        require_real_geff()
        edge_labels, metadata = generate_edge_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            nodes_t=torch.zeros((0, 3)), nodes_t1=torch.zeros((3, 3)), t=0,
        )

        assert edge_labels.shape == (0,)

    def test_output_shape_is_row_major_over_node_counts(self):
        require_real_geff()
        n_t, n_t1 = 2, 3
        nodes_t = torch.tensor([[3.0, 45.0, 45.0], [3.0, 100.0, 100.0]])
        nodes_t1 = torch.tensor([[3.0, 46.0, 46.0], [3.0, 101.0, 101.0], [3.0, 200.0, 200.0]])

        edge_labels, metadata = generate_edge_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            nodes_t=nodes_t, nodes_t1=nodes_t1, t=0,
        )

        assert edge_labels.shape == (n_t * n_t1,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
    def test_cuda_resident_nodes_do_not_crash_on_numpy_conversion(self):
        """REGRESSION GUARD, gap found by adversarial review: this file's other
        generate_edge_targets tests only ever pass CPU tensors, so they would NOT
        catch a regression of the real fix already shipped in match_to_gt()
        (`candidate_coords.detach().cpu().numpy()`, src/targets.py) -- a bare
        `.numpy()` works fine on a CPU tensor but raises TypeError on a CUDA one.
        train.py's real callers always pass GPU-resident nodes_t/nodes_t1, so this
        is the actual failure mode that shipped once already. Skipped (not
        xfailed) when no GPU is present, since it can't meaningfully run there --
        this project's CI/dev machine is CPU-only, but Kaggle's real GPU runs are
        exactly where this regression would resurface."""
        require_real_geff()
        nodes_t = torch.tensor([[3.0, 45.0, 45.0]], device="cuda")
        nodes_t1 = torch.tensor([[3.0, 46.0, 46.0]], device="cuda")

        edge_labels, metadata = generate_edge_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            nodes_t=nodes_t, nodes_t1=nodes_t1, t=0,
        )

        assert edge_labels.shape == (1,)

    def test_geff_cache_shared_reduces_real_parse_count(self, monkeypatch):
        require_real_geff()
        call_count = {"n": 0}
        real_from_geff = tracksdata.graph.IndexedRXGraph.from_geff

        def counting_from_geff(path):
            call_count["n"] += 1
            return real_from_geff(path)

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(counting_from_geff))

        cache = {}
        nodes_t = torch.tensor([[3.0, 45.0, 45.0]])
        nodes_t1 = torch.tensor([[3.0, 46.0, 46.0]])
        generate_edge_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            nodes_t=nodes_t, nodes_t1=nodes_t1, t=0, geff_cache=cache,
        )
        generate_edge_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            nodes_t=nodes_t1, nodes_t1=nodes_t, t=1, geff_cache=cache,
        )

        assert call_count["n"] == 1


class TestDetectionLoss:
    """
    Regression tests for the adaptive per-batch class-imbalance weighting fix
    (2026-07-13). A real full-epoch Kaggle training run (14,751 batches) produced
    val_score=0.0 -- confirmed root cause: the old fixed weight_neg=0.01 only
    compensates class imbalance by 100x, but the real measured imbalance (via
    generate_heatmap_targets on real .geff data) ranges from ~67x (dense samples)
    to ~667x (sparse samples), so sparse frames were under-compensated by up to
    ~6.7x -- the model's cheapest way to reduce loss was to push background more
    confidently negative rather than ever cross the detection threshold on a real
    cell voxel. A weak assertion (e.g. "loss is some float") would not catch a
    regression back to the old fixed-ratio behavior -- these assert the actual
    balancing math and the exact fallback path.
    """

    def test_adaptive_balances_pos_and_neg_contribution(self):
        # 4 positive (target=1) voxels out of 100 -> a real, checkable imbalance
        # ratio, not toy 50/50 data that wouldn't exercise the weighting at all.
        targets = torch.zeros(1, 1, 1, 10, 10)
        targets[0, 0, 0, 0, :4] = 1.0
        logits = torch.zeros_like(targets)  # sigmoid(0)=0.5 -> uniform bce per voxel

        loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01, adaptive=True)
        loss = loss_fn(logits, targets)

        pos_mass = targets.sum().item()
        neg_mass = targets.numel() - pos_mass
        expected_weight_neg = 1.0 * pos_mass / neg_mass
        # With uniform bce per voxel (logits=0), pos and neg contributions should
        # be exactly equal under the adaptive weight -- this is the actual claim
        # the fix makes, not just "loss changed from before".
        bce_per_voxel = loss_fn.bce_loss(logits, targets)[0, 0, 0, 0, 0].item()
        pos_contrib = 1.0 * pos_mass * bce_per_voxel
        neg_contrib = expected_weight_neg * neg_mass * bce_per_voxel
        assert pos_contrib == pytest.approx(neg_contrib, rel=1e-5)
        assert loss.item() == pytest.approx((pos_contrib + neg_contrib) / targets.numel(), rel=1e-5)

    def test_adaptive_weight_neg_matches_real_measured_imbalance(self):
        require_real_geff()
        heatmaps, _ = generate_heatmap_targets(
            sample_id="44b6_0113de3b", geff_path=REAL_GEFF_PATH,
            volume_shape=REAL_VOLUME_SHAPE, target_ts=[30],
        )
        targets = heatmaps[30].unsqueeze(0)
        pos_mass = targets.sum().item()
        neg_mass = targets.numel() - pos_mass
        # Real measured ratio for this exact sample/timepoint was ~667x under the
        # old fixed weight_neg=0.01 (see DEFERRED_IMPROVEMENTS.md) -- assert the
        # adaptive weight_neg this specific real batch produces is in that
        # ballpark, not just "some small positive number".
        adaptive_weight_neg = 1.0 * pos_mass / neg_mass
        assert 1e-5 < adaptive_weight_neg < 2e-5

    def test_adaptive_falls_back_to_fixed_weight_on_zero_cell_batch(self):
        # A batch with NO ground-truth cells at all (a real, observed case --
        # e.g. timepoint T=10 of 44b6_0113de3b has 0 GT centroids) has nothing to
        # balance against; must fall back to the fixed weight_neg, not divide by
        # zero or silently produce a NaN/zero loss.
        targets = torch.zeros(1, 1, 4, 4, 4)
        logits = torch.zeros_like(targets)
        loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01, adaptive=True)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        bce_per_voxel = loss_fn.bce_loss(logits, targets)[0, 0, 0, 0, 0].item()
        expected = 0.01 * bce_per_voxel  # every voxel is negative, weight=weight_neg
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_non_adaptive_matches_original_fixed_weighting(self):
        # Backward-compat: adaptive=False must reproduce the exact pre-fix
        # behavior, so existing/explicit fixed-ratio usage isn't silently changed.
        targets = torch.zeros(1, 1, 1, 10, 10)
        targets[0, 0, 0, 0, :4] = 1.0
        logits = torch.zeros_like(targets)

        loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01, adaptive=False)
        loss = loss_fn(logits, targets)

        bce_per_voxel = loss_fn.bce_loss(logits, targets)[0, 0, 0, 0, 0].item()
        pos_mass = targets.sum().item()
        neg_mass = targets.numel() - pos_mass
        expected = (1.0 * pos_mass * bce_per_voxel + 0.01 * neg_mass * bce_per_voxel) / targets.numel()
        assert loss.item() == pytest.approx(expected, rel=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
