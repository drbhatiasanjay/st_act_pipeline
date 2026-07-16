"""
Unit tests for src/train.py's TrainingLoop helper methods, using the same
__new__ bypass pattern already established elsewhere in this codebase
(generate_submission.py / inference_kernel.py) to construct a minimal
TrainingLoop without needing real models/DataLoaders -- these methods only
touch a handful of attributes, not the full training apparatus.

Written to close a real, previously-flagged test-coverage gap (no test_train.py
existed at all before this file). Directly regression-tests bug 1.2 (the geff
cache existed but was never wired up) at TrainingLoop._get_gt_nodes -- the
actual call site train_epoch()/validate_epoch() use.

Run: py -m pytest tests/test_train.py -v
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tracksdata

import src.train as train_module
from src.train import TrainingLoop, extract_peaks_from_volume

REAL_TRAIN_DIR = "data/staging/train"
REAL_SAMPLE_ID = "44b6_0113de3b"


def make_bare_training_loop(**extra_attrs):
    """Bypass __init__ (which needs real models/loaders) and set only the
    attributes the method under test actually reads."""
    loop = TrainingLoop.__new__(TrainingLoop)
    for key, value in extra_attrs.items():
        setattr(loop, key, value)
    return loop


class TestGetGtNodesGeffCache:
    def require_real_geff(self):
        if not os.path.exists(os.path.join(REAL_TRAIN_DIR, f"{REAL_SAMPLE_ID}.geff")):
            pytest.skip(f"Real staged .geff not found for {REAL_SAMPLE_ID} in this environment")

    def test_geff_cache_reused_across_repeated_calls_same_sample(self, monkeypatch):
        """REGRESSION GUARD for bug 1.2: _get_gt_nodes() at the real call site
        (train_epoch/validate_epoch call this twice per batch, for t and t+1) must
        reuse self._geff_cache instead of re-parsing the .geff every time."""
        self.require_real_geff()
        call_count = {"n": 0}
        real_from_geff = tracksdata.graph.IndexedRXGraph.from_geff

        def counting_from_geff(path):
            call_count["n"] += 1
            return real_from_geff(path)

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(counting_from_geff))

        loop = make_bare_training_loop(data_dir=Path(REAL_TRAIN_DIR), _geff_cache={})

        loop._get_gt_nodes(REAL_SAMPLE_ID, t_idx=0)
        loop._get_gt_nodes(REAL_SAMPLE_ID, t_idx=1)  # different t, SAME sample/.geff file

        assert call_count["n"] == 1, (
            "two _get_gt_nodes() calls for the same sample at different timepoints "
            "must share one .geff parse via self._geff_cache, not re-parse per call"
        )
        assert len(loop._geff_cache) == 1

    def test_missing_geff_file_returns_none_not_a_crash(self):
        loop = make_bare_training_loop(
            data_dir=Path("data/staging/train"), _geff_cache={}
        )

        result = loop._get_gt_nodes("sample_id_with_no_geff_anywhere", t_idx=0)

        assert result is None

    def test_returns_zero_row_tensor_for_timepoint_with_no_gt_nodes(self):
        """Real geff has GT nodes only at specific t values (e.g. t=0..2, 27-33,
        ...) -- a valid timepoint with zero GT nodes must return an empty (0,3)
        tensor, not None or a crash, since None specifically means load FAILURE."""
        self.require_real_geff()
        loop = make_bare_training_loop(data_dir=Path(REAL_TRAIN_DIR), _geff_cache={})

        # t=99 is within the real volume's 100 timepoints but confirmed to have no
        # GT nodes in this sample's real geff (max real t with nodes is 75)
        result = loop._get_gt_nodes(REAL_SAMPLE_ID, t_idx=99)

        assert result is not None, "a valid timepoint with no GT nodes must return empty tensor, not None"
        assert result.shape == (0, 3)


class TestGenerateAndValidateHeatmapTarget:
    """P0-1 fix (2026-07-16), Section 6 training-side invariant: CompetitionDataset's
    pair index now guarantees every retained training pair has >=1 GT node at both
    t_idx and t_idx+1, so TrainingLoop must fail loudly -- not silently substitute an
    all-zero target -- if a retained pair's heatmap generation fails OR succeeds but
    produces zero target mass in either channel. This is exactly the gap that let a
    real full-epoch run (14,751 batches) silently train on all-background targets
    with no crash, ever (see DetectionLoss docstring)."""

    def make_loop(self):
        return make_bare_training_loop(
            data_dir=Path(REAL_TRAIN_DIR),
            _geff_cache={},
            device=torch.device("cpu"),
        )

    def test_raises_when_heatmap_generation_raises(self, monkeypatch):
        def failing_generate_heatmap_targets(*args, **kwargs):
            raise RuntimeError("simulated GEFF parse failure")

        monkeypatch.setattr(train_module, "generate_heatmap_targets", failing_generate_heatmap_targets)
        loop = self.make_loop()

        with pytest.raises(RuntimeError, match="Heatmap target generation failed"):
            loop._generate_and_validate_heatmap_target(
                REAL_SAMPLE_ID, 5, (7, 8, 16, 16), 8, 16, 16
            )

    def test_raises_when_retained_pair_produces_all_zero_heatmap(self, monkeypatch):
        """The exact regression this invariant guards against: heatmap generation
        SUCCEEDS (no exception) but returns an all-zero target for a pair the
        dataset claims is retained/annotated -- must abort, not silently continue
        training on a fabricated all-background example."""
        def zero_generate_heatmap_targets(sample_id, geff_path, volume_shape, **kwargs):
            z, y, x = volume_shape[1], volume_shape[2], volume_shape[3]
            target_ts = kwargs["target_ts"]
            return {t: torch.zeros((1, z, y, x), dtype=torch.float32) for t in target_ts}, {}

        monkeypatch.setattr(train_module, "generate_heatmap_targets", zero_generate_heatmap_targets)
        loop = self.make_loop()

        with pytest.raises(RuntimeError, match="all-zero heatmap target"):
            loop._generate_and_validate_heatmap_target(
                REAL_SAMPLE_ID, 5, (7, 8, 16, 16), 8, 16, 16
            )

    def test_raises_when_only_channel0_is_all_zero(self, monkeypatch):
        """Independent of the both-channels-zero test: channel 1 (t_idx+1) alone
        having real mass must NOT mask channel 0 (t_idx) being all-zero -- both
        channels are checked with `or`, not just their combined/either sum."""
        def mixed_generate_heatmap_targets(sample_id, geff_path, volume_shape, **kwargs):
            z, y, x = volume_shape[1], volume_shape[2], volume_shape[3]
            t0, t1 = kwargs["target_ts"]
            h1 = torch.zeros((1, z, y, x), dtype=torch.float32)
            h1[0, 0, 0, 0] = 1.0
            return {t0: torch.zeros((1, z, y, x), dtype=torch.float32), t1: h1}, {}

        monkeypatch.setattr(train_module, "generate_heatmap_targets", mixed_generate_heatmap_targets)
        loop = self.make_loop()

        with pytest.raises(RuntimeError, match="all-zero heatmap target"):
            loop._generate_and_validate_heatmap_target(
                REAL_SAMPLE_ID, 5, (7, 8, 16, 16), 8, 16, 16
            )

    def test_raises_when_only_channel1_is_all_zero(self, monkeypatch):
        """Mirror of the channel-0 case: a real, nonzero channel 0 must NOT mask
        channel 1 being all-zero."""
        def mixed_generate_heatmap_targets(sample_id, geff_path, volume_shape, **kwargs):
            z, y, x = volume_shape[1], volume_shape[2], volume_shape[3]
            t0, t1 = kwargs["target_ts"]
            h0 = torch.zeros((1, z, y, x), dtype=torch.float32)
            h0[0, 0, 0, 0] = 1.0
            return {t0: h0, t1: torch.zeros((1, z, y, x), dtype=torch.float32)}, {}

        monkeypatch.setattr(train_module, "generate_heatmap_targets", mixed_generate_heatmap_targets)
        loop = self.make_loop()

        with pytest.raises(RuntimeError, match="all-zero heatmap target"):
            loop._generate_and_validate_heatmap_target(
                REAL_SAMPLE_ID, 5, (7, 8, 16, 16), 8, 16, 16
            )

    def test_returns_concatenated_target_for_a_valid_nonzero_heatmap(self, monkeypatch):
        """Positive case: a real (nonzero) heatmap for both channels must pass
        through untouched as a (1, 2, Z, Y, X) tensor -- this invariant check must
        not reject legitimately annotated pairs."""
        def real_generate_heatmap_targets(sample_id, geff_path, volume_shape, **kwargs):
            z, y, x = volume_shape[1], volume_shape[2], volume_shape[3]
            target_ts = kwargs["target_ts"]
            heatmaps = {}
            for t in target_ts:
                h = torch.zeros((1, z, y, x), dtype=torch.float32)
                h[0, 0, 0, 0] = 1.0
                heatmaps[t] = h
            return heatmaps, {}

        monkeypatch.setattr(train_module, "generate_heatmap_targets", real_generate_heatmap_targets)
        loop = self.make_loop()

        result = loop._generate_and_validate_heatmap_target(
            REAL_SAMPLE_ID, 5, (7, 8, 16, 16), 8, 16, 16
        )

        assert result.shape == (1, 2, 8, 16, 16)
        assert result.sum().item() == pytest.approx(2.0)


class TestExtractPeaksFromVolumeSubvoxelRefinement:
    """A `vol == pooled` tied plateau is mathematically guaranteed to be
    uniform-valued internally (proven: any two adjacent True voxels are each
    other's local max under a symmetric maximum_filter, forcing equal value;
    confirmed empirically over 20k random-volume trials with zero
    counterexamples). So weighting the centroid by `vol` restricted to just
    the plateau (an earlier, rejected version of this fix) is a no-op --
    real sub-voxel information only exists in the falloff just OUTSIDE the
    plateau. These tests use nms_radius_um=2.0 with voxel_size=(1,1,1),
    giving a real kernel radius of 1 (kernel size 3) -- a degenerate
    nms_radius_um=1.0 gives kernel size 1 (no actual neighborhood), which
    trivially marks every above-threshold voxel as its own "peak" and
    doesn't exercise the tied-plateau behavior these tests target.

    Also covers the two competitor-validated additions (background
    subtraction, max-shift safety cap) layered onto the padded-window
    refinement -- see COMPETITOR_RESEARCH_2026-07-13.md item 1 and
    DEFERRED_IMPROVEMENTS.md item 0."""

    def test_zero_peaks_returns_empty_list(self):
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        assert extract_peaks_from_volume(vol, threshold=0.4, voxel_size=(1, 1, 1), nms_radius_um=2.0) == []

    def test_symmetric_falloff_leaves_centroid_at_geometric_center(self):
        """Sanity check: refinement must not introduce a bias when the
        falloff around the plateau is symmetric."""
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[4, 4, 4] = 10.0
        vol[4, 4, 5] = 10.0  # 2-voxel tied plateau along x, geometric center x=4.5
        vol[4, 4, 3] = 5.0
        vol[4, 4, 6] = 5.0   # symmetric falloff on both sides

        peaks = extract_peaks_from_volume(vol, threshold=0.4, voxel_size=(1, 1, 1), nms_radius_um=2.0)

        assert len(peaks) == 1
        assert np.allclose(peaks[0], [4.0, 4.0, 4.5])

    def test_asymmetric_falloff_pulls_centroid_toward_it(self):
        """REGRESSION-relevant: the actual point of the feature. A brighter
        falloff on one side of an otherwise-tied plateau must pull the
        refined centroid measurably off the plain geometric center -- not
        leave it at the old value (which is what the rejected is_peak-
        weighted version would have done, since it's mathematically a
        no-op)."""
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[4, 4, 4] = 10.0
        vol[4, 4, 5] = 10.0  # tied plateau, geometric center x=4.5
        vol[4, 4, 3] = 2.0   # weak falloff on -x side
        vol[4, 4, 6] = 8.0   # strong falloff on +x side

        peaks = extract_peaks_from_volume(vol, threshold=0.4, voxel_size=(1, 1, 1), nms_radius_um=2.0)

        assert len(peaks) == 1
        assert np.allclose(peaks[0], [4.0, 4.0, 4.8])
        assert not np.allclose(peaks[0], [4.0, 4.0, 4.5]), "must differ from the plain (old) geometric center"

    def test_nearby_peak_does_not_bleed_into_neighbor(self):
        """Two separate single-voxel peaks close enough that their padded
        refinement windows overlap must NOT contaminate each other's
        centroid -- each peak's weight_mask must exclude voxels already
        claimed by a DIFFERENT peak's label."""
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[4, 4, 4] = 10.0  # peak A
        vol[4, 4, 5] = 3.0   # background dip between the two peaks
        vol[4, 4, 6] = 10.0  # peak B

        peaks = extract_peaks_from_volume(
            vol, threshold=0.4, voxel_size=(1, 1, 1), nms_radius_um=2.0, subvoxel_refine_radius=2,
        )

        assert len(peaks) == 2
        peaks_sorted = sorted(peaks, key=lambda p: p[2])
        # Without exclusion, peak A's centroid would be pulled all the way to
        # x=5.0 (the midpoint) by peak B's plateau bleeding into its window.
        assert np.allclose(peaks_sorted[0], [4.0, 4.0, 4.230769230769231])
        assert np.allclose(peaks_sorted[1], [4.0, 4.0, 5.769230769230769])

    def test_shift_exceeding_cap_reverts_to_plateau_center(self):
        """REGRESSION-relevant: the safety bound. The same asymmetric-falloff
        setup that normally refines to x=4.8 must fall back to the plain
        plateau centroid (x=4.5) when max_shift_um is set below the actual
        shift distance."""
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[4, 4, 4] = 10.0
        vol[4, 4, 5] = 10.0
        vol[4, 4, 3] = 2.0
        vol[4, 4, 6] = 8.0

        peaks = extract_peaks_from_volume(
            vol, threshold=0.4, voxel_size=(1, 1, 1), nms_radius_um=2.0, max_shift_um=0.01,
        )

        assert len(peaks) == 1
        assert np.allclose(peaks[0], [4.0, 4.0, 4.5]), "must revert to the plateau centroid, not the capped shift"

    def test_background_subtraction_removes_uniform_local_floor(self):
        """REGRESSION-relevant: the actual point of background subtraction.
        A uniform intensity floor added across the whole refinement window
        (e.g. out-of-focus glow, baseline offset) must be subtracted out via
        the local percentile estimate, recovering the SAME refined centroid
        as the floor-free case -- not a result diluted/pulled toward the
        window's own geometric center by the extra uniform mass."""
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[3:6, 3:6, 3:7] = 3.0  # uniform floor across the eventual padded window
        vol[4, 4, 4] = 13.0  # 10 + floor
        vol[4, 4, 5] = 13.0  # 10 + floor
        vol[4, 4, 3] = 5.0   # 2 + floor (weak falloff)
        vol[4, 4, 6] = 11.0  # 8 + floor (strong falloff)

        peaks = extract_peaks_from_volume(
            vol, threshold=0.4, voxel_size=(1, 1, 1), nms_radius_um=2.0, background_percentile=20.0,
        )

        assert len(peaks) == 1
        assert np.allclose(peaks[0], [4.0, 4.0, 4.8]), "floor must be subtracted, matching the floor-free result"


class TestComputeWarmupLr:
    """REGRESSION GUARD: an earlier version of this ramp used
    global_step/warmup_steps (not (global_step+1)/warmup_steps), which never
    actually reached target_lr -- it topped out at target_lr's 90% for
    warmup_steps=10 and silently stayed there for the rest of training.
    Caught via direct local testing before shipping (2026-07-14)."""

    def test_first_step_is_above_start_lr_not_exactly_start_lr(self):
        lr = TrainingLoop._compute_warmup_lr(global_step=0, warmup_steps=10, start_lr=1e-4, target_lr=1e-2)
        assert lr > 1e-4

    def test_last_warmup_step_reaches_exactly_target_lr(self):
        """The critical regression case: global_step=9 is the LAST call
        made under warmup_steps=10 (the caller stops once
        global_step==warmup_steps) -- this step must land exactly on
        target_lr, not some fraction short of it."""
        lr = TrainingLoop._compute_warmup_lr(global_step=9, warmup_steps=10, start_lr=1e-4, target_lr=1e-2)
        assert lr == pytest.approx(1e-2)

    def test_ramp_is_monotonically_increasing(self):
        lrs = [
            TrainingLoop._compute_warmup_lr(global_step=s, warmup_steps=10, start_lr=1e-4, target_lr=1e-2)
            for s in range(10)
        ]
        assert lrs == sorted(lrs)
        assert len(set(lrs)) == 10, "every step must produce a distinct lr, not a flat/repeated ramp"

    def test_single_step_warmup_reaches_target_immediately(self):
        lr = TrainingLoop._compute_warmup_lr(global_step=0, warmup_steps=1, start_lr=1e-4, target_lr=1e-2)
        assert lr == pytest.approx(1e-2)


class TestPeaksForChannel:
    def make_loop_with_hyperparams(self, **hyperparam_overrides):
        hyperparams = {
            "detection_threshold": 0.5,
            "max_positive_voxel_fraction": 0.005,
            "nms_radius_um": 5.0,
        }
        hyperparams.update(hyperparam_overrides)
        return make_bare_training_loop(hyperparams=hyperparams, device=torch.device("cpu"))

    def test_well_calibrated_volume_uses_fixed_threshold(self):
        """A volume where very few voxels exceed the threshold (well-calibrated,
        trained model) should NOT trigger the adaptive-threshold fallback."""
        loop = self.make_loop_with_hyperparams()
        vol = np.zeros((1, 2, 8, 16, 16), dtype=np.float32)
        vol[0, 0, 4, 8, 8] = 0.9  # a single clear peak, everything else near 0
        detection_probs = torch.from_numpy(vol)

        peaks = loop._peaks_for_channel(detection_probs, channel=0, t_idx=0)

        assert len(peaks) >= 1

    def test_undertrained_volume_triggers_adaptive_threshold_not_a_hang(self):
        """REGRESSION-relevant: an undertrained model's near-uniform sigmoid output
        (all voxels just above the fixed threshold) must trigger the adaptive
        threshold path and still return in reasonable time -- this is the exact
        failure mode the docstring describes (ndimage.label() over near-total-noise
        volumes hanging/ballooning memory)."""
        loop = self.make_loop_with_hyperparams(detection_threshold=0.5)
        vol = np.full((1, 2, 8, 16, 16), 0.6, dtype=np.float32)  # EVERY voxel above threshold
        detection_probs = torch.from_numpy(vol)

        # must complete without hanging; result may be empty or small after
        # adaptive re-thresholding, but must not attempt to label ~100% of voxels
        peaks = loop._peaks_for_channel(detection_probs, channel=0, t_idx=0)

        assert isinstance(peaks, list)


class TestNodesAndFeaturesAtPeaks:
    def test_zero_peaks_returns_empty_tensors_with_correct_feature_dim(self):
        loop = make_bare_training_loop(device=torch.device("cpu"))
        features = torch.randn(1, 16, 4, 8, 8)

        nodes, feats = loop._nodes_and_features_at_peaks(features, peaks=[])

        assert nodes.shape == (0, 3)
        assert feats.shape == (0, 16)

    def test_peak_coordinates_are_clamped_to_feature_map_bounds(self):
        """An out-of-bounds peak (possible from NMS on a small/edge volume) must be
        clamped, not index out of range."""
        loop = make_bare_training_loop(device=torch.device("cpu"))
        features = torch.randn(1, 16, 4, 8, 8)  # (B, C, Z=4, Y=8, X=8)

        # peak coordinates deliberately beyond the valid Z/Y/X range
        nodes, feats = loop._nodes_and_features_at_peaks(features, peaks=[[100.0, 100.0, 100.0]])

        assert nodes.shape == (1, 3)
        assert feats.shape == (1, 16)  # must not have raised an index error

    def test_feature_vector_matches_real_value_at_peak_location(self):
        loop = make_bare_training_loop(device=torch.device("cpu"))
        features = torch.zeros(1, 4, 2, 4, 4)
        features[0, :, 1, 2, 3] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        nodes, feats = loop._nodes_and_features_at_peaks(features, peaks=[[1.0, 2.0, 3.0]])

        assert torch.equal(feats[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


class _FakeDegenerateUNet3D(torch.nn.Module):
    """Always returns deeply-negative logits (sigmoid~0 everywhere) regardless
    of input -- simulates the exact collapsed-model behavior confirmed on a
    real full-epoch Kaggle run (2026-07-13): sigmoid stuck at [0.0000, 0.0000],
    zero peaks on every validation batch."""

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.full((batch, 2, 8, 16, 16), -100.0)
        features = torch.zeros(batch, 4, 8, 16, 16)
        return logits, features


def _make_fake_val_batch(t_idx: int) -> dict:
    return {
        "frame_t": torch.zeros(1, 1, 8, 16, 16),
        "frame_t1": torch.zeros(1, 1, 8, 16, 16),
        "sample_id": ["fake_sample"],
        "t_idx": [t_idx],
    }


class _FakeLateBloomerUNet3D(torch.nn.Module):
    """Returns zero-detection logits for its first `zero_calls` forward()
    calls, then a real high-confidence spike afterward -- simulates a sample
    whose early chronological frames are genuinely empty (no cells yet /
    boundary frames), not a structurally dead model. Under P0-3's
    chronological (shuffle=False) requirement, this exact shape is what the
    OLD first-N-batches circuit breaker would have falsely aborted on."""

    def __init__(self, zero_calls: int):
        super().__init__()
        self.zero_calls = zero_calls
        self.calls = 0

    def forward(self, x):
        batch = x.shape[0]
        self.calls += 1
        if self.calls <= self.zero_calls:
            logits = torch.full((batch, 2, 8, 16, 16), -100.0)
        else:
            logits = torch.full((batch, 2, 8, 16, 16), -100.0)
            logits[:, :, 4, 8, 8] = 100.0  # one clear peak per channel
        features = torch.zeros(batch, 4, 8, 16, 16)
        return logits, features


class _FakeEdgeTransformer(torch.nn.Module):
    """Minimal stand-in for SimpleNodeTransformer: returns a fixed
    high-probability flat (n_t * n_t1,) edge-probability tensor, matching
    greedy_edge_assignment()'s expected shape when candidate_edges=None.
    Real transformer forward-pass numerics aren't under test here."""

    def forward(self, nodes_t, nodes_t1, features_t, features_t1):
        n_t = nodes_t.shape[0]
        n_t1 = nodes_t1.shape[0]
        return torch.full((n_t * n_t1,), 0.9)


class TestValidateEpochCircuitBreaker:
    """REGRESSION GUARD for the exact incident this test class is named after:
    a real 5.6-hour Kaggle run continued through all ~4,950 validation batches
    after the first few already showed zero detections, because validate_epoch()
    had no way to notice a structural failure early.

    P0-3 fix (2026-07-17): the circuit breaker moved from a mid-loop
    first-10-batches check to a POST-PASS check over the complete run's
    UNIQUE node total. The old first-N-batches version relied on
    shuffle=True precisely so those N batches weren't just one sample's
    real, legitimately-empty early/boundary frames -- P0-3 now REQUIRES
    chronological (shuffle=False) order, under which the first N batches
    genuinely ARE one sample's earliest frames, so an early-abort would
    misfire on real empty frames rather than a structurally dead model.
    validate_epoch() still uses a FROZEN model -- a fully-zero pass still
    cannot self-correct, so the check still fires, just only after
    confirming the ENTIRE pass, not an early guess."""

    def make_loop(self, unet3d, num_batches: int = 15, transformer=None):
        if transformer is None:
            transformer = torch.nn.Identity()  # never reached when peaks are always empty
        return make_bare_training_loop(
            unet3d=unet3d,
            transformer=transformer,
            device=torch.device("cpu"),
            _amp_enabled=False,  # validate_epoch()'s autocast() call reads this
            # Non-existent path: geff_path.exists() is False, so GT loading is
            # skipped cleanly (no exception, no evaluation_failure count) for
            # tests that reach that far (i.e. don't raise on zero nodes first).
            data_dir=Path("nonexistent_test_dir_for_circuit_breaker_test"),
            hyperparams={
                "detection_threshold": 0.5,
                "max_positive_voxel_fraction": 0.005,
                "nms_radius_um": 5.0,
                "edge_threshold": 0.5,
                "max_batches_per_epoch": None,
            },
            val_loader=[_make_fake_val_batch(i) for i in range(num_batches)],
            epoch_fallback_counts={
                "heatmap_generation_failure": 0,
                "edge_target_generation_failure": 0,
                "edge_loss_computation_failure": 0,
                "evaluation_failure": 0,
            },
        )

    def test_raises_after_full_pass_on_a_structurally_zero_model(self):
        loop = self.make_loop(_FakeDegenerateUNet3D(), num_batches=15)

        with pytest.raises(RuntimeError, match="ZERO unique graph nodes"):
            loop.validate_epoch()

    def test_full_pass_is_processed_before_raising_not_an_early_abort(self, monkeypatch):
        """Proves the check is now a genuine post-pass check, not a
        disguised early abort -- every batch's forward pass must run before
        the RuntimeError fires."""
        loop = self.make_loop(_FakeDegenerateUNet3D(), num_batches=15)
        seen_batches = []
        real_peaks_for_channel = loop._peaks_for_channel

        def counting_peaks_for_channel(*args, **kwargs):
            seen_batches.append(1)
            return real_peaks_for_channel(*args, **kwargs)

        monkeypatch.setattr(loop, "_peaks_for_channel", counting_peaks_for_channel)

        with pytest.raises(RuntimeError):
            loop.validate_epoch()

        # 2 calls per batch (channel 0 and channel 1) x all 15 batches --
        # the old version raised after only 20 (10 batches).
        assert len(seen_batches) == 30

    def test_chronologically_empty_boundary_frames_do_not_cause_a_false_failure(self):
        """The exact P0-3 regression case: a model whose first several
        chronological batches are genuinely empty (real biological boundary
        frames), but later batches show real detections. Under
        shuffle=False, the OLD first-10-batches breaker would have falsely
        aborted here. The new post-pass check must NOT raise."""
        loop = self.make_loop(
            _FakeLateBloomerUNet3D(zero_calls=12), num_batches=15, transformer=_FakeEdgeTransformer(),
        )

        val_metrics = loop.validate_epoch()

        assert val_metrics["predicted_nodes_total"] > 0
        assert val_metrics["is_structural_zero"] is False


class _SyncThread:
    """Test double for threading.Thread that runs the target synchronously on
    .start() instead of spawning a real thread -- lets tests assert on the
    outcome without sleeping/joining a real background thread."""

    def __init__(self, target, daemon=True):
        self._target = target

    def start(self):
        self._target()


class TestPostNtfyHeartbeat:
    """Regression coverage for the live ntfy.sh progress channel added to
    close the "kaggle kernels output returns stale/empty data mid-run" gap
    (see CLAUDE.md's Kaggle Training Run Monitoring Checklist). Verified
    against a real Kaggle sandbox kernel before being wired in; these tests
    mock requests.post so no real network call happens in CI."""

    def test_posts_exact_payload_to_ntfy_topic(self, monkeypatch):
        calls = []

        def fake_post(url, data=None, timeout=None):
            calls.append((url, data, timeout))

        monkeypatch.setattr(train_module.requests, "post", fake_post)
        monkeypatch.setattr(train_module.threading, "Thread", _SyncThread)

        train_module._post_ntfy_heartbeat({"epoch": 1, "train_loss": 0.5})

        assert len(calls) == 1
        url, data, timeout = calls[0]
        assert url == f"https://ntfy.sh/{train_module.NTFY_TOPIC}"
        assert json.loads(data) == {"epoch": 1, "train_loss": 0.5}
        assert timeout == 5

    def test_network_failure_never_raises(self, monkeypatch):
        def fake_post(*args, **kwargs):
            raise ConnectionError("simulated network failure")

        monkeypatch.setattr(train_module.requests, "post", fake_post)
        monkeypatch.setattr(train_module.threading, "Thread", _SyncThread)

        # Must not raise -- a network hiccup can never be allowed to affect
        # a real training run just to report progress.
        train_module._post_ntfy_heartbeat({"epoch": 1})

    def test_write_batch_heartbeat_triggers_ntfy_post(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            train_module, "_post_ntfy_heartbeat", lambda payload: calls.append(payload)
        )
        loop = make_bare_training_loop(
            progress_file=tmp_path / "progress.json", deployed_sha="abc123",
        )

        loop._write_batch_heartbeat(batch_idx=5, effective_total=100, loss=0.42, max_sigmoid=0.1)

        assert len(calls) == 1
        assert calls[0]["batch"] == 5
        assert calls[0]["train_loss"] == 0.42

    def test_write_batch_heartbeat_skips_ntfy_when_no_progress_file(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            train_module, "_post_ntfy_heartbeat", lambda payload: calls.append(payload)
        )
        loop = make_bare_training_loop(progress_file=None)

        loop._write_batch_heartbeat(batch_idx=5, effective_total=100, loss=0.42, max_sigmoid=0.1)

        assert calls == []


class TestSaveCheckpointSplitIdentity:
    """P0-2 checkpoint/split-identity fix (2026-07-16): save_checkpoint() must
    embed self.split_identity as 'split_membership_sha256' in every saved
    checkpoint, so evaluate_checkpoint.py can later detect a mismatch against
    whatever split is active at evaluation time (see
    src/split_utils.py's validate_checkpoint_split_compatibility()).
    Uses tiny real nn.Module/optimizer stand-ins (not the full UNet3D/
    SimpleNodeTransformer) -- save_checkpoint() only calls .state_dict() on
    them, so a minimal real module is enough and much faster than the real
    models."""

    def _make_loop(self, tmp_path, split_identity):
        model_a = torch.nn.Linear(2, 2)
        model_b = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(list(model_a.parameters()) + list(model_b.parameters()), lr=0.01)
        return make_bare_training_loop(
            unet3d=model_a,
            transformer=model_b,
            optimizer=optimizer,
            checkpoint_dir=tmp_path,
            hyperparams={"seed": 42},
            split_identity=split_identity,
        )

    def test_checkpoint_embeds_split_identity_when_known(self, tmp_path):
        loop = self._make_loop(tmp_path, split_identity="a" * 64)
        loop.save_checkpoint(epoch=1, metrics={"adjusted_edge_jaccard": 0.5})

        saved = list(tmp_path.glob("epoch_*.pt"))
        assert len(saved) == 1
        checkpoint = torch.load(saved[0], weights_only=False)
        assert checkpoint["split_membership_sha256"] == "a" * 64

    def test_checkpoint_omits_split_identity_key_when_unknown(self, tmp_path):
        """"unknown" (TrainingLoop's default) must NOT be written as a literal
        placeholder value -- the key must be absent entirely, so
        validate_checkpoint_split_compatibility()'s legacy-checkpoint
        (checkpoint.get(...) is None) branch fires, not a confusing hard
        mismatch against the string "unknown"."""
        loop = self._make_loop(tmp_path, split_identity="unknown")
        loop.save_checkpoint(epoch=1, metrics={"adjusted_edge_jaccard": 0.5})

        saved = list(tmp_path.glob("epoch_*.pt"))
        checkpoint = torch.load(saved[0], weights_only=False)
        assert "split_membership_sha256" not in checkpoint


class TestSaveLastCheckpointSplitIdentity:
    """P0-2 checkpoint/split-identity fix (2026-07-16), round 2: last_checkpoint.pt
    (the unconditional per-epoch checkpoint saved by _save_last_checkpoint(),
    independent of save_checkpoint()'s val-score-keyed files) must follow the
    exact same identity-embedding contract as save_checkpoint() -- easy to
    miss since it's a separate code path with its own torch.save() call."""

    def _make_loop(self, tmp_path, split_identity):
        model_a = torch.nn.Linear(2, 2)
        model_b = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(list(model_a.parameters()) + list(model_b.parameters()), lr=0.01)
        return make_bare_training_loop(
            unet3d=model_a,
            transformer=model_b,
            optimizer=optimizer,
            checkpoint_dir=tmp_path,
            hyperparams={"seed": 42},
            split_identity=split_identity,
        )

    def test_last_checkpoint_embeds_split_identity_when_known(self, tmp_path):
        loop = self._make_loop(tmp_path, split_identity="a" * 64)
        loop._save_last_checkpoint(epoch=1, train_loss=0.5)

        checkpoint = torch.load(tmp_path / "last_checkpoint.pt", weights_only=False)
        assert checkpoint["split_membership_sha256"] == "a" * 64

    def test_last_checkpoint_omits_split_identity_key_when_unknown(self, tmp_path):
        loop = self._make_loop(tmp_path, split_identity="unknown")
        loop._save_last_checkpoint(epoch=1, train_loss=0.5)

        checkpoint = torch.load(tmp_path / "last_checkpoint.pt", weights_only=False)
        assert "split_membership_sha256" not in checkpoint


class TestLoadCheckpointSplitIdentity:
    """P0-2 checkpoint/split-identity fix (2026-07-16), round 2:
    load_checkpoint() must fail loud by DEFAULT on a missing or mismatched
    checkpoint split identity -- stricter than evaluate_checkpoint.py's
    warn-only legacy handling, since resuming TRAINING from a mismatched
    checkpoint can directly contaminate the currently held-out embryo's
    weights (see src/split_utils.py's validate_resume_checkpoint_split_
    identity() docstring)."""

    def _make_loop(self, split_identity):
        model_a = torch.nn.Linear(2, 2)
        model_b = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(list(model_a.parameters()) + list(model_b.parameters()), lr=0.01)
        return make_bare_training_loop(
            unet3d=model_a,
            transformer=model_b,
            optimizer=optimizer,
            device=torch.device("cpu"),
            split_identity=split_identity,
        )

    def _save_checkpoint_with_identity(self, tmp_path, identity) -> Path:
        """Build a minimal, real torch.save()'d checkpoint (matching the real
        shape load_checkpoint() expects) with the given saved identity.
        identity=None omits the key entirely (legacy checkpoint)."""
        model_a = torch.nn.Linear(2, 2)
        model_b = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(list(model_a.parameters()) + list(model_b.parameters()), lr=0.01)
        checkpoint = {
            "epoch": 1,
            "unet3d_state_dict": model_a.state_dict(),
            "transformer_state_dict": model_b.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": {"adjusted_edge_jaccard": 0.5},
        }
        if identity is not None:
            checkpoint["split_membership_sha256"] = identity
        path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, path)
        return path

    def test_matching_identity_loads_normally(self, tmp_path):
        """Test 1."""
        ckpt_path = self._save_checkpoint_with_identity(tmp_path, "a" * 64)
        loop = self._make_loop(split_identity="a" * 64)

        val_metrics = loop.load_checkpoint(str(ckpt_path))

        assert val_metrics == {"adjusted_edge_jaccard": 0.5}

    def test_mismatching_identity_raises_by_default(self, tmp_path):
        """Test 2."""
        ckpt_path = self._save_checkpoint_with_identity(tmp_path, "b" * 64)
        loop = self._make_loop(split_identity="a" * 64)

        with pytest.raises(RuntimeError, match="different embryo-disjoint folds"):
            loop.load_checkpoint(str(ckpt_path))

    def test_missing_legacy_identity_raises_by_default(self, tmp_path):
        """Test 3."""
        ckpt_path = self._save_checkpoint_with_identity(tmp_path, None)
        loop = self._make_loop(split_identity="a" * 64)

        with pytest.raises(RuntimeError, match="historical, embryo-leaking data_split.json"):
            loop.load_checkpoint(str(ckpt_path))

    def test_explicit_mismatch_override_works(self, tmp_path):
        """Test 4."""
        ckpt_path = self._save_checkpoint_with_identity(tmp_path, "b" * 64)
        loop = self._make_loop(split_identity="a" * 64)

        val_metrics = loop.load_checkpoint(str(ckpt_path), allow_split_mismatch=True)

        assert val_metrics == {"adjusted_edge_jaccard": 0.5}

    def test_explicit_legacy_override_works(self, tmp_path):
        """Test 5."""
        ckpt_path = self._save_checkpoint_with_identity(tmp_path, None)
        loop = self._make_loop(split_identity="a" * 64)

        val_metrics = loop.load_checkpoint(str(ckpt_path), allow_legacy_split=True)

        assert val_metrics == {"adjusted_edge_jaccard": 0.5}

    def test_current_loop_identity_unknown_preserves_backward_compatibility(self, tmp_path, caplog):
        """Test 6: when THIS TrainingLoop itself has no configured split
        identity (not the checkpoint), loading must succeed regardless of
        the checkpoint's own identity -- but log a prominent warning rather
        than silently proceeding, since a caller in this state hasn't been
        updated to pass split_identity=... yet."""
        ckpt_path = self._save_checkpoint_with_identity(tmp_path, "b" * 64)
        loop = self._make_loop(split_identity="unknown")

        with caplog.at_level("WARNING"):
            val_metrics = loop.load_checkpoint(str(ckpt_path))

        assert val_metrics == {"adjusted_edge_jaccard": 0.5}
        assert any("no configured split_identity" in r.message for r in caplog.records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
