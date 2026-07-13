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
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tracksdata

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
    doesn't exercise the tied-plateau behavior these tests target."""

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


class TestValidateEpochCircuitBreaker:
    """REGRESSION GUARD for the exact incident this test class is named after:
    a real 5.6-hour Kaggle run continued through all ~4,950 validation batches
    after the first few already showed zero detections, because validate_epoch()
    had no way to notice a structural failure early. validate_epoch() uses a
    FROZEN model (no weight updates happen during validation) -- if the first N
    batches predict zero nodes, no later batch can differ, so this must raise
    fast rather than run to completion confirming what's already certain."""

    def make_degenerate_loop(self, num_batches: int = 15):
        return make_bare_training_loop(
            unet3d=_FakeDegenerateUNet3D(),
            transformer=torch.nn.Identity(),  # never reached -- peaks are always empty
            device=torch.device("cpu"),
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

    def test_raises_within_10_batches_on_a_structurally_zero_model(self):
        loop = self.make_degenerate_loop(num_batches=15)

        with pytest.raises(RuntimeError, match="ZERO nodes"):
            loop.validate_epoch()

    def test_does_not_raise_before_the_check_point(self, monkeypatch):
        """The breaker must check AT batch 10, not before -- a model that's
        merely slow to warm up shouldn't be killed on batch 1."""
        loop = self.make_degenerate_loop(num_batches=15)
        seen_batches = []
        real_peaks_for_channel = loop._peaks_for_channel

        def counting_peaks_for_channel(*args, **kwargs):
            seen_batches.append(1)
            return real_peaks_for_channel(*args, **kwargs)

        monkeypatch.setattr(loop, "_peaks_for_channel", counting_peaks_for_channel)

        with pytest.raises(RuntimeError):
            loop.validate_epoch()

        # 2 calls per batch (channel 0 and channel 1) x 10 batches processed
        # before the breaker fires on the 10th
        assert len(seen_batches) == 20


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
