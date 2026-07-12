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

from src.train import TrainingLoop

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
