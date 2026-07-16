"""
Unit tests for scripts/local_smoke_train.py's build_and_validate_targets --
the P0-2 fix (2026-07-16) that replaces the old silent
heatmaps.get(t_idx, zero_ch) fallback with the same fail-loud supervision
invariant TrainingLoop._generate_and_validate_heatmap_target enforces in the
real training loop (src/train.py). Fully synthetic -- no real model, dataset,
or GEFF file required.

Run: py -m pytest tests/test_local_smoke_train.py -v
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.local_smoke_train import build_and_validate_targets, validate_resume_split_identity


def make_heatmap(fill_value: float = 0.0) -> torch.Tensor:
    h = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    if fill_value:
        h[0, 0, 0, 0] = fill_value
    return h


class TestBuildAndValidateTargets:
    def test_raises_when_t_idx_missing(self):
        heatmaps = {6: make_heatmap(1.0)}  # missing key 5 (t_idx)
        with pytest.raises(RuntimeError, match="did not return both expected timepoints"):
            build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

    def test_raises_when_t_idx_plus_one_missing(self):
        heatmaps = {5: make_heatmap(1.0)}  # missing key 6 (t_idx+1)
        with pytest.raises(RuntimeError, match="did not return both expected timepoints"):
            build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

    def test_raises_when_both_timepoints_missing(self):
        heatmaps = {}
        with pytest.raises(RuntimeError, match="did not return both expected timepoints"):
            build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

    def test_raises_when_channel0_is_all_zero(self):
        heatmaps = {5: make_heatmap(0.0), 6: make_heatmap(1.0)}
        with pytest.raises(RuntimeError, match="all-zero heatmap target"):
            build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

    def test_raises_when_channel1_is_all_zero(self):
        heatmaps = {5: make_heatmap(1.0), 6: make_heatmap(0.0)}
        with pytest.raises(RuntimeError, match="all-zero heatmap target"):
            build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

    def test_raises_when_both_channels_are_all_zero(self):
        heatmaps = {5: make_heatmap(0.0), 6: make_heatmap(0.0)}
        with pytest.raises(RuntimeError, match="all-zero heatmap target"):
            build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

    def test_valid_targets_are_concatenated_correctly(self):
        heatmaps = {5: make_heatmap(1.0), 6: make_heatmap(2.0)}
        result = build_and_validate_targets(heatmaps, t_idx=5, sample_id="sample_a")

        assert result.shape == (1, 2, 4, 8, 8)
        assert result[0, 0, 0, 0, 0].item() == pytest.approx(1.0)
        assert result[0, 1, 0, 0, 0].item() == pytest.approx(2.0)


class TestValidateResumeSplitIdentity:
    """P0-2 checkpoint/split-identity fix (2026-07-16): resuming this
    script's own checkpoint under a DIFFERENT active split than it started
    with is always a bug -- unlike evaluate_checkpoint.py's cross-fold case,
    there is no legitimate reason to resume mid-training under a different
    split."""

    CKPT_PATH = Path("dummy_checkpoint.pt")

    def test_matching_identity_does_not_raise(self):
        state = {"split_membership_sha256": "a" * 64}
        validate_resume_split_identity(state, "a" * 64, self.CKPT_PATH)

    def test_missing_identity_raises_by_default(self):
        """P0-2 round 2 (2026-07-16): changed from warn-and-continue to
        fail-by-default -- resuming training on a legacy (identity-less)
        checkpoint could be continuing to train weights that already
        accumulated gradient signal from the historical, embryo-leaking
        data_split.json."""
        state = {}  # predates the P0-2 fix
        with pytest.raises(RuntimeError, match="historical, embryo-leaking data_split.json"):
            validate_resume_split_identity(state, "a" * 64, self.CKPT_PATH)

    def test_missing_identity_with_allow_legacy_resume_warns_instead_of_raising(self, capsys):
        state = {}  # predates the P0-2 fix
        validate_resume_split_identity(state, "a" * 64, self.CKPT_PATH, allow_legacy_resume=True)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "no saved split_membership_sha256" in captured.out

    def test_mismatched_identity_raises(self):
        state = {"split_membership_sha256": "b" * 64}
        with pytest.raises(RuntimeError, match="always a bug, never intentional"):
            validate_resume_split_identity(state, "a" * 64, self.CKPT_PATH)

    def test_mismatched_identity_is_never_bypassable_by_allow_legacy_resume(self):
        """allow_legacy_resume only covers a MISSING identity, never a real
        mismatch -- a known identity mismatch always indicates a real bug
        with no legitimate override, unlike evaluate_checkpoint.py's
        cross-fold evaluation case."""
        state = {"split_membership_sha256": "b" * 64}
        with pytest.raises(RuntimeError, match="always a bug, never intentional"):
            validate_resume_split_identity(state, "a" * 64, self.CKPT_PATH, allow_legacy_resume=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
