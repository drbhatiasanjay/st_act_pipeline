"""
Unit tests for src/dataset.py: CompetitionDataset, against REAL local staged data
(data/staging/train/*.zarr), matching this project's established preference for
testing against real data over synthetic/mocked stores (see test_data_loader_real.py).

Written to close a real, previously-flagged test-coverage gap (no test_dataset.py
existed at all before this file).

Run: py -m pytest tests/test_dataset.py -v
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import AnisotropicZarrLoader
from src.dataset import CompetitionDataset

REAL_TRAIN_DIR = "data/staging/train"
REAL_SAMPLE_ID = "44b6_0113de3b"


def make_dataset(sample_ids):
    """Build a CompetitionDataset without needing a real data_split.json file,
    using the same __new__ bypass pattern already established in this codebase
    (generate_submission.py / inference_kernel.py) for the same reason: split_file
    isn't relevant to what these tests exercise."""
    if not os.path.exists(os.path.join(REAL_TRAIN_DIR, f"{REAL_SAMPLE_ID}.zarr")):
        pytest.skip(f"Real staged data not found at {REAL_TRAIN_DIR} in this environment")

    dataset = CompetitionDataset.__new__(CompetitionDataset)
    dataset.data_dir = Path(REAL_TRAIN_DIR)
    dataset.split_type = "train"
    dataset.normalize = True
    dataset.anisotropy = (4.0, 1.0, 1.0)
    dataset.physical_voxel_size = (1.625, 0.40625, 0.40625)
    dataset.zip_path = None
    dataset.sample_ids = sample_ids
    dataset.pairs = []
    dataset._loader_cache = {}
    dataset._build_pair_index()
    return dataset


class TestPairIndex:
    def test_pairs_built_from_real_local_sample(self):
        dataset = make_dataset([REAL_SAMPLE_ID])

        assert len(dataset) > 0, "expected at least one (frame_t, frame_t+1) pair from real data"
        for sample_id, frame_idx in dataset.pairs:
            assert sample_id == REAL_SAMPLE_ID
            assert frame_idx >= 0

    def test_missing_sample_id_is_skipped_not_crashed(self):
        """A sample_id with no matching .zarr on disk must be silently skipped
        (logged, not raised) -- real behavior needed since data_split.json lists
        149 train samples but only a handful exist in any given local/CI checkout."""
        dataset = make_dataset(["this_sample_id_does_not_exist_anywhere"])

        assert len(dataset) == 0
        assert dataset.pairs == []

    def test_mixed_real_and_missing_sample_ids(self):
        dataset = make_dataset([REAL_SAMPLE_ID, "totally_fake_sample_xyz"])

        assert len(dataset) > 0
        assert all(sample_id == REAL_SAMPLE_ID for sample_id, _ in dataset.pairs)


class TestLoaderCaching:
    def test_get_loader_returns_the_same_instance_on_repeat_calls(self):
        """REGRESSION GUARD for bug 2.1: CompetitionDataset used to construct a
        fresh AnisotropicZarrLoader (real zarr.open() + quantile-attrs extraction)
        on every single __getitem__ call, confirmed live in real Kaggle logs as
        repeated "Opening real Zarr v3 store..." for the same sample at
        closely-spaced timestamps. An `is` identity check is deliberately used
        (not equality) -- a subtly-wrong fix that builds a new-but-equal loader
        each time would still pass a weaker check."""
        dataset = make_dataset([REAL_SAMPLE_ID])

        loader_1 = dataset._get_loader(REAL_SAMPLE_ID)
        loader_2 = dataset._get_loader(REAL_SAMPLE_ID)

        assert loader_1 is loader_2, "expected the cached loader instance to be reused, not rebuilt"
        assert isinstance(loader_1, AnisotropicZarrLoader)

    def test_build_pair_index_and_getitem_share_the_same_cached_loader(self):
        """The loader opened during _build_pair_index() (to read num_frames) must be
        the SAME instance __getitem__ later reuses, not a second independent open."""
        dataset = make_dataset([REAL_SAMPLE_ID])
        loader_from_index_build = dataset._loader_cache[REAL_SAMPLE_ID]

        _ = dataset[0]

        assert dataset._loader_cache[REAL_SAMPLE_ID] is loader_from_index_build


class TestGetItem:
    def test_frame_shape_preserves_full_z_depth_with_added_channel_dim(self):
        """REGRESSION GUARD for the Phase 2 Wave 1 bug already documented in
        CLAUDE.md: frame_t[0:1, :, :] on a (Z,Y,X) array slices axis 0 (Z) down to
        a single plane instead of adding a channel axis, silently discarding 63 of
        64 Z-slices. A weak ndim==3/4-only check would pass for both the buggy and
        correct output -- this asserts the exact shape."""
        dataset = make_dataset([REAL_SAMPLE_ID])
        item = dataset[0]

        assert item["frame_t"].ndim == 4, "expected (C, Z, Y, X)"
        assert item["frame_t"].shape[0] == 1, "channel dim must be added (size 1), not sliced from Z"
        assert item["frame_t"].shape[1] == 64, (
            f"Z dimension must be the real full depth (64), got {item['frame_t'].shape[1]} -- "
            f"this is exactly the failure mode of the Phase 2 Wave 1 slicing bug"
        )
        assert item["frame_t1"].shape == item["frame_t"].shape

    def test_frame_dtype_is_float32(self):
        dataset = make_dataset([REAL_SAMPLE_ID])
        item = dataset[0]

        assert item["frame_t"].dtype.__str__() == "torch.float32"
        assert item["frame_t1"].dtype.__str__() == "torch.float32"

    def test_frame_t1_is_the_next_consecutive_timepoint(self):
        dataset = make_dataset([REAL_SAMPLE_ID])
        item = dataset[5]
        sample_id, frame_idx = dataset.pairs[5]

        assert item["sample_id"] == sample_id
        assert item["t_idx"] == frame_idx
        assert item["metadata"]["t_idx"] == frame_idx

        # frame_t1 must be the loader's own value for timepoint frame_idx+1,
        # not e.g. a repeat of frame_t (which would corrupt inter-frame motion)
        loader = dataset._get_loader(sample_id)
        expected_t1 = loader.load_timepoint_block(frame_idx + 1, normalize=True)
        assert item["frame_t1"].squeeze(0).numpy() == pytest.approx(expected_t1, abs=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
