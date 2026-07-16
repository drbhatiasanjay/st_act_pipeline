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


def make_dataset(sample_ids, split_type="train", filter_unannotated_pairs=False):
    """Build a CompetitionDataset without needing a real data_split.json file,
    using the same __new__ bypass pattern already established in this codebase
    (generate_submission.py / inference_kernel.py) for the same reason: split_file
    isn't relevant to what these tests exercise. filter_unannotated_pairs defaults
    to False, matching CompetitionDataset.__init__'s own default -- split_type
    alone must never imply filtering (P0-1 follow-up fix, 2026-07-16)."""
    if not os.path.exists(os.path.join(REAL_TRAIN_DIR, f"{REAL_SAMPLE_ID}.zarr")):
        pytest.skip(f"Real staged data not found at {REAL_TRAIN_DIR} in this environment")

    dataset = CompetitionDataset.__new__(CompetitionDataset)
    dataset.data_dir = Path(REAL_TRAIN_DIR)
    dataset.split_type = split_type
    dataset.normalize = True
    dataset.anisotropy = (4.0, 1.0, 1.0)
    dataset.physical_voxel_size = (1.625, 0.40625, 0.40625)
    dataset.zip_path = None
    dataset.filter_unannotated_pairs = filter_unannotated_pairs
    dataset.sample_ids = sample_ids
    dataset.pairs = []
    dataset._loader_cache = {}
    dataset._gt_counts_by_time_cache = {}
    dataset.annotation_pair_stats = None
    dataset._build_pair_index()
    return dataset


class _FakeLoader:
    """Reports a controlled frame count and returns deterministic fake volumes,
    without touching a real Zarr store."""

    def __init__(self, num_frames):
        self._num_frames = num_frames

    def get_shape(self):
        return (self._num_frames, 64, 256, 256)

    def load_timepoint_block(self, t, normalize=True):
        import numpy as np
        # Fill value == t so a test can confirm frame_t1 is really timepoint t+1's
        # own data, not e.g. a repeat of frame_t.
        return np.full((64, 256, 256), fill_value=float(t), dtype=np.float32)


class _FakeGraph:
    """Minimal stand-in for tracksdata's IndexedRXGraph -- only implements what
    CompetitionDataset._get_gt_counts_by_time actually calls: node_attrs(attr_keys=...)
    returning a polars DataFrame with a 't' column."""

    def __init__(self, t_values):
        self._t_values = list(t_values)

    def node_attrs(self, attr_keys=None):
        import polars as pl
        return pl.DataFrame({"t": self._t_values})


def _new_bare_dataset(data_dir, sample_ids, split_type="train", filter_unannotated_pairs=False):
    """Construct a CompetitionDataset via the __new__ bypass with all attributes
    _build_pair_index()/_get_gt_counts_by_time() read, but WITHOUT calling
    _build_pair_index() yet -- callers finish setup (monkeypatching, fixture
    files) then call it themselves. filter_unannotated_pairs defaults to False,
    matching CompetitionDataset.__init__'s own default."""
    dataset = CompetitionDataset.__new__(CompetitionDataset)
    dataset.data_dir = data_dir
    dataset.split_type = split_type
    dataset.normalize = True
    dataset.anisotropy = (4.0, 1.0, 1.0)
    dataset.physical_voxel_size = (1.625, 0.40625, 0.40625)
    dataset.zip_path = None
    dataset.filter_unannotated_pairs = filter_unannotated_pairs
    dataset.sample_ids = sample_ids
    dataset.pairs = []
    dataset._loader_cache = {}
    dataset._gt_counts_by_time_cache = {}
    dataset.annotation_pair_stats = None
    return dataset


def make_dataset_with_gt_counts(
    monkeypatch, tmp_path, sample_ids, gt_counts_by_sample, *,
    filter_unannotated_pairs, num_frames_by_sample=None, split_type="train",
):
    """Build a CompetitionDataset entirely from synthetic/mocked pieces -- no real
    staged Kaggle data required, so these tests never skip (and never depend on
    which samples happen to be staged) in any environment, including CI.

    filter_unannotated_pairs is keyword-only with NO default -- every caller must
    state its intent explicitly. This mirrors the production fix (P0-1 follow-up,
    2026-07-16): split_type alone must never imply filtering, so a test helper
    that silently defaulted this flag one way or the other would hide exactly the
    class of bug (deriving filtering from split_type) this fix corrects.

    Creates an empty {sample_id}.zarr directory under tmp_path (only needs to
    exist for _build_pair_index()'s existence check to pass), monkeypatches
    _get_loader() to return a fake loader reporting a controlled frame count, and
    monkeypatches _get_gt_counts_by_time() to return controlled, synthetic
    per-sample {timepoint: count} maps. Only TestAnnotationFilteringRealData's
    designated integration test still depends on data/staging.
    """
    if num_frames_by_sample is None:
        num_frames_by_sample = dict.fromkeys(sample_ids, 10)

    for sample_id in sample_ids:
        (tmp_path / f"{sample_id}.zarr").mkdir(parents=True, exist_ok=True)

    dataset = _new_bare_dataset(
        tmp_path, sample_ids, split_type=split_type,
        filter_unannotated_pairs=filter_unannotated_pairs,
    )

    def fake_get_gt_counts_by_time(self, sample_id):
        return gt_counts_by_sample[sample_id]

    def fake_get_loader(self, sample_id):
        return _FakeLoader(num_frames_by_sample[sample_id])

    monkeypatch.setattr(CompetitionDataset, "_get_gt_counts_by_time", fake_get_gt_counts_by_time)
    monkeypatch.setattr(CompetitionDataset, "_get_loader", fake_get_loader)

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


class TestAnnotationFiltering:
    """P0-1 fix (2026-07-16): CompetitionDataset used to build (t, t+1) pairs from
    Zarr frame count alone, with no idea whether either frame had a GT-labeled cell.
    Combined with generate_heatmap_targets() returning an all-zero heatmap for an
    unlabeled timepoint and DetectionLoss's adaptive fallback branch reducing to
    plain BCE-against-zero on an all-zero target (see DetectionLoss docstring), a
    completely unannotated frame pair was silently trained as an all-background
    example -- actively teaching the model "no cells exist here" for windows that
    were simply never labeled, not actually empty. These tests pin down the fix:
    _build_pair_index() now retains a training pair only when BOTH frame_idx and
    frame_idx+1 have >=1 GT node, using fully synthetic GT counts/loaders/GEFF data
    (via make_dataset_with_gt_counts / tmp_path) so every test here is deterministic
    and independent of real staged Kaggle data -- none of them can skip in CI. Only
    TestAnnotationFilteringRealData's designated test still depends on
    data/staging."""

    def test_pair_retained_when_both_frames_have_gt_nodes(self, monkeypatch, tmp_path):
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 3, 3: 4}},
            filter_unannotated_pairs=True,
        )

        assert (REAL_SAMPLE_ID, 2) in dataset.pairs
        stats = dataset.annotation_pair_stats['per_sample'][REAL_SAMPLE_ID]
        assert stats['retained'] >= 1

    def test_pair_excluded_when_t_has_zero_gt_nodes(self, monkeypatch, tmp_path):
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 0, 3: 4}},
            filter_unannotated_pairs=True,
        )

        assert (REAL_SAMPLE_ID, 2) not in dataset.pairs
        stats = dataset.annotation_pair_stats['per_sample'][REAL_SAMPLE_ID]
        assert stats['excluded_t_zero'] >= 1

    def test_pair_excluded_when_t1_has_zero_gt_nodes(self, monkeypatch, tmp_path):
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 3, 3: 0}},
            filter_unannotated_pairs=True,
        )

        assert (REAL_SAMPLE_ID, 2) not in dataset.pairs
        stats = dataset.annotation_pair_stats['per_sample'][REAL_SAMPLE_ID]
        assert stats['excluded_t1_zero'] >= 1

    def test_pair_excluded_when_both_frames_have_zero_gt_nodes(self, monkeypatch, tmp_path):
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 0, 3: 0}},
            filter_unannotated_pairs=True,
        )

        assert (REAL_SAMPLE_ID, 2) not in dataset.pairs
        stats = dataset.annotation_pair_stats['per_sample'][REAL_SAMPLE_ID]
        assert stats['excluded_both_zero'] >= 1

    def test_only_valid_consecutive_pairs_retained(self, monkeypatch, tmp_path):
        """GT timepoints = {2, 3, 5, 6, 7} (each with >=1 node, all other t implicitly
        zero). Expected retained pair indices: EXACTLY 2 (2,3 both nonzero), 5 (5,6),
        6 (6,7) -- nothing else. NOT 3 (t=4 is missing/zero -- the gap breaks the
        chain), NOT 4 (t=4 itself is zero)."""
        gt_counts = {2: 1, 3: 1, 5: 1, 6: 1, 7: 1}
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: gt_counts},
            filter_unannotated_pairs=True,
        )

        retained_frame_indices = {frame_idx for sid, frame_idx in dataset.pairs if sid == REAL_SAMPLE_ID}

        assert retained_frame_indices == {2, 5, 6}

    def test_geff_parsed_once_per_sample_and_cache_prevents_reparsing(self, monkeypatch, tmp_path):
        """The .geff for a sample must be parsed exactly once while building the
        pair index for that sample, AND repeated direct calls to
        _get_gt_counts_by_time() afterward must reuse the cache, not re-parse.
        Fully synthetic: IndexedRXGraph.from_geff itself is mocked (never delegates
        to the real parser), so this never depends on real staged .geff data."""
        import tracksdata

        sample_id = REAL_SAMPLE_ID
        (tmp_path / f"{sample_id}.zarr").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"{sample_id}.geff").touch()

        call_count = {"n": 0}

        def counting_from_geff(path):
            call_count["n"] += 1
            return _FakeGraph([2, 3, 4, 5]), object()

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(counting_from_geff))
        monkeypatch.setattr(CompetitionDataset, "_get_loader", lambda self, sid: _FakeLoader(6))

        dataset = _new_bare_dataset(tmp_path, [sample_id], filter_unannotated_pairs=True)
        dataset._build_pair_index()

        assert call_count["n"] == 1, (
            f"expected exactly one .geff parse while building the pair index for "
            f"one sample, got {call_count['n']}"
        )

        # Prove the CACHE itself works, not just that _build_pair_index() happens
        # to call the helper once: three more direct calls for the SAME sample
        # must not trigger any additional IndexedRXGraph.from_geff calls.
        dataset._get_gt_counts_by_time(sample_id)
        dataset._get_gt_counts_by_time(sample_id)
        dataset._get_gt_counts_by_time(sample_id)

        assert call_count["n"] == 1, (
            f"expected _get_gt_counts_by_time's cache to prevent re-parsing across "
            f"repeated calls for the same sample, got {call_count['n']} total "
            f"IndexedRXGraph.from_geff calls after 3 additional direct calls"
        )

    def test_missing_geff_for_existing_zarr_sample_fails_loudly_during_indexing(self, monkeypatch, tmp_path):
        """Case 2, genuine integration test: a sample with an EXISTING Zarr
        directory but NO matching .geff must make _build_pair_index() itself raise
        FileNotFoundError -- exercised through the real indexing path (not by
        calling _get_gt_counts_by_time() directly), so this actually proves the
        "existing Zarr + missing GEFF during indexing" scenario the test name
        claims."""
        sample_id = "sample_with_zarr_but_no_geff"
        (tmp_path / f"{sample_id}.zarr").mkdir(parents=True, exist_ok=True)
        # Deliberately do NOT create f"{sample_id}.geff" under tmp_path.

        monkeypatch.setattr(CompetitionDataset, "_get_loader", lambda self, sid: _FakeLoader(10))

        dataset = _new_bare_dataset(tmp_path, [sample_id], filter_unannotated_pairs=True)
        expected_geff_path = tmp_path / f"{sample_id}.geff"

        with pytest.raises(FileNotFoundError) as exc_info:
            dataset._build_pair_index()

        assert sample_id in str(exc_info.value)
        assert str(expected_geff_path) in str(exc_info.value)

    def test_geff_parse_exception_raises_runtime_error_with_sample_and_path(self, monkeypatch, tmp_path):
        """Case 3: .geff exists but IndexedRXGraph.from_geff() itself raises (e.g.
        corrupt file, schema mismatch) -- must propagate as a RuntimeError naming
        both the sample and the geff path, not silently proceed."""
        import tracksdata

        sample_id = "sample_with_corrupt_geff"
        (tmp_path / f"{sample_id}.geff").touch()

        def failing_from_geff(path):
            raise ValueError("simulated corrupt GEFF")

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(failing_from_geff))

        dataset = _new_bare_dataset(tmp_path, [sample_id])
        expected_geff_path = tmp_path / f"{sample_id}.geff"

        with pytest.raises(RuntimeError) as exc_info:
            dataset._get_gt_counts_by_time(sample_id)

        assert sample_id in str(exc_info.value)
        assert str(expected_geff_path) in str(exc_info.value)

    def test_geff_with_zero_nodes_raises_runtime_error(self, monkeypatch, tmp_path):
        """Case 4: .geff parses successfully but contains zero GT nodes -- must
        raise RuntimeError, not silently build zero training pairs."""
        import tracksdata

        sample_id = "sample_with_empty_geff"
        (tmp_path / f"{sample_id}.geff").touch()

        def empty_from_geff(path):
            return _FakeGraph([]), object()

        monkeypatch.setattr(tracksdata.graph.IndexedRXGraph, "from_geff", staticmethod(empty_from_geff))

        dataset = _new_bare_dataset(tmp_path, [sample_id])

        with pytest.raises(RuntimeError, match="zero GT nodes"):
            dataset._get_gt_counts_by_time(sample_id)

    def test_statistics_are_internally_consistent(self, monkeypatch, tmp_path):
        """candidate_pairs must equal the sum of retained + all three excluded
        categories, with no double-counting."""
        gt_counts = {2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0}
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: gt_counts},
            filter_unannotated_pairs=True,
        )

        s = dataset.annotation_pair_stats
        assert s['total_candidate_pairs'] == (
            s['retained_annotated_pairs']
            + s['excluded_both_zero']
            + s['excluded_t_zero']
            + s['excluded_t1_zero']
        )
        per_sample = s['per_sample'][REAL_SAMPLE_ID]
        assert per_sample['candidate_pairs'] == (
            per_sample['retained']
            + per_sample['excluded_both_zero']
            + per_sample['excluded_t_zero']
            + per_sample['excluded_t1_zero']
        )

    def test_retained_item_preserves_existing_output_shape(self, monkeypatch, tmp_path):
        """Filtering which pairs are BUILT must not change what __getitem__ RETURNS
        for a retained pair -- same shape/dtype/keys contract as before this fix.
        Fully synthetic, same fake loader/GT-count pattern as the rest of this
        class -- no real staged data required."""
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 3, 3: 4}},
            filter_unannotated_pairs=True,
        )
        assert len(dataset) > 0, "expected at least one retained pair"

        item = dataset[0]
        sample_id, frame_idx = dataset.pairs[0]

        assert item["frame_t"].ndim == 4
        assert item["frame_t"].shape[0] == 1
        assert item["frame_t"].shape[1] == 64
        assert item["frame_t1"].shape == item["frame_t"].shape
        assert item["frame_t"].dtype.__str__() == "torch.float32"
        assert item["frame_t1"].dtype.__str__() == "torch.float32"
        assert item["sample_id"] == sample_id
        assert item["t_idx"] == frame_idx
        assert item["metadata"]["t_idx"] == frame_idx

    def test_train_split_without_filter_flag_retains_every_pair(self, monkeypatch, tmp_path):
        """The core regression this follow-up fix exists for: split_type == "train"
        ALONE must never imply filtering. With filter_unannotated_pairs explicitly
        False, a train-split dataset must retain every consecutive pair regardless
        of GT coverage -- exactly what evaluate_checkpoint.py's
        run_evaluation(split_type="train", ...) needs (pure inference/graph
        construction, never backprop)."""
        num_frames = 10
        dataset = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 0, 3: 0}},
            filter_unannotated_pairs=False,
            num_frames_by_sample={REAL_SAMPLE_ID: num_frames},
        )

        assert len(dataset) == num_frames - 1
        assert {frame_idx for _sid, frame_idx in dataset.pairs} == set(range(num_frames - 1))
        assert dataset.annotation_pair_stats is None

    def test_validation_split_with_default_flag_retains_every_pair(self, tmp_path):
        """split_type == "validation" with filter_unannotated_pairs left at its
        DEFAULT (not explicitly passed at all) must retain every consecutive pair
        -- exact count and exact pair indices, not merely "more than some other
        filtered count" (which could pass even if some pairs were wrongly
        dropped)."""
        sample_id = REAL_SAMPLE_ID
        num_frames = 10
        (tmp_path / f"{sample_id}.zarr").mkdir(parents=True, exist_ok=True)

        dataset = _new_bare_dataset(tmp_path, [sample_id], split_type="validation")
        # filter_unannotated_pairs deliberately left unset by the caller here --
        # _new_bare_dataset's own parameter default (False) is exactly what's
        # under test, mirroring CompetitionDataset.__init__'s real default.
        dataset._loader_cache[sample_id] = _FakeLoader(num_frames)
        dataset._build_pair_index()

        assert len(dataset) == num_frames - 1
        assert {frame_idx for _sid, frame_idx in dataset.pairs} == set(range(num_frames - 1))
        assert dataset.annotation_pair_stats is None

    def test_validation_and_test_split_never_call_get_gt_counts_by_time(self, monkeypatch, tmp_path):
        """When filter_unannotated_pairs is False, _get_gt_counts_by_time must
        never be invoked at all -- not "called but its result discarded," genuinely
        never reached. A monkeypatch that raises on any call proves this directly
        rather than inferring it from the resulting pair count."""
        sample_id = REAL_SAMPLE_ID
        (tmp_path / f"{sample_id}.zarr").mkdir(parents=True, exist_ok=True)

        def poison_get_gt_counts_by_time(self, sid):
            raise AssertionError(
                "_get_gt_counts_by_time must not be called when "
                "filter_unannotated_pairs is False"
            )

        monkeypatch.setattr(CompetitionDataset, "_get_gt_counts_by_time", poison_get_gt_counts_by_time)
        monkeypatch.setattr(CompetitionDataset, "_get_loader", lambda self, sid: _FakeLoader(10))

        validation_dataset = _new_bare_dataset(tmp_path, [sample_id], split_type="validation")
        validation_dataset._build_pair_index()  # must not raise

        test_dataset = _new_bare_dataset(tmp_path, [sample_id], split_type="test")
        test_dataset._build_pair_index()  # must not raise

        assert len(validation_dataset) == 9
        assert len(test_dataset) == 9

    def test_annotation_pair_stats_populated_only_when_filtering_enabled(self, monkeypatch, tmp_path):
        """annotation_pair_stats must be the dict when filter_unannotated_pairs is
        True, and None whenever it's False -- regardless of split_type."""
        filtered = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 1, 3: 1}},
            filter_unannotated_pairs=True,
        )
        assert filtered.annotation_pair_stats is not None

        unfiltered_train = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 1, 3: 1}},
            filter_unannotated_pairs=False,
        )
        assert unfiltered_train.annotation_pair_stats is None

        unfiltered_validation = make_dataset_with_gt_counts(
            monkeypatch, tmp_path, [REAL_SAMPLE_ID], {REAL_SAMPLE_ID: {2: 1, 3: 1}},
            filter_unannotated_pairs=False, split_type="validation",
        )
        assert unfiltered_validation.annotation_pair_stats is None


class TestAnnotationFilteringRealData:
    """Test 12 (real-data integration): confirms the filtering behaves correctly
    against the REAL staged .geff, not just synthetic mocks."""

    def test_retained_pairs_have_real_gt_at_both_endpoints_and_some_pairs_excluded(self):
        dataset = make_dataset([REAL_SAMPLE_ID], filter_unannotated_pairs=True)
        gt_counts = dataset._get_gt_counts_by_time(REAL_SAMPLE_ID)

        assert len(dataset) > 0
        for _sample_id, frame_idx in dataset.pairs:
            assert gt_counts.get(frame_idx, 0) > 0
            assert gt_counts.get(frame_idx + 1, 0) > 0

        stats = dataset.annotation_pair_stats['per_sample'][REAL_SAMPLE_ID]
        excluded = stats['excluded_both_zero'] + stats['excluded_t_zero'] + stats['excluded_t1_zero']
        assert excluded > 0, (
            f"expected real sample {REAL_SAMPLE_ID} (known-sparse GT, see "
            f"data/staging/README.md) to have at least one excluded candidate pair"
        )


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
