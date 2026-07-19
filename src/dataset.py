"""
PyTorch Dataset class for ST-ACT competition.

CompetitionDataset loads Zarr v3 volumes and .geff ground truth.
Produces (frame_t, frame_t+1) pairs with anisotropic metadata.
Respects the train/val split named by whichever split_file is passed in.

P0-2 correction (2026-07-16): the historical, pre-P0-2 content of
data_split.json was NOT embryo-disjoint -- it stratified by embryo prefix
rather than excluding an embryo's samples from one side, per Kaggle's own
competition documentation ("multiple samples may share the same embryo").
The current root data_split.json has since been replaced with a compatibility
alias (an exact copy of data_splits/embryo_44b6_validation.json), so it is
now genuinely embryo-disjoint too. Prefer resolving the active fold via
scripts/build_train_val_split.py's leave-one-embryo-out generator /
src/split_utils.py's resolve_split_file_path() rather than hardcoding this
filename.
"""

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tracksdata
from torch.utils.data import Dataset

from src.data_loader import AnisotropicZarrLoader

logger = logging.getLogger(__name__)


class CompetitionDataset(Dataset):
    """
    PyTorch Dataset for ST-ACT competition.

    Loads Zarr v3 volumes + .geff ground truth from either:
    - Local staged directory (for development)
    - Kaggle-mounted competition path (for training)

    Produces (frame_t, frame_t+1) pairs with anisotropic metadata.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        split_type: str = "train",
        normalize: bool = True,
        anisotropy: tuple[float, float, float] = (4.0, 1.0, 1.0),
        zip_path: str | Path | None = None,
        filter_unannotated_pairs: bool = False,
        strict_sample_coverage: bool = False,
        sample_id_allowlist: list[str] | None = None,
    ):
        """
        Initialize CompetitionDataset.

        Args:
            data_dir: Path to directory containing Zarr/geff data
                      (local staging or Kaggle-mounted competition path)
            split_file: Path to data_split.json
            split_type: which sample-id membership list to load from split_file
                (e.g. "train", "validation", "test") -- selects WHICH samples this
                dataset covers, nothing else. Does NOT control annotation
                filtering (see filter_unannotated_pairs); a P0-1 regression
                (2026-07-16) derived filtering from `split_type == "train"`, which
                incorrectly filtered train-split samples used for pure inference
                (e.g. evaluate_checkpoint.py's run_evaluation(split_type="train",
                ...)) -- fixed by decoupling the two concerns.
            normalize: Whether to apply normalization (default: True)
            anisotropy: Anisotropy ratio (Z:Y:X), default (4.0, 1.0, 1.0)
            zip_path: Optional path to zip file for extracting samples on-the-fly
            filter_unannotated_pairs: If True, _build_pair_index() retains a
                (t, t+1) pair only when BOTH timepoints have >=1 GT node (see
                _get_gt_counts_by_time). Set True ONLY for the actual training
                dataset used for optimizer/backpropagation (see DetectionLoss
                docstring for why an all-zero target is actively harmful during
                training). Every other caller -- validation inference, train-split
                evaluation, test/submission inference -- must leave this False so
                pair coverage matches the real frame timeline, not GT coverage.
                Default False: the pre-P0-1 unconditional behavior.
            strict_sample_coverage: P0-7 (2026-07-19). If True, a missing/
                unreadable expected Zarr, or a sample that opens successfully but
                produces zero usable (frame_t, frame_t+1) pairs, raises RuntimeError
                during __init__ instead of being silently soft-skipped -- see
                expected_sample_ids/successfully_opened_sample_ids/
                zero_pairs_sample_ids/failed_sample_ids below. Default False
                preserves the pre-P0-7 soft-skip behavior (expected for local/CI
                checkouts where only a handful of a split's samples are staged).
                Kaggle production training/validation datasets must pass True.
            sample_id_allowlist: GPU sanity gate (2026-07-19). Fail-closed contract,
                deliberately stricter than an ordinary optional filter:
                - None (default): no restriction, unchanged pre-existing behavior.
                - Explicitly [] (empty list): raises ValueError. A GPU sanity gate
                  run is only ever meaningful against a real, non-empty configured
                  subset -- an empty allowlist almost certainly means a caller bug
                  upstream (e.g. an unpopulated K-expansion result), not "train on
                  zero samples", so this is refused rather than silently accepted
                  as a valid (if useless) zero-sample dataset.
                - Duplicate IDs: raises ValueError listing the exact duplicates.
                  Never silently collapsed via set() -- a caller passing duplicates
                  has a real bug (e.g. double-counting in a candidate-expansion
                  loop) that a silent dedup would hide.
                - IDs absent from split_file's split_type list: raises ValueError
                  listing them (distinct from strict_sample_coverage, which is
                  about a listed sample's Zarr being missing/unreadable on disk,
                  not about the allowlist itself naming an ID the split doesn't
                  even claim to have).
                - Valid, duplicate-free, fully-present IDs: self.sample_ids is
                  filtered to exactly this set, applied BEFORE _build_pair_index()
                  runs -- every downstream coverage primitive (expected_sample_ids
                  etc.) reflects only the allowlisted subset, in split_file's own
                  original relative order (never re-ordered to match the
                  allowlist argument's order -- see the post-BUILD invariant
                  check below for why this matters).
                - Post-BUILD invariant (mandatory, always executed when
                  sample_id_allowlist is not None, AFTER self._build_pair_index()
                  returns -- not merely right after the pre-build filter step):
                  set(self.sample_ids) must exactly equal set(sample_id_allowlist)
                  -- raises RuntimeError with a diagnostic otherwise. Checking
                  after the build, not just after the filter, is what lets this
                  catch self.sample_ids being mutated during index construction,
                  not only a bypassed pre-build check -- a pre-build-only check
                  can never observe a post-build mutation (Codex review, PR #4,
                  2026-07-19). Provably unreachable in the current
                  implementation given the duplicate/missing-ID checks above and
                  _build_pair_index() not itself reassigning self.sample_ids,
                  kept anyway per this codebase's "should be unreachable but
                  checked anyway" convention (e.g. train.py's
                  edge_targets.numel()==0 check) -- proven effective, not just
                  unreachable, by
                  test_post_build_invariant_catches_mutation_during_build_pair_index.
                strict_sample_coverage composes on top of this, unaffected: it
                still governs only whether a genuinely missing/unreadable Zarr
                *within the filtered selection* raises vs. soft-skips -- it never
                reaches samples excluded by the allowlist, which are filtered out
                of self.sample_ids before _build_pair_index()'s per-sample loop
                even starts (excluded samples' Zarr/.geff are never opened,
                touched, or even existence-checked).
        """
        self.data_dir = Path(data_dir)
        self.split_type = split_type
        self.normalize = normalize
        self.anisotropy = anisotropy
        self.physical_voxel_size = (1.625, 0.40625, 0.40625)  # um
        self.zip_path = Path(zip_path) if zip_path else None
        self.filter_unannotated_pairs = filter_unannotated_pairs
        self.strict_sample_coverage = strict_sample_coverage
        self.sample_id_allowlist = sample_id_allowlist

        # Load split file
        split_file = Path(split_file)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            split_data = json.load(f)

        self.sample_ids = split_data[split_type]
        logger.info(
            f"Loaded {len(self.sample_ids)} samples for split '{split_type}' "
            f"from {split_file}"
        )

        if sample_id_allowlist is not None:
            if len(sample_id_allowlist) == 0:
                raise ValueError(
                    "sample_id_allowlist must not be an empty list -- pass None "
                    "to disable filtering. An explicitly empty allowlist is "
                    "refused rather than silently accepted as a valid "
                    "zero-sample dataset."
                )

            seen: set[str] = set()
            duplicates: set[str] = set()
            for sid in sample_id_allowlist:
                if sid in seen:
                    duplicates.add(sid)
                seen.add(sid)
            if duplicates:
                raise ValueError(
                    f"sample_id_allowlist contains duplicate sample IDs "
                    f"(not silently collapsed): {sorted(duplicates)}"
                )

            allowlist_set = set(sample_id_allowlist)
            split_set = set(self.sample_ids)
            missing_from_split = allowlist_set - split_set
            if missing_from_split:
                raise ValueError(
                    f"sample_id_allowlist contains sample IDs not present in "
                    f"split '{split_type}' of {split_file}: "
                    f"{sorted(missing_from_split)}"
                )
            self.sample_ids = [s for s in self.sample_ids if s in allowlist_set]
            logger.info(
                f"sample_id_allowlist active: filtered {len(split_set)} -> "
                f"{len(self.sample_ids)} samples for split '{split_type}'"
            )

        # Build index of (frame_t, frame_t+1) pairs
        self.pairs = []
        # One AnisotropicZarrLoader per sample_id, reused across _build_pair_index and
        # every __getitem__ call for that sample instead of reopening the store (real
        # zarr.open() + quantile-attrs extraction + several logger.info() calls) on
        # every single frame access -- confirmed live in Kaggle logs this session
        # (repeated "Opening real Zarr v3 store..." at closely-spaced timestamps for the
        # same sample_id, since shuffle=False means ~100 consecutive pairs per sample).
        self._loader_cache: dict[str, AnisotropicZarrLoader] = {}
        # Per-sample {timepoint: gt_node_count}, parsed from .geff once and reused --
        # see _get_gt_counts_by_time. Only populated/consulted when
        # filter_unannotated_pairs is True (see _build_pair_index).
        self._gt_counts_by_time_cache: dict[str, dict[int, int]] = {}
        # P0-1 fix (2026-07-16): audit trail of which candidate (t, t+1) pairs were
        # kept vs. dropped for lacking GT coverage. None when filter_unannotated_pairs
        # is False (nothing is filtered there -- see _build_pair_index). See
        # CLAUDE.md for the underlying bug this exists to prevent.
        self.annotation_pair_stats: dict[str, Any] | None = None
        self._build_pair_index()

        # Mandatory post-BUILD invariant (Codex review, PR #4, 2026-07-19):
        # design §3.0's "assert actual_ids == configured_ids" must run AFTER
        # _build_pair_index(), not merely re-check the filter step that
        # immediately preceded it -- checking right after the filter can only
        # ever re-validate that same filter and can never detect
        # self.sample_ids being mutated during index construction. Checking
        # here, after _build_pair_index() has actually run, is the only
        # position that can catch that class of bug. See
        # TestSampleIdAllowlist::test_post_build_invariant_catches_mutation_during_build_pair_index
        # for a regression test that proves this position is effective, not
        # just unreachable-by-construction.
        if sample_id_allowlist is not None:
            allowlist_set = set(sample_id_allowlist)
            if set(self.sample_ids) != allowlist_set:
                raise RuntimeError(
                    f"sample_id_allowlist invariant violated: self.sample_ids "
                    f"{sorted(set(self.sample_ids))} does not exactly equal the "
                    f"configured allowlist {sorted(allowlist_set)} after "
                    f"_build_pair_index() ran. This should be unreachable given "
                    f"the checks performed before filtering -- if it fires, "
                    f"self.sample_ids was mutated during index construction."
                )

    def _get_loader(self, sample_id: str) -> AnisotropicZarrLoader:
        """Return this instance's cached loader for sample_id, opening it on first use."""
        loader = self._loader_cache.get(sample_id)
        if loader is None:
            zarr_path = self.data_dir / f"{sample_id}.zarr"
            loader = AnisotropicZarrLoader(str(zarr_path))
            self._loader_cache[sample_id] = loader
        return loader

    def _get_gt_counts_by_time(self, sample_id: str) -> dict[int, int]:
        """
        Return {timepoint: gt_node_count} for sample_id, parsed from its .geff once
        and cached for the lifetime of this dataset instance.

        P0-1 fix (2026-07-16): CompetitionDataset used to build (t, t+1) pairs from
        Zarr frame count alone, with no idea whether either frame actually had a
        labeled GT cell. Combined with generate_heatmap_targets() returning an
        all-zero heatmap for an unlabeled timepoint (correct in isolation) and
        DetectionLoss's adaptive branch falling back to a FIXED weight_neg on an
        all-zero target (see DetectionLoss docstring) -- which cancels out of
        loss.sum()/weights.sum() exactly -- a completely unannotated frame pair was
        silently trained as ordinary mean BCE against an all-background target,
        actively teaching the model "no cells exist here" for windows that were
        simply never labeled, not actually empty. Real GT coverage is genuinely
        sparse (~0.2% of the estimated true cell count in the densest local sample --
        see data/staging/README.md's "Ground truth is far sparser than the PRD's
        prose suggested" section, written 2026-07-03), so this was not a rare edge
        case.

        Deliberately fails loudly rather than substituting a default, matching this
        project's established "silent fallback masks real breakage" lesson (the
        polars/GroupNorm incidents in CLAUDE.md): a training run must not silently
        proceed on annotation data it cannot verify.
        """
        if sample_id in self._gt_counts_by_time_cache:
            return self._gt_counts_by_time_cache[sample_id]

        geff_path = self.data_dir / f"{sample_id}.geff"

        try:
            graph, _ = self.load_geff_gt(sample_id)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Sample {sample_id}: Zarr volume exists but no matching .geff "
                f"ground truth was found at {geff_path} -- cannot determine which "
                f"timepoints are annotated, so training cannot safely proceed for "
                f"this sample."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Sample {sample_id}: failed to parse .geff at {geff_path}: {e}"
            ) from e

        try:
            node_attrs_df = graph.node_attrs(attr_keys=["t"])
            t_vals = node_attrs_df['t'].to_list()
        except Exception as e:
            raise RuntimeError(
                f"Sample {sample_id}: failed to read node 't' attribute from parsed "
                f".geff at {geff_path}: {e}"
            ) from e

        if len(t_vals) == 0:
            raise RuntimeError(
                f"Sample {sample_id}: .geff at {geff_path} parsed successfully but "
                f"contains zero GT nodes -- cannot build any annotated training "
                f"pairs for this sample."
            )

        # Explicit int() cast: node_attrs()'s polars column has been observed to
        # yield plain Python int via .to_list() for this dtype (verified empirically
        # 2026-07-16), but that's an implementation detail of polars, not a
        # contract -- casting explicitly makes the counts dict's key type an
        # invariant of THIS function, not an accident of the upstream library.
        counts: dict[int, int] = {}
        for t in t_vals:
            t_idx = int(t)
            counts[t_idx] = counts.get(t_idx, 0) + 1

        self._gt_counts_by_time_cache[sample_id] = counts
        return counts

    def _build_pair_index(self) -> None:
        """
        Build index of (frame_t, frame_t+1) pairs.

        When self.filter_unannotated_pairs is True: a pair is retained only when
        BOTH frame_idx and frame_idx + 1 have at least one GT node (see
        _get_gt_counts_by_time for why). When False (the default): unchanged,
        unconditional behavior -- every consecutive pair is added regardless of GT
        coverage.

        This is deliberately gated on filter_unannotated_pairs, NOT on
        `split_type == "train"` -- an earlier version of this fix conflated the
        two, which broke evaluate_checkpoint.py's run_evaluation(split_type="train",
        ...) (pure inference/graph construction over train-split samples, no
        backprop) by silently handing it a GT-filtered, incomplete frame timeline
        just because split_type happened to be "train". split_type only selects
        WHICH samples this dataset covers; filter_unannotated_pairs is the only
        thing that should ever gate this filtering. Callers: only the real
        training dataset used for optimizer/backpropagation
        (kaggle_kernel/train_kernel.py's train_dataset) sets this True. Validation
        inference, train-split evaluation, and test/submission inference must all
        leave it False -- filtering any of those would alter graph construction /
        official metric aggregation, out of scope for the P0-1 fix this filtering
        exists for.

        P0-7 (2026-07-19) coverage primitives: every sample_id in self.sample_ids
        (== self.expected_sample_ids) is classified into exactly one of
        successfully_opened_sample_ids / zero_pairs_sample_ids / failed_sample_ids
        by the time this method returns (without raising) -- see
        self.strict_sample_coverage docstring in __init__. A pre-existing,
        unconditional (not gated on strict_sample_coverage) failure mode is
        untouched: a missing/unparseable/empty .geff for a filter_unannotated_pairs
        sample whose Zarr DOES exist still always raises out of
        _get_gt_counts_by_time (Cases 2/3/4 below) -- that is a training-data
        correctness failure regardless of strict_sample_coverage, not a coverage
        gap this flag is about.
        """
        if self.filter_unannotated_pairs:
            self.annotation_pair_stats = {
                'total_candidate_pairs': 0,
                'retained_annotated_pairs': 0,
                'excluded_both_zero': 0,
                'excluded_t_zero': 0,
                'excluded_t1_zero': 0,
                'per_sample': {},
            }

        self.expected_sample_ids: list[str] = list(self.sample_ids)
        self.successfully_opened_sample_ids: list[str] = []
        self.zero_pairs_sample_ids: list[str] = []
        self.failed_sample_ids: list[str] = []

        for sample_id in self.sample_ids:
            # Case 1: sample has no local Zarr at all -- expected in local/CI
            # checkouts where only a handful of the split's samples are staged.
            # Soft-skip (recorded, not raised) unless strict_sample_coverage.
            zarr_path = self.data_dir / f"{sample_id}.zarr"
            if not zarr_path.exists():
                self.failed_sample_ids.append(sample_id)
                # getattr(..., False): a bare CompetitionDataset.__new__() test
                # double (bypassing __init__) may not set self.strict_sample_coverage --
                # real production instances always go through __init__, which
                # defaults it to False anyway.
                if getattr(self, 'strict_sample_coverage', False):
                    raise RuntimeError(
                        f"strict_sample_coverage=True: expected Zarr not found for "
                        f"sample {sample_id} at {zarr_path}."
                    )
                logger.debug(
                    f"Sample {sample_id} not found locally (OK for local testing)"
                )
                continue

            try:
                loader = self._get_loader(sample_id)
                num_frames = loader.get_shape()[0]
            except Exception as e:
                self.failed_sample_ids.append(sample_id)
                if getattr(self, 'strict_sample_coverage', False):
                    raise RuntimeError(
                        f"strict_sample_coverage=True: failed to open expected Zarr "
                        f"for sample {sample_id} at {zarr_path}: {e}"
                    ) from e
                logger.warning(f"Failed to open Zarr for sample {sample_id}: {e}")
                continue

            pairs_before = len(self.pairs)

            if not self.filter_unannotated_pairs:
                # Unconditional behavior, unchanged from before this fix.
                for frame_idx in range(num_frames - 1):
                    self.pairs.append((sample_id, frame_idx))
                logger.debug(
                    f"Sample {sample_id}: {num_frames} frames → {num_frames - 1} pairs"
                )
            else:
                # Cases 2/3/4 (deliberately NOT caught here): a missing/unparseable/
                # empty .geff for a sample whose Zarr DOES exist is a training-data
                # correctness failure, not a benign "sample not staged" gap -- must
                # propagate all the way out of __init__, not be swallowed into a
                # logger.warning like the Zarr-existence check above.
                gt_counts = self._get_gt_counts_by_time(sample_id)

                sample_stats = {
                    'candidate_pairs': 0,
                    'retained': 0,
                    'excluded_both_zero': 0,
                    'excluded_t_zero': 0,
                    'excluded_t1_zero': 0,
                }
                for frame_idx in range(num_frames - 1):
                    sample_stats['candidate_pairs'] += 1
                    count_t = gt_counts.get(frame_idx, 0)
                    count_t1 = gt_counts.get(frame_idx + 1, 0)

                    if count_t > 0 and count_t1 > 0:
                        self.pairs.append((sample_id, frame_idx))
                        sample_stats['retained'] += 1
                    elif count_t == 0 and count_t1 == 0:
                        sample_stats['excluded_both_zero'] += 1
                    elif count_t == 0:
                        sample_stats['excluded_t_zero'] += 1
                    else:
                        sample_stats['excluded_t1_zero'] += 1

                self.annotation_pair_stats['per_sample'][sample_id] = sample_stats
                self.annotation_pair_stats['total_candidate_pairs'] += sample_stats['candidate_pairs']
                self.annotation_pair_stats['retained_annotated_pairs'] += sample_stats['retained']
                self.annotation_pair_stats['excluded_both_zero'] += sample_stats['excluded_both_zero']
                self.annotation_pair_stats['excluded_t_zero'] += sample_stats['excluded_t_zero']
                self.annotation_pair_stats['excluded_t1_zero'] += sample_stats['excluded_t1_zero']

                logger.info(
                    f"Sample {sample_id}: candidate_pairs={sample_stats['candidate_pairs']} "
                    f"retained={sample_stats['retained']} "
                    f"excluded_both_zero={sample_stats['excluded_both_zero']} "
                    f"excluded_t_zero={sample_stats['excluded_t_zero']} "
                    f"excluded_t1_zero={sample_stats['excluded_t1_zero']}"
                )

            # P0-7 Section F: opened successfully but produced zero usable pairs is
            # a COVERAGE-INTEGRITY failure, distinct from the technical
            # missing/unreadable-Zarr failures above -- never placed in
            # failed_sample_ids.
            pairs_added = len(self.pairs) - pairs_before
            if pairs_added == 0:
                self.zero_pairs_sample_ids.append(sample_id)
                if getattr(self, 'strict_sample_coverage', False):
                    raise RuntimeError(
                        f"strict_sample_coverage=True: sample {sample_id} opened "
                        f"successfully but produced zero usable (frame_t, frame_t+1) "
                        f"pairs (num_frames={num_frames}, "
                        f"filter_unannotated_pairs={self.filter_unannotated_pairs})."
                    )
                logger.warning(
                    f"Sample {sample_id} opened successfully but produced zero "
                    f"usable pairs (num_frames={num_frames}, "
                    f"filter_unannotated_pairs={self.filter_unannotated_pairs})."
                )
            else:
                self.successfully_opened_sample_ids.append(sample_id)

        logger.info(
            f"Coverage: expected={len(self.expected_sample_ids)} "
            f"successfully_opened={len(self.successfully_opened_sample_ids)} "
            f"zero_pairs={len(self.zero_pairs_sample_ids)} "
            f"failed={len(self.failed_sample_ids)}"
        )

        if self.filter_unannotated_pairs:
            s = self.annotation_pair_stats
            excluded_total = (
                s['excluded_both_zero'] + s['excluded_t_zero'] + s['excluded_t1_zero']
            )
            logger.info(
                f"Dataset total: candidate_pairs={s['total_candidate_pairs']} "
                f"retained={s['retained_annotated_pairs']} excluded={excluded_total} "
                f"(both_zero={s['excluded_both_zero']} t_zero={s['excluded_t_zero']} "
                f"t1_zero={s['excluded_t1_zero']})"
            )

        logger.info(f"Built index: {len(self.pairs)} (frame_t, frame_t+1) pairs")

    def __len__(self) -> int:
        """Return total number of (frame_t, frame_t+1) pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load (frame_t, frame_t+1) pair by index.

        Returns:
            {
                "frame_t": (C=1, Z=64, Y=256, X=256) float32, full 3D volume,
                "frame_t1": (C=1, Z=64, Y=256, X=256) float32, full 3D volume,
                "sample_id": str,
                "t_idx": int (frame index for frame_t),
                "metadata": {
                    "sample_id": str,
                    "t_idx": int,
                    "volume_shape": tuple,
                    "physical_voxel_size": tuple,
                    "anisotropy_ratio": tuple,
                }
            }
        """
        sample_id, frame_idx = self.pairs[idx]

        # Load volume (reuses this instance's cached loader -- see _get_loader)
        loader = self._get_loader(sample_id)

        # Extract frame_t and frame_t+1
        frame_t = loader.load_timepoint_block(frame_idx, normalize=self.normalize)
        frame_t1 = loader.load_timepoint_block(frame_idx + 1, normalize=self.normalize)

        # Ensure float32
        frame_t = frame_t.astype(np.float32)
        frame_t1 = frame_t1.astype(np.float32)

        # load_timepoint_block() returns (Z, Y, X) = (64, 256, 256). Add a leading
        # channel dimension -> (1, Z, Y, X) = (1, 64, 256, 256), preserving the full
        # 3D volume. Task 2.1's UNet3D needs the real Z depth, not a single slice --
        # an earlier version of this code used frame_t[0:1, :, :] here, which SLICES
        # axis 0 (Z) down to one plane instead of adding a new axis, silently
        # discarding 63 of 64 Z-slices. Caught during plan verification, not by this
        # file's own test (which logged shapes but never asserted them).
        if frame_t.ndim == 3:
            frame_t = frame_t[np.newaxis, :, :, :]
            frame_t1 = frame_t1[np.newaxis, :, :, :]

        # Convert to torch
        frame_t = torch.from_numpy(frame_t).float()
        frame_t1 = torch.from_numpy(frame_t1).float()

        volume_shape = loader.get_shape()
        metadata = {
            "sample_id": sample_id,
            "t_idx": frame_idx,
            "volume_shape": volume_shape,
            "physical_voxel_size": self.physical_voxel_size,
            "anisotropy_ratio": self.anisotropy,
        }

        return {
            "frame_t": frame_t,
            "frame_t1": frame_t1,
            "sample_id": sample_id,
            "t_idx": frame_idx,
            "metadata": metadata,
        }

    def load_geff_gt(self, sample_id: str) -> tuple[Any, Any]:
        """
        Load ground truth graph from .geff file for a sample.

        Returns:
            (graph, metadata) tuple from tracksdata.graph.IndexedRXGraph.from_geff()
        """
        geff_path = self.data_dir / f"{sample_id}.geff"
        if not geff_path.exists():
            if self.zip_path:
                # Extract from zip
                return self._load_geff_from_zip(sample_id)
            raise FileNotFoundError(f"GEFF file not found: {geff_path}")

        return tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))

    def _load_geff_from_zip(
        self, sample_id: str
    ) -> tuple[Any, Any]:
        """Load .geff file from zip archive."""
        if not self.zip_path:
            raise ValueError("zip_path not configured")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                # Extract .geff entry
                geff_entry = f"train/{sample_id}.geff/"
                members = [m for m in zf.namelist() if m.startswith(geff_entry)]

                if not members:
                    raise FileNotFoundError(f"No .geff entries for {sample_id}")

                # Extract to temp directory
                for member in members:
                    zf.extract(member, tmpdir)

                # Load from temp location
                geff_path = Path(tmpdir) / geff_entry
                return tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))


class AugmentedCompetitionDataset(CompetitionDataset):
    """
    CompetitionDataset with optional augmentations.

    Scaffolded for Wave 3. Currently no-op augmentations.
    """

    def __init__(self, *args, augment: bool = False, **kwargs):
        """
        Initialize with augmentation flag.

        Args:
            augment: Whether to apply augmentations (not yet implemented)
            *args, **kwargs: Passed to CompetitionDataset
        """
        super().__init__(*args, **kwargs)
        self.augment = augment

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and optionally augment."""
        item = super().__getitem__(idx)

        if not self.augment:
            return item

        # TODO: Wave 3 augmentations
        # - Elastic deformation (respect anisotropy)
        # - Y/X rotation (not Z)
        # - Intensity jitter
        # - Patch dropout

        return item
