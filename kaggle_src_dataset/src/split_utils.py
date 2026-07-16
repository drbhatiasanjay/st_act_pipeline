"""
Split-file resolution and embryo-disjointness validation for CompetitionDataset
train/validation splits.

P0-2 fix (2026-07-16): centralizes logic that previously lived ad hoc in each
script that read data_split.json, so kaggle_kernel/train_kernel.py (the real
optimizer/backprop training path) and evaluate_checkpoint.py (checkpoint
evaluation) cannot silently diverge on which split file -- and therefore which
embryo(s) are held out -- they're using.

Kaggle's own competition documentation (fetched live via
`kaggle competitions pages -c biohub-cell-tracking-during-development
--page-name data-description --content`) states: folder names follow
`{embryo_id}_{field_of_view}`, the first underscore-delimited segment is the
embryo ID, and "multiple samples may share the same embryo." The old
data_split.json (stratified by that prefix, not disjoint by it) therefore had
real embryo-level leakage between what a model trained on and what its
validation score (used for checkpoint selection, LR scheduling, and early
stopping -- see src/train.py) was computed against. See
scripts/build_train_val_split.py for the leave-one-embryo-out fold generator
this module consumes the output of.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_FILE = "data_splits/embryo_44b6_validation.json"
SPLIT_FILE_ENV_VAR = "ST_ACT_SPLIT_FILE"


def extract_embryo_id(sample_id: str) -> str:
    """Return the embryo ID for a sample ID, per Kaggle's own naming
    convention `{embryo_id}_{field_of_view}` -- the leading segment up to
    (not including) the FIRST underscore."""
    return sample_id.split("_", 1)[0]


def compute_membership_sha256(train: list[str], validation: list[str]) -> str:
    """
    Canonical fingerprint of exactly which samples are in train vs.
    validation. Deterministic regardless of input list order (sorted before
    hashing), so any change to split membership -- adding, removing, or
    moving even a single sample between train and validation -- produces a
    different hash. This is the identity value checkpoints embed
    (TrainingLoop.save_checkpoint(), the partial-checkpoint handlers in
    kaggle_kernel/train_kernel.py and scripts/local_smoke_train.py) and that
    evaluation-time code (evaluate_checkpoint.py, verify_eval_fixed.py)
    compares against, to catch a checkpoint being evaluated against a
    different split than it was trained on without having to diff full file
    contents.
    """
    canonical = json.dumps({"train": sorted(train), "validation": sorted(validation)}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_split_identity(split_path: Path) -> str:
    """Load+validate the split at split_path and return its canonical
    membership_sha256 identity -- the single value checkpoint-saving and
    evaluation-time code should use, rather than each re-deriving the hash
    themselves."""
    split_data = load_and_validate_split(split_path)
    return compute_membership_sha256(split_data["train"], split_data["validation"])


def resolve_split_file_path(
    repo_root: Path | None = None,
    kaggle_src_dataset_dir: str | Path | None = None,
) -> Path:
    """
    Resolve the active split file path from the ST_ACT_SPLIT_FILE environment
    variable (falling back to DEFAULT_SPLIT_FILE), handling three cases:

    - the resolved value is an absolute path: used as-is;
    - kaggle_src_dataset_dir is given (Kaggle environment): resolved relative
      to it, matching how kaggle_kernel/train_kernel.py already resolves
      data_split.json today;
    - otherwise (local environment): resolved relative to repo_root (defaults
      to this file's repo root if not given).
    """
    raw = os.environ.get(SPLIT_FILE_ENV_VAR, DEFAULT_SPLIT_FILE)
    raw_path = Path(raw)

    if raw_path.is_absolute():
        return raw_path

    if kaggle_src_dataset_dir is not None:
        return Path(kaggle_src_dataset_dir) / raw_path

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    return repo_root / raw_path


def _find_duplicates(items: list[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for item in items:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return dupes


def load_and_validate_split(
    split_path: Path,
    full_inventory: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load a split JSON file and validate embryo-disjointness. Raises
    RuntimeError -- never just a logged warning -- on any violation, per the
    project's established "silent fallback masks real breakage" discipline.

    Checks (in order): file exists; train nonempty; validation nonempty; no
    duplicate sample IDs within train; no duplicate sample IDs within
    validation; no sample ID appears in both; no embryo ID appears in both;
    metadata group_key/train_embryos/validation_embryos/per_embryo_counts/
    membership_sha256 (each, when present) match what's actually derivable
    from the train/validation lists themselves -- P0-2 fix (2026-07-16), so a
    hand-edited or stale metadata block can never silently disagree with the
    real data;
    metadata counts (train_count/validation_count/total_samples, when present)
    match the actual list lengths; and, if full_inventory is given, that
    train+validation covers it exactly once (this last check is skipped, not
    failed, when full_inventory is None -- the complete inventory isn't always
    available, e.g. without the full competition zip).

    Every metadata field check above is skipped (not failed) when that field
    is absent from metadata -- matches the existing train_count/
    validation_count/total_samples pattern, so a minimal synthetic split dict
    (as used throughout this project's tests) without a full metadata block
    still passes.

    Returns the parsed split dict on success and logs the selected split's
    file, method, train/validation embryos, and sample counts.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path) as f:
        split_data = json.load(f)

    train: list[str] = split_data.get("train", [])
    validation: list[str] = split_data.get("validation", [])
    metadata: dict[str, Any] = split_data.get("metadata", {})

    if len(train) == 0:
        raise RuntimeError(f"Split file {split_path}: train list is empty.")
    if len(validation) == 0:
        raise RuntimeError(f"Split file {split_path}: validation list is empty.")

    train_dupes = _find_duplicates(train)
    if train_dupes:
        raise RuntimeError(
            f"Split file {split_path}: duplicate sample IDs in train: {sorted(train_dupes)}"
        )

    val_dupes = _find_duplicates(validation)
    if val_dupes:
        raise RuntimeError(
            f"Split file {split_path}: duplicate sample IDs in validation: {sorted(val_dupes)}"
        )

    sample_overlap = set(train) & set(validation)
    if sample_overlap:
        raise RuntimeError(
            f"Split file {split_path}: {len(sample_overlap)} sample ID(s) appear "
            f"in BOTH train and validation: {sorted(sample_overlap)}"
        )

    train_embryos = {extract_embryo_id(s) for s in train}
    val_embryos = {extract_embryo_id(s) for s in validation}
    embryo_overlap = train_embryos & val_embryos
    if embryo_overlap:
        raise RuntimeError(
            f"Split file {split_path}: embryo-disjointness violated -- embryo "
            f"ID(s) {sorted(embryo_overlap)} appear in BOTH train and "
            f"validation. Per Kaggle's own competition documentation, multiple "
            f"sample IDs can share one embryo -- a valid split must never let "
            f"one embryo's samples be split across train and validation."
        )

    if "group_key" in metadata:
        expected_group_key = "sample_id.split('_', 1)[0]"
        if metadata["group_key"] != expected_group_key:
            raise RuntimeError(
                f"Split file {split_path}: metadata group_key="
                f"{metadata['group_key']!r} does not match the expected "
                f"grouping function {expected_group_key!r}"
            )

    if "train_embryos" in metadata:
        expected_train_embryos = sorted(train_embryos)
        if metadata["train_embryos"] != expected_train_embryos:
            raise RuntimeError(
                f"Split file {split_path}: metadata train_embryos="
                f"{metadata['train_embryos']} does not match the embryo IDs "
                f"actually derived from the train list {expected_train_embryos}"
            )

    if "validation_embryos" in metadata:
        expected_val_embryos = sorted(val_embryos)
        if metadata["validation_embryos"] != expected_val_embryos:
            raise RuntimeError(
                f"Split file {split_path}: metadata validation_embryos="
                f"{metadata['validation_embryos']} does not match the embryo "
                f"IDs actually derived from the validation list {expected_val_embryos}"
            )

    if "per_embryo_counts" in metadata:
        per_embryo_counts = metadata["per_embryo_counts"]
        all_embryos = train_embryos | val_embryos
        declared_embryos = set(per_embryo_counts.keys())

        missing_embryos = all_embryos - declared_embryos
        if missing_embryos:
            raise RuntimeError(
                f"Split file {split_path}: metadata per_embryo_counts is "
                f"missing entries for embryo ID(s) {sorted(missing_embryos)} "
                f"that actually appear in train/validation."
            )

        unexpected_embryos = declared_embryos - all_embryos
        if unexpected_embryos:
            raise RuntimeError(
                f"Split file {split_path}: metadata per_embryo_counts declares "
                f"embryo ID(s) {sorted(unexpected_embryos)} that do not appear "
                f"in either train or validation."
            )

        for embryo_id in sorted(all_embryos):
            actual_in_train = sum(1 for s in train if extract_embryo_id(s) == embryo_id)
            actual_in_validation = sum(1 for s in validation if extract_embryo_id(s) == embryo_id)
            actual_total = actual_in_train + actual_in_validation
            declared = per_embryo_counts[embryo_id]

            if declared.get("total") != actual_total:
                raise RuntimeError(
                    f"Split file {split_path}: metadata per_embryo_counts"
                    f"[{embryo_id!r}]['total']={declared.get('total')!r} does "
                    f"not match the actual total {actual_total}"
                )
            if declared.get("in_train") != actual_in_train:
                raise RuntimeError(
                    f"Split file {split_path}: metadata per_embryo_counts"
                    f"[{embryo_id!r}]['in_train']={declared.get('in_train')!r} "
                    f"does not match the actual in_train count {actual_in_train}"
                )
            if declared.get("in_validation") != actual_in_validation:
                raise RuntimeError(
                    f"Split file {split_path}: metadata per_embryo_counts"
                    f"[{embryo_id!r}]['in_validation']="
                    f"{declared.get('in_validation')!r} does not match the "
                    f"actual in_validation count {actual_in_validation}"
                )

    if "membership_sha256" in metadata:
        expected_membership_sha256 = compute_membership_sha256(train, validation)
        if metadata["membership_sha256"] != expected_membership_sha256:
            raise RuntimeError(
                f"Split file {split_path}: metadata membership_sha256="
                f"{metadata['membership_sha256']!r} does not match the "
                f"computed identity {expected_membership_sha256!r} for the "
                f"actual train/validation lists -- the split file's content "
                f"was changed without regenerating this hash."
            )

    if metadata.get("train_count") is not None and metadata["train_count"] != len(train):
        raise RuntimeError(
            f"Split file {split_path}: metadata train_count={metadata['train_count']} "
            f"does not match actual train list length {len(train)}"
        )
    if metadata.get("validation_count") is not None and metadata["validation_count"] != len(validation):
        raise RuntimeError(
            f"Split file {split_path}: metadata validation_count={metadata['validation_count']} "
            f"does not match actual validation list length {len(validation)}"
        )
    if metadata.get("total_samples") is not None and metadata["total_samples"] != len(train) + len(validation):
        raise RuntimeError(
            f"Split file {split_path}: metadata total_samples={metadata['total_samples']} "
            f"does not match actual train+validation length {len(train) + len(validation)}"
        )

    if full_inventory is not None:
        combined = set(train) | set(validation)
        expected = set(full_inventory)
        missing = expected - combined
        unexpected = combined - expected
        if missing or unexpected:
            raise RuntimeError(
                f"Split file {split_path}: train+validation does not cover the "
                f"full expected inventory exactly once. "
                f"Missing ({len(missing)}): {sorted(missing)[:10]}"
                f"{'...' if len(missing) > 10 else ''}. "
                f"Unexpected ({len(unexpected)}): {sorted(unexpected)[:10]}"
                f"{'...' if len(unexpected) > 10 else ''}."
            )
        if len(train) + len(validation) != len(full_inventory):
            raise RuntimeError(
                f"Split file {split_path}: train+validation has "
                f"{len(train) + len(validation)} entries but the expected full "
                f"inventory has {len(full_inventory)} -- possible duplicate "
                f"across train/validation not caught above, or full_inventory "
                f"itself has duplicates."
            )

    logger.info(f"Selected split file: {split_path}")
    logger.info(f"Split method: {metadata.get('method', 'unknown')}")
    logger.info(f"Train embryos: {sorted(train_embryos)} | Validation embryos: {sorted(val_embryos)}")
    logger.info(f"Train samples: {len(train)} | Validation samples: {len(validation)}")

    return split_data


def validate_checkpoint_split_compatibility(
    checkpoint: dict[str, Any],
    active_split_identity: str,
    active_split_path: Path,
    allow_mismatch: bool = False,
) -> None:
    """
    Compare a loaded checkpoint's saved 'split_membership_sha256' (if any)
    against the currently active split's identity (see
    compute_membership_sha256()/get_split_identity()). Evaluating a
    checkpoint against a split different from the one it was trained on
    silently produces a meaningless score -- the checkpoint's val_score was
    selected/early-stopped against ONE specific embryo held out, and scoring
    it against a different held-out embryo (or a genuinely different split
    entirely) is not the same measurement.

    Three outcomes:
    - checkpoint has no 'split_membership_sha256' key at all: it predates
      this fix. Logs a WARNING (does not raise) -- see
      DEFERRED_IMPROVEMENTS.md's LEGACY ARTIFACT WARNING, which already
      documents that historical val_score for such checkpoints is not
      comparable to scores under the leave-one-embryo-out split.
    - checkpoint's identity matches active_split_identity: logs INFO,
      returns normally.
    - checkpoint's identity does NOT match: raises RuntimeError unless
      allow_mismatch=True (an explicit, deliberate cross-fold evaluation --
      e.g. intentionally scoring a 44b6-fold checkpoint against the 6bba
      fold), in which case it logs a WARNING and returns normally instead.
    """
    checkpoint_identity = checkpoint.get("split_membership_sha256")

    if checkpoint_identity is None:
        logger.warning(
            f"Checkpoint has no saved split_membership_sha256 (trained before "
            f"the P0-2 checkpoint/split identity fix) -- cannot verify it was "
            f"trained against the same split now active at "
            f"{active_split_path}. See DEFERRED_IMPROVEMENTS.md's LEGACY "
            f"ARTIFACT WARNING: historical val_score for this checkpoint is "
            f"not comparable to scores computed under the leave-one-embryo-out "
            f"split."
        )
        return

    if checkpoint_identity != active_split_identity:
        if allow_mismatch:
            logger.warning(
                f"Checkpoint split_membership_sha256={checkpoint_identity} "
                f"does NOT match the active split's identity "
                f"{active_split_identity} at {active_split_path} -- "
                f"proceeding anyway because allow_mismatch=True was "
                f"explicitly set (deliberate cross-fold evaluation)."
            )
            return
        raise RuntimeError(
            f"Checkpoint was trained against a split with identity "
            f"{checkpoint_identity}, but the active split at "
            f"{active_split_path} has identity {active_split_identity} -- "
            f"these are different embryo-disjoint folds (or a different "
            f"split entirely). Evaluating a checkpoint against a mismatched "
            f"split silently produces a meaningless score. Pass "
            f"allow_mismatch=True (or the --allow-split-mismatch CLI flag) "
            f"if this cross-fold evaluation is intentional."
        )

    logger.info(
        f"Checkpoint split_membership_sha256 matches the active split at "
        f"{active_split_path} -- compatible."
    )


def validate_resume_checkpoint_split_identity(
    checkpoint: dict[str, Any],
    active_split_identity: str,
    checkpoint_path: Path | str,
    allow_split_mismatch: bool = False,
    allow_legacy_split: bool = False,
) -> None:
    """
    Stricter, DELIBERATELY SEPARATE counterpart to
    validate_checkpoint_split_compatibility() for resuming TRAINING (e.g.
    TrainingLoop.load_checkpoint()), not evaluation. Resuming training from
    a checkpoint trained under a different embryo-disjoint fold can
    DIRECTLY CONTAMINATE the currently held-out embryo -- the model's
    weights would keep accumulating gradient signal derived from data that
    should have stayed held out. Evaluation's risk (a wrong number gets
    reported) is real but categorically smaller than corrupting the weights
    themselves, so this fails loud on a MISSING identity too, not just a
    mismatch -- do not collapse this into validate_checkpoint_split_
    compatibility()'s warn-only legacy handling.

    active_split_identity == "unknown" is a distinct backward-compatibility
    case: the CALLING TrainingLoop itself has no configured split identity
    (not the checkpoint), e.g. an older call site that hasn't been updated
    to pass split_identity=... to TrainingLoop.__init__. There is nothing to
    validate against, so this logs a prominent WARNING and returns
    immediately, regardless of the checkpoint's own identity -- update the
    caller instead of relying on this fallback.

    Otherwise:
    - checkpoint's saved identity matches active_split_identity: logs INFO,
      returns normally.
    - checkpoint's saved identity is absent (predates this fix -- may have
      been trained under the historical, embryo-leaking data_split.json):
      raises RuntimeError unless allow_legacy_split=True.
    - checkpoint's saved identity differs: raises RuntimeError unless
      allow_split_mismatch=True.
    """
    if active_split_identity == "unknown":
        logger.warning(
            f"Resuming from {checkpoint_path} on a TrainingLoop with no "
            f"configured split_identity -- cannot verify this checkpoint was "
            f"trained under the same embryo-disjoint fold. Update the caller "
            f"to pass split_identity=... to TrainingLoop.__init__ instead of "
            f"relying on this warning."
        )
        return

    saved_identity = checkpoint.get("split_membership_sha256")

    if saved_identity is None:
        if not allow_legacy_split:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no saved "
                f"split_membership_sha256 -- it may have been trained under "
                f"the historical, embryo-leaking data_split.json (see "
                f"DEFERRED_IMPROVEMENTS.md's LEGACY ARTIFACT WARNING). "
                f"Resuming TRAINING from it (unlike evaluation) can directly "
                f"contaminate the currently held-out embryo's weights. Pass "
                f"allow_legacy_split=True only for a deliberate legacy warm "
                f"start."
            )
        logger.warning(
            f"Checkpoint {checkpoint_path} has no saved "
            f"split_membership_sha256 -- resuming anyway because "
            f"allow_legacy_split=True was explicitly set."
        )
        return

    if saved_identity != active_split_identity:
        if not allow_split_mismatch:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} was trained under split "
                f"identity {saved_identity}, but the active split identity "
                f"is {active_split_identity} -- these are different "
                f"embryo-disjoint folds. Resuming TRAINING from it (unlike "
                f"evaluation) can directly contaminate the currently "
                f"held-out embryo's weights. Pass allow_split_mismatch=True "
                f"only for a deliberate cross-fold resume."
            )
        logger.warning(
            f"Checkpoint {checkpoint_path} split_membership_sha256="
            f"{saved_identity} does NOT match the active identity "
            f"{active_split_identity} -- resuming anyway because "
            f"allow_split_mismatch=True was explicitly set."
        )
        return

    logger.info(
        f"Checkpoint {checkpoint_path} split_membership_sha256 matches the "
        f"active split identity -- compatible for resume."
    )
