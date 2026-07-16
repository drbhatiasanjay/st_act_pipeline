#!/usr/bin/env python3
"""
Build leave-one-embryo-out train/validation folds for Task 1.3.

P0-2 fix (2026-07-16): the previous version of this script assumed a full
sample ID (e.g. "44b6_0113de3b") was one independent "embryo" unit, and only
stratified (balanced proportions) by the leading prefix rather than treating it
as a group to keep disjoint. That assumption was verified WRONG against
Kaggle's own official competition documentation, fetched live via
`kaggle competitions pages -c biohub-cell-tracking-during-development
--page-name data-description --content`:

    "### Embryo Identity
    Folder names follow the pattern `{embryo_id}_{field_of_view}` (e.g.,
    `44b6_0049_0438_1330_1273`). The first segment identifies which embryo the
    sample comes from. Multiple samples may share the same embryo. Train and
    test sets are embryo-disjoint -- no embryo appears in both."

So the leading underscore-delimited segment (e.g. "44b6") IS the embryo ID, and
multiple sample IDs sharing that segment come from the SAME embryo. The old
`data_split.json` (stratified split, kept both embryos present in both its
train and validation lists) therefore had real embryo-level leakage between
what was used for training and what was used for checkpoint
selection/LR-scheduling/early-stopping validation -- see P0-2 audit.

This script now implements a deterministic leave-one-embryo-out (LOEO) fold
generator: for each embryo E, one fold holds out ALL of E's samples for
validation and trains on every other embryo's samples. No sample ID is ever
split across train/validation within the same fold, and no embryo ID is ever
split across train/validation within the same fold.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.enumerate_dataset import enumerate_dataset_from_zip
from src.split_utils import compute_membership_sha256

logger = logging.getLogger("[ST-ACT Split]")
logging.basicConfig(
    level=logging.INFO,
    format="[ST-ACT Split] %(asctime)s - %(levelname)s: %(message)s"
)

GROUP_KEY_DESCRIPTION = "sample_id.split('_', 1)[0]"


def extract_embryo_id(sample_id: str) -> str:
    """Return the embryo ID for a sample ID, per Kaggle's own naming
    convention `{embryo_id}_{field_of_view}` -- the leading segment up to
    (not including) the FIRST underscore. Using split("_", 1) rather than a
    bare split("_")[0] is equivalent for the current 2-segment IDs but is
    explicit about which underscore is the boundary, in case a future
    field_of_view segment itself contains underscores."""
    return sample_id.split("_", 1)[0]


def build_leave_one_embryo_out_folds(samples: list[str]) -> dict[str, dict]:
    """
    Build one leave-one-embryo-out fold per distinct embryo present in
    `samples`. Deterministic: no shuffling, no random seed -- group assignment
    is fully determined by embryo ID membership, and every sample list is
    sorted before being written out.

    Returns {fold_name: {"train": [...], "validation": [...], "metadata": {...}}}.
    """
    by_embryo: dict[str, list[str]] = defaultdict(list)
    for sample_id in samples:
        by_embryo[extract_embryo_id(sample_id)].append(sample_id)

    embryo_ids = sorted(by_embryo.keys())
    logger.info(f"Distinct embryo IDs: {embryo_ids}")
    for embryo_id in embryo_ids:
        logger.info(f"  {embryo_id}: {len(by_embryo[embryo_id])} samples")

    folds: dict[str, dict] = {}
    for held_out in embryo_ids:
        train: list[str] = []
        validation: list[str] = []
        for embryo_id in embryo_ids:
            target = validation if embryo_id == held_out else train
            target.extend(by_embryo[embryo_id])

        train = sorted(train)
        validation = sorted(validation)

        assert len(set(train) & set(validation)) == 0, \
            f"Sample-ID overlap detected for fold held_out={held_out}!"
        assert len(train) + len(validation) == len(samples), \
            f"Fold held_out={held_out} sample count mismatch!"

        per_embryo_counts = {
            embryo_id: {
                "total": len(by_embryo[embryo_id]),
                "in_train": 0 if embryo_id == held_out else len(by_embryo[embryo_id]),
                "in_validation": len(by_embryo[embryo_id]) if embryo_id == held_out else 0,
            }
            for embryo_id in embryo_ids
        }

        metadata = {
            "method": "leave_one_embryo_out",
            "group_key": GROUP_KEY_DESCRIPTION,
            "train_embryos": [e for e in embryo_ids if e != held_out],
            "validation_embryos": [held_out],
            "total_samples": len(train) + len(validation),
            "train_count": len(train),
            "validation_count": len(validation),
            "per_embryo_counts": per_embryo_counts,
            # Canonical fingerprint of this exact train/validation membership --
            # see src/split_utils.py's compute_membership_sha256() docstring.
            # Checkpoints trained against this fold embed this same value.
            "membership_sha256": compute_membership_sha256(train, validation),
        }

        fold_name = f"embryo_{held_out}_validation"
        folds[fold_name] = {
            "train": train,
            "validation": validation,
            "metadata": metadata,
        }

        logger.info(
            f"Fold {fold_name}: train_embryos={metadata['train_embryos']} "
            f"({metadata['train_count']} samples), "
            f"validation_embryos={metadata['validation_embryos']} "
            f"({metadata['validation_count']} samples)"
        )

    return folds


def write_folds(folds: dict[str, dict], output_dir: Path = Path("data_splits")) -> list[Path]:
    """Write each fold to output_dir/{fold_name}.json. Returns the list of
    written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fold_name, split_data in folds.items():
        output_file = output_dir / f"{fold_name}.json"
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)
        written.append(output_file)
        logger.info(f"Wrote {output_file}")
    return written


def main():
    print("=" * 80)
    print("TASK 1.3 (P0-2 REVISION): BUILD LEAVE-ONE-EMBRYO-OUT FOLDS")
    print("=" * 80)
    print()

    zip_path = Path("C:\\Users\\hemas\\Downloads\\biohub-cell-tracking-during-development.zip")
    logger.info(f"Enumerating samples from {zip_path}...")

    result = enumerate_dataset_from_zip(zip_path)
    if result is None:
        logger.error("Failed to enumerate samples")
        return
    _zf, samples, _sample_dict = result
    logger.info(f"Enumerated {len(samples)} samples")

    folds = build_leave_one_embryo_out_folds(samples)
    written = write_folds(folds)

    print()
    print("=" * 80)
    print("FOLD SUMMARY")
    print("=" * 80)
    for path in written:
        with open(path) as f:
            data = json.load(f)
        m = data["metadata"]
        print(
            f"{path.name}: train_embryos={m['train_embryos']} "
            f"({m['train_count']}) validation_embryos={m['validation_embryos']} "
            f"({m['validation_count']}) total={m['total_samples']}"
        )
    print()


if __name__ == "__main__":
    main()
