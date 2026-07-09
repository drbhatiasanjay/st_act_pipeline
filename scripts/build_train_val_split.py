#!/usr/bin/env python3
"""
Build embryo-disjoint train/val split for Task 1.3.

Partitions 199 individual samples into ~149 train / ~50 held-out validation,
stratified by movie prefix (44b6/6bba).

Per RESEARCH.md S2.3 and CONTEXT.md: "embryo" means each individual sample ID,
NOT the movie prefix. Split is at sample level, drawing from both prefixes in each set.
"""

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.enumerate_dataset import enumerate_dataset_from_zip

logger = logging.getLogger("[ST-ACT Split]")
logging.basicConfig(
    level=logging.INFO,
    format="[ST-ACT Split] %(asctime)s - %(levelname)s: %(message)s"
)

def build_stratified_split(
    samples: list[str],
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[list[str], list[str], dict]:
    """
    Stratified split by movie prefix.

    Args:
        samples: List of sample IDs (e.g., "44b6_0113de3b")
        train_ratio: Target ratio for train set (0.75 = ~149 train, ~50 validation)
        seed: Random seed for reproducibility

    Returns:
        (train_samples, validation_samples, metadata)
    """
    random.seed(seed)

    # Group by prefix
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        prefix = sample.split("_")[0]  # e.g., "44b6" from "44b6_0113de3b"
        by_prefix[prefix].append(sample)

    logger.info(f"Prefix distribution: {dict(by_prefix)}")

    train_samples = []
    validation_samples = []

    # Stratified split per prefix
    for prefix in sorted(by_prefix.keys()):
        prefix_samples = by_prefix[prefix]
        n_prefix = len(prefix_samples)
        n_train = int(n_prefix * train_ratio)
        n_val = n_prefix - n_train

        # Shuffle and split
        shuffled = prefix_samples.copy()
        random.shuffle(shuffled)

        train_samples.extend(shuffled[:n_train])
        validation_samples.extend(shuffled[n_train:])

        logger.info(
            f"  {prefix}: {n_prefix} total → {n_train} train, {n_val} validation"
        )

    # Verification
    assert len(train_samples) + len(validation_samples) == len(samples), \
        "Split mismatch!"
    assert len(set(train_samples) & set(validation_samples)) == 0, \
        "Overlap detected!"

    metadata = {
        "total_samples": len(samples),
        "train_count": len(train_samples),
        "validation_count": len(validation_samples),
        "44b6_train": len([s for s in train_samples if s.startswith("44b6")]),
        "44b6_validation": len([s for s in validation_samples if s.startswith("44b6")]),
        "6bba_train": len([s for s in train_samples if s.startswith("6bba")]),
        "6bba_validation": len([s for s in validation_samples if s.startswith("6bba")]),
        "seed": seed
    }

    return train_samples, validation_samples, metadata


def main():
    """Execute Task 1.3: Build embryo-disjoint train/val split."""

    print("=" * 80)
    print("TASK 1.3: BUILD EMBRYO-DISJOINT TRAIN/VAL SPLIT")
    print("=" * 80)
    print()

    # Enumerate samples from zip (returns zf, valid_samples, sample_dict)
    zip_path = Path("C:\\Users\\hemas\\Downloads\\biohub-cell-tracking-during-development.zip")
    logger.info(f"Enumerating samples from {zip_path}...")

    result = enumerate_dataset_from_zip(zip_path)
    if result is None:
        logger.error("Failed to enumerate samples")
        return
    zf, samples, sample_dict = result
    logger.info(f"Enumerated {len(samples)} samples")

    # Build stratified split
    logger.info("Building stratified split by movie prefix...")
    train, validation, metadata = build_stratified_split(samples, train_ratio=0.75, seed=42)

    # Write to JSON
    output_file = Path("data_split.json")
    split_data = {
        "train": sorted(train),
        "validation": sorted(validation),
        "metadata": metadata
    }

    with open(output_file, "w") as f:
        json.dump(split_data, f, indent=2)

    logger.info(f"Wrote split to {output_file.absolute()}")
    print()

    # Summary
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Train samples: {metadata['train_count']} ({metadata['train_count']/metadata['total_samples']*100:.1f}%)")
    print(f"Validation samples: {metadata['validation_count']} ({metadata['validation_count']/metadata['total_samples']*100:.1f}%)")
    print()
    print("Prefix distribution:")
    print(f"  44b6: {metadata['44b6_train']} train, {metadata['44b6_validation']} validation (total: {metadata['44b6_train']+metadata['44b6_validation']})")
    print(f"  6bba: {metadata['6bba_train']} train, {metadata['6bba_validation']} validation (total: {metadata['6bba_train']+metadata['6bba_validation']})")
    print()
    print("Verification:")
    print(f"  No overlap: {len(set(train) & set(validation)) == 0}")
    print(f"  Total matches: {len(train) + len(validation) == metadata['total_samples']}")
    print()
    print(f"Output: {output_file.absolute()}")
    print()


if __name__ == "__main__":
    main()
