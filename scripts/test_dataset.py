#!/usr/bin/env python3
"""
Test CompetitionDataset on locally-available real samples (Task 1.4).

Tests against the 4 originally-staged samples + up-to-5 spot-checked samples.
Full 199-sample I/O validation deferred to Wave 3's Kaggle sanity-check run.
"""

import json
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import CompetitionDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset():
    """Test CompetitionDataset on locally-available real samples."""

    print("=" * 80)
    print("TASK 1.4: TEST PYTORCH DATASET CLASS")
    print("=" * 80)
    print()

    # Paths
    data_dir = Path("data/staging/train")
    split_file = Path("data_split.json")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False

    if not split_file.exists():
        logger.error(f"Split file not found: {split_file}")
        return False

    # Check available samples
    zarr_files = list(data_dir.glob("*.zarr"))
    available_samples = [f.stem for f in zarr_files]
    logger.info(f"Available local samples: {available_samples}")
    print()

    # Load split
    with open(split_file) as f:
        split_data = json.load(f)

    # Create datasets for both splits
    success = True

    for split_type in ["train", "validation"]:
        print(f"\n--- Testing {split_type} split ---")

        try:
            dataset = CompetitionDataset(
                data_dir=str(data_dir),
                split_file=str(split_file),
                split_type=split_type,
                normalize=True,
                anisotropy=(4.0, 1.0, 1.0),
            )

            logger.info(f"Created {split_type} dataset")
            logger.info(f"Total samples in split: {len(dataset.sample_ids)}")
            logger.info(f"Available locally: {len(available_samples)}")
            logger.info(f"(Frame_t, frame_t+1) pairs: {len(dataset)}")
            print()

            if len(dataset) == 0:
                logger.warning(
                    f"No pairs available in local data for {split_type} split "
                    "(OK: full 199-sample validation deferred to Wave 3)"
                )
                continue

            # Test loading first few items
            for i in range(min(3, len(dataset))):
                try:
                    item = dataset[i]

                    frame_t = item["frame_t"]
                    frame_t1 = item["frame_t1"]
                    sample_id = item["sample_id"]
                    metadata = item["metadata"]

                    logger.info(f"  Item {i}:")
                    logger.info(f"    Sample: {sample_id}")
                    logger.info(f"    Frame_t shape: {frame_t.shape}, dtype: {frame_t.dtype}")
                    logger.info(
                        f"    Frame_t1 shape: {frame_t1.shape}, dtype: {frame_t1.dtype}"
                    )
                    logger.info(
                        f"    Physical voxel size: {metadata['physical_voxel_size']}"
                    )
                    logger.info(f"    Anisotropy: {metadata['anisotropy_ratio']}")

                    # Verify shapes and dtypes
                    assert frame_t.ndim == 3, (
                        f"Expected 3D tensor, got {frame_t.ndim}D"
                    )
                    assert frame_t1.ndim == 3, (
                        f"Expected 3D tensor, got {frame_t1.ndim}D"
                    )
                    assert frame_t.dtype in [torch.float32], (
                        f"Expected float32, got {frame_t.dtype}"
                    )
                    assert frame_t1.dtype in [torch.float32], (
                        f"Expected float32, got {frame_t1.dtype}"
                    )

                    # Check metadata
                    assert "sample_id" in metadata
                    assert "t_idx" in metadata
                    assert "volume_shape" in metadata
                    assert metadata["physical_voxel_size"] == (
                        1.625,
                        0.40625,
                        0.40625,
                    )
                    assert metadata["anisotropy_ratio"] == (4.0, 1.0, 1.0)

                    logger.info("    PASSED")

                except Exception as e:
                    logger.error(f"  Item {i} FAILED: {e}")
                    success = False

            # Test split filtering logic
            print("\n--- Split filtering verification ---")
            split_samples = set(dataset.sample_ids)
            for sample_id in available_samples:
                in_train = sample_id in split_data["train"]
                in_val = sample_id in split_data["validation"]

                if split_type == "train":
                    if sample_id in split_samples:
                        assert in_train, (
                            f"{sample_id} in {split_type} dataset "
                            f"but not in split file"
                        )
                        logger.info(f"  {sample_id}: correctly in {split_type}")
                    elif in_train:
                        logger.info(f"  {sample_id}: in split but not loaded locally")

                elif split_type == "validation":
                    if sample_id in split_samples:
                        assert in_val, (
                            f"{sample_id} in {split_type} dataset "
                            f"but not in split file"
                        )
                        logger.info(f"  {sample_id}: correctly in {split_type}")
                    elif in_val:
                        logger.info(f"  {sample_id}: in split but not loaded locally")

        except Exception as e:
            logger.error(f"Failed to test {split_type} dataset: {e}")
            success = False

    print()
    print("=" * 80)
    print("DATASET TEST RESULTS")
    print("=" * 80)

    if success:
        print("All tests PASSED")
        print()
        print(f"Locally-available samples tested: {len(available_samples)}")
        print(
            "Full 199-sample I/O validation explicitly deferred to Wave 3's "
            "Kaggle sanity-check run (where full dataset is mounted)"
        )
    else:
        print("Some tests FAILED")

    print()
    return success


if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
