# ruff: noqa: E402
"""
Test if repeatedly reopening Zarr stores (instead of caching) causes the crash.

This tests whether the crash is related to AnisotropicZarrLoader's internal caching.
Two variants:
1. WITH caching (normal): one zarr.open() per sample, reused
2. WITHOUT caching: zarr.open() on EVERY load_timepoint_block call

Run with:
    python test_zarr_reopen.py --use-cache --steps 40
    python test_zarr_reopen.py --no-cache --steps 40
"""

import argparse
import sys
import time
from pathlib import Path

import blosc2
import numpy as np
import zarr

# Replicate the blosc2 fix from data_loader.py
blosc2.set_nthreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import AnisotropicZarrLoader
from src.dataset import CompetitionDataset


def load_timepoint_no_cache(store_path: str, t: int, normalize: bool = True) -> np.ndarray:
    """
    Load a timepoint WITHOUT caching the zarr.open() call.
    This reopens the zarr store on EVERY call.
    """
    # Reopen every time (no cache)
    root = zarr.open(store_path, mode='r')
    if hasattr(root, 'shape') and hasattr(root, 'chunks'):
        dataset = root
    elif hasattr(root, '__getitem__'):
        try:
            dataset = root['0']
        except (KeyError, TypeError):
            array_keys = list(root.array_keys()) if hasattr(root, 'array_keys') else []
            if array_keys:
                dataset = root[array_keys[0]]
            else:
                raise ValueError("No openable arrays") from None
    else:
        raise ValueError("Cannot open")

    vol = dataset[t, :, :, :]
    return vol.astype(np.float32)


def test_pattern(use_cache: bool, steps: int, data_dir: str, split_file: str):
    """Run the dataset iteration with or without zarr caching."""

    dataset = CompetitionDataset(
        data_dir=data_dir, split_file=split_file, split_type="train", normalize=True,
    )

    print(f"\nTesting with caching={use_cache}, {steps} steps")

    if use_cache:
        # Normal path: use AnisotropicZarrLoader's single-item cache
        loaders = {}
        def get_loader(sample_id: str) -> AnisotropicZarrLoader:
            if sample_id not in loaders:
                zarr_path = f"{data_dir}/{sample_id}.zarr"
                loaders[sample_id] = AnisotropicZarrLoader(zarr_path)
            return loaders[sample_id]

        def load_fn(sample_id: str, t: int):
            return get_loader(sample_id).load_timepoint_block(t, normalize=True)
    else:
        # No-cache path: reopen zarr.open() every time
        def load_fn(sample_id: str, t: int):
            zarr_path = f"{data_dir}/{sample_id}.zarr"
            return load_timepoint_no_cache(zarr_path, t, normalize=True)

    start = time.time()
    for step in range(steps):
        if step >= len(dataset):
            break

        sample_id, frame_idx = dataset.pairs[step]
        t = frame_idx
        t1 = frame_idx + 1

        try:
            vol_t = load_fn(sample_id, t)
            vol_t1 = load_fn(sample_id, t1)

            print(f"  step={step:2d}: sample={sample_id}, t=[{t}, {t1}], "
                  f"shapes=[{vol_t.shape}, {vol_t1.shape}]")
        except Exception as e:
            print(f"  step={step:2d}: ERROR - {e}")
            raise

    elapsed = time.time() - start
    print(f"Completed {steps} steps in {elapsed:.1f}s (avg {elapsed/steps:.2f}s per step)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use zarr caching")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable zarr caching (reopen every time)")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    args = parser.parse_args()

    test_pattern(args.use_cache, args.steps, args.data_dir, args.split_file)


if __name__ == "__main__":
    main()
