"""
Minimal Zarr decompression stress test -- NO torch/training, just pure Zarr loading.
This isolates whether the crash is zarr/blosc2 specific or torch-related.

Run with:
    python test_zarr_loop.py --count 50 --sample 44b6_0113de3b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import AnisotropicZarrLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Number of sequential decompression calls")
    parser.add_argument("--sample", default="44b6_0113de3b", help="Sample ID to load")
    args = parser.parse_args()

    sample_id = args.sample
    zarr_path = f"data/staging/train/{sample_id}.zarr"

    print(f"Loading Zarr from {zarr_path}")
    loader = AnisotropicZarrLoader(zarr_path)
    shape = loader.get_shape()
    print(f"Shape: {shape} (T, Z, Y, X)")

    num_t = shape[0]
    print(f"\nLooping through {args.count} sequential timepoint loads...")

    for i in range(args.count):
        t = i % num_t  # Wrap around if we exceed num_t
        try:
            # This is the exact call that the training loop makes
            vol = loader.load_timepoint_block(t, normalize=True)
            print(f"  step={i:3d}: loaded T={t:3d}, shape={vol.shape}, dtype={vol.dtype}, "
                  f"min={vol.min():.4f}, max={vol.max():.4f}")
        except Exception as e:
            print(f"  step={i:3d}: ERROR at T={t}: {e}")
            raise

    print(f"\nCompleted {args.count} loads without crash!")


if __name__ == "__main__":
    main()
