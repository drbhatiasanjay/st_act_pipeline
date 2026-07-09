#!/usr/bin/env python3
"""
Enumerate and verify full competition dataset from zip file (Task 1.2).

Enumerates all 199 train samples directly from zip (no full extraction).
Spot-checks 5 samples to verify Zarr v3 format and .geff parsability.
"""

import logging
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tracksdata

from data_loader import AnisotropicZarrLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enumerate_dataset_from_zip(zip_path):
    """Enumerate all train samples from the competition zip file."""
    print(f"\n{'='*80}")
    print("TASK 1.2: ENUMERATE & VERIFY FULL COMPETITION DATASET")
    print(f"{'='*80}\n")

    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"ERROR: Zip file not found at {zip_path}")
        return None

    print(f"Zip file: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1e9:.1f} GB\n")

    print("Enumerating train samples from zip (no extraction)...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List all entries starting with 'train/'
            train_entries = [name for name in zf.namelist() if name.startswith('train/')]

            # Extract sample IDs and group by type (.geff, .zarr)
            sample_dict = defaultdict(lambda: {"geff": False, "zarr": False})

            for entry in train_entries:
                # Expected format: train/{sample_id}.{geff,zarr}/...
                parts = entry.split('/')
                if len(parts) >= 2:
                    name_with_ext = parts[1]  # e.g., "44b6_0113de3b.geff" or "44b6_0113de3b.zarr"

                    if name_with_ext.endswith('.geff'):
                        sample_id = name_with_ext.replace('.geff', '')
                        sample_dict[sample_id]['geff'] = True
                    elif name_with_ext.endswith('.zarr'):
                        sample_id = name_with_ext.replace('.zarr', '')
                        sample_dict[sample_id]['zarr'] = True

            # Filter to samples that have BOTH .geff and .zarr
            valid_samples = [
                sample_id for sample_id, data in sample_dict.items()
                if data['geff'] and data['zarr']
            ]

            print(f"Total valid samples (with both .geff and .zarr): {len(valid_samples)}")

            # Analyze prefix distribution
            prefix_counts = defaultdict(int)
            for sample_id in valid_samples:
                prefix = sample_id.split('_')[0]
                prefix_counts[prefix] += 1

            print("\nPrefix distribution:")
            for prefix in sorted(prefix_counts.keys()):
                count = prefix_counts[prefix]
                print(f"  {prefix}: {count} samples")

            print(f"\nTotal: {sum(prefix_counts.values())} samples")

            # Expected distribution from plan
            expected = {"44b6": 71, "6bba": 128}
            all_match = all(prefix_counts.get(p, 0) == expected[p] for p in expected)
            if all_match:
                print("MATCH: Distribution matches expected (44b6=71, 6bba=128)")
            else:
                print("WARNING: Distribution differs from expected")
                for p in expected:
                    actual = prefix_counts.get(p, 0)
                    if actual != expected[p]:
                        print(f"  {p}: expected {expected[p]}, got {actual}")

            return zf, valid_samples, sample_dict

    except Exception as e:
        print(f"ERROR: Failed to enumerate zip: {e}")
        return None


def spot_check_samples(zip_path, valid_samples, n_samples=5):
    """Spot-check n random samples to verify format."""
    import random

    print(f"\n{'='*80}")
    print(f"SPOT-CHECKING {n_samples} RANDOM SAMPLES")
    print(f"{'='*80}\n")

    # Select 5 random samples ensuring both prefixes
    prefix_samples = defaultdict(list)
    for sample in valid_samples:
        prefix = sample.split('_')[0]
        prefix_samples[prefix].append(sample)

    # Ensure we get both prefixes
    selected = []
    for prefix in sorted(prefix_samples.keys()):
        prefix_list = prefix_samples[prefix]
        # Take up to ceil(n_samples/2) from each prefix
        n_per_prefix = (n_samples + 1) // 2
        selected.extend(random.sample(prefix_list, min(n_per_prefix, len(prefix_list))))

    selected = selected[:n_samples]

    print(f"Selected samples: {selected}\n")

    results = {}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for sample_id in selected:
            print(f"--- Spot-checking: {sample_id} ---")

            # Create temp directory for extraction
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Extract this one sample's zarr and geff to temp location
                zarr_prefix = f"train/{sample_id}.zarr/"
                geff_prefix = f"train/{sample_id}.geff/"

                zarr_extracted = False
                geff_extracted = False

                try:
                    # Extract zarr files
                    for name in zf.namelist():
                        if name.startswith(zarr_prefix):
                            zf.extract(name, tmpdir)
                            zarr_extracted = True

                    # Extract geff files
                    for name in zf.namelist():
                        if name.startswith(geff_prefix):
                            zf.extract(name, tmpdir)
                            geff_extracted = True

                    if not zarr_extracted or not geff_extracted:
                        print(f"  ERROR: zarr_extracted={zarr_extracted}, geff_extracted={geff_extracted}")
                        results[sample_id] = {"status": "FAILED", "reason": "extraction failed"}
                        continue

                    # Now verify the extracted data
                    zarr_path = tmpdir / "train" / f"{sample_id}.zarr"
                    geff_path = tmpdir / "train" / f"{sample_id}.geff"

                    # Test Zarr loading
                    try:
                        loader = AnisotropicZarrLoader(str(zarr_path), simulate=False)
                        shape = loader.get_shape()
                        print(f"  Zarr shape: {shape}")

                        if shape[0] != 100:
                            print(f"  WARNING: Expected T=100, got T={shape[0]}")

                        if shape[1:] != (64, 256, 256):
                            print(f"  WARNING: Expected shape (*, 64, 256, 256), got {shape}")

                        # Try loading one frame
                        frame = loader.load_timepoint_block(0, normalize=True)
                        print(f"  Frame loaded: shape={frame.shape}, dtype={frame.dtype}")

                    except Exception as e:
                        print(f"  ERROR loading Zarr: {e}")
                        results[sample_id] = {"status": "FAILED", "reason": f"zarr load error: {e}"}
                        continue

                    # Test geff parsing
                    try:
                        graph, metadata = tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))
                        print(f"  GEFF loaded: {len(graph.node_ids())} nodes")

                    except Exception as e:
                        print(f"  ERROR loading GEFF: {e}")
                        results[sample_id] = {"status": "FAILED", "reason": f"geff parse error: {e}"}
                        continue

                    print("  PASSED")
                    results[sample_id] = {"status": "PASSED"}

                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[sample_id] = {"status": "FAILED", "reason": str(e)}

    print(f"\n{'='*80}")
    print("SPOT-CHECK SUMMARY")
    print(f"{'='*80}\n")

    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in results.values() if r["status"] == "FAILED")

    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed samples:")
        for sample_id, result in results.items():
            if result["status"] == "FAILED":
                print(f"  {sample_id}: {result.get('reason', 'unknown error')}")

    return results


def main():
    zip_path = r"C:\Users\hemas\Downloads\biohub-cell-tracking-during-development.zip"

    result = enumerate_dataset_from_zip(zip_path)
    if result is None:
        return

    zf, valid_samples, sample_dict = result

    # Spot-check samples
    spot_check_results = spot_check_samples(zip_path, valid_samples, n_samples=5)

    # Summary
    print(f"\n{'='*80}")
    print("TASK 1.2 COMPLETION SUMMARY")
    print(f"{'='*80}\n")

    print("Dataset enumeration: COMPLETE")
    print(f"  Total samples: {len(valid_samples)}")
    print(f"  Prefix distribution: 44b6={sum(1 for s in valid_samples if s.startswith('44b6'))}, "
          f"6bba={sum(1 for s in valid_samples if s.startswith('6bba'))}")
    print("\nSpot-check results:")
    print(f"  Samples tested: {len(spot_check_results)}")
    print(f"  Passed: {sum(1 for r in spot_check_results.values() if r['status'] == 'PASSED')}")
    print(f"  Failed: {sum(1 for r in spot_check_results.values() if r['status'] == 'FAILED')}")

    print("\nLocal disk usage: No full extraction performed (by design)")
    print("  Only 5 spot-check samples temporarily extracted (~50MB each)")

if __name__ == "__main__":
    main()
