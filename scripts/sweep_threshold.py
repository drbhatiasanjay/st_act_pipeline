"""
Phase 1 threshold-calibration sweep for the new max_pool3d/maximum_filter-based
NMS peak-finding (see run_pipeline.py's extract_peaks_from_volume). Detection-only
(no tracking) across a handful of threshold values, sampling several timepoints
per dataset, reporting candidate counts so a threshold can be chosen with real
data in hand instead of a guess -- see .planning/phases/01-baseline-parity/
01-CONTEXT.md, "Threshold calibration" decision.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import extract_peaks_from_volume  # noqa: E402
from src.data_loader import AnisotropicZarrLoader  # noqa: E402
from src.evaluation import DEFAULT_SCALE  # noqa: E402

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
SAMPLE_TIMEPOINTS = [0, 25, 50, 75, 99]

TRAIN_DIR = PROJECT_ROOT / "data" / "staging" / "train"


def main():
    zarr_paths = sorted(TRAIN_DIR.glob("*.zarr"))
    print(f"Sweeping {len(THRESHOLDS)} thresholds x {len(SAMPLE_TIMEPOINTS)} timepoints "
          f"x {len(zarr_paths)} datasets\n")

    for zarr_path in zarr_paths:
        dataset_id = zarr_path.stem
        loader = AnisotropicZarrLoader(store_path=str(zarr_path), simulate=False)
        print(f"=== {dataset_id} ===")

        for threshold in THRESHOLDS:
            counts = []
            for t in SAMPLE_TIMEPOINTS:
                vol = loader.load_timepoint_block(t)
                peaks = extract_peaks_from_volume(vol, threshold=threshold, voxel_size=DEFAULT_SCALE)
                counts.append(len(peaks))
            avg = sum(counts) / len(counts)
            print(f"  threshold={threshold}: counts={counts} avg={avg:.0f}")
        print()


if __name__ == "__main__":
    main()
