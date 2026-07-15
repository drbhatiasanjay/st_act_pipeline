"""
Diagnostic script to understand which sample/timepoint corresponds to step 34.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import CompetitionDataset

# Load dataset exactly as the smoke test does
dataset = CompetitionDataset(
    data_dir="data/staging/train",
    split_file="data_split.json",
    split_type="train",
    normalize=True,
)

print(f"Total (frame_t, frame_t+1) pairs: {len(dataset)}")
print("\nFirst 40 pairs (step index -> (sample_id, frame_idx)):")

for step in range(min(40, len(dataset))):
    sample_id, frame_idx = dataset.pairs[step]
    t_idx = frame_idx
    t_next = frame_idx + 1
    print(f"  step {step:2d}: sample={sample_id}, frames=[{t_idx}, {t_next}]")

# Show details around step 34
if len(dataset.pairs) > 34:
    print("\n=== DETAIL AROUND STEP 34 (CRASH POINT) ===")
    for step in range(32, min(36, len(dataset))):
        sample_id, frame_idx = dataset.pairs[step]
        print(f"Step {step}: sample={sample_id}, frame_idx={frame_idx}")
        # Try to get shape for this sample
        try:
            loader = dataset._get_loader(sample_id)
            shape = loader.get_shape()
            print(f"  -> Sample shape: {shape} (T, Z, Y, X)")
        except Exception as e:
            print(f"  -> Error getting shape: {e}")
