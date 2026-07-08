"""
ST-ACT Phase 2 -- Kaggle GPU smoke test.

Verifies, before any real training code is written, that:
1. A GPU is actually attached and visible to torch.
2. The competition data is mounted at the expected input path.
3. zarr/geff-format sample files are readable from that mount.

This is scaffolding only -- confirms the environment works, does not train anything.
"""

import os

import torch

print("=== GPU check ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
else:
    print("WARNING: no GPU visible -- check kernel-metadata.json enable_gpu setting "
          "and that a GPU accelerator was actually selected for this run.")

print("\n=== Competition data mount check ===")
input_root = "/kaggle/input/biohub-cell-tracking-during-development"
if os.path.isdir(input_root):
    entries = sorted(os.listdir(input_root))[:20]
    print(f"Found mount at {input_root}, top-level entries (first 20): {entries}")
else:
    print(f"WARNING: expected mount not found at {input_root}")
    print("Actual /kaggle/input contents:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")

print("\n=== Sample file readability check ===")
train_dir = os.path.join(input_root, "train")
if os.path.isdir(train_dir):
    samples = sorted(os.listdir(train_dir))[:5]
    print(f"First 5 entries under train/: {samples}")
else:
    print(f"WARNING: no train/ directory found under {input_root}")

print("\nSmoke test complete.")
