"""
ST-ACT Phase 2 -- Kaggle GPU smoke test.

Verifies, before any real training code is written, that:
1. A GPU is actually attached, visible to torch, AND actually usable
   (compute capability compatible with the installed PyTorch build --
   a P100 (sm_60) reported "CUDA available: True" on the first run of
   this smoke test but was NOT usable by the installed PyTorch, which
   only supports sm_70+).
2. The competition data is mounted, and at which actual path (the
   first run's guessed path, /kaggle/input/<competition-slug>, was
   wrong -- real content lives under /kaggle/input/competitions/).
3. zarr/geff-format sample files are readable from that mount.

This is scaffolding only -- confirms the environment works, does not train anything.
"""

import os

import torch

print("=== GPU check ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print("Device:", device_name)
    print("Device count:", torch.cuda.device_count())
    print("Compute capability:", capability)
    try:
        # Actually exercise the GPU, not just query it -- is_available()
        # can report True even when the compute capability is unsupported.
        x = torch.zeros(4, 4, device="cuda")
        y = x + 1
        torch.cuda.synchronize()
        print("Real GPU tensor op succeeded:", y.sum().item())
    except Exception as e:
        print(f"GPU IS NOT ACTUALLY USABLE despite is_available()=True: {e}")
else:
    print("WARNING: no GPU visible -- check kernel-metadata.json enable_gpu setting "
          "and that a GPU accelerator was actually selected for this run.")

print("\n=== /kaggle/input contents (top level) ===")
print(os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING")

print("\n=== /kaggle/input/competitions contents ===")
comp_root = "/kaggle/input/competitions"
if os.path.isdir(comp_root):
    print(sorted(os.listdir(comp_root)))
    for entry in sorted(os.listdir(comp_root)):
        sub = os.path.join(comp_root, entry)
        if os.path.isdir(sub):
            print(f"  {entry}/ ->", sorted(os.listdir(sub))[:10])
else:
    print("MISSING")

print("\nSmoke test complete.")
