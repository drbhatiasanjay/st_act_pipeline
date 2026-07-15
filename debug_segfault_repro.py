"""
Minimal, deterministic reproduction script for investigating native Windows segfaults.
Tight loop of real forward + backward + optimizer-step iterations on staged data.
"""

import argparse
import ctypes
import gc
import sys
import time
from ctypes import wintypes
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import CompetitionDataset
from src.model import UNet3D
from src.targets import DetectionLoss, generate_heatmap_targets


# Windows Process Memory Utility using ctypes
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_memory_usage_mb() -> float:
    try:
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess

        process = GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        if GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# Utility to replace GroupNorm with nn.Identity
def replace_group_norm_with_identity(model: nn.Module):
    replaced_count = 0
    for name, child in model.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(model, name, nn.Identity())
            replaced_count += 1
        else:
            replaced_count += replace_group_norm_with_identity(child)
    return replaced_count


# Patch torch.utils.checkpoint.checkpoint to disable gradient checkpointing
orig_checkpoint = torch.utils.checkpoint.checkpoint

def patch_checkpoint_disabled(*args, **kwargs):
    # args[0] is module, args[1] is input
    module = args[0]
    inp = args[1]
    return module(inp)


def run_repro(
    steps: int,
    mode: str,
    no_groupnorm: bool,
    no_checkpointing: bool,
    lr: float,
    data_dir: str,
    split_file: str,
    seed: int,
):
    import faulthandler
    print("Enabling faulthandler explicitly...", flush=True)
    faulthandler.enable(all_threads=True)

    print(f"Setting seed={seed}")
    torch.manual_seed(seed)

    print("Creating CompetitionDataset...", flush=True)
    dataset = CompetitionDataset(
        data_dir=data_dir, split_file=split_file, split_type="train", normalize=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Initializing UNet3D model...", flush=True)
    model = UNet3D(in_channels=2, channels=(32, 64, 128))

    if no_groupnorm:
        count = replace_group_norm_with_identity(model)
        print(f"Replaced {count} GroupNorm modules with nn.Identity", flush=True)
    else:
        print("Using standard GroupNorm layers", flush=True)

    if no_checkpointing:
        torch.utils.checkpoint.checkpoint = patch_checkpoint_disabled
        print("Disabled Gradient Checkpointing (patched checkpoint function)", flush=True)
    else:
        print("Gradient Checkpointing enabled where training=True", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01, adaptive=True)

    geff_cache = {}
    model.train()
    start_time = time.time()

    print(f"\nStarting repro loop. Mode: {mode}, Steps: {steps}", flush=True)
    print(f"Initial Memory Usage: {get_memory_usage_mb():.2f} MB", flush=True)

    try:
        for step, batch in enumerate(loader):
            if step >= steps:
                break

            step_start = time.time()
            frame_t, frame_t1 = batch["frame_t"], batch["frame_t1"]
            sample_id, t_idx = batch["sample_id"][0], int(batch["t_idx"][0])
            x = torch.cat([frame_t, frame_t1], dim=1)

            # Target generation
            heatmaps, _ = generate_heatmap_targets(
                sample_id, f"{data_dir}/{sample_id}.geff", (t_idx + 2, 64, 256, 256),
                target_type="gaussian", target_ts=[t_idx, t_idx + 1], geff_cache=geff_cache,
            )
            zero_ch = torch.zeros((1, 64, 256, 256))
            ch0 = heatmaps.get(t_idx, zero_ch)
            ch1 = heatmaps.get(t_idx + 1, zero_ch)
            targets = torch.cat([ch0, ch1], dim=0).unsqueeze(0)

            # Forward pass
            logits, unused_features = model(x)
            del unused_features  # Explicitly release features tensor
            loss = loss_fn(logits, targets)

            if mode == "full":
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            elif mode == "forward-only":
                pass
            elif mode == "forward-backward":
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            with torch.no_grad():
                max_sigmoid = torch.sigmoid(logits).max().item()

            mem_mb = get_memory_usage_mb()
            step_elapsed = time.time() - step_start
            total_elapsed = time.time() - start_time
            print(
                f"step={step:2d} sample={sample_id} t={t_idx:2d} loss={loss.item():.4f} "
                f"max_sig={max_sigmoid:.6f} mem={mem_mb:.1f}MB step_took={step_elapsed:.2f}s total={total_elapsed:.1f}s",
                flush=True,
            )

            # Ensure tensors are cleaned up
            del logits, loss, targets, x, frame_t, frame_t1, batch
            if (step + 1) % 5 == 0:
                gc.collect()

        print(f"\nSUCCESS: Completed {steps} steps without crash!", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nException occurred: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Reproduction script for PyTorch CPU native segfault on Windows")
    parser.add_argument("--steps", type=int, default=50, help="Number of loop iterations")
    parser.add_argument("--mode", choices=["full", "forward-only", "forward-backward"], default="full")
    parser.add_argument("--no-groupnorm", action="store_true", help="Replace GroupNorm layers with nn.Identity")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable Gradient Checkpointing")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_repro(
        steps=args.steps,
        mode=args.mode,
        no_groupnorm=args.no_groupnorm,
        no_checkpointing=args.no_checkpointing,
        lr=args.lr,
        data_dir=args.data_dir,
        split_file=args.split_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
