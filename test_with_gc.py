"""
Test with explicit garbage collection between steps to see if the crash is
related to memory management or C extension memory not being released.

Run with:
    python test_with_gc.py --steps 40 --gc-interval 1
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import CompetitionDataset
from src.model import UNet3D
from src.targets import DetectionLoss, generate_heatmap_targets


def run_test(steps: int, lr: float, gc_interval: int, data_dir: str, split_file: str):
    """Run with explicit garbage collection between steps."""

    torch.manual_seed(42)
    dataset = CompetitionDataset(
        data_dir=data_dir, split_file=split_file, split_type="train", normalize=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=2, channels=(32, 64, 128))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01, adaptive=True)

    geff_cache = {}
    model.train()
    start = time.time()

    print(f"Running with gc_interval={gc_interval} (collect every {gc_interval} steps)")

    for step, batch in enumerate(loader):
        if step >= steps:
            break

        frame_t, frame_t1 = batch["frame_t"], batch["frame_t1"]
        sample_id, t_idx = batch["sample_id"][0], int(batch["t_idx"][0])
        x = torch.cat([frame_t, frame_t1], dim=1)

        heatmaps, _ = generate_heatmap_targets(
            sample_id, f"{data_dir}/{sample_id}.geff", (t_idx + 2, 64, 256, 256),
            target_type="gaussian", target_ts=[t_idx, t_idx + 1], geff_cache=geff_cache,
        )
        zero_ch = torch.zeros((1, 64, 256, 256))
        ch0 = heatmaps.get(t_idx, zero_ch)
        ch1 = heatmaps.get(t_idx + 1, zero_ch)
        targets = torch.cat([ch0, ch1], dim=0).unsqueeze(0)

        opt.zero_grad()
        logits, unused_features = model(x)
        del unused_features
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Explicit garbage collection
        if (step + 1) % gc_interval == 0:
            collected = gc.collect()
            print(f"  [step {step}] gc.collect() returned {collected}", flush=True)

        with torch.no_grad():
            max_sigmoid = torch.sigmoid(logits).max().item()

        elapsed = time.time() - start
        print(
            f"step={step:2d} sample={sample_id} t={t_idx} loss={loss.item():.4f} "
            f"max_sigmoid={max_sigmoid:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    print(f"\nCompleted {min(steps, len(dataset))} steps without crash")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gc-interval", type=int, default=1, help="Call gc.collect() every N steps")
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    args = parser.parse_args()

    run_test(args.steps, args.lr, args.gc_interval, args.data_dir, args.split_file)


if __name__ == "__main__":
    main()
