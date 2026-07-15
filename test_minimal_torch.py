"""
Minimal torch test to isolate torch operations.
Tests if the crash requires full backward pass or happens even with forward-only.

Variants:
1. Full (forward + backward + optim.step)
2. Forward only (no backward, no optim.step)
3. Forward + backward but NO optim.step

Run with:
    python test_minimal_torch.py --mode full --steps 40
    python test_minimal_torch.py --mode forward-only --steps 40
    python test_minimal_torch.py --mode forward-backward --steps 40
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import CompetitionDataset
from src.model import UNet3D
from src.targets import DetectionLoss, generate_heatmap_targets


def run_test(mode: str, steps: int, lr: float, data_dir: str, split_file: str):
    """
    Run training with different torch operation patterns.

    Args:
        mode: "full", "forward-only", or "forward-backward"
    """
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

    print(f"Running with mode={mode}")

    for step, batch in enumerate(loader):
        if step >= steps:
            break

        frame_t, frame_t1 = batch["frame_t"], batch["frame_t1"]
        sample_id, t_idx = batch["sample_id"][0], int(batch["t_idx"][0])
        x = torch.cat([frame_t, frame_t1], dim=1)

        # Generate targets (this involves zarr loading via generate_heatmap_targets)
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
        del unused_features
        loss = loss_fn(logits, targets)

        if mode == "full":
            # Full training: zero_grad, backward, clip, step
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        elif mode == "forward-only":
            # Just forward, no backward
            pass
        elif mode == "forward-backward":
            # Forward + backward, but NO optim.step
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Skip opt.step()
        else:
            raise ValueError(f"Unknown mode: {mode}")

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
    parser.add_argument("--mode", choices=["full", "forward-only", "forward-backward"], default="full")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    args = parser.parse_args()

    run_test(args.mode, args.steps, args.lr, args.data_dir, args.split_file)


if __name__ == "__main__":
    main()
