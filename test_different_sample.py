"""
Test if the crash is sample-specific or iteration-count-specific.
Run with reduced steps starting from a different sample.

Run with:
    python test_different_sample.py --steps 40 --start-step 99
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import CompetitionDataset
from src.model import UNet3D
from src.targets import DetectionLoss, generate_heatmap_targets


def run_test(
    steps: int,
    start_step: int,
    lr: float,
    data_dir: str,
    split_file: str,
) -> list[dict]:
    torch.manual_seed(42)
    dataset = CompetitionDataset(
        data_dir=data_dir, split_file=split_file, split_type="train", normalize=True,
    )

    # Create a subset starting from start_step
    indices = list(range(start_step, min(start_step + steps, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=2, channels=(32, 64, 128))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01, adaptive=True)

    geff_cache = {}
    model.train()
    start = time.time()
    results = []

    for local_step, batch in enumerate(loader):
        global_step = start_step + local_step

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

        with torch.no_grad():
            max_sigmoid = torch.sigmoid(logits).max().item()

        elapsed = time.time() - start
        results.append({
            "global_step": global_step, "local_step": local_step,
            "sample_id": sample_id, "t_idx": t_idx,
            "loss": loss.item(), "max_sigmoid": max_sigmoid, "elapsed": elapsed,
        })
        print(
            f"global_step={global_step} local_step={local_step} sample={sample_id} "
            f"t={t_idx} loss={loss.item():.4f} max_sigmoid={max_sigmoid:.6f}",
            flush=True,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40, help="Number of steps to run")
    parser.add_argument("--start-step", type=int, default=0, help="Global dataset step to start from")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    args = parser.parse_args()

    print(f"Starting from global_step={args.start_step}, running {args.steps} steps")
    results = run_test(
        steps=args.steps, start_step=args.start_step, lr=args.lr,
        data_dir=args.data_dir, split_file=args.split_file,
    )

    if results:
        print(f"\nCompleted {len(results)} steps without crash")
    else:
        print("No steps completed")


if __name__ == "__main__":
    main()
