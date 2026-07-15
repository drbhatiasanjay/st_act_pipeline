# ruff: noqa: E402
"""
Diagnostic version of local_smoke_train.py with faulthandler enabled.
This captures native stack traces when a segfault happens, NOT just Python tracebacks.

Run with:
    python test_faulthandler.py --steps 40 --lr 1e-3
"""

import argparse
import faulthandler
import sys
import time
from pathlib import Path

# Enable faulthandler immediately at module load time, before any C extension imports
faulthandler.enable()

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import CompetitionDataset
from src.model import UNet3D
from src.targets import DetectionLoss, generate_heatmap_targets


def run_smoke_test(
    steps: int,
    lr: float,
    data_dir: str,
    split_file: str,
    weight_pos: float,
    weight_neg: float,
    seed: int,
) -> list[dict]:
    torch.manual_seed(seed)
    dataset = CompetitionDataset(
        data_dir=data_dir, split_file=split_file, split_type="train", normalize=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=2, channels=(32, 64, 128))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DetectionLoss(weight_pos=weight_pos, weight_neg=weight_neg, adaptive=True)

    geff_cache = {}
    model.train()
    start = time.time()
    results = []
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

        with torch.no_grad():
            max_sigmoid = torch.sigmoid(logits).max().item()

        elapsed = time.time() - start
        results.append({
            "step": step, "sample_id": sample_id, "t_idx": t_idx,
            "loss": loss.item(), "max_sigmoid": max_sigmoid, "elapsed": elapsed,
        })
        print(
            f"step={step} sample={sample_id} t={t_idx} loss={loss.item():.4f} "
            f"max_sigmoid={max_sigmoid:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    parser.add_argument("--weight-pos", type=float, default=1.0)
    parser.add_argument("--weight-neg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_smoke_test(
        steps=args.steps, lr=args.lr, data_dir=args.data_dir, split_file=args.split_file,
        weight_pos=args.weight_pos, weight_neg=args.weight_neg, seed=args.seed,
    )

    if results:
        print(f"\nCompleted {len(results)} steps without crash")
    else:
        print("No steps completed")


if __name__ == "__main__":
    main()
