"""
Local, no-GPU-needed real-data training smoke test -- runs N real gradient
steps against the actual CompetitionDataset/UNet3D/DetectionLoss and reports
whether the model is genuinely learning (max_sigmoid trending up from the
1e-4 prior-bias floor) before spending Kaggle GPU quota on a longer run.

Written after rewriting this same ~40-line diagnostic inline three times in
one session (2026-07-13/14) while debugging the v39 zero-detections
collapse -- see CLAUDE.md's "Kaggle Training Run Monitoring Checklist" and
the concurrency/masked-exit-code lesson in Operational Lessons for why a
reusable, properly-isolated version of this was worth building.

IMPORTANT: run this alone. Do not launch a second heavy python/torch job
while this is running -- confirmed this session to cause a real native
Segmentation fault from CPU/thread-pool contention (see CLAUDE.md), not a
bug in this script or the model. Check `Get-Process python` (Windows) /
`ps aux | grep python` first.

Usage:
    python scripts/local_smoke_train.py --steps 40 --lr 1e-3
    python scripts/local_smoke_train.py --steps 100 --lr 1e-3 --log-file out.log
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    checkpoint_path: str | None = None,
    checkpoint_every: int = 10,
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
    results: list[dict] = []
    resume_from_step = 0
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None

    # Resume support: this script's own known unresolved native segfault
    # (see module docstring / CLAUDE.md) recurs reliably around step 33-36,
    # so any run past that needs to survive a crash without losing already
    # -trained state. Opt-in via --checkpoint-path so default behavior
    # (no checkpoint file) is unchanged for every existing caller.
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        results = state["results"]
        resume_from_step = state["next_step"]
        print(f"Resumed from checkpoint at step {resume_from_step} ({len(results)} prior results)", flush=True)

    model.train()
    start = time.time()
    for step, batch in enumerate(loader):
        if step < resume_from_step:
            continue
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
        # UNet3D's second output is a (1,128,64,256,256) ~2.14GB tensor this
        # script never uses. Left as `_`, it's held for a full extra
        # iteration by loop-variable reuse, and PyTorch's autograd grad_fn
        # cycles aren't reliably freed by refcounting alone -- suspected as
        # the cause of a native segfault recurring around step 35-36 (Gemini
        # cross-check, 2026-07-14; unlike this script, TrainingLoop in
        # src/train.py genuinely uses its `features` output, so this specific
        # leak doesn't apply there). Explicit del forces immediate release.
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

        if ckpt_path is not None and (step + 1) % checkpoint_every == 0:
            tmp_path = ckpt_path.with_suffix(".tmp")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "results": results,
                "next_step": step + 1,
            }, tmp_path)
            tmp_path.replace(ckpt_path)  # atomic overwrite, matches src/train.py's heartbeat pattern

    return results


def summarize(results: list[dict]) -> None:
    if not results:
        print("No steps completed.")
        return

    prior_floor = 1e-4  # sigmoid(RetinaNet-style prior bias) at init, see src/model.py
    first_half = results[: len(results) // 2] or results
    second_half = results[len(results) // 2:] or results
    first_avg = sum(r["max_sigmoid"] for r in first_half) / len(first_half)
    second_avg = sum(r["max_sigmoid"] for r in second_half) / len(second_half)
    final = results[-1]["max_sigmoid"]

    print(f"\n--- SUMMARY ({len(results)} steps) ---")
    print(f"max_sigmoid: first-half avg={first_avg:.6f}, second-half avg={second_avg:.6f}, final={final:.6f}")
    print(f"prior-bias floor (sigmoid at init): {prior_floor:.6f}")

    if final <= prior_floor * 2:
        print("VERDICT: still at/near the init prior floor -- no real learning signal yet.")
    elif second_avg > first_avg * 1.5:
        print("VERDICT: max_sigmoid trending up -- model is learning.")
    else:
        print("VERDICT: above the init floor but not clearly trending -- inconclusive, run more steps.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data/staging/train")
    parser.add_argument("--split-file", default="data_split.json")
    parser.add_argument("--weight-pos", type=float, default=1.0)
    parser.add_argument("--weight-neg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default=None,
                         help="If given, save/resume model+optimizer state here every "
                              "--checkpoint-every steps -- lets a run survive this script's "
                              "known unresolved native segfault (~step 33-36) by resuming "
                              "instead of restarting from step 0.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    results = run_smoke_test(
        steps=args.steps, lr=args.lr, data_dir=args.data_dir, split_file=args.split_file,
        weight_pos=args.weight_pos, weight_neg=args.weight_neg, seed=args.seed,
        checkpoint_path=args.checkpoint_path, checkpoint_every=args.checkpoint_every,
    )
    summarize(results)


if __name__ == "__main__":
    main()
