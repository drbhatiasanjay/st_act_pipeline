"""
Single-frame overfit diagnostic: does the reference implementation's exact
detection-loss recipe (hard point target + fixed neg_weight=0.01, non-adaptive
DetectionLoss) separate GT-cell probability from background probability better
than our current recipe (Gaussian target + adaptive per-batch DetectionLoss),
under IDENTICAL training conditions (same tiny model, same single frame, same
step count)?

This is deliberately narrower and lower-noise than scripts/benchmark_heatmap_targets.py
(which already ran and was inconclusive -- every config there saturated to
edge_jaccard=0.0, oscillating between "predicts nothing" and "predicts
everything", a known undertrained-toy-model NMS-threshold artifact per that
script's own comments, not a real signal about target/loss recipe). This
script removes the NMS/ILP-tracker/scoring pipeline entirely and measures raw
sigmoid probability directly: mean prob at exact GT voxels vs mean prob at
confirmed-background voxels (>=15 voxels away from every GT point in this
frame), tracked over training steps on a genuine single-frame overfit (not
2 epochs over 6 different frames like the old benchmark -- that undertrains
by construction).

Real reference recipe verified 2026-08-14 by fetching and reading
royerlab/kaggle-cell-tracking-competition/scripts/train_unet_transformer.py
directly (not paraphrased): compute_detection_loss() marks only annotated GT
voxels positive (hard point target, target[b,zi,yi,xi]=1.0, no Gaussian/sigma/
blur anywhere in that script), and uses per-sample normalized BCE with a fixed
neg_weight (actual training default 1e-2, i.e. 0.01) -- not our adaptive
per-batch weight_neg formula.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import AnisotropicZarrLoader
from src.targets import DetectionLoss, generate_heatmap_targets

ANISOTROPY = (4.0, 1.0, 1.0)
DATA_DIR = Path("data/staging/train")
SAMPLE_ID = "6bba_05db0fb1"  # denser sample (12.3 annotated/frame) for better SNR
TRAIN_STEPS = 300
LOG_EVERY = 25
BACKGROUND_EXCLUSION_VOXELS = 15  # min distance from any GT point to count as "confirmed background"
N_BACKGROUND_SAMPLES = 200
SEED = 0


class ThrowawayTinyUNet(nn.Module):
    """Same architecture as scripts/benchmark_heatmap_targets.py's throwaway
    model -- deliberately small for a fast CPU overfit test, not Task 2.1's
    real UNet3D. Kept IDENTICAL across both configs in this script so the
    only variable is target type + loss recipe, not architecture."""

    def __init__(self, in_channels=1, base_channels=4):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(in_channels, base_channels, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv3d(base_channels * 2 + base_channels, base_channels, 3, padding=1), nn.ReLU())
        self.out = nn.Conv3d(base_channels, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        u1 = self.up(e2)
        cat = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat)
        return self.out(d1)


def run_config(name, target_type, adaptive, weight_neg, x, gt_voxel_coords, bg_coords, device):
    torch.manual_seed(SEED)
    model = ThrowawayTinyUNet(base_channels=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=weight_neg, adaptive=adaptive)

    heatmaps, _ = generate_heatmap_targets(
        SAMPLE_ID, GEFF_PATH, VOLUME_SHAPE, anisotropy=ANISOTROPY,
        target_type=target_type, target_ts=[FRAME_T],
    )
    target = heatmaps[FRAME_T][None, ...].to(device)  # (1,1,Z,Y,X)

    print(f"\n=== {name} (target={target_type}, adaptive={adaptive}, weight_neg={weight_neg}) ===")
    history = []
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x))[0, 0]
                gt_probs = [probs[z, y, xx].item() for z, y, xx in gt_voxel_coords]
                bg_probs = [probs[z, y, xx].item() for z, y, xx in bg_coords]
            mean_gt = float(np.mean(gt_probs)) if gt_probs else float("nan")
            mean_bg = float(np.mean(bg_probs)) if bg_probs else float("nan")
            print(f"  step {step:4d}: loss={loss.item():.4f}  mean_prob_at_GT={mean_gt:.4f}  mean_prob_at_BG={mean_bg:.4f}  separation={mean_gt - mean_bg:+.4f}")
            history.append({"step": step, "loss": loss.item(), "mean_gt": mean_gt, "mean_bg": mean_bg})
            model.train()

    return history


def main():
    global GEFF_PATH, VOLUME_SHAPE, FRAME_T

    device = torch.device("cpu")
    zarr_path = DATA_DIR / f"{SAMPLE_ID}.zarr"
    GEFF_PATH = DATA_DIR / f"{SAMPLE_ID}.geff"

    loader = AnisotropicZarrLoader(str(zarr_path), simulate=False)
    t_dim, z_dim, y_dim, x_dim = loader.get_shape()
    VOLUME_SHAPE = (t_dim, z_dim, y_dim, x_dim)

    import tracksdata
    graph, _ = tracksdata.graph.IndexedRXGraph.from_geff(str(GEFF_PATH))
    attrs = graph.node_attrs(attr_keys=["t", "z", "y", "x"])
    t_vals = attrs["t"].to_list()
    labeled_ts = sorted(set(t_vals))
    FRAME_T = labeled_ts[0]

    # Real GT voxel coordinates for this single frame (rounded to nearest voxel)
    rows = [(int(row["t"]), int(round(row["z"])), int(round(row["y"])), int(round(row["x"])))
            for row in attrs.to_dicts()]
    gt_voxel_coords = [(z, y, x) for (t, z, y, x) in rows if t == FRAME_T]
    print(f"Sample {SAMPLE_ID}, frame t={FRAME_T}: {len(gt_voxel_coords)} real annotated cells")

    # Confirmed-background voxels: random points at least BACKGROUND_EXCLUSION_VOXELS
    # from every GT point in this frame (so they're not accidentally landing in
    # the Gaussian blur radius or right next to an annotated cell).
    rng = np.random.default_rng(SEED)
    gt_arr = np.array(gt_voxel_coords) if gt_voxel_coords else np.zeros((0, 3))
    bg_coords = []
    attempts = 0
    while len(bg_coords) < N_BACKGROUND_SAMPLES and attempts < N_BACKGROUND_SAMPLES * 50:
        attempts += 1
        z = rng.integers(0, z_dim)
        y = rng.integers(0, y_dim)
        x = rng.integers(0, x_dim)
        if len(gt_arr) > 0:
            d = np.sqrt(((gt_arr - np.array([z, y, x])) ** 2).sum(axis=1))
            if d.min() < BACKGROUND_EXCLUSION_VOXELS:
                continue
        bg_coords.append((z, y, x))
    print(f"Sampled {len(bg_coords)} confirmed-background voxels (>= {BACKGROUND_EXCLUSION_VOXELS} voxels from any GT point)")

    vol = loader.load_timepoint_block(FRAME_T, normalize=True).astype(np.float32)
    x = torch.from_numpy(vol)[None, None, ...].to(device)

    results = {}
    results["current_gaussian_adaptive"] = run_config(
        "CURRENT (ours)", "gaussian", True, 0.01, x, gt_voxel_coords, bg_coords, device
    )
    results["reference_point_fixed"] = run_config(
        "REFERENCE recipe", "point", False, 0.01, x, gt_voxel_coords, bg_coords, device
    )

    print("\n=== FINAL COMPARISON (last logged step) ===")
    for name, hist in results.items():
        if hist:
            last = hist[-1]
            print(f"  {name}: separation={last['mean_gt'] - last['mean_bg']:+.4f} (GT={last['mean_gt']:.4f}, BG={last['mean_bg']:.4f})")


if __name__ == "__main__":
    main()
