"""
Task 2.3 empirical benchmark: point vs. dilated-Gaussian heatmap targets.

Trains a throwaway tiny 3D UNet (2-3 conv layers, NOT Task 2.1's real UNet3D --
per 02-PLAN.md explicit instruction) on each heatmap target type, runs the
resulting detections through Phase 1's real STHypergraphTracker (ILP), and
scores the result against real .geff ground truth via src/evaluation.py --
the same evaluate_submission() used for the actual competition score.

Scope: 4 real staged samples (all currently available locally -- 44b6/6bba,
train+validation) rather than the plan's 10 validation samples, since staging
more from the 87GB competition zip was out of scope for this throwaway
target-type comparison. A short window of consecutive timepoints per sample
(not the full ~100-frame sequence) keeps train+track+eval tractable; this is
a relative comparison between two target encodings, not a final-model
evaluation, so the reduced scope does not compromise the decision.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tracksdata

from run_pipeline import convert_nx_to_tracksdata, extract_peaks_from_volume
from src.data_loader import AnisotropicZarrLoader
from src.evaluation import DEFAULT_SCALE, evaluate_submission
from src.targets import DetectionLoss, generate_heatmap_targets
from src.tracker import STHypergraphTracker

logging.basicConfig(level=logging.WARNING)  # keep tracker/loader noise down
logger = logging.getLogger("benchmark_heatmap_targets")
logger.setLevel(logging.INFO)

ANISOTROPY = (4.0, 1.0, 1.0)
MAX_CANDIDATES_PER_TIMEPOINT = 75
WINDOW_SIZE = 15  # consecutive timepoints for detection+tracking eval
TRAIN_EPOCHS = 2
DATA_DIR = Path("data/staging/train")

SAMPLES = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]
HEATMAP_TYPES = ["point", "gaussian"]


class ThrowawayTinyUNet(nn.Module):
    """2-3 layer 3D UNet, single frame in/out. Task 2.3 benchmark only -- not Task 2.1's UNet3D."""

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


def pick_train_and_window_timepoints(labeled_ts: list[int], t_dim: int) -> tuple[list[int], list[int]]:
    """First 6 labeled timepoints for training; a WINDOW_SIZE-frame consecutive
    window starting at the earliest labeled timepoint for detection+tracking eval."""
    train_ts = labeled_ts[:6]
    w_start = labeled_ts[0]
    w_end = min(w_start + WINDOW_SIZE, t_dim)
    window_ts = list(range(w_start, w_end))
    return train_ts, window_ts


def run_one_config(sample_id: str, heatmap_type: str) -> dict:
    t0 = time.time()
    zarr_path = DATA_DIR / f"{sample_id}.zarr"
    geff_path = DATA_DIR / f"{sample_id}.geff"

    loader = AnisotropicZarrLoader(str(zarr_path), simulate=False)
    t_dim, z_dim, y_dim, x_dim = loader.get_shape()
    volume_shape = (t_dim, z_dim, y_dim, x_dim)

    graph, geff_metadata = tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))
    labeled_ts = sorted(set(graph.node_attrs(attr_keys=["t"])["t"].to_list()))

    train_ts, window_ts = pick_train_and_window_timepoints(labeled_ts, t_dim)
    logger.info(f"[{sample_id}/{heatmap_type}] train_ts={train_ts} window_ts={window_ts[0]}..{window_ts[-1]}")

    heatmaps, heatmap_meta = generate_heatmap_targets(
        sample_id, geff_path, volume_shape, anisotropy=ANISOTROPY, target_type=heatmap_type
    )

    model = ThrowawayTinyUNet(base_channels=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01)

    # --- Train ---
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        for t in train_ts:
            vol = loader.load_timepoint_block(t, normalize=True).astype(np.float32)
            x = torch.from_numpy(vol)[None, None, ...]
            target = heatmaps[t][None, ...]  # (1,1,Z,Y,X)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        logger.info(f"[{sample_id}/{heatmap_type}] epoch {epoch}: mean_loss={epoch_loss / len(train_ts):.4f}")

    # --- Detect over the eval window ---
    # A 2-epoch/6-frame "throwaway" model's raw sigmoid output is not calibrated
    # like a real trained detector's -- it can sit at ~0.5 almost everywhere
    # (near-zero logits), which turns a fixed threshold=0.5 into "half the volume
    # is a peak" and makes extract_peaks_from_volume's ndimage.label() explode
    # (hundreds of thousands of tiny tied components before any cap applies).
    # This is the exact "recalibrate thresholds against the real distribution,
    # don't guess" failure mode this project hit once already (CLAUDE.md) --
    # guard against it with a hard voxel-fraction sanity check before NMS, and
    # fall back to an adaptive high-percentile threshold when the fixed one is
    # clearly miscalibrated for this particular undertrained model's output.
    model.eval()
    centroids_by_t = {}
    motion_vectors_by_t = {}
    total_candidates = 0
    MAX_POSITIVE_VOXEL_FRACTION = 0.005  # 0.5% of volume, generous for real cell density
    for t in window_ts:
        vol = loader.load_timepoint_block(t, normalize=True).astype(np.float32)
        x = torch.from_numpy(vol)[None, None, ...]
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0, 0].numpy()

        threshold = 0.5
        positive_fraction = float((probs > threshold).mean())
        if positive_fraction > MAX_POSITIVE_VOXEL_FRACTION:
            adaptive_threshold = float(np.percentile(probs, 100 * (1 - MAX_POSITIVE_VOXEL_FRACTION)))
            logger.warning(
                f"[{sample_id}/{heatmap_type}] t={t}: threshold=0.5 flags "
                f"{positive_fraction*100:.2f}% of voxels (undertrained-model miscalibration) "
                f"-- using adaptive threshold={adaptive_threshold:.4f} instead"
            )
            threshold = max(adaptive_threshold, 0.5)

        peaks = extract_peaks_from_volume(probs, threshold=threshold, voxel_size=DEFAULT_SCALE, nms_radius_um=5.0)
        if len(peaks) > MAX_CANDIDATES_PER_TIMEPOINT:
            peaks = peaks[:MAX_CANDIDATES_PER_TIMEPOINT]
        centroids_by_t[t] = peaks
        motion_vectors_by_t[t] = [[0.0, 0.0, 0.0] for _ in peaks]
        total_candidates += len(peaks)

    logger.info(f"[{sample_id}/{heatmap_type}] total candidates over window: {total_candidates}")

    # --- Track (real Phase 1 ILP tracker) ---
    tracker = STHypergraphTracker(birth_cost=15.0, death_cost=15.0, division_reward=-8.0)
    lineage_graph = tracker.solve_lineage(centroids_by_t, motion_vectors_by_t, anisotropy=np.array(ANISOTROPY), max_gap_frames=2)
    pred_graph = convert_nx_to_tracksdata(lineage_graph, sample_id)

    # --- Restrict GT to the same window for a fair comparison ---
    gt_node_df = graph.node_attrs(attr_keys=["node_id", "t"])
    window_node_ids = gt_node_df.filter(
        (gt_node_df["t"] >= window_ts[0]) & (gt_node_df["t"] <= window_ts[-1])
    )["node_id"].to_list()
    gt_window_graph = graph.filter(node_ids=window_node_ids).subgraph()

    # --- Score via the real evaluation harness ---
    result = evaluate_submission(
        pred_graphs={sample_id: pred_graph},
        gt_graphs={sample_id: gt_window_graph},
    )

    elapsed = time.time() - t0
    logger.info(
        f"[{sample_id}/{heatmap_type}] DONE in {elapsed:.1f}s: "
        f"edge_jaccard={result['edge_jaccard']:.4f} adjusted={result['adjusted_edge_jaccard']:.4f} "
        f"pred_nodes={result['num_pred_nodes_total']} gt_nodes={result['num_gt_nodes_total']}"
    )

    return {
        "sample_id": sample_id,
        "heatmap_type": heatmap_type,
        "edge_jaccard": result["edge_jaccard"],
        "adjusted_edge_jaccard": result["adjusted_edge_jaccard"],
        "num_pred_nodes": result["num_pred_nodes_total"],
        "num_gt_nodes": result["num_gt_nodes_total"],
        "total_candidates": total_candidates,
        "elapsed_s": elapsed,
    }


def main():
    results_path = Path("scripts/benchmark_heatmap_results.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["sample_id"], r["heatmap_type"]) for r in results if "edge_jaccard" in r}
    if done:
        logger.info(f"Resuming: {len(done)} configs already completed -- {sorted(done)}")

    overall_start = time.time()
    for sample_id in SAMPLES:
        for heatmap_type in HEATMAP_TYPES:
            if (sample_id, heatmap_type) in done:
                continue
            try:
                r = run_one_config(sample_id, heatmap_type)
            except Exception as e:
                logger.error(f"[{sample_id}/{heatmap_type}] FAILED: {e}")
                r = {"sample_id": sample_id, "heatmap_type": heatmap_type, "error": str(e)}
            results.append(r)
            results_path.write_text(json.dumps(results, indent=2))

    logger.info(f"ALL DONE in {time.time() - overall_start:.1f}s")

    for heatmap_type in HEATMAP_TYPES:
        rows = [r for r in results if r.get("heatmap_type") == heatmap_type and "edge_jaccard" in r]
        if not rows:
            logger.info(f"{heatmap_type}: no successful runs")
            continue
        mean_ej = sum(r["edge_jaccard"] for r in rows) / len(rows)
        mean_adj = sum(r["adjusted_edge_jaccard"] for r in rows) / len(rows)
        logger.info(f"{heatmap_type}: mean edge_jaccard={mean_ej:.4f} mean adjusted={mean_adj:.4f} (n={len(rows)})")


if __name__ == "__main__":
    main()
