# ruff: noqa: E402
"""
Diagnostic verification script for evaluate_checkpoint.py.
Incorporates proposed explicit variable deletion and garbage collection to prevent native crashes.
"""

import argparse
import ctypes

# Enable faulthandler explicitly
import faulthandler
import gc
import logging
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

print("Enabling faulthandler explicitly...", flush=True)
faulthandler.enable(all_threads=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracksdata.graph import IndexedRXGraph

from src.dataset import CompetitionDataset
from src.inference import greedy_edge_assignment
from src.model import SimpleNodeTransformer, UNet3D
from src.split_utils import (
    get_split_identity,
    load_and_validate_split,
    resolve_split_file_path,
    validate_checkpoint_split_compatibility,
)
from src.train import (
    DEFAULT_SCALE,
    extract_peaks_from_volume,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def extract_peaks(detection_probs: torch.Tensor, channel: int, t_idx: int, hyperparams: dict) -> list:
    vol_np = detection_probs[0, channel].cpu().numpy()
    threshold = hyperparams['detection_threshold']
    positive_fraction = float((vol_np > threshold).mean())
    max_positive_fraction = hyperparams.get('max_positive_voxel_fraction', 0.005)

    if positive_fraction > max_positive_fraction:
        adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
        logger.warning(
            f"t_idx={t_idx} ch={channel}: threshold={threshold} flags "
            f"{positive_fraction*100:.2f}% of voxels (undertrained-model miscalibration) "
            f"-- using adaptive threshold={adaptive_threshold:.4f} instead"
        )
        threshold = max(adaptive_threshold, threshold)
    elif positive_fraction == 0.0:
        adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
        logger.warning(
            f"t_idx={t_idx} ch={channel}: threshold={threshold} flags 0% of voxels "
            f"(severe under-confidence) -- using adaptive threshold={adaptive_threshold:.6f} instead"
        )
        threshold = adaptive_threshold

    return extract_peaks_from_volume(
        vol_np,
        threshold=threshold,
        voxel_size=DEFAULT_SCALE,
        nms_radius_um=hyperparams['nms_radius_um']
    )


def get_nodes_and_features(features: torch.Tensor, peaks: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(peaks) == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, features.shape[1]), dtype=torch.float32, device=device),
        )
    nodes = torch.tensor(peaks, dtype=torch.float32, device=device)
    zc = torch.clamp(nodes[:, 0].long(), 0, features.shape[2] - 1)
    yc = torch.clamp(nodes[:, 1].long(), 0, features.shape[3] - 1)
    xc = torch.clamp(nodes[:, 2].long(), 0, features.shape[4] - 1)
    feats = features[0, :, zc, yc, xc].t()
    return nodes, feats


def run_evaluation(
    checkpoint_path: str = "checkpoint_dataset/epoch_1_v48_groupnorm_gradckpt_lr3e3.pt",
    data_dir: str = "data/staging/train",
    split_file: str | None = None,
    split_type: str = "validation",
    dataset_id_filter: list[str] = None,
    max_pairs: int = None,
    allow_split_mismatch: bool = False,
):
    """
    split_file: if omitted (None), resolved via ST_ACT_SPLIT_FILE (same active
    fold as evaluate_checkpoint.py / kaggle_kernel/train_kernel.py). Pass an
    explicit path to intentionally override the environment for this one call
    -- P0-2 fix (2026-07-16): checkpoint evaluation must not silently use a
    fold different from the one training used.

    allow_split_mismatch: P0-2 checkpoint/split-identity fix (2026-07-16).
    By default, evaluating a checkpoint against a split with a different
    membership_sha256 than the one it was trained under raises RuntimeError.
    Pass True (or the --allow-split-mismatch CLI flag) only for a deliberate
    cross-fold evaluation.
    """
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    logger.info("Initializing models...")
    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128))
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)

    unet3d.load_state_dict(checkpoint['unet3d_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])

    unet3d.to(device)
    transformer.to(device)

    unet3d.eval()
    transformer.eval()

    resolved_split_file = Path(split_file) if split_file is not None else resolve_split_file_path()
    load_and_validate_split(resolved_split_file)
    # P0-2 checkpoint/split-identity fix (2026-07-16): detect a checkpoint
    # being evaluated against a different split than it was trained under.
    active_split_identity = get_split_identity(resolved_split_file)
    validate_checkpoint_split_compatibility(
        checkpoint, active_split_identity, resolved_split_file, allow_mismatch=allow_split_mismatch,
    )

    logger.info(f"Creating CompetitionDataset for split '{split_type}'")
    dataset = CompetitionDataset(
        data_dir=Path(data_dir),
        split_file=resolved_split_file,
        split_type=split_type,
        normalize=True,
        # Checkpoint evaluation does pure inference/graph construction, never
        # backprop -- must always see every real consecutive pair regardless
        # of GT coverage.
        filter_unannotated_pairs=False,
    )

    if dataset_id_filter:
        dataset.sample_ids = [s for s in dataset.sample_ids if s in dataset_id_filter]
        dataset._build_pair_index()
        logger.info(f"Filtered dataset pairs to {len(dataset.pairs)} for sample(s) {dataset_id_filter}")

    if max_pairs is not None and len(dataset.pairs) > max_pairs:
        dataset.pairs = dataset.pairs[:max_pairs]
        logger.info(f"Capped to first {max_pairs} pairs (max_pairs)")

    if len(dataset) == 0:
        logger.warning(f"No pairs found to evaluate for split {split_type} and filter {dataset_id_filter}")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_pred_graphs = {}

    hyperparams = checkpoint.get('hyperparams', {
        'edge_threshold': 0.5,
        'detection_threshold': 0.5,
        'nms_radius_um': 5.0,
    })
    logger.info(f"Using hyperparameters from checkpoint: {hyperparams}")

    total_peaks_t = 0
    total_peaks_t1 = 0
    total_edges = 0

    logger.info(f"Running inference over {len(loader)} batches...")
    total_start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_start_time = time.time()
            frame_t = batch['frame_t'].to(device)
            frame_t1 = batch['frame_t1'].to(device)
            sample_id = batch['sample_id'][0]
            t_idx = int(batch.get('t_idx', [0])[0])

            if sample_id not in all_pred_graphs:
                pred_graph = IndexedRXGraph()
                for key in ('t', 'x', 'y', 'z'):
                    try:
                        pred_graph.add_node_attr_key(key, pl.Int64, 0)
                    except ValueError:
                        pass
                all_pred_graphs[sample_id] = pred_graph
            pred_graph = all_pred_graphs[sample_id]

            # Forward pass
            x = torch.cat([frame_t, frame_t1], dim=1)
            logits, features = unet3d(x)
            detection_probs = torch.sigmoid(logits)

            p_min, p_max = detection_probs.min().item(), detection_probs.max().item()

            # Extract peaks
            peaks_t = extract_peaks(detection_probs, channel=0, t_idx=t_idx, hyperparams=hyperparams)
            peaks_t1 = extract_peaks(detection_probs, channel=1, t_idx=t_idx, hyperparams=hyperparams)

            total_peaks_t += len(peaks_t)
            total_peaks_t1 += len(peaks_t1)

            # Extract features at peaks
            nodes_t, features_t = get_nodes_and_features(features, peaks_t, device)
            nodes_t1, features_t1 = get_nodes_and_features(features, peaks_t1, device)

            # Proposed Fix: Immediately delete features tensor since we already extracted peak features
            del features

            if len(peaks_t) > 0 and len(peaks_t1) > 0:
                edge_probs = transformer(nodes_t, nodes_t1, features_t, features_t1)
                assignment = greedy_edge_assignment(
                    edge_probs,
                    nodes_t.cpu(),
                    nodes_t1.cpu(),
                    threshold=hyperparams['edge_threshold'],
                    max_children=2,
                    max_parents=1
                )
                edges = assignment['edges']
            else:
                edges = []

            total_edges += len(edges)

            # Add nodes and edges
            node_id_map_t = {}
            for i, (z, y, x_coord) in enumerate(peaks_t):
                node_id_map_t[i] = pred_graph.add_node({
                    't': t_idx, 'x': int(round(x_coord)), 'y': int(round(y)), 'z': int(round(z)),
                })
            node_id_map_t1 = {}
            for j, (z, y, x_coord) in enumerate(peaks_t1):
                node_id_map_t1[j] = pred_graph.add_node({
                    't': t_idx + 1, 'x': int(round(x_coord)), 'y': int(round(y)), 'z': int(round(z)),
                })

            for src_idx, tgt_idx, _prob in edges:
                pred_graph.add_edge(node_id_map_t[src_idx], node_id_map_t1[tgt_idx], {})

            batch_elapsed = time.time() - batch_start_time
            mem_mb = get_memory_usage_mb()
            logger.info(
                f"Batch {batch_idx+1:02d}/{len(loader)} | t_idx={t_idx:02d} | "
                f"Sigmoid: [{p_min:.4f}, {p_max:.4f}] | "
                f"Peaks: {len(peaks_t)} (ch0), {len(peaks_t1)} (ch1) | "
                f"Edges: {len(edges)} | Mem: {mem_mb:.1f}MB | Took {batch_elapsed:.2f}s"
            )

            # Proposed Fix: Explicitly delete all other loop tensors
            del logits, detection_probs, frame_t, frame_t1, x, batch, nodes_t, features_t, nodes_t1, features_t1, peaks_t, peaks_t1, edges
            if (batch_idx + 1) % 5 == 0:
                gc.collect()

    total_elapsed = time.time() - total_start_time
    logger.info(f"Inference complete in {total_elapsed:.2f}s. Total peaks_t: {total_peaks_t}, peaks_t1: {total_peaks_t1}, edges: {total_edges}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=50, help="Cap the number of pairs evaluated")
    parser.add_argument("--checkpoint", default="checkpoint_dataset/epoch_1_v48_groupnorm_gradckpt_lr3e3.pt")
    parser.add_argument(
        "--allow-split-mismatch", action="store_true",
        help="Bypass the checkpoint/active-split identity check for a deliberate cross-fold evaluation.",
    )
    args = parser.parse_args()

    # Run on first validation sample (the one that always crashed at batch ~33-34)
    run_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_id_filter=["44b6_0b24845f"],
        max_pairs=args.max_pairs,
        allow_split_mismatch=args.allow_split_mismatch,
    )


if __name__ == "__main__":
    main()
