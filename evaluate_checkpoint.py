"""
Local evaluation script for Task 3.4.
Evaluates the downloaded Kaggle sanity-check model checkpoint on real staged data.
"""

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tracksdata.graph import IndexedRXGraph

from src.dataset import CompetitionDataset
from src.evaluation import (
    DEFAULT_SCALE,
    evaluate_submission,
    load_geff_ground_truth,
)
from src.inference import greedy_edge_assignment

# Import project modules
from src.model import SimpleNodeTransformer, UNet3D
from src.train import extract_peaks_from_volume

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate_checkpoint")


def find_latest_local_checkpoint(search_root: Path = Path(".")) -> Path | None:
    """Find the most-recently-modified epoch_*.pt anywhere under search_root.

    Doesn't assume a fixed filename or directory -- save_checkpoint() only
    writes a new file when val_score improves, so the newest epoch_*.pt is
    the best one, whether it landed in checkpoint_dataset/,
    kaggle_sanity_outputs/checkpoints_sanity/, or wherever a fresh Kaggle
    download was dropped.
    """
    candidates = sorted(search_root.rglob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_evaluation(split_type: str, dataset_id_filter: list = None):
    device = torch.device("cpu")
    checkpoint_path = find_latest_local_checkpoint()

    if checkpoint_path is None:
        logger.error("No epoch_*.pt checkpoint found anywhere under the project root")
        return
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize models
    logger.info("Initializing UNet3D and SimpleNodeTransformer")
    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128))
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)

    # Load state dicts
    unet3d.load_state_dict(checkpoint['unet3d_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])

    unet3d.eval()
    transformer.eval()

    # Load dataset
    data_dir = Path("data/staging/train")
    split_file = Path("data_split.json")

    logger.info(f"Creating CompetitionDataset for split '{split_type}'")
    dataset = CompetitionDataset(
        data_dir=data_dir,
        split_file=split_file,
        split_type=split_type,
        normalize=True
    )

    if dataset_id_filter:
        # Filter the pairs to only include selected sample IDs
        filtered_pairs = [p for p in dataset.pairs if p[0] in dataset_id_filter]
        dataset.pairs = filtered_pairs
        logger.info(f"Filtered dataset pairs to {len(dataset.pairs)} for sample(s) {dataset_id_filter}")

    if len(dataset) == 0:
        logger.warning(f"No pairs found to evaluate for split {split_type} and filter {dataset_id_filter}")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_pred_graphs = {}
    all_gt_graphs = {}
    all_gt_metadata = {}

    # Hyperparameters from checkpoint
    hyperparams = checkpoint.get('hyperparams', {
        'edge_threshold': 0.5,
        'detection_threshold': 0.5,
        'nms_radius_um': 5.0,
    })
    logger.info(f"Using hyperparameters from checkpoint: {hyperparams}")

    # Store adaptive threshold statistics
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

            # Extract peaks for channel 0 and 1
            peaks_t = extract_peaks(detection_probs, channel=0, t_idx=t_idx, hyperparams=hyperparams)
            peaks_t1 = extract_peaks(detection_probs, channel=1, t_idx=t_idx, hyperparams=hyperparams)

            total_peaks_t += len(peaks_t)
            total_peaks_t1 += len(peaks_t1)

            # Extract features at peaks
            nodes_t, features_t = get_nodes_and_features(features, peaks_t, device)
            nodes_t1, features_t1 = get_nodes_and_features(features, peaks_t1, device)

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

            # Add nodes and edges to the prediction graph
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
            logger.info(
                f"Batch {batch_idx+1:02d}/{len(loader)} | t_idx={t_idx:02d} | "
                f"Sigmoid: [{p_min:.4f}, {p_max:.4f}] | "
                f"Peaks: {len(peaks_t)} (ch0), {len(peaks_t1)} (ch1) | "
                f"Edges: {len(edges)} | Took {batch_elapsed:.2f}s"
            )

    total_elapsed = time.time() - total_start_time
    logger.info(f"Inference complete in {total_elapsed:.2f}s. Total peaks_t: {total_peaks_t}, peaks_t1: {total_peaks_t1}, edges: {total_edges}")

    # Load GT graphs
    for sample_id in all_pred_graphs:
        geff_path = data_dir / f"{sample_id}.geff"
        if geff_path.exists():
            logger.info(f"Loading GT for {sample_id} from {geff_path}")
            gt_graph, gt_metadata = load_geff_ground_truth(str(geff_path))
            all_gt_graphs[sample_id] = gt_graph
            all_gt_metadata[sample_id] = gt_metadata
            logger.info(f"GT {sample_id} loaded: {gt_graph.num_nodes()} nodes, {gt_graph.num_edges()} edges")

    # Run evaluation
    if all_gt_graphs:
        try:
            logger.info("Evaluating predicted graphs against GT...")
            val_metrics = evaluate_submission(
                all_pred_graphs,
                all_gt_graphs,
                gt_metadata=all_gt_metadata
            )

            val_metrics_clean = {}
            for key, val in val_metrics.items():
                if isinstance(val, float) and math.isnan(val):
                    val_metrics_clean[key] = 0.0
                else:
                    val_metrics_clean[key] = val

            logger.info(f"RESULTS FOR {split_type.upper()} SPLIT (Filter: {dataset_id_filter}):")
            logger.info(f"  Edge Jaccard:          {val_metrics_clean['edge_jaccard']:.6f}")
            logger.info(f"  Adjusted Edge Jaccard: {val_metrics_clean['adjusted_edge_jaccard']:.6f}")
            logger.info(f"  Division Jaccard:      {val_metrics_clean['division_jaccard']:.6f}")
            logger.info(f"  Combined Score:        {val_metrics_clean['score']:.6f}")
            logger.info(f"  Predicted Nodes:       {val_metrics_clean['num_pred_nodes_total']}")
            logger.info(f"  GT Nodes:              {val_metrics_clean['num_gt_nodes_total']}")
            logger.info(f"  Datasets Evaluated:    {val_metrics_clean['num_datasets']}")

            # Print individual sample info
            for sample_id in all_pred_graphs:
                pred_g = all_pred_graphs[sample_id]
                gt_g = all_gt_graphs[sample_id]
                logger.info(f"  Sample {sample_id}: pred_nodes={pred_g.num_nodes()}, pred_edges={pred_g.num_edges()} | gt_nodes={gt_g.num_nodes()}, gt_edges={gt_g.num_edges()}")

            return val_metrics_clean
        except Exception:
            logger.error("Evaluation failed with exception:", exc_info=True)
    else:
        logger.warning("No GT graphs loaded, skipping evaluation.")

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

if __name__ == "__main__":
    logger.info("=== EVALUATING VALIDATION SAMPLE (44b6_0b24845f) ===")
    run_evaluation(split_type="validation", dataset_id_filter=["44b6_0b24845f"])

    logger.info("\n=== EVALUATING TRAIN SAMPLE (6bba_05b6850b - Smallest staged sample) ===")
    run_evaluation(split_type="train", dataset_id_filter=["6bba_05b6850b"])
