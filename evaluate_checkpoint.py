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
import torch
from torch.utils.data import DataLoader

from src.dataset import CompetitionDataset
from src.evaluation import (
    DEFAULT_SCALE,
    evaluate_submission,
    load_geff_ground_truth,
)
from src.inference import greedy_edge_assignment

# Import project modules
from src.model import SimpleNodeTransformer, UNet3D
from src.prediction_graph import PredictionGraphAssembler
from src.split_utils import (
    get_split_identity,
    load_and_validate_split,
    resolve_split_file_path,
    validate_checkpoint_split_compatibility,
)
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


def run_evaluation(
    split_type: str,
    dataset_id_filter: list = None,
    max_pairs: int | None = None,
    allow_split_mismatch: bool = False,
):
    """
    max_pairs: cap the number of (frame_t, frame_t1) pairs evaluated per call.
    Exists to work around the project's known unresolved native Windows
    segfault (see CLAUDE.md), which has reproduced 3/3 times in this exact
    script at batch ~33 -- staying under that threshold lets
    evaluate_submission() actually execute instead of losing all progress on
    every attempt. Real score, real (partial) coverage -- not a full-sample
    number, but the actual scoring code runs end to end.

    allow_split_mismatch: P0-2 checkpoint/split-identity fix (2026-07-16).
    By default, evaluating a checkpoint against a split with a different
    membership_sha256 than the one it was trained under raises RuntimeError
    -- the checkpoint's val_score was selected against a specific held-out
    embryo, so scoring it against a different split is not the same
    measurement. Pass True only for a deliberate cross-fold evaluation.
    """
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
    # P0-2 fix (2026-07-16): use the SAME active-fold resolution as
    # kaggle_kernel/train_kernel.py (ST_ACT_SPLIT_FILE, validated for
    # embryo-disjointness) so a checkpoint's training run and its evaluation
    # here can never accidentally consult different split files.
    split_file = resolve_split_file_path()
    load_and_validate_split(split_file)
    # P0-2 checkpoint/split-identity fix (2026-07-16): detect a checkpoint
    # being evaluated against a different split than it was trained under.
    active_split_identity = get_split_identity(split_file)
    validate_checkpoint_split_compatibility(
        checkpoint, active_split_identity, split_file, allow_mismatch=allow_split_mismatch,
    )

    logger.info(f"Creating CompetitionDataset for split '{split_type}'")
    dataset = CompetitionDataset(
        data_dir=data_dir,
        split_file=split_file,
        split_type=split_type,
        normalize=True,
        # P0-1 fix (2026-07-16): explicit, not just relying on the default --
        # this function does pure inference/graph evaluation for BOTH
        # split_type values it's ever called with (including "train", e.g.
        # run_evaluation(split_type="train", ...) below), never backprop, so it
        # must always see every real consecutive pair regardless of GT coverage.
        filter_unannotated_pairs=False,
    )

    if dataset_id_filter:
        # Filter the pairs to only include selected sample IDs
        filtered_pairs = [p for p in dataset.pairs if p[0] in dataset_id_filter]
        dataset.pairs = filtered_pairs
        logger.info(f"Filtered dataset pairs to {len(dataset.pairs)} for sample(s) {dataset_id_filter}")

    if max_pairs is not None and len(dataset.pairs) > max_pairs:
        dataset.pairs = dataset.pairs[:max_pairs]
        logger.info(f"Capped to first {max_pairs} pairs (max_pairs)")

    if len(dataset) == 0:
        logger.warning(f"No pairs found to evaluate for split {split_type} and filter {dataset_id_filter}")
        return

    # shuffle=False is mandatory (not just historical convenience): P0-3's
    # PredictionGraphAssembler requires strict chronological per-sample
    # window order and raises RuntimeError otherwise.
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # P0-3 fix (2026-07-17): replaces the unconditional dual-frame add_node()
    # block that previously created two distinct graph nodes for timepoint
    # t_idx+1 (once as one batch's channel-1, once as the next batch's own
    # channel-0) -- see src/prediction_graph.py's module docstring.
    assembler = PredictionGraphAssembler()
    all_gt_graphs = {}
    all_gt_metadata = {}

    # Hyperparameters from checkpoint
    hyperparams = checkpoint.get('hyperparams', {
        'edge_threshold': 0.5,
        'detection_threshold': 0.5,
        'nms_radius_um': 5.0,
    })
    logger.info(f"Using hyperparameters from checkpoint: {hyperparams}")

    logger.info(f"Running inference over {len(loader)} batches...")

    total_start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_start_time = time.time()
            frame_t = batch['frame_t'].to(device)
            frame_t1 = batch['frame_t1'].to(device)
            sample_id = batch['sample_id'][0]
            t_idx = int(batch.get('t_idx', [0])[0])

            # Fail loud, immediately, on any out-of-order window for this sample.
            assembler.validate_window_order(sample_id, t_idx)

            # Forward pass
            x = torch.cat([frame_t, frame_t1], dim=1)
            logits, features = unet3d(x)
            detection_probs = torch.sigmoid(logits)

            p_min, p_max = detection_probs.min().item(), detection_probs.max().item()

            # Extract peaks for channel 0 and 1
            peaks_t = extract_peaks(detection_probs, channel=0, t_idx=t_idx, hyperparams=hyperparams)
            peaks_t1 = extract_peaks(detection_probs, channel=1, t_idx=t_idx, hyperparams=hyperparams)

            # Canonical graph identity (P0-3): see
            # PredictionGraphAssembler.process_window()'s docstring -- frame
            # t_idx's source set is peaks_t only for this sample's first
            # window, otherwise the already-canonical nodes from the prior
            # window's channel-1 output (peaks_t counted diagnostically only).
            source_ids, source_coords, target_ids, target_coords = assembler.process_window(
                sample_id, t_idx, peaks_t, peaks_t1,
            )

            # Extract features at the CANONICAL coordinates from this
            # window's feature tensor (not necessarily peaks_t/peaks_t1).
            nodes_t, features_t = get_nodes_and_features(features, source_coords, device)
            nodes_t1, features_t1 = get_nodes_and_features(features, target_coords, device)

            if len(source_coords) > 0 and len(target_coords) > 0:
                edge_logits = transformer(nodes_t, nodes_t1, features_t, features_t1)
                edge_probs = torch.sigmoid(edge_logits)
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

            assembler.add_edges(sample_id, source_ids, target_ids, edges)

            batch_elapsed = time.time() - batch_start_time
            logger.info(
                f"Batch {batch_idx+1:02d}/{len(loader)} | t_idx={t_idx:02d} | "
                f"Sigmoid: [{p_min:.4f}, {p_max:.4f}] | "
                f"Raw peaks: {len(peaks_t)} (ch0), {len(peaks_t1)} (ch1) | "
                f"Canonical: {len(source_coords)} (frame {t_idx}), {len(target_coords)} (frame {t_idx + 1}) | "
                f"Edges: {len(edges)} | Took {batch_elapsed:.2f}s"
            )

    total_elapsed = time.time() - total_start_time
    diag = assembler.diagnostics()
    logger.info(
        f"Inference complete in {total_elapsed:.2f}s. "
        f"Unique nodes: {diag['predicted_nodes_total']}, unique edges: {diag['predicted_edges_total']} | "
        f"Raw peaks (diagnostic only, NOT the graph node count): "
        f"ch0={diag['raw_channel0_peaks_total']}, ch1={diag['raw_channel1_peaks_total']}"
    )
    all_pred_graphs = assembler.pred_graphs()

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
    elif positive_fraction == 0.0:
        # Opposite failure mode: raw confidence never crosses the fixed
        # threshold anywhere -- see src/train.py::_peaks_for_channel for the
        # full rationale (same duplicated logic).
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

if __name__ == "__main__":
    logger.info("=== EVALUATING VALIDATION SAMPLE (44b6_0b24845f) ===")
    run_evaluation(split_type="validation", dataset_id_filter=["44b6_0b24845f"])

    logger.info("\n=== EVALUATING TRAIN SAMPLE (6bba_05b6850b - Smallest staged sample) ===")
    run_evaluation(split_type="train", dataset_id_filter=["6bba_05b6850b"])
