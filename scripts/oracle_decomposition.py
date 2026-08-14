#!/usr/bin/env python
"""
Oracle score decomposition runner.

Runs all four oracle modes against the validation split using the most recent
usable checkpoint, producing a JSON report with per-sample and aggregated metrics.

Usage:
    python scripts/oracle_decomposition.py [--checkpoint-path PATH] [--output-dir DIR]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.checkpoint_manifest import load_verified_checkpoint, sha256_file
from src.data_loader import AnisotropicZarrLoader
from src.evaluation import load_geff_ground_truth
from src.inference import greedy_edge_assignment
from src.oracle_evaluation import (
    compute_detection_metrics,
    evaluate_oracle_modes,
)
from src.prediction_graph import PredictionGraphAssembler
from src.split_utils import load_and_validate_split, resolve_split_file_path
from src.train import extract_inference_peaks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: Path = Path("models")) -> Path | None:
    """Find the most recent usable checkpoint (by modification time)."""
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None

    valid_checkpoints = []
    for ckpt_file in checkpoint_dir.glob("*.pt"):
        try:
            # Try to load the manifest to verify it's valid
            manifest_file = ckpt_file.parent / ckpt_file.stem / "checkpoint_manifest.json"
            if manifest_file.exists():
                valid_checkpoints.append((ckpt_file, ckpt_file.stat().st_mtime))
        except Exception:
            pass

    if not valid_checkpoints:
        return None

    valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
    return valid_checkpoints[0][0]


def run_oracle_decomposition(
    checkpoint_path: Path | None = None,
    split_file: Path | None = None,
    data_dir: Path = Path("data/staging/train"),
    output_dir: Path = Path("oracle_results"),
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Run oracle score decomposition on validation split.

    Args:
        checkpoint_path: Path to checkpoint file. If None, finds latest.
        split_file: Path to split JSON. If None, uses default.
        data_dir: Root data directory containing .zarr and .geff files.
        output_dir: Where to write results JSON.
        max_samples: Limit to this many samples (for testing). None = all.

    Returns:
        Dictionary with aggregated results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    if split_file is None:
        split_file = resolve_split_file_path()
    split_data = load_and_validate_split(Path(split_file))
    split_identity = f"md5:{Path(split_file).stem}"

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            logger.error("No checkpoint found and none specified")
            sys.exit(1)

    logger.info(f"Using checkpoint: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)

    # Try to load verified checkpoint (will fail cleanly if manifest doesn't exist yet)
    try:
        checkpoint_data = load_verified_checkpoint(checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
    except FileNotFoundError:
        logger.warning(f"Checkpoint manifest not found for {checkpoint_path}; proceeding without SHA verification")
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_sha = None

    hyperparams = checkpoint_data.get('hyperparams', {})
    model_state = checkpoint_data.get('unet3d_state_dict')
    transformer_state = checkpoint_data.get('transformer_state_dict')

    if model_state is None:
        logger.error("Checkpoint missing unet3d_state_dict")
        sys.exit(1)

    # Load model
    from src.model import SimpleNodeTransformer, UNet3D

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    unet = UNet3D(
        in_channels=2,
        channels=(32, 64, 128),
    ).to(device)
    unet.load_state_dict(model_state)
    unet.eval()

    transformer = SimpleNodeTransformer(
        input_channels=32,
        hidden_channels=32,
        output_channels=1,
    ).to(device)
    if transformer_state:
        transformer.load_state_dict(transformer_state)
    transformer.eval()

    # Setup validation data
    data_loader = AnisotropicZarrLoader(
        data_dir=str(data_dir),
        cache_frames=1,
        normalize=True,
    )

    validation_ids = split_data['validation'][:max_samples] if max_samples else split_data['validation']
    logger.info(f"Running oracle decomposition on {len(validation_ids)} validation samples")

    per_sample_results = {}
    all_precisions = []
    all_recalls = []
    all_localizations = []

    for sample_idx, sample_id in enumerate(validation_ids):
        logger.info(f"[{sample_idx+1}/{len(validation_ids)}] Processing {sample_id}")

        try:
            # Load GT
            gt_path = data_dir / f"{sample_id}.geff"
            gt_graph, gt_metadata = load_geff_ground_truth(str(gt_path))

            # Load volume
            volume_path = data_dir / f"{sample_id}.zarr"
            volume = data_loader.load_zarr(str(volume_path), normalize=True)
            if volume is None:
                logger.warning(f"Failed to load volume for {sample_id}")
                continue

            # Run inference to get predictions
            assembler = PredictionGraphAssembler()
            pred_nodes_by_frame = {}

            with torch.no_grad():
                volume_tensor = torch.from_numpy(volume).float().to(device)

                for t_idx in range(volume.shape[0] - 1):
                    assembler.validate_window_order(sample_id, t_idx)

                    frame_t = volume_tensor[t_idx:t_idx+1, ...]  # (1, Z, Y, X)
                    frame_t1 = volume_tensor[t_idx+1:t_idx+2, ...]

                    # Get detection probabilities via TTA
                    x = torch.cat([frame_t, frame_t1], dim=0).unsqueeze(0)  # (1, 2, Z, Y, X)
                    detection_logits, feature_map = unet(x)

                    # Extract NMS peaks
                    peaks_t = extract_inference_peaks(
                        detection_logits,
                        channel=0,
                        t_idx=t_idx,
                        hyperparams=hyperparams,
                    )
                    peaks_t1 = extract_inference_peaks(
                        detection_logits,
                        channel=1,
                        t_idx=t_idx + 1,
                        hyperparams=hyperparams,
                    )

                    # Process window
                    source_ids, source_coords, target_ids, target_coords = \
                        assembler.process_window(sample_id, t_idx, peaks_t, peaks_t1)

                    # Store predicted nodes for oracle evaluation
                    if t_idx not in pred_nodes_by_frame:
                        pred_nodes_by_frame[t_idx] = peaks_t
                    if t_idx + 1 not in pred_nodes_by_frame:
                        pred_nodes_by_frame[t_idx + 1] = peaks_t1

                    # Run edge prediction if we have nodes
                    if len(source_ids) > 0 and len(target_ids) > 0:
                        # Sample features
                        source_features = []
                        target_features = []
                        for coord in source_coords:
                            z, y, x = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
                            feat = feature_map[0, :, z, y, x].unsqueeze(0)  # (1, C)
                            source_features.append(feat)
                        for coord in target_coords:
                            z, y, x = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
                            feat = feature_map[0, :, z, y, x].unsqueeze(0)  # (1, C)
                            target_features.append(feat)

                        if source_features and target_features:
                            source_features = torch.cat(source_features, dim=0)  # (n, C)
                            target_features = torch.cat(target_features, dim=0)  # (m, C)

                            # Predict edges
                            edge_probs = []
                            for src_feat in source_features:
                                for tgt_feat in target_features:
                                    combined = torch.cat([src_feat, tgt_feat], dim=0).unsqueeze(0)  # (1, 2C)
                                    prob = torch.sigmoid(transformer(combined)).item()
                                    edge_probs.append(prob)

                            edge_probs = torch.tensor(edge_probs, device=device)

                            # Greedy assignment
                            assignment_result = greedy_edge_assignment(
                                edge_probs,
                                torch.tensor(source_coords, device=device),
                                torch.tensor(target_coords, device=device),
                                threshold=hyperparams.get('edge_threshold', 0.5),
                            )

                            # Add edges to assembler
                            edges_to_add = []
                            for src_id, tgt_id, _prob in assignment_result['edges']:
                                edges_to_add.append((source_ids[src_id], target_ids[tgt_id]))

                            assembler.add_edges(sample_id, source_ids, target_ids, edges_to_add)

            # Get predicted graph
            pred_graphs = assembler.pred_graphs()
            pred_graph = pred_graphs.get(sample_id)

            if pred_graph is None:
                logger.warning(f"Failed to build prediction graph for {sample_id}")
                continue

            # Compute detection metrics
            precision, recall, localization_errors = compute_detection_metrics(
                pred_nodes_by_frame,
                gt_graph,
            )
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_localizations.extend(localization_errors)

            # Evaluate oracle modes
            mode_results = evaluate_oracle_modes(
                sample_id=sample_id,
                pred_graph=pred_graph,
                pred_nodes_by_frame=pred_nodes_by_frame,
                gt_graph=gt_graph,
                gt_metadata=gt_metadata,
            )

            per_sample_results[sample_id] = mode_results
            logger.info(f"  GT+GT: {mode_results['gt_nodes_gt_edges']['score']:.4f} | "
                       f"  GT+Model: {mode_results['gt_nodes_model_edges']['score']:.4f} | "
                       f"  Model+Oracle: {mode_results['model_nodes_oracle_edges']['score']:.4f} | "
                       f"  Model+Model: {mode_results['model_nodes_model_edges']['score']:.4f}")

        except Exception as e:
            logger.error(f"Failed to process {sample_id}: {e}", exc_info=True)
            continue

    # Aggregate results
    aggregated = {
        'gt_nodes_gt_edges': {'score': 0.0, 'edge_jaccard': 0.0, 'adjusted_edge_jaccard': 0.0},
        'gt_nodes_model_edges': {'score': 0.0, 'edge_jaccard': 0.0, 'adjusted_edge_jaccard': 0.0},
        'model_nodes_oracle_edges': {'score': 0.0, 'edge_jaccard': 0.0, 'adjusted_edge_jaccard': 0.0},
        'model_nodes_model_edges': {'score': 0.0, 'edge_jaccard': 0.0, 'adjusted_edge_jaccard': 0.0},
    }

    for mode in aggregated.keys():
        scores = [results[mode]['score'] for results in per_sample_results.values()]
        edge_jaccards = [results[mode]['edge_jaccard'] for results in per_sample_results.values()]
        adjusted_jaccards = [results[mode]['adjusted_edge_jaccard'] for results in per_sample_results.values()]

        if scores:
            aggregated[mode]['score'] = float(np.mean(scores))
            aggregated[mode]['edge_jaccard'] = float(np.mean(edge_jaccards))
            aggregated[mode]['adjusted_edge_jaccard'] = float(np.mean(adjusted_jaccards))

    # Compute aggregate detection metrics
    avg_precision = float(np.mean(all_precisions)) if all_precisions else 0.0
    avg_recall = float(np.mean(all_recalls)) if all_recalls else 0.0
    localization_p50 = float(np.percentile(all_localizations, 50)) if all_localizations else 0.0
    localization_p95 = float(np.percentile(all_localizations, 95)) if all_localizations else 0.0

    # Final report
    report = {
        'metadata': {
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_sha': checkpoint_sha,
            'split_file': str(split_file),
            'split_identity': split_identity,
            'num_samples': len(per_sample_results),
            'num_validation_samples': len(validation_ids),
            'inference_config': hyperparams,
        },
        'detection_metrics': {
            'precision_7um': avg_precision,
            'recall_7um': avg_recall,
            'localization_p50_um': localization_p50,
            'localization_p95_um': localization_p95,
        },
        'aggregated_modes': aggregated,
        'per_sample': per_sample_results,
    }

    # Write report
    output_file = output_dir / "oracle_decomposition.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Results written to {output_file}")
    logger.info(f"GT+GT score: {aggregated['gt_nodes_gt_edges']['score']:.4f}")
    logger.info(f"GT+Model score: {aggregated['gt_nodes_model_edges']['score']:.4f}")
    logger.info(f"Model+Oracle score: {aggregated['model_nodes_oracle_edges']['score']:.4f}")
    logger.info(f"Model+Model score: {aggregated['model_nodes_model_edges']['score']:.4f}")
    logger.info(f"Detection P50 localization error: {localization_p50:.2f}µm")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run oracle score decomposition')
    parser.add_argument('--checkpoint-path', type=Path, help='Path to checkpoint file')
    parser.add_argument('--split-file', type=Path, help='Path to split JSON')
    parser.add_argument('--data-dir', type=Path, default=Path("data/staging/train"))
    parser.add_argument('--output-dir', type=Path, default=Path("oracle_results"))
    parser.add_argument('--max-samples', type=int, help='Limit to N samples (for testing)')
    args = parser.parse_args()

    run_oracle_decomposition(
        checkpoint_path=args.checkpoint_path,
        split_file=args.split_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
