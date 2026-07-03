"""
Phase 0 Pipeline Orchestrator: Multi-Dataset Integration and End-to-End Testing

Refactored to:
1. Accept --test-dir and --train-dir arguments (two distinct passes)
2. Load real Zarr data via AnisotropicZarrLoader
3. Run STHypergraphTracker on each dataset
4. Export submission CSV via export_submission()
5. Evaluate locally against .geff ground truth (train/ only)
"""

import os
import sys
import argparse
import logging
import time
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import networkx as nx
import tracksdata as td

from src.data_loader import AnisotropicZarrLoader
from src.tracker import STHypergraphTracker
from src.submission_exporter import export_submission, validate_submission
from src.evaluation import evaluate_submission, load_geff_ground_truth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Pipeline] %(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger("Pipeline")


def extract_peaks_from_volume(vol: np.ndarray, threshold=0.4, offset_bias=0.0):
    """
    Simulates CNN/U-Net heatmap thresholding and peak local max finding.
    Returns list of 3D coordinates [z, y, x].
    """
    nz, ny, nx = vol.shape
    z_indices = np.arange(nz)
    y_indices = np.arange(4, ny - 4, 8)
    x_indices = np.arange(4, nx - 4, 8)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return []

    zz, yy, xx = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
    values = vol[zz, yy, xx]
    mask = values > threshold

    z_hits = zz[mask]
    y_hits = yy[mask].astype(float) + offset_bias
    x_hits = xx[mask].astype(float) + offset_bias

    return np.column_stack([z_hits, y_hits, x_hits]).tolist()


def ensemble_consensus_centroids(cnn_centroids, unet_centroids, anisotropy, eps_microns=6.0):
    """
    Ensemble Consensus Centroid Clustering (DBSCAN):
    Combines cell centroid predictions from multiple networks.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.warning("scikit-learn not installed. Skipping DBSCAN, returning unified peaks.")
        return cnn_centroids if cnn_centroids else unet_centroids

    if not cnn_centroids or not unet_centroids:
        return cnn_centroids if cnn_centroids else unet_centroids

    scaled_cnn = np.array(cnn_centroids) * anisotropy
    scaled_unet = np.array(unet_centroids) * anisotropy

    all_points_scaled = np.vstack([scaled_cnn, scaled_unet])

    db = DBSCAN(eps=eps_microns, min_samples=2).fit(all_points_scaled)
    labels = db.labels_

    consensus_centroids = []
    unique_labels = set(labels) - {-1}

    for label in unique_labels:
        cluster_points = all_points_scaled[labels == label]
        mean_physical = np.mean(cluster_points, axis=0)
        voxel_centroid = mean_physical / anisotropy
        consensus_centroids.append(voxel_centroid.tolist())

    if not consensus_centroids:
        return cnn_centroids

    return consensus_centroids


def run_dataset(zarr_path: str, dataset_id: str, anisotropy: np.ndarray) -> nx.DiGraph:
    """
    Load a single Zarr dataset, run detector+tracker, return lineage graph.

    Args:
        zarr_path: Path to the Zarr store
        dataset_id: Dataset identifier (for logging)
        anisotropy: Physical voxel scale (Z, Y, X)

    Returns:
        nx.DiGraph: Tracked lineage graph with nodes and edges
    """
    logger.info(f"[Dataset {dataset_id}] Loading Zarr data...")

    try:
        loader = AnisotropicZarrLoader(store_path=zarr_path, simulate=False)
    except Exception as e:
        logger.error(f"[Dataset {dataset_id}] Failed to load Zarr: {e}")
        raise

    t_dim, z_dim, y_dim, x_dim = loader.get_shape()
    logger.info(f"[Dataset {dataset_id}] Volume shape: (T={t_dim}, Z={z_dim}, Y={y_dim}, X={x_dim})")

    # Run detector on each timepoint
    logger.info(f"[Dataset {dataset_id}] Running detection...")
    centroids_by_t = {}
    motion_vectors_by_t = {}

    for t in range(t_dim):
        vol_3d = loader.load_timepoint_block(t)

        cnn_centroids = extract_peaks_from_volume(vol_3d, threshold=0.4, offset_bias=0.0)
        unet_centroids = extract_peaks_from_volume(vol_3d, threshold=0.45, offset_bias=0.2)

        consensus_centroids = ensemble_consensus_centroids(cnn_centroids, unet_centroids, anisotropy)
        centroids_by_t[t] = consensus_centroids

        motion_vectors = [[0.05, 0.2, 0.3] for _ in consensus_centroids]
        motion_vectors_by_t[t] = motion_vectors

        logger.info(f"[Dataset {dataset_id}] Timepoint {t:02d}: Detected {len(consensus_centroids)} centroids")

    # Run tracker
    logger.info(f"[Dataset {dataset_id}] Running tracker...")
    tracker = STHypergraphTracker(birth_cost=15.0, death_cost=15.0, division_reward=-8.0)
    lineage_graph = tracker.solve_lineage(
        centroids_by_t,
        motion_vectors_by_t,
        anisotropy=anisotropy,
        max_gap_frames=2
    )

    # Smooth mitosis edges
    logger.info(f"[Dataset {dataset_id}] Smoothing mitosis edges...")
    lineage_graph = tracker.smooth_mitosis_edges(lineage_graph, centroids_by_t, window_size=2)

    logger.info(f"[Dataset {dataset_id}] Tracker complete: {lineage_graph.number_of_nodes()} nodes, {lineage_graph.number_of_edges()} edges")

    # Convert to tracksdata graph (networkx -> tracksdata)
    logger.info(f"[Dataset {dataset_id}] Converting to tracksdata format...")
    td_graph = convert_nx_to_tracksdata(lineage_graph, dataset_id)

    return td_graph


def convert_nx_to_tracksdata(nx_graph: nx.DiGraph, dataset_id: str) -> td.graph.BaseGraph:
    """
    Convert networkx DiGraph to tracksdata BaseGraph format.

    Args:
        nx_graph: NetworkX directed graph with nodes (t, node_id) and edge attributes
        dataset_id: Dataset identifier

    Returns:
        tracksdata BaseGraph with proper node/edge structure
    """
    # Create a new tracksdata graph
    td_g = td.RXGraph()

    # Add nodes
    for node, attrs in nx_graph.nodes(data=True):
        t, node_id = node
        coords = attrs.get('coords', [0, 0, 0])

        td_g.add_node(
            node_id=node_id,
            attrs={
                't': int(t),
                'z': int(coords[0]),
                'y': int(coords[1]),
                'x': int(coords[2])
            }
        )

    # Add edges
    for source, target in nx_graph.edges():
        source_t, source_node_id = source
        target_t, target_node_id = target
        td_g.add_edge(source_node_id, target_node_id)

    return td_g


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Pipeline Orchestrator")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/staging/test",
        help="Directory containing test data (submission pass)"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/staging/train",
        help="Directory containing train data with ground truth (scoring pass)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="submissions/phase_0_baseline_submission.csv",
        help="Output path for submission CSV"
    )

    args = parser.parse_args()

    anisotropy = np.array([4.0, 1.0, 1.0])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    logger.info("=" * 80)
    logger.info("Phase 0 Pipeline: Multi-Dataset End-to-End Test")
    logger.info("=" * 80)

    # PASS 1: Submission pass (test/ data)
    logger.info(f"\n[PASS 1: SUBMISSION] Processing test data from {args.test_dir}")
    logger.info("-" * 80)

    test_dir_path = Path(args.test_dir)
    submission_graphs = {}

    # Glob all .zarr directories in test/
    zarr_stores = sorted(test_dir_path.glob("*.zarr"))
    logger.info(f"Found {len(zarr_stores)} Zarr stores in {args.test_dir}")

    for zarr_path in zarr_stores:
        dataset_id = zarr_path.stem  # e.g., "44b6_0113de3b"
        try:
            graph = run_dataset(str(zarr_path), dataset_id, anisotropy)
            submission_graphs[dataset_id] = graph
            logger.info(f"✓ Dataset {dataset_id} processed successfully\n")
        except Exception as e:
            logger.error(f"✗ Dataset {dataset_id} failed: {e}\n")
            continue

    logger.info(f"[PASS 1 COMPLETE] Processed {len(submission_graphs)} datasets")

    # Export submission CSV
    logger.info(f"\n[EXPORT] Generating submission CSV at {args.output_path}")
    logger.info("-" * 80)

    try:
        export_submission(submission_graphs, args.output_path)
        logger.info(f"✓ Submission CSV exported: {args.output_path}")

        # Validate submission
        validate_submission(args.output_path)
        logger.info(f"✓ Submission validation passed (schema-compliant)")

        # Report submission stats
        import pandas as pd
        df = pd.read_csv(args.output_path)
        num_rows = len(df)
        num_datasets = df['dataset'].nunique()
        logger.info(f"  - Total rows: {num_rows}")
        logger.info(f"  - Datasets: {num_datasets}")
        logger.info(f"  - Sample: \n{df.head(5).to_string()}\n")

    except Exception as e:
        logger.error(f"✗ Submission export/validation failed: {e}")
        raise

    # PASS 2: Local scoring pass (train/ data)
    logger.info(f"\n[PASS 2: SCORING] Processing train data from {args.train_dir}")
    logger.info("-" * 80)

    train_dir_path = Path(args.train_dir)
    scoring_graphs = {}
    gt_graphs = {}
    gt_metadata = {}

    zarr_stores = sorted(train_dir_path.glob("*.zarr"))
    logger.info(f"Found {len(zarr_stores)} Zarr stores in {args.train_dir}")

    for zarr_path in zarr_stores:
        dataset_id = zarr_path.stem
        try:
            # Run tracker on train data
            graph = run_dataset(str(zarr_path), dataset_id, anisotropy)
            scoring_graphs[dataset_id] = graph

            # Load ground truth
            geff_path = train_dir_path / f"{dataset_id}.geff"
            if geff_path.exists():
                gt_graph, metadata = load_geff_ground_truth(str(geff_path))
                gt_graphs[dataset_id] = gt_graph
                gt_metadata[dataset_id] = metadata
                logger.info(f"✓ Dataset {dataset_id} scored (GT loaded)\n")
            else:
                logger.warning(f"⚠ Dataset {dataset_id}: No .geff found, skipping evaluation")

        except Exception as e:
            logger.error(f"✗ Dataset {dataset_id} failed: {e}\n")
            continue

    logger.info(f"[PASS 2 COMPLETE] Processed {len(scoring_graphs)} datasets")

    # Evaluate
    logger.info(f"\n[EVALUATION] Computing scores")
    logger.info("-" * 80)

    if gt_graphs:
        try:
            results = evaluate_submission(
                scoring_graphs,
                gt_graphs,
                scale=(1.625, 0.40625, 0.40625),
                max_distance=7.0,
                gt_metadata=gt_metadata
            )

            logger.info(f"Edge Jaccard: {results['edge_jaccard']:.4f}")
            logger.info(f"Adjusted Edge Jaccard: {results['adjusted_edge_jaccard']:.4f}")
            logger.info(f"Division Jaccard: {results['division_jaccard']:.4f}")
            logger.info(f"Combined Score: {results['score']:.4f}")
            logger.info(f"Baseline: 0.763")
            logger.info(f"Above baseline: {'YES' if results['score'] > 0.763 else 'NO'}")
            logger.info(f"Datasets evaluated: {results['num_datasets']}")

        except Exception as e:
            logger.error(f"✗ Evaluation failed: {e}")
            raise
    else:
        logger.warning("No ground truth available for evaluation")

    logger.info("\n" + "=" * 80)
    logger.info("Phase 0 Pipeline Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
