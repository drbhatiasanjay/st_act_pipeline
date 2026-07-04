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
import polars as pl
import tracksdata as td

from src.data_loader import AnisotropicZarrLoader
from src.tracker import STHypergraphTracker
from src.submission_exporter import export_submission, validate_submission
from src.evaluation import evaluate_submission, load_geff_ground_truth
from src.run_tracker import (
    RunTracker,
    detection_cache_key,
    load_detection_cache,
    save_detection_cache,
    dataset_checkpoint_key,
    load_dataset_checkpoint,
    save_dataset_checkpoint,
)

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


# Detection thresholds and tracker costs live here (module level) so cache keys computed
# in run_dataset() and the main loop always agree on what "the same config" means.
CNN_THRESHOLD = 0.92
UNET_THRESHOLD = 0.94
CNN_OFFSET = 0.0
UNET_OFFSET = 0.2
# Safety cap: prevents pathological ILP blowup regardless of detector quality (a real
# trained detector in Phase 2 could also occasionally over-predict, this isn't just a
# placeholder-threshold patch). Value chosen from direct profiling of
# STHypergraphTracker.solve_lineage on synthetic data at this density: ILP solve time
# scales super-linearly with candidates/timepoint (50->100->150->200 measured at
# 0.85s->3.68s->6.94s->17.72s for a 10-timepoint slice) -- 500/timepoint (the original,
# unbounded value) took 112.7s for just 10 timepoints and was the direct cause of a
# 2.5+ hour stuck run with no output. 100 keeps a full 100-timepoint dataset in the
# tens-of-seconds range. This is the SAME risk already flagged in STATE.md as a Phase 3
# "ILP solve time at scale" concern -- it's just materializing in Phase 0 at much lower
# candidate density than expected, via the detector's over-prediction, not real cell count.
MAX_CANDIDATES_PER_TIMEPOINT = 100
BIRTH_COST = 15.0
DEATH_COST = 15.0
DIVISION_REWARD = -8.0
MAX_GAP_FRAMES = 2


def _dataset_full_config() -> dict:
    """Everything that affects a dataset's output -- feeds both cache-key functions.
    Bump nothing here for code-only changes (e.g. an ILP formulation fix); use
    --force-rerun for those, since they aren't captured by a config hash."""
    return {
        "cnn_threshold": CNN_THRESHOLD, "unet_threshold": UNET_THRESHOLD,
        "cnn_offset": CNN_OFFSET, "unet_offset": UNET_OFFSET,
        "max_candidates": MAX_CANDIDATES_PER_TIMEPOINT,
        "birth_cost": BIRTH_COST, "death_cost": DEATH_COST,
        "division_reward": DIVISION_REWARD, "max_gap_frames": MAX_GAP_FRAMES,
    }


def run_dataset(zarr_path: str, dataset_id: str, anisotropy: np.ndarray,
                 force_rerun: bool = False) -> Tuple[nx.DiGraph, Dict[str, float], bool]:
    """
    Load a single Zarr dataset, run detector+tracker, return lineage graph + timing info.

    Checkpointed: if a prior run completed this exact dataset under the exact same
    config (thresholds, costs, gap frames -- see _dataset_full_config), returns the
    cached result immediately. Detection is ALSO cached separately from the full
    dataset result, since tracker-cost-only changes (the common case during Phase 4
    tuning) don't need to re-run detection at all.

    Args:
        zarr_path: Path to the Zarr store
        dataset_id: Dataset identifier (for logging)
        anisotropy: Physical voxel scale (Z, Y, X)
        force_rerun: bypass both caches (e.g. after a code change the config hash can't see)

    Returns:
        (nx.DiGraph, dict, bool): lineage graph, timing info, whether the full-dataset
        checkpoint was used (if True, `timings` is the ORIGINAL run's timings, not this call's)
    """
    full_config = _dataset_full_config()
    ckpt_key = dataset_checkpoint_key(dataset_id, zarr_path, full_config)

    if not force_rerun:
        cached = load_dataset_checkpoint(ckpt_key)
        if cached is not None:
            logger.info(f"[Dataset {dataset_id}] Using cached checkpoint (config unchanged) -- skipping detection+tracking entirely")
            return cached["lineage_graph"], cached["timings"], True

    timings = {}
    start_dataset = time.time()

    # Load phase
    start_load = time.time()
    logger.info(f"[Dataset {dataset_id}] Loading Zarr data...")

    try:
        loader = AnisotropicZarrLoader(store_path=zarr_path, simulate=False)
    except Exception as e:
        logger.error(f"[Dataset {dataset_id}] Failed to load Zarr: {e}")
        raise

    t_dim, z_dim, y_dim, x_dim = loader.get_shape()
    logger.info(f"[Dataset {dataset_id}] Volume shape: (T={t_dim}, Z={z_dim}, Y={y_dim}, X={x_dim})")
    timings['load'] = time.time() - start_load

    # Detection phase -- cached separately (see module docstring): the volume is static,
    # detection thresholds change rarely, so this is the expensive step worth skipping on
    # the common "just retuning tracker costs" rerun.
    start_detect = time.time()
    det_key = detection_cache_key(zarr_path, CNN_THRESHOLD, UNET_THRESHOLD, CNN_OFFSET, UNET_OFFSET, MAX_CANDIDATES_PER_TIMEPOINT)
    detection_cached = None if force_rerun else load_detection_cache(det_key)

    if detection_cached is not None:
        logger.info(f"[Dataset {dataset_id}] Using cached detection output ({len(detection_cached['centroids_by_t'])} timepoints)")
        centroids_by_t = detection_cached["centroids_by_t"]
        motion_vectors_by_t = detection_cached["motion_vectors_by_t"]
    else:
        logger.info(f"[Dataset {dataset_id}] Running detection...")
        centroids_by_t = {}
        motion_vectors_by_t = {}

        for t in range(t_dim):
            vol_3d = loader.load_timepoint_block(t)

            # NOTE 2026-07-04: thresholds raised from the original 0.4/0.45. Real
            # quantile-normalized data has a saturated plateau near 1.0 (dense embryonic
            # tissue, not the old simulated [0,1]-uniform data these constants were tuned
            # against) -- 0.4 produced ~18,000 candidates in a single timepoint of real
            # data and caused the ILP tracker to blow up combinatorially (2.5+ hour stuck
            # run, 30GB+ RAM, no output). 0.92/0.94 cuts that to a few thousand grid hits;
            # the MAX_CANDIDATES cap below is the real backstop.
            cnn_centroids = extract_peaks_from_volume(vol_3d, threshold=CNN_THRESHOLD, offset_bias=CNN_OFFSET)
            unet_centroids = extract_peaks_from_volume(vol_3d, threshold=UNET_THRESHOLD, offset_bias=UNET_OFFSET)

            consensus_centroids = ensemble_consensus_centroids(cnn_centroids, unet_centroids, anisotropy)

            if len(consensus_centroids) > MAX_CANDIDATES_PER_TIMEPOINT:
                logger.warning(
                    f"[Dataset {dataset_id}] Timepoint {t:02d}: {len(consensus_centroids)} candidates "
                    f"exceeds cap ({MAX_CANDIDATES_PER_TIMEPOINT}) -- truncating. This means the "
                    f"placeholder detector is over-predicting for this frame; expect degraded local "
                    f"score, not a crash."
                )
                consensus_centroids = consensus_centroids[:MAX_CANDIDATES_PER_TIMEPOINT]

            centroids_by_t[t] = consensus_centroids

            motion_vectors = [[0.05, 0.2, 0.3] for _ in consensus_centroids]
            motion_vectors_by_t[t] = motion_vectors

            logger.debug(f"[Dataset {dataset_id}] Timepoint {t:02d}: Detected {len(consensus_centroids)} centroids")

        save_detection_cache(det_key, centroids_by_t, motion_vectors_by_t)

    timings['detection'] = time.time() - start_detect
    logger.info(f"[Dataset {dataset_id}] Detection complete ({timings['detection']:.1f}s)")

    # Tracking phase
    start_track = time.time()
    logger.info(f"[Dataset {dataset_id}] Running tracker...")
    tracker = STHypergraphTracker(birth_cost=BIRTH_COST, death_cost=DEATH_COST, division_reward=DIVISION_REWARD)
    lineage_graph = tracker.solve_lineage(
        centroids_by_t,
        motion_vectors_by_t,
        anisotropy=anisotropy,
        max_gap_frames=MAX_GAP_FRAMES
    )

    # Smooth mitosis edges
    logger.info(f"[Dataset {dataset_id}] Smoothing mitosis edges...")
    lineage_graph = tracker.smooth_mitosis_edges(lineage_graph, centroids_by_t, window_size=2)

    timings['tracking'] = time.time() - start_track
    logger.info(f"[Dataset {dataset_id}] Tracking complete ({timings['tracking']:.1f}s): {lineage_graph.number_of_nodes()} nodes, {lineage_graph.number_of_edges()} edges")

    # Conversion phase
    start_convert = time.time()
    logger.info(f"[Dataset {dataset_id}] Converting to tracksdata format...")
    td_graph = convert_nx_to_tracksdata(lineage_graph, dataset_id)
    timings['conversion'] = time.time() - start_convert

    # Total time for this dataset
    timings['total'] = time.time() - start_dataset

    logger.info(f"[Dataset {dataset_id}] TIMING SUMMARY:")
    logger.info(f"  Load:       {timings['load']:7.2f}s")
    logger.info(f"  Detection:  {timings['detection']:7.2f}s")
    logger.info(f"  Tracking:   {timings['tracking']:7.2f}s")
    logger.info(f"  Conversion: {timings['conversion']:7.2f}s")
    logger.info(f"  ---")
    logger.info(f"  TOTAL:      {timings['total']:7.2f}s")

    save_dataset_checkpoint(ckpt_key, td_graph, timings)

    return td_graph, timings, False


def convert_nx_to_tracksdata(nx_graph: nx.DiGraph, dataset_id: str) -> td.graph.BaseGraph:
    """
    Convert networkx DiGraph to tracksdata BaseGraph format.

    Args:
        nx_graph: NetworkX directed graph with nodes (t, node_id) and edge attributes
        dataset_id: Dataset identifier

    Returns:
        tracksdata BaseGraph with proper node/edge structure
    """
    # Create a new tracksdata graph (IndexedRXGraph)
    td_g = td.graph.IndexedRXGraph()

    # Register attribute keys
    for key in ('z', 'y', 'x'):
        try:
            td_g.add_node_attr_key(key, pl.Int64, 0)
        except ValueError:
            pass  # Key already exists

    # Map networkx node ids to tracksdata node ids
    node_mapping = {}
    for node, attrs in nx_graph.nodes(data=True):
        t, node_idx = node
        coords = attrs.get('coords', [0, 0, 0])

        # Add node and track the returned node_id
        attrs_dict = {
            't': int(t),
            'z': int(coords[0]),
            'y': int(coords[1]),
            'x': int(coords[2])
        }
        td_node_id = td_g.add_node(attrs_dict)
        node_mapping[(t, node_idx)] = td_node_id

    # Add edges using the mapped node ids
    for source, target in nx_graph.edges():
        source_td_id = node_mapping[source]
        target_td_id = node_mapping[target]
        td_g.add_edge(source_td_id, target_td_id, {})

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
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Bypass detection and dataset checkpoints (e.g. after a code change the "
             "config hash can't detect, like an ILP formulation edit). Without this flag, "
             "a dataset already completed under the identical config is skipped entirely."
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
    submission_timings = {}

    # Glob all .zarr directories in test/
    zarr_stores = sorted(test_dir_path.glob("*.zarr"))
    logger.info(f"Found {len(zarr_stores)} Zarr stores in {args.test_dir}")

    pass1_start = time.time()
    pass1_tracker = RunTracker("pass1_submission", total_units=len(zarr_stores))
    for zarr_path in zarr_stores:
        dataset_id = zarr_path.stem  # e.g., "44b6_0113de3b"
        unit = pass1_tracker.start_unit(dataset_id)
        try:
            graph, timings, was_cached = run_dataset(str(zarr_path), dataset_id, anisotropy, force_rerun=args.force_rerun)
            submission_graphs[dataset_id] = graph
            submission_timings[dataset_id] = timings
            unit.done(extra={"nodes": graph.num_nodes(), "edges": graph.num_edges()}, cached=was_cached)
            logger.info(f"✓ Dataset {dataset_id} processed successfully\n")
        except Exception as e:
            unit.failed(str(e))
            logger.error(f"✗ Dataset {dataset_id} failed: {e}\n")
            continue

    pass1_duration = time.time() - pass1_start
    pass1_tracker.run_end(summary={"datasets_completed": len(submission_graphs), "total": len(zarr_stores)})
    logger.info(f"[PASS 1 COMPLETE] Processed {len(submission_graphs)} datasets in {pass1_duration:.1f}s")

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
    scoring_timings = {}

    zarr_stores = sorted(train_dir_path.glob("*.zarr"))
    logger.info(f"Found {len(zarr_stores)} Zarr stores in {args.train_dir}")

    pass2_start = time.time()
    pass2_tracker = RunTracker("pass2_scoring", total_units=len(zarr_stores))
    for zarr_path in zarr_stores:
        dataset_id = zarr_path.stem
        unit = pass2_tracker.start_unit(dataset_id)
        try:
            # Run tracker on train data
            graph, timings, was_cached = run_dataset(str(zarr_path), dataset_id, anisotropy, force_rerun=args.force_rerun)
            scoring_graphs[dataset_id] = graph
            scoring_timings[dataset_id] = timings

            # Load ground truth
            geff_path = train_dir_path / f"{dataset_id}.geff"
            if geff_path.exists():
                gt_graph, metadata = load_geff_ground_truth(str(geff_path))
                gt_graphs[dataset_id] = gt_graph
                gt_metadata[dataset_id] = metadata
                logger.info(f"✓ Dataset {dataset_id} scored (GT loaded)\n")
            else:
                logger.warning(f"⚠ Dataset {dataset_id}: No .geff found, skipping evaluation")
            unit.done(extra={"nodes": graph.num_nodes(), "edges": graph.num_edges()}, cached=was_cached)

        except Exception as e:
            unit.failed(str(e))
            logger.error(f"✗ Dataset {dataset_id} failed: {e}\n")
            continue

    pass2_duration = time.time() - pass2_start
    pass2_tracker.run_end(summary={"datasets_completed": len(scoring_graphs), "total": len(zarr_stores)})
    logger.info(f"[PASS 2 COMPLETE] Processed {len(scoring_graphs)} datasets in {pass2_duration:.1f}s")

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

    # Final timing report
    logger.info("\n" + "=" * 80)
    logger.info("TIMING ANALYSIS")
    logger.info("=" * 80)

    if submission_timings:
        logger.info("\n[PASS 1 - SUBMISSION PASS]")
        total_load = sum(t.get('load', 0) for t in submission_timings.values())
        total_detect = sum(t.get('detection', 0) for t in submission_timings.values())
        total_track = sum(t.get('tracking', 0) for t in submission_timings.values())
        total_convert = sum(t.get('conversion', 0) for t in submission_timings.values())

        for dataset_id, timings in sorted(submission_timings.items()):
            logger.info(f"  {dataset_id}: {timings['total']:7.2f}s (load={timings['load']:.1f}s, detect={timings['detection']:.1f}s, track={timings['tracking']:.1f}s)")

        logger.info(f"\n  Subtotals (across all {len(submission_timings)} datasets):")
        logger.info(f"    Load:       {total_load:7.2f}s")
        logger.info(f"    Detection:  {total_detect:7.2f}s")
        logger.info(f"    Tracking:   {total_track:7.2f}s ({total_track / len(submission_timings):.1f}s per dataset avg)")
        logger.info(f"    Conversion: {total_convert:7.2f}s")
        logger.info(f"    ---")
        logger.info(f"    Pass Total: {pass1_duration:7.2f}s")

    if scoring_timings:
        logger.info("\n[PASS 2 - SCORING PASS]")
        total_track = sum(t.get('tracking', 0) for t in scoring_timings.values())

        for dataset_id, timings in sorted(scoring_timings.items()):
            logger.info(f"  {dataset_id}: {timings['total']:7.2f}s (track={timings['tracking']:.1f}s)")

        logger.info(f"\n  Tracking time (which dominates): {total_track:7.2f}s ({total_track / len(scoring_timings):.1f}s per dataset avg)")
        logger.info(f"  Pass Total: {pass2_duration:7.2f}s")

    # Bottleneck analysis
    logger.info("\n[BOTTLENECK ANALYSIS]")
    if submission_timings:
        total_track = sum(t.get('tracking', 0) for t in submission_timings.values())
        total_time = sum(t.get('total', 0) for t in submission_timings.values())
        if total_time > 0:
            track_pct = 100 * total_track / total_time
            logger.info(f"  ILP Solving (Tracking): {track_pct:.1f}% of total time")
            if track_pct > 50:
                logger.info(f"  -> PRIMARY BOTTLENECK: ILP solver dominates execution time")
                logger.info(f"  -> Optimization: Consider ILP solver parameters, gap closing strategy, or approximation algorithms for Phase 2/3")

    logger.info("\n" + "=" * 80)
    logger.info("Phase 0 Pipeline Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
