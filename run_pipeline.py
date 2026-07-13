"""
Phase 0 Pipeline Orchestrator: Multi-Dataset Integration and End-to-End Testing

Refactored to:
1. Accept --test-dir and --train-dir arguments (two distinct passes)
2. Load real Zarr data via AnisotropicZarrLoader
3. Run STHypergraphTracker on each dataset
4. Export submission CSV via export_submission()
5. Evaluate locally against .geff ground truth (train/ only)
"""

import argparse
import logging
import os
import time
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import tracksdata as td
from scipy import ndimage

from src.data_loader import AnisotropicZarrLoader
from src.evaluation import DEFAULT_SCALE, evaluate_submission, load_geff_ground_truth
from src.run_tracker import (
    RunTracker,
    dataset_checkpoint_key,
    detection_cache_key,
    load_dataset_checkpoint,
    load_detection_cache,
    save_dataset_checkpoint,
    save_detection_cache,
)
from src.submission_exporter import export_submission, validate_submission
from src.tracker import STHypergraphTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Pipeline] %(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger("Pipeline")


def pool_kernel_from_um(um: float, voxel_size: tuple) -> tuple:
    """Convert a physical-micron NMS radius into an odd per-axis pooling kernel
    size in voxels, given each axis's physical voxel size (Z, Y, X) in um."""
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def extract_peaks_from_volume(vol: np.ndarray, threshold=0.4, voxel_size=DEFAULT_SCALE, nms_radius_um=5.0):
    """
    Real 3D non-max suppression via max_pool3d, kernel sized from physical
    micrometers (not a fixed voxel size) -- replicates the host's own reference
    NMS approach (REFERENCE_IMPLEMENTATION.md S5). Applied directly to the raw
    (already quantile-normalized) intensity volume, since no learned detector
    exists yet in Phase 1; Phase 2 swaps this input for real detection logits.

    IMPORTANT deviation from the host's version: the host applies this to a
    trained model's logits, which are naturally sharp/point-like. Raw real
    microscopy intensity has broad, flat-topped bright regions (whole cell
    bodies, out-of-focus glow), so a strict `vol == pooled` comparison ties
    across every voxel in a plateau -- verified directly on real data, this
    produced ~282,000 "peaks" (6.7% of all voxels) at every threshold from 0.5
    to 0.98, virtually threshold-independent. Collapsing each connected tied
    region to a single centroid (via scipy.ndimage.label) fixes this without
    abandoning the host's kernel-sizing approach.

    Uses scipy.ndimage.maximum_filter rather than torch's max_pool3d: verified
    ~22x faster on real data (0.6s vs 13s per timepoint) for this kernel size,
    with identical peak count -- naive 3D max-pooling doesn't scale well to a
    (3,13,13) kernel on CPU, whereas scipy's separable filter does.
    Returns list of 3D coordinates [z, y, x].
    """
    kernel = pool_kernel_from_um(nms_radius_um, voxel_size)
    pooled = ndimage.maximum_filter(vol, size=kernel, mode='constant', cval=-np.inf)
    is_peak = (vol == pooled) & (vol > threshold)

    labeled, num_labels = ndimage.label(is_peak)
    if num_labels == 0:
        return []
    centroids = ndimage.center_of_mass(is_peak, labeled, range(1, num_labels + 1))
    return [list(c) for c in centroids]


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
# Recalibrated 2026-07-06 (Phase 1) for the new max_pool3d/maximum_filter-based
# NMS peak-finding -- the old 0.92/0.94 values were tuned for the retired
# stride-8 grid scan and are meaningless for this algorithm. Chosen via
# scripts/sweep_threshold.py across all 4 staged train datasets, 5 timepoints
# each, 7 threshold values: candidate counts are fairly threshold-insensitive
# in the 0.8-0.95 range (real signal is dominated by per-timepoint/per-dataset
# cell density, not threshold choice) so 0.85/0.9 keeps the CNN/UNET ensemble's
# two-distinct-views design without landing at an extreme.
CNN_THRESHOLD = 0.85
UNET_THRESHOLD = 0.9
# Safety cap: prevents pathological ILP blowup regardless of detector quality.
# 2026-07-04 (Phase 0, CBC solver): set to 30 because the placeholder detector's
# grid-scan over-predicted catastrophically (up to ~18,000 candidates/timepoint)
# and CBC could not handle more than ~30-100/timepoint in reasonable time.
# 2026-07-07 (Phase 1, SCIP solver): raised to 75. Two things changed the
# calculus: (1) the CBC->SCIP swap gave an 11.7x real-data solver speedup, and
# (2) the new real max_pool3d/maximum_filter peak-finding (replacing the grid
# scan) produces genuinely higher real candidate counts on dense/late-development
# timepoints (avg ~1110/timepoint measured on the densest staged dataset's t=85-99
# tail) -- these are NOT detector over-prediction, they're real embryo cell
# density, so cap=30 was discarding ~97% of legitimate candidates every frame,
# confirmed as the dominant reason the Phase 1 local score stayed near-zero
# despite working peak-finding (786/786 timepoints hit the cap in a full 4-dataset
# run). Direct profiling on that same dense tail (STHypergraphTracker.solve_lineage,
# SCIP): cap=30 -> 1.97s, cap=50 -> 4.99s, cap=75 -> 13.44s, cap=100 -> 27.09s for
# a 15-timepoint slice -- confirms solve time still scales super-linearly, so this
# is a considered, profiled increase, not an unbounded fix. Full windowed/min-cost-flow
# scaling remains Phase 3 scope for when real cell counts (not just this placeholder
# gap) demand it.
MAX_CANDIDATES_PER_TIMEPOINT = 75
BIRTH_COST = 15.0
DEATH_COST = 15.0
DIVISION_REWARD = -8.0
MAX_GAP_FRAMES = 2

# Post-solve graph refinement (COMPETITOR_RESEARCH_2026-07-13.md items 3/4,
# verified real techniques from 2 independent top public Kaggle notebooks --
# see DEFERRED_IMPROVEMENTS.md's priority matrix). Both are low-risk: pruning
# is a pure graph op with a division-protection safety check, smoothing only
# mutates coordinates and cannot touch edges/T_pred/division_jaccard.
MIN_TRACK_LEN = 4  # the two source notebooks disagreed (4 vs 7); tune against our own data later
KEEP_DIVISION_COMPONENTS = True
LINEFIT_WEIGHT = 0.76  # confirmed exact value in the source notebook
LINEFIT_WINDOW = 2


def _dataset_full_config() -> dict:
    """Everything that affects a dataset's output -- feeds both cache-key functions.
    Bump nothing here for code-only changes (e.g. an ILP formulation fix); use
    --force-rerun for those, since they aren't captured by a config hash."""
    return {
        "cnn_threshold": CNN_THRESHOLD, "unet_threshold": UNET_THRESHOLD,
        "max_candidates": MAX_CANDIDATES_PER_TIMEPOINT,
        "birth_cost": BIRTH_COST, "death_cost": DEATH_COST,
        "division_reward": DIVISION_REWARD, "max_gap_frames": MAX_GAP_FRAMES,
    }


def run_dataset(zarr_path: str, dataset_id: str, anisotropy: np.ndarray,
                 force_rerun: bool = False) -> tuple[nx.DiGraph, dict[str, float], bool]:
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
    det_key = detection_cache_key(zarr_path, CNN_THRESHOLD, UNET_THRESHOLD, MAX_CANDIDATES_PER_TIMEPOINT)
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
            if t % 10 == 0:
                logger.info(f"[Dataset {dataset_id}] Detection progress: timepoint {t}/{t_dim}")
            vol_3d = loader.load_timepoint_block(t)

            # NOTE 2026-07-05 (Phase 1): CNN_THRESHOLD/UNET_THRESHOLD's 0.92/0.94 values
            # were tuned against the OLD stride-8 grid-scan's candidate distribution, not
            # against real max_pool3d-based NMS peaks -- the two produce very different
            # candidate counts at the same threshold. Re-sweep before trusting these values
            # (see scripts/sweep_threshold.py); the MAX_CANDIDATES cap below remains the
            # hard backstop against a repeat of the 2.5+ hour ILP-blowup incident either way.
            cnn_centroids = extract_peaks_from_volume(vol_3d, threshold=CNN_THRESHOLD, voxel_size=DEFAULT_SCALE)
            unet_centroids = extract_peaks_from_volume(vol_3d, threshold=UNET_THRESHOLD, voxel_size=DEFAULT_SCALE)

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

            # Phase 1 decision (01-CONTEXT.md): zero motion vectors, not a hand-rolled
            # heuristic. Motion vectors only warp a cell's position before the tracker
            # measures next-frame distance; real measured inter-frame displacement (see
            # STATE.md) shows only 3.1% of true edges exceed the tracker's link/break-even
            # threshold, so zero should already be sufficient for the vast majority of
            # links. Phase 2 replaces this with a real learned motion field.
            motion_vectors = [[0.0, 0.0, 0.0] for _ in consensus_centroids]
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

    # Post-solve graph refinement (competitor-validated, low-risk -- see
    # DEFERRED_IMPROVEMENTS.md priority matrix). Pruning first (reduces graph
    # size before spending time smoothing nodes that'll be dropped anyway);
    # order doesn't affect correctness since smoothing only moves coordinates.
    nodes_before_prune = lineage_graph.number_of_nodes()
    lineage_graph = prune_short_tracks(lineage_graph)
    logger.info(
        f"[Dataset {dataset_id}] Pruned short tracks: "
        f"{nodes_before_prune} -> {lineage_graph.number_of_nodes()} nodes"
    )
    lineage_graph = linefit_smooth_coordinates(lineage_graph)

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
    logger.info("  ---")
    logger.info(f"  TOTAL:      {timings['total']:7.2f}s")

    save_dataset_checkpoint(ckpt_key, td_graph, timings)

    return td_graph, timings, False


def prune_short_tracks(
    lineage_graph: nx.DiGraph,
    min_track_len: int = MIN_TRACK_LEN,
    keep_division_components: bool = KEEP_DIVISION_COMPONENTS,
) -> nx.DiGraph:
    """
    Component-based short-track pruning, with division protection.

    Drops entire weakly-connected components shorter than min_track_len, EXCEPT
    any component containing a division event (a node with out-degree >= 2) is
    always kept regardless of length. This is the real technique two independent
    top public Kaggle notebooks for this exact competition converged on
    (COMPETITOR_RESEARCH_2026-07-13.md item 3, verified against the real source,
    not just Gemini's summary) -- it directly answers this project's own earlier
    concern (see DEFERRED_IMPROVEMENTS.md's old item 1) that naive isolated-node
    pruning would risk cutting real births/deaths alongside noise: operating on
    whole connected components with a structural division safety check avoids
    that specific failure mode instead of relying on a per-node confidence
    threshold.

    Reduces the `T_pred` over-prediction penalty in `adjusted_edge_jaccard` by
    removing likely-spurious short fragments, without touching any node that's
    part of a real division event.

    Args:
        lineage_graph: post-solve lineage graph (before tracksdata conversion)
        min_track_len: components with fewer than this many nodes are dropped,
            unless keep_division_components exempts them. The two source
            notebooks disagreed (4 vs 7) -- treat as a tunable, not a constant
            to copy verbatim; needs empirical tuning against our own real
            checkpoint data (PRD.md Phase 4).
        keep_division_components: if True, any component containing a node
            with out-degree >= 2 is kept regardless of length.

    Returns:
        A new graph containing only the kept nodes/edges (input is not mutated).
    """
    if lineage_graph.number_of_nodes() == 0:
        return lineage_graph

    keep_nodes: set = set()
    for component in nx.weakly_connected_components(lineage_graph):
        has_division = any(lineage_graph.out_degree(n) >= 2 for n in component)
        if len(component) >= min_track_len or (keep_division_components and has_division):
            keep_nodes.update(component)

    return lineage_graph.subgraph(keep_nodes).copy()


def linefit_smooth_coordinates(
    lineage_graph: nx.DiGraph,
    weight: float = LINEFIT_WEIGHT,
    window: int = LINEFIT_WINDOW,
) -> nx.DiGraph:
    """
    Topology-preserving line-fit coordinate smoothing.

    Mutates only node 'coords' attributes -- never adds/removes nodes or edges,
    so it cannot affect `T_pred` or `division_jaccard` at all. Verified real
    technique (COMPETITOR_RESEARCH_2026-07-13.md item 4, w=0.76 confirmed as
    the exact value used in the source notebook, not approximated). Directly
    targets the competition's strict 7.0µm `DistanceMatching` gate by removing
    frame-to-frame jitter along already-correctly-linked tracks.

    For each node, walks up to `window` steps backward/forward through STRICTLY
    single-predecessor/single-successor chains of CONSECUTIVE-frame edges only
    (a chain stops at any branch, merge, or gap-closing edge that skips a
    frame -- including a gap edge would corrupt the linear-fit's dt
    assumption). Fits a degree-1 polynomial per axis (z, y, x) against relative
    frame offset over that local neighborhood, then blends the fitted position
    with the raw one.

    Args:
        lineage_graph: post-solve lineage graph, mutated in place and returned
        weight: blend weight toward the fitted line (0=no smoothing, 1=fully
            replace with the fit). Clamped to [0, 1].
        window: how many single-successor/predecessor steps to walk in each
            direction before stopping.

    Returns:
        The same graph object, with 'coords' updated in place.
    """
    if lineage_graph.number_of_edges() == 0:
        return lineage_graph

    weight = max(0.0, min(1.0, weight))
    original_coords = {
        n: np.asarray(attrs["coords"], dtype=np.float64)
        for n, attrs in lineage_graph.nodes(data=True)
    }

    # Only strictly-consecutive-frame edges (t -> t+1) define the linear
    # neighborhood -- gap-closing edges (t -> t+2/t+3) would corrupt the fit.
    predecessor: dict = {}
    successor: dict = {}
    for u, v in lineage_graph.edges():
        if v[0] == u[0] + 1:
            successor.setdefault(u, []).append(v)
            predecessor.setdefault(v, []).append(u)

    updated_coords: dict = {}
    for node in lineage_graph.nodes():
        neighborhood = [(0, node)]

        current = node
        for step in range(1, window + 1):
            preds = predecessor.get(current, [])
            if len(preds) != 1:
                break
            current = preds[0]
            neighborhood.append((-step, current))

        current = node
        for step in range(1, window + 1):
            succs = successor.get(current, [])
            if len(succs) != 1:
                break
            current = succs[0]
            neighborhood.append((step, current))

        if len(neighborhood) < 3:
            continue  # not enough points for a meaningful line fit

        dts = np.array([delta for delta, _ in neighborhood], dtype=np.float64)
        coords = np.stack([original_coords[n] for _, n in neighborhood])
        fitted = np.array([
            np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0)
            for axis in range(3)
        ])
        if not np.isfinite(fitted).all():
            continue
        updated_coords[node] = (1.0 - weight) * original_coords[node] + weight * fitted

    for node, coords in updated_coords.items():
        lineage_graph.nodes[node]["coords"] = coords

    return lineage_graph


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

    # DEFAULT_SCALE (real physical microns, (1.625, 0.40625, 0.40625)) not the
    # (4.0, 1.0, 1.0) Z:Y:X ratio -- ensemble_consensus_centroids()/run_dataset()'s
    # tracker calls compare scaled distances against real micron thresholds
    # (eps_microns, max_z_micron, max_xy_micron, the 40um search radius), and the
    # ratio inflates every computed distance by ~2.46x, silently tightening all
    # four gates below their intended values. Same bug class already fixed once in
    # evaluation.py's DEFAULT_SCALE; this was the still-unfixed second instance.
    anisotropy = np.array(DEFAULT_SCALE)

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
        logger.info("✓ Submission validation passed (schema-compliant)")

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
    logger.info("\n[EVALUATION] Computing scores")
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
            logger.info("Baseline: 0.763")
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
        logger.info("    ---")
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
                logger.info("  -> PRIMARY BOTTLENECK: ILP solver dominates execution time")
                logger.info("  -> Optimization: Consider ILP solver parameters, gap closing strategy, or approximation algorithms for Phase 2/3")

    logger.info("\n" + "=" * 80)
    logger.info("Phase 0 Pipeline Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
