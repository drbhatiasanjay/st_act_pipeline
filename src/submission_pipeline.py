"""
Shared production submission inference pipeline (P0-6, Part A).

Owns the actual graph-building path used by BOTH generate_submission.py (local
smoke-test generator) and kaggle_kernel_inference/inference_kernel.py (the
real graded Kaggle submission path), so the two production callers can never
silently diverge into two separately-maintained inference/graph-building
loops. Reuses the exact same canonical-node-identity rule
(src/prediction_graph.py's PredictionGraphAssembler) already used by
TrainingLoop.validate_epoch(), and the exact same NMS/threshold/feature-
sampling helpers already used there (src/train.py's extract_inference_peaks /
nodes_and_features_at_peaks, Part A5).
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tracksdata.graph import IndexedRXGraph

from src.checkpoint_manifest import validate_inference_hyperparams
from src.dataset import CompetitionDataset
from src.inference import greedy_edge_assignment
from src.prediction_graph import PredictionGraphAssembler
from src.train import extract_inference_peaks, nodes_and_features_at_peaks

logger = logging.getLogger(__name__)


def build_test_dataset(test_dir: str | Path, sample_id: str) -> CompetitionDataset:
    """Construct a CompetitionDataset covering exactly one test sample (Part
    A1's exact required construction) -- test samples have no split JSON
    membership and no .geff ground truth, so __init__ is bypassed in favor
    of the same manual-attribute pattern already used (pre-P0-6, now
    centralized here) by generate_submission.py and inference_kernel.py."""
    test_dir = Path(test_dir)
    dataset = CompetitionDataset.__new__(CompetitionDataset)
    dataset.data_dir = test_dir
    dataset.split_type = "test"
    dataset.normalize = True
    dataset.anisotropy = (4.0, 1.0, 1.0)
    dataset.physical_voxel_size = (1.625, 0.40625, 0.40625)
    dataset.zip_path = None
    dataset.sample_ids = [sample_id]
    dataset.pairs = []
    dataset._loader_cache = {}
    # Submission inference must see every real consecutive pair -- test
    # samples have no .geff at all, so GT-count filtering isn't even
    # meaningful here, let alone desirable (matches the pre-P0-6 callers).
    dataset.filter_unannotated_pairs = False
    dataset._gt_counts_by_time_cache = {}
    dataset.annotation_pair_stats = None
    dataset._build_pair_index()
    return dataset


def run_sample_loader_inference(
    *,
    sample_id: str,
    loader: DataLoader,
    expected_pair_count: int,
    assembler: PredictionGraphAssembler,
    unet3d: nn.Module,
    transformer: nn.Module,
    device: torch.device,
    hyperparams: dict[str, Any],
) -> dict[str, Any]:
    """Run one sample's complete (t, t+1) window inference, feeding every
    window through the shared PredictionGraphAssembler (Part A2/A3). Enforces
    complete, correctly-scoped loader coverage per Part A2 -- any mismatched
    sample ID, mixed-sample batch, batch size > 1, or incomplete coverage
    raises RuntimeError rather than silently producing a partial graph."""
    if expected_pair_count <= 0:
        raise RuntimeError(
            f"run_sample_loader_inference: expected_pair_count must be positive for "
            f"sample {sample_id!r}, got {expected_pair_count}"
        )
    if len(loader) != expected_pair_count:
        raise RuntimeError(
            f"run_sample_loader_inference: loader length {len(loader)} does not match "
            f"expected_pair_count {expected_pair_count} for sample {sample_id!r}"
        )

    processed_pair_count = 0
    total_candidate_edges = 0
    total_accepted_edges = 0

    for batch in loader:
        batch_sample_ids = batch['sample_id']
        if len(batch_sample_ids) != 1:
            raise RuntimeError(
                f"run_sample_loader_inference: expected batch size 1, got "
                f"{len(batch_sample_ids)} sample IDs {list(batch_sample_ids)!r} "
                f"(sample {sample_id!r})"
            )
        batch_sample_id = batch_sample_ids[0]
        if batch_sample_id != sample_id:
            raise RuntimeError(
                f"run_sample_loader_inference: loader produced a batch for sample_id "
                f"{batch_sample_id!r}, but this loader was built for sample_id "
                f"{sample_id!r} -- mismatched or mixed-sample loader."
            )

        t_idx_values = batch['t_idx']
        if len(t_idx_values) != 1:
            raise RuntimeError(
                f"run_sample_loader_inference: expected exactly one t_idx per batch, "
                f"got {len(t_idx_values)} for sample {sample_id!r}"
            )
        t_idx = int(t_idx_values[0])

        frame_t = batch['frame_t']
        frame_t1 = batch['frame_t1']
        if frame_t.shape[0] != 1 or frame_t1.shape[0] != 1:
            raise RuntimeError(
                f"run_sample_loader_inference: expected batch dimension 1 for "
                f"frame_t/frame_t1, got {frame_t.shape[0]}/{frame_t1.shape[0]} for "
                f"sample {sample_id!r} t_idx={t_idx}"
            )

        frame_t = frame_t.to(device)
        frame_t1 = frame_t1.to(device)

        # Fail loud, immediately, on any out-of-order window for this sample
        # (P0-3 chronological-order contract -- see PredictionGraphAssembler).
        assembler.validate_window_order(sample_id, t_idx)

        x = torch.cat([frame_t, frame_t1], dim=1)
        with torch.no_grad():
            detection_logits, features = unet3d(x)
            # Exactly one Sigmoid between raw detection logits and NMS peak
            # extraction (Part A4).
            detection_probs = torch.sigmoid(detection_logits.float())

            peaks_t = extract_inference_peaks(
                detection_probs, channel=0, t_idx=t_idx, hyperparams=hyperparams,
            )
            peaks_t1 = extract_inference_peaks(
                detection_probs, channel=1, t_idx=t_idx, hyperparams=hyperparams,
            )

            # Canonical graph identity (P0-3, via PredictionGraphAssembler):
            # frame t_idx's source set is peaks_t only for this sample's
            # first window; otherwise it is the already-canonical nodes from
            # the prior window's channel-1 output (peaks_t counted
            # diagnostically only inside the assembler). Frame t_idx+1 is
            # always newly owned here from peaks_t1.
            source_ids, source_coords, target_ids, target_coords = assembler.process_window(
                sample_id, t_idx, peaks_t, peaks_t1,
            )

            nodes_t, features_t = nodes_and_features_at_peaks(features, source_coords, device)
            nodes_t1, features_t1 = nodes_and_features_at_peaks(features, target_coords, device)

            if len(source_coords) > 0 and len(target_coords) > 0:
                edge_logits = transformer(nodes_t, nodes_t1, features_t, features_t1)
                # Exactly one Sigmoid between raw transformer edge logits and
                # greedy edge assignment (Part A4).
                edge_probs = torch.sigmoid(edge_logits.float())
                assignment = greedy_edge_assignment(
                    edge_probs, nodes_t.cpu(), nodes_t1.cpu(),
                    threshold=hyperparams['edge_threshold'], max_children=2, max_parents=1,
                )
                edges = assignment['edges']
                total_candidate_edges += assignment['stats'].get('num_candidate_edges', edge_logits.numel())
            else:
                edges = []

        # The shared submission pipeline never calls graph.add_node()/
        # add_edge() directly -- PredictionGraphAssembler owns all graph
        # mutation (Part A3).
        accepted = assembler.add_edges(sample_id, source_ids, target_ids, edges)
        total_accepted_edges += accepted
        processed_pair_count += 1

    if processed_pair_count == 0:
        raise RuntimeError(f"run_sample_loader_inference: zero batches processed for sample {sample_id!r}.")
    if processed_pair_count != expected_pair_count or processed_pair_count != len(loader):
        raise RuntimeError(
            f"run_sample_loader_inference: processed {processed_pair_count} batches but "
            f"expected {expected_pair_count} (loader length {len(loader)}) for sample "
            f"{sample_id!r} -- incomplete loader coverage."
        )

    return {
        'sample_id': sample_id,
        'expected_pair_count': expected_pair_count,
        'loader_length': len(loader),
        'processed_pair_count': processed_pair_count,
        'total_candidate_edges': total_candidate_edges,
        'total_accepted_edges': total_accepted_edges,
    }


def run_submission_inference(
    *,
    test_dir: str | Path,
    test_zarrs: list[str | Path],
    unet3d: nn.Module,
    transformer: nn.Module,
    device: torch.device,
    hyperparams: dict[str, Any],
) -> tuple[dict[str, IndexedRXGraph], dict[str, Any]]:
    """
    Production submission inference entry point (Part A). Both
    generate_submission.py and kaggle_kernel_inference/inference_kernel.py
    must call this and only this for graph construction.

    Returns (pred_graphs, diagnostics):
      - pred_graphs: dict[sample_id, IndexedRXGraph], one canonical graph per
        required test sample.
      - diagnostics: dict with required_dataset_ids, per-sample coverage/
        node/edge/timing detail, and run-wide totals (Part A7).

    Raises RuntimeError on any structural failure (Part A1/A8): empty
    test_zarrs, duplicate/empty sample IDs, a sample with zero frame pairs,
    a required sample producing zero nodes, or a zero total predicted-edge
    count across the whole run. Never fabricates nodes/edges and never
    silently omits a required sample.
    """
    test_dir = Path(test_dir)
    hyperparams = validate_inference_hyperparams(hyperparams)

    sorted_zarrs = sorted(Path(z) for z in test_zarrs)
    if not sorted_zarrs:
        raise RuntimeError(
            "run_submission_inference: test_zarrs is empty -- no test samples to run "
            "inference on."
        )

    required_dataset_ids: list[str] = []
    seen_ids: set[str] = set()
    for zarr_path in sorted_zarrs:
        sample_id = zarr_path.stem
        if not sample_id:
            raise RuntimeError(f"run_submission_inference: empty sample ID derived from {zarr_path}")
        if sample_id in seen_ids:
            raise RuntimeError(
                f"run_submission_inference: duplicate sample ID {sample_id!r} derived "
                f"from test_zarrs {sorted_zarrs}"
            )
        seen_ids.add(sample_id)
        required_dataset_ids.append(sample_id)

    unet3d.eval()
    transformer.eval()

    # Exactly ONE PredictionGraphAssembler for the complete submission
    # inference run (Part A3).
    assembler = PredictionGraphAssembler()

    per_sample_diagnostics: dict[str, Any] = {}
    cuda_active = (device.type == 'cuda')
    if cuda_active:
        torch.cuda.synchronize(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            logger.warning("Could not reset CUDA peak-memory stats before submission inference.")
    total_start = time.monotonic()

    for zarr_path, sample_id in zip(sorted_zarrs, required_dataset_ids, strict=True):
        dataset = build_test_dataset(test_dir, sample_id)
        if len(dataset) == 0:
            raise RuntimeError(
                f"run_submission_inference: sample {sample_id!r} (from {zarr_path}) "
                f"produced ZERO frame pairs -- refusing to silently omit it from the "
                f"submission."
            )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # GPU ops are asynchronous -- without synchronizing immediately
        # before starting and stopping this sample's timer, elapsed_seconds
        # would not reliably represent completed GPU execution even though
        # the run-wide total_elapsed_seconds (synchronized above/below) does.
        if cuda_active:
            torch.cuda.synchronize(device)
        sample_start = time.monotonic()
        sample_diag = run_sample_loader_inference(
            sample_id=sample_id,
            loader=loader,
            expected_pair_count=len(dataset),
            assembler=assembler,
            unet3d=unet3d,
            transformer=transformer,
            device=device,
            hyperparams=hyperparams,
        )
        if cuda_active:
            torch.cuda.synchronize(device)
        sample_diag['elapsed_seconds'] = time.monotonic() - sample_start
        per_sample_diagnostics[sample_id] = sample_diag

    if cuda_active:
        torch.cuda.synchronize(device)
    total_elapsed_seconds = time.monotonic() - total_start

    diagnostics = assembler.diagnostics()
    pred_graphs = assembler.pred_graphs()

    for sid in required_dataset_ids:
        per_sample_diagnostics[sid]['unique_node_count'] = len(pred_graphs[sid].node_ids())
        per_sample_diagnostics[sid]['unique_edge_count'] = len(pred_graphs[sid].edge_list())

    # Part A8: structural circuit breakers.
    actual_keys = set(pred_graphs.keys())
    required_keys = set(required_dataset_ids)
    missing_keys = sorted(required_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - required_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"run_submission_inference: graph-key set does not match required dataset "
            f"IDs. Missing: {missing_keys}. Unexpected: {unexpected_keys}."
        )

    zero_node_samples = [sid for sid in required_dataset_ids if len(pred_graphs[sid].node_ids()) == 0]
    if zero_node_samples:
        node_counts = {sid: len(pred_graphs[sid].node_ids()) for sid in required_dataset_ids}
        expected_counts = {sid: per_sample_diagnostics[sid]['expected_pair_count'] for sid in required_dataset_ids}
        processed_counts = {sid: per_sample_diagnostics[sid]['processed_pair_count'] for sid in required_dataset_ids}
        raise RuntimeError(
            f"run_submission_inference: required sample(s) {zero_node_samples} produced "
            f"ZERO nodes. node counts: {node_counts}, expected pair counts: "
            f"{expected_counts}, processed pair counts: {processed_counts}."
        )

    total_predicted_nodes = diagnostics['predicted_nodes_total']
    total_predicted_edges = diagnostics['predicted_edges_total']
    if total_predicted_nodes == 0:
        raise RuntimeError("run_submission_inference: total predicted node count across all samples is ZERO.")
    if total_predicted_edges == 0:
        # A sample may legitimately have zero edges as long as another
        # required sample has valid edges -- only the RUN-WIDE total is
        # gated here, never a per-sample edge count.
        raise RuntimeError("run_submission_inference: total predicted edge count across all samples is ZERO.")

    total_accepted_edges = sum(d['total_accepted_edges'] for d in per_sample_diagnostics.values())
    total_candidate_edges = sum(d['total_candidate_edges'] for d in per_sample_diagnostics.values())

    result_diagnostics: dict[str, Any] = {
        'required_dataset_ids': required_dataset_ids,
        'per_sample': per_sample_diagnostics,
        'total_unique_nodes': total_predicted_nodes,
        'total_unique_edges': total_predicted_edges,
        'total_candidate_edges': total_candidate_edges,
        'total_accepted_edges': total_accepted_edges,
        'total_elapsed_seconds': total_elapsed_seconds,
    }
    if cuda_active:
        try:
            result_diagnostics['cuda_max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        except Exception:
            logger.warning("Could not read CUDA max_memory_allocated after submission inference.")

    logger.info(
        f"run_submission_inference complete: {len(required_dataset_ids)} sample(s), "
        f"{total_predicted_nodes} unique nodes, {total_predicted_edges} unique edges, "
        f"{total_accepted_edges}/{total_candidate_edges} edges accepted, "
        f"{total_elapsed_seconds:.2f}s total."
    )

    return pred_graphs, result_diagnostics
