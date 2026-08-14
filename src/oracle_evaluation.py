"""
Oracle score decomposition: isolate detector vs linker bottlenecks.

Implements four scoring modes to measure whether the bottleneck preventing this
pipeline from reaching the competition's classical baseline (0.763) is the detector
or the linker/tracker:

1. GT nodes + GT edges (control - should score ~1.0)
2. GT nodes + model-predicted edges (linker ceiling)
3. Model-predicted nodes + oracle edges (detector ceiling)
4. Model-predicted nodes + model-predicted edges (actual end-to-end)

All modes use the same evaluation path (evaluate_submission()) to ensure fair
comparison. Oracle edges in mode 3 are induced by uniquely matching each
predicted node to its nearest GT node within the competition's real match
tolerance (7.0µm).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.graph import IndexedRXGraph

from src.evaluation import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_SCALE,
    evaluate_submission,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OracleModeResult:
    """Results for a single oracle mode on a single sample."""
    mode: str  # 'gt_nodes_gt_edges', 'gt_nodes_model_edges', 'model_nodes_oracle_edges', 'model_nodes_model_edges'
    sample_id: str
    edge_jaccard: float
    adjusted_edge_jaccard: float
    division_jaccard: float | None  # None if no GT divisions
    score: float
    predicted_nodes_total: int
    gt_nodes_total: int


@dataclass(frozen=True)
class OracleDecomposition:
    """Aggregated oracle score decomposition results."""
    gt_nodes_gt_edges: dict[str, float]
    gt_nodes_model_edges: dict[str, float]
    model_nodes_oracle_edges: dict[str, float]
    model_nodes_model_edges: dict[str, float]
    detection_precision_7um: float
    detection_recall_7um: float
    localization_p50_um: float
    localization_p95_um: float
    checkpoint_sha: str | None
    split_identity: str
    inference_config: dict[str, Any]


def _compute_physical_distance(
    coord1: tuple[float, float, float],
    coord2: tuple[float, float, float],
    scale: tuple[float, float, float] = DEFAULT_SCALE,
) -> float:
    """Compute physical distance in micrometers between two coordinates (z, y, x)."""
    diffs = [(c1 - c2) * s for c1, c2, s in zip(coord1, coord2, scale, strict=False)]
    return float(np.sqrt(sum(d**2 for d in diffs)))


def _find_nearest_gt_node(
    pred_coord: tuple[float, float, float],
    gt_nodes: dict[int, dict[str, float]],  # node_id -> {t, z, y, x}
    max_distance_um: float = DEFAULT_MAX_DISTANCE,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
) -> tuple[int | None, float]:
    """
    Find nearest GT node to a predicted node within max_distance_um.

    Returns (gt_node_id, distance_um) or (None, np.inf) if no GT node within tolerance.
    """
    best_gt_id = None
    best_distance = np.inf

    for gt_id, gt_props in gt_nodes.items():
        gt_coord = (gt_props['z'], gt_props['y'], gt_props['x'])
        dist = _compute_physical_distance(pred_coord, gt_coord, scale)
        if dist < max_distance_um and dist < best_distance:
            best_gt_id = gt_id
            best_distance = dist

    return best_gt_id, best_distance


def _build_oracle_graph(
    gt_graph: td.graph.BaseGraph,
    pred_nodes_by_frame: dict[int, list[tuple[float, float, float]]],  # t -> [(z, y, x), ...]
) -> tuple[IndexedRXGraph, dict[int, dict[int, int]]]:
    """
    Build oracle graph by matching predicted nodes to nearest GT nodes and
    using GT edge structure over matched predicted nodes.

    Returns (oracle_graph, pred_to_gt_matching) where pred_to_gt_matching is
    a dict mapping t_idx -> {pred_node_idx -> gt_node_id}.
    """
    oracle_graph = IndexedRXGraph()
    for key in ('t', 'x', 'y', 'z'):
        try:
            oracle_graph.add_node_attr_key(key, pl.Int64, 0)
        except ValueError:
            pass

    pred_to_gt_matching: dict[int, dict[int, int]] = {}
    localization_errors_um: list[float] = []

    # Get all GT nodes indexed by t, keyed by REAL graph node ID (not row
    # position). node_attrs() returns no ID column -- node_ids() and
    # node_attrs() share row order, so real IDs must come from zipping them.
    # Using enumerate() row-position indices here previously meant
    # pred_to_gt_matching stored fabricated small integers that could never
    # equal the real (large, tracksdata-assigned) node IDs returned by
    # gt_graph.edge_list() below -- silently zeroing every oracle edge match.
    gt_nodes_by_t: dict[int, list[dict[str, float]]] = {}
    gt_real_ids = gt_graph.node_ids()
    gt_all_attrs = gt_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
    for real_id, row in zip(gt_real_ids, gt_all_attrs.to_dicts(), strict=True):
        t = int(row['t'])
        if t not in gt_nodes_by_t:
            gt_nodes_by_t[t] = []
        gt_nodes_by_t[t].append({
            'id': real_id,
            't': t,
            'z': float(row['z']),
            'y': float(row['y']),
            'x': float(row['x']),
        })

    # Match each predicted node to nearest GT node
    pred_node_id_counter = 0
    pred_node_mapping: dict[int, tuple[int, int]] = {}  # pred_node_id -> (t, gt_node_id)

    for t_idx in sorted(pred_nodes_by_frame.keys()):
        pred_to_gt_matching[t_idx] = {}
        pred_nodes = pred_nodes_by_frame[t_idx]

        gt_nodes_list = gt_nodes_by_t.get(t_idx, [])

        for pred_idx, pred_coord in enumerate(pred_nodes):
            if gt_nodes_list:
                gt_nodes_dict = {node['id']: node for node in gt_nodes_list}
                gt_id, distance = _find_nearest_gt_node(pred_coord, gt_nodes_dict)
                if gt_id is not None:
                    localization_errors_um.append(distance)
                    pred_to_gt_matching[t_idx][pred_idx] = gt_id
                    pred_node_mapping[pred_node_id_counter] = (t_idx, gt_id)

            oracle_graph.add_node({
                't': t_idx,
                'x': int(round(pred_coord[2])),
                'y': int(round(pred_coord[1])),
                'z': int(round(pred_coord[0])),
            })
            pred_node_id_counter += 1

    # Add GT edges, but only if both endpoints have matched predicted nodes
    # Build a mapping from oracle graph node ID to predicted node details
    oracle_node_to_pred: dict[int, tuple[int, int]] = {}  # oracle_node_id -> (t_idx, pred_idx)
    pred_counter = 0
    for t_idx in sorted(pred_nodes_by_frame.keys()):
        for pred_idx in range(len(pred_nodes_by_frame[t_idx])):
            oracle_node_to_pred[pred_counter] = (t_idx, pred_idx)
            pred_counter += 1

    # Also build reverse mapping: (t_idx, pred_idx) -> oracle_node_id
    pred_to_oracle_id: dict[tuple[int, int], int] = {v: k for k, v in oracle_node_to_pred.items()}

    # Iterate through GT edges (simple list, not indexed by node ID)
    gt_edge_list = gt_graph.edge_list()
    for gt_src, gt_tgt in gt_edge_list:
        # Search for matched predicted nodes for this edge
        for oracle_id_src, (t_src, _gt_matched_src) in oracle_node_to_pred.items():
            # Check if this predicted node was matched to the GT source
            if t_src in pred_to_gt_matching and any(
                gt_id == gt_src for gt_id in pred_to_gt_matching[t_src].values()
            ):
                # Found a predicted node matched to gt_src
                # Now find the corresponding gt_tgt node in the next timeframe
                if t_src + 1 in pred_to_gt_matching:
                    for pred_idx_tgt, gt_matched_tgt in pred_to_gt_matching[t_src + 1].items():
                        if gt_matched_tgt == gt_tgt:
                            oracle_id_tgt = pred_to_oracle_id.get((t_src + 1, pred_idx_tgt))
                            if oracle_id_tgt is not None:
                                try:
                                    oracle_graph.add_edge(oracle_id_src, oracle_id_tgt, {})
                                except (ValueError, RuntimeError, TypeError):
                                    pass  # Edge might already exist
                            break

    return oracle_graph, localization_errors_um


def compute_detection_metrics(
    pred_nodes_by_frame: dict[int, list[tuple[float, float, float]]],
    gt_graph: td.graph.BaseGraph,
) -> tuple[float, float, list[float]]:
    """
    Compute detection precision/recall at 7µm and localization errors.

    Precision: fraction of predicted nodes with a GT node within 7µm.
    Recall: fraction of GT nodes with a predicted node within 7µm.
    """
    gt_nodes_by_t: dict[int, list[dict[str, float]]] = {}
    gt_all_attrs = gt_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
    for row in gt_all_attrs.to_dicts():
        t = int(row['t'])
        if t not in gt_nodes_by_t:
            gt_nodes_by_t[t] = []
        gt_nodes_by_t[t].append({
            't': t,
            'z': float(row['z']),
            'y': float(row['y']),
            'x': float(row['x']),
        })

    localization_errors_um: list[float] = []
    num_pred_matched = 0
    total_pred = 0

    for t_idx in sorted(pred_nodes_by_frame.keys()):
        pred_nodes = pred_nodes_by_frame[t_idx]
        gt_nodes_list = gt_nodes_by_t.get(t_idx, [])

        for pred_coord in pred_nodes:
            total_pred += 1
            if gt_nodes_list:
                # Convert list to dict for _find_nearest_gt_node
                gt_nodes_dict = {i: node for i, node in enumerate(gt_nodes_list)}
                gt_id, distance = _find_nearest_gt_node(pred_coord, gt_nodes_dict)
                if gt_id is not None:
                    num_pred_matched += 1
                    localization_errors_um.append(distance)

    num_gt_total = sum(len(nodes) for nodes in gt_nodes_by_t.values())
    gt_matched = 0

    for t_idx in sorted(pred_nodes_by_frame.keys()):
        pred_nodes = pred_nodes_by_frame[t_idx]
        gt_nodes_list = gt_nodes_by_t.get(t_idx, [])

        matched_at_t = set()
        for pred_coord in pred_nodes:
            if gt_nodes_list:
                # Convert list to dict for _find_nearest_gt_node
                gt_nodes_dict = {i: node for i, node in enumerate(gt_nodes_list)}
                gt_id, distance = _find_nearest_gt_node(pred_coord, gt_nodes_dict)
                if gt_id is not None:
                    matched_at_t.add(gt_id)
        gt_matched += len(matched_at_t)

    precision = float(num_pred_matched / total_pred) if total_pred > 0 else 0.0
    recall = float(gt_matched / num_gt_total) if num_gt_total > 0 else 0.0

    return precision, recall, localization_errors_um


def evaluate_oracle_modes(
    sample_id: str,
    pred_graph: td.graph.BaseGraph,
    pred_nodes_by_frame: dict[int, list[tuple[float, float, float]]],
    gt_graph: td.graph.BaseGraph,
    gt_metadata: 'td.geff.GeffMetadata',
) -> dict[str, dict[str, float]]:
    """
    Evaluate all four oracle modes for a single sample.

    Returns a dict mapping mode name -> {edge_jaccard, adjusted_edge_jaccard, division_jaccard, score}.
    """
    results = {}

    # Mode 1: GT nodes + GT edges (control)
    gt_copy = gt_graph.copy()
    result_1 = evaluate_submission(
        [gt_copy],
        [gt_graph],
        gt_metadata=[gt_metadata],
    )
    results['gt_nodes_gt_edges'] = {
        'edge_jaccard': result_1['edge_jaccard'],
        'adjusted_edge_jaccard': result_1['adjusted_edge_jaccard'],
        'division_jaccard': result_1['division_jaccard'],
        'score': result_1['score'],
        'predicted_nodes_total': result_1['num_pred_nodes_total'],
        'gt_nodes_total': result_1['num_gt_nodes_total'],
    }

    # Mode 2: GT nodes + model-predicted edges (linker ceiling)
    # Start with GT nodes, replace edges with the model's predicted edges --
    # translated into GT node-ID space via nearest-coordinate matching, NOT
    # by reusing pred_graph's own edge (src_id, tgt_id) pairs directly. Those
    # IDs live in pred_graph's own independent ID space (assigned by whatever
    # built pred_graph, e.g. PredictionGraphAssembler's internal numbering)
    # and are only coincidentally valid GT node IDs when pred_graph happens
    # to be a literal copy of gt_graph (as in this module's own test
    # fixtures) -- against a real, independently-assembled prediction graph
    # this raised KeyError inside tracksdata's add_edge(), the same node-ID
    # confusion class as the mode 3 bug fixed in _build_oracle_graph above.
    gt_nodes_graph = gt_graph.copy()

    # Clear all GT edges
    for edge in list(gt_nodes_graph.edge_list()):
        gt_nodes_graph.remove_edge(*edge)

    # Build a real-ID -> (z, y, x) lookup for the predicted graph's own nodes.
    pred_real_ids = pred_graph.node_ids()
    pred_attrs = pred_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
    pred_coords_by_id = {
        real_id: (float(row['z']), float(row['y']), float(row['x']))
        for real_id, row in zip(pred_real_ids, pred_attrs.to_dicts(), strict=True)
    }
    pred_t_by_id = {
        real_id: int(row['t'])
        for real_id, row in zip(pred_real_ids, pred_attrs.to_dicts(), strict=True)
    }

    # GT nodes grouped by t, keyed by real GT node ID (for nearest-match).
    gt_real_ids = gt_graph.node_ids()
    gt_all_attrs = gt_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
    gt_nodes_by_t: dict[int, dict[int, dict[str, float]]] = {}
    for real_id, row in zip(gt_real_ids, gt_all_attrs.to_dicts(), strict=True):
        t = int(row['t'])
        gt_nodes_by_t.setdefault(t, {})[real_id] = {
            'z': float(row['z']), 'y': float(row['y']), 'x': float(row['x']),
        }

    # Translate each predicted edge into GT node-ID space via nearest match;
    # only add it if BOTH endpoints have a real GT match within tolerance.
    for pred_edge in pred_graph.edge_list():
        src_pred_id, tgt_pred_id = pred_edge
        src_coord = pred_coords_by_id.get(src_pred_id)
        tgt_coord = pred_coords_by_id.get(tgt_pred_id)
        if src_coord is None or tgt_coord is None:
            continue
        src_t = pred_t_by_id[src_pred_id]
        tgt_t = pred_t_by_id[tgt_pred_id]
        src_gt_id, _ = _find_nearest_gt_node(src_coord, gt_nodes_by_t.get(src_t, {}))
        tgt_gt_id, _ = _find_nearest_gt_node(tgt_coord, gt_nodes_by_t.get(tgt_t, {}))
        if src_gt_id is None or tgt_gt_id is None:
            continue
        try:
            gt_nodes_graph.add_edge(src_gt_id, tgt_gt_id, {})
        except (ValueError, RuntimeError, TypeError):
            pass  # Edge might already exist or be invalid

    result_2 = evaluate_submission(
        [gt_nodes_graph],
        [gt_graph],
        gt_metadata=[gt_metadata],
    )
    results['gt_nodes_model_edges'] = {
        'edge_jaccard': result_2['edge_jaccard'],
        'adjusted_edge_jaccard': result_2['adjusted_edge_jaccard'],
        'division_jaccard': result_2['division_jaccard'],
        'score': result_2['score'],
        'predicted_nodes_total': result_2['num_pred_nodes_total'],
        'gt_nodes_total': result_2['num_gt_nodes_total'],
    }

    # Mode 3: Model-predicted nodes + oracle edges
    oracle_graph, localization_errors = _build_oracle_graph(gt_graph, pred_nodes_by_frame)
    result_3 = evaluate_submission(
        [oracle_graph],
        [gt_graph],
        gt_metadata=[gt_metadata],
    )
    results['model_nodes_oracle_edges'] = {
        'edge_jaccard': result_3['edge_jaccard'],
        'adjusted_edge_jaccard': result_3['adjusted_edge_jaccard'],
        'division_jaccard': result_3['division_jaccard'],
        'score': result_3['score'],
        'predicted_nodes_total': result_3['num_pred_nodes_total'],
        'gt_nodes_total': result_3['num_gt_nodes_total'],
    }

    # Mode 4: Model-predicted nodes + model-predicted edges (actual system)
    result_4 = evaluate_submission(
        [pred_graph],
        [gt_graph],
        gt_metadata=[gt_metadata],
    )
    results['model_nodes_model_edges'] = {
        'edge_jaccard': result_4['edge_jaccard'],
        'adjusted_edge_jaccard': result_4['adjusted_edge_jaccard'],
        'division_jaccard': result_4['division_jaccard'],
        'score': result_4['score'],
        'predicted_nodes_total': result_4['num_pred_nodes_total'],
        'gt_nodes_total': result_4['num_gt_nodes_total'],
    }

    return results
