"""
Local evaluation harness for the Kaggle cell tracking competition.

Provides a clean Python API for computing the exact same score as Kaggle's official
scorer using the vendored tracksdata metrics (edge Jaccard, division Jaccard) and
the adjustment formula for predicted node count.

Main entry point:
    evaluate_submission(pred_graphs, gt_graphs, scale=(4.0, 1.0, 1.0), max_distance=7.0)
        -> dict with edge_jaccard, adjusted_edge_jaccard, division_jaccard, score, etc.

The module handles:
- Loading .geff files into tracksdata graphs
- Micro-averaged metric computation across multiple datasets
- Adjustment formula for predicted-vs-ground-truth node count mismatch
- Graceful handling of zero-division cases (no GT divisions)
"""

from typing import Dict, List, Tuple, Union
import tracksdata as td
from tracksdata.graph import IndexedRXGraph

from src.tracking_cellmot import (
    evaluate_datasets,
    evaluate_divisions,
    ADJUSTMENT_ALPHA,
    SCORE_DIVISION_WEIGHT,
)


# Default physical voxel scale (z, y, x) — 4.0 µm in Z, 0.40625 µm in Y/X
DEFAULT_SCALE: Tuple[float, float, float] = (4.0, 1.0, 1.0)
DEFAULT_MAX_DISTANCE: float = 7.0


def load_geff_ground_truth(geff_path: str) -> Tuple[td.graph.BaseGraph, 'td.geff.GeffMetadata']:
    """
    Load a .geff ground-truth file into a tracksdata graph.

    Parameters
    ----------
    geff_path : str
        Path to the .geff file (directory or file).

    Returns
    -------
    tuple[tracksdata.graph.BaseGraph, tracksdata.geff.GeffMetadata]
        Returns a tuple (graph, metadata). The metadata object contains
        estimated_number_of_nodes (T_true) used for the adjustment formula.

    Raises
    ------
    FileNotFoundError
        If the .geff file does not exist.
    ValueError
        If the .geff file is corrupted or empty.
    """
    try:
        graph, metadata = IndexedRXGraph.from_geff(geff_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"GEFF file not found: {geff_path}")
    except Exception as e:
        raise ValueError(f"Failed to load GEFF file {geff_path}: {e}")

    # Validation
    if graph is None or graph.num_nodes() == 0:
        raise ValueError(f"Loaded GEFF graph from {geff_path} is empty (0 nodes)")

    return graph, metadata


def load_gt_for_dataset(dataset_id: str, geff_dir: str) -> td.graph.BaseGraph:
    """
    Load ground-truth .geff file for a specific dataset by ID.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g., "44b6_0113de3b").
    geff_dir : str
        Directory containing .geff files (e.g., "data/staging/train/").

    Returns
    -------
    tracksdata.graph.BaseGraph
        The loaded ground-truth graph.

    Raises
    ------
    FileNotFoundError
        If the .geff file for this dataset_id doesn't exist.
    """
    import os
    geff_path = os.path.join(geff_dir, f"{dataset_id}.geff")
    if not os.path.exists(geff_path):
        raise FileNotFoundError(
            f"GEFF file not found for dataset '{dataset_id}' in {geff_dir}\n"
            f"Expected: {geff_path}"
        )
    graph, _ = load_geff_ground_truth(geff_path)
    return graph


def evaluate_submission(
    pred_graphs: Union[List[td.graph.BaseGraph], Dict[str, td.graph.BaseGraph]],
    gt_graphs: Union[List[td.graph.BaseGraph], Dict[str, td.graph.BaseGraph]],
    scale: Tuple[float, ...] = DEFAULT_SCALE,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    gt_metadata: Union[List['td.geff.GeffMetadata'], Dict[str, 'td.geff.GeffMetadata']] = None,
) -> Dict[str, Union[float, int]]:
    """
    Evaluate predicted tracking graphs against ground-truth graphs.

    Computes the exact same score as Kaggle's official scorer:
    - Edge Jaccard via tracksdata.evaluate_datasets()
    - Division Jaccard via tracksdata.evaluate_divisions()
    - Adjusted edge Jaccard using the node-count adjustment formula
    - Combined score = adjusted_edge_jaccard + 0.1 * division_jaccard
      (or just adjusted_edge_jaccard if no GT divisions exist)

    Parameters
    ----------
    pred_graphs : list[tracksdata.graph.BaseGraph] or dict[str, tracksdata.graph.BaseGraph]
        Predicted graphs. Either a list (must align with gt_graphs order)
        or a dict keyed by dataset ID.
    gt_graphs : list[tracksdata.graph.BaseGraph] or dict[str, tracksdata.graph.BaseGraph]
        Ground-truth graphs. Same structure as pred_graphs.
    scale : tuple[float, ...], optional
        Physical voxel scale (e.g., (z, y, x)) for anisotropy correction.
        Default: (4.0, 1.0, 1.0) for typical light-sheet microscopy.
    max_distance : float, optional
        Maximum centroid distance for node matching (µm). Default: 7.0.
    gt_metadata : list[GeffMetadata] or dict[str, GeffMetadata], optional
        GEFF metadata for each ground-truth graph (needed for T_true in adjustment formula).
        If provided as dict, keys must match pred_graphs/gt_graphs keys.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'edge_jaccard': float — micro-averaged edge Jaccard
        - 'adjusted_edge_jaccard': float — edge Jaccard with node-count penalty
        - 'division_jaccard': float — micro-averaged division Jaccard (NaN if no divisions)
        - 'score': float — final submission score
        - 'num_pred_nodes_total': int — total predicted nodes across all datasets
        - 'num_gt_nodes_total': int — total ground-truth nodes
        - 'num_datasets': int — number of graph pairs evaluated

    Raises
    ------
    ValueError
        If pred_graphs and gt_graphs are misaligned, or if either is empty.
    """
    # Normalize to lists for uniform processing
    if isinstance(pred_graphs, dict):
        dataset_ids = list(pred_graphs.keys())
        pred_list = [pred_graphs[did] for did in dataset_ids]
        gt_list = [gt_graphs[did] for did in dataset_ids]
        if gt_metadata is not None and isinstance(gt_metadata, dict):
            metadata_list = [gt_metadata[did] for did in dataset_ids]
        else:
            metadata_list = None
    else:
        pred_list = pred_graphs
        gt_list = gt_graphs
        metadata_list = gt_metadata

    # Validation
    if len(pred_list) == 0 or len(gt_list) == 0:
        raise ValueError("pred_graphs and gt_graphs must not be empty")
    if len(pred_list) != len(gt_list):
        raise ValueError(
            f"pred_graphs and gt_graphs must have the same length. "
            f"Got {len(pred_list)} pred and {len(gt_list)} gt."
        )

    # Compute micro-averaged Jaccard scores across all datasets
    graph_pairs = list(zip(pred_list, gt_list))
    datasets_result = evaluate_datasets(graph_pairs, scale=scale, max_distance=max_distance)

    # Compute adjusted edge Jaccard using node-count mismatch penalty
    # Formula: J_adj = max(0, J * (1 - ALPHA * (T_pred - T_true) / T_true))
    # Where T_true is the estimated GT node count and T_pred is the total predicted nodes
    num_pred_nodes_total = sum(g.num_nodes() for g in pred_list)
    num_gt_nodes_total = sum(g.num_nodes() for g in gt_list)

    # Extract T_true from metadata if available; otherwise use GT node count
    if metadata_list is not None:
        t_true_list = []
        for m, g in zip(metadata_list, gt_list):
            if hasattr(m, 'extra') and m.extra and 'estimated_number_of_nodes' in m.extra:
                t_true_list.append(m.extra['estimated_number_of_nodes'])
            else:
                t_true_list.append(g.num_nodes())
    else:
        t_true_list = [g.num_nodes() for g in gt_list]

    # Compute total T_true across all datasets
    t_true = sum(t_true_list)

    # Apply adjustment formula (only if we have GT nodes for normalization)
    if t_true > 0:
        node_ratio = (num_pred_nodes_total - t_true) / t_true
        adjusted_edge_jaccard = max(0.0, datasets_result.edge_jaccard * (1.0 - ADJUSTMENT_ALPHA * node_ratio))
    else:
        adjusted_edge_jaccard = datasets_result.edge_jaccard

    # Compute final score
    # If no GT divisions anywhere, drop the division term (division_jaccard will be NaN)
    import math
    if not math.isnan(datasets_result.division_jaccard):
        # We have divisions
        score = adjusted_edge_jaccard + SCORE_DIVISION_WEIGHT * datasets_result.division_jaccard
    else:
        # No divisions in any GT graph
        score = adjusted_edge_jaccard

    return {
        'edge_jaccard': datasets_result.edge_jaccard,
        'adjusted_edge_jaccard': adjusted_edge_jaccard,
        'division_jaccard': datasets_result.division_jaccard,
        'score': score,
        'num_pred_nodes_total': num_pred_nodes_total,
        'num_gt_nodes_total': num_gt_nodes_total,
        'num_datasets': len(pred_list),
    }
