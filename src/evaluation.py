"""
Local evaluation harness for the Kaggle cell tracking competition.

Provides a clean Python API for computing the exact same score as Kaggle's official
scorer using the vendored tracksdata metrics (edge Jaccard, division Jaccard) and
the adjustment formula for predicted node count.

Main entry point:
    evaluate_submission(pred_graphs, gt_graphs, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)
        -> dict with edge_jaccard, adjusted_edge_jaccard, division_jaccard, score, etc.

The module handles:
- Loading .geff files into tracksdata graphs
- Micro-averaged metric computation across multiple datasets
- Adjustment formula for predicted-vs-ground-truth node count mismatch
- Graceful handling of zero-division cases (no GT divisions)
"""


import tracksdata as td
from tracksdata.graph import IndexedRXGraph

from src.tracking_cellmot import (
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)

# Real physical voxel scale (z, y, x) in micrometers — z=1.625um, y=x=0.40625um.
# NOT the anisotropy RATIO (4.0,1.0,1.0) -- that ratio describes Z:Y:X relative
# coarseness but the 7.0um gating threshold needs real physical distances, and
# using the ratio instead of real microns inflates every distance by ~2.46x
# (1/0.40625), silently corrupting node matching. Confirmed against io.py's own
# DEFAULT_SCALE (vendored from the host's reference implementation, Plan 00).
DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
DEFAULT_MAX_DISTANCE: float = 7.0


def load_geff_ground_truth(geff_path: str) -> tuple[td.graph.BaseGraph, 'td.geff.GeffMetadata']:
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
    except FileNotFoundError as e:
        raise FileNotFoundError(f"GEFF file not found: {geff_path}") from e
    except Exception as e:
        raise ValueError(f"Failed to load GEFF file {geff_path}: {e}") from e

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
    pred_graphs: list[td.graph.BaseGraph] | dict[str, td.graph.BaseGraph],
    gt_graphs: list[td.graph.BaseGraph] | dict[str, td.graph.BaseGraph],
    scale: tuple[float, ...] = DEFAULT_SCALE,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    gt_metadata: list['td.geff.GeffMetadata'] | dict[str, 'td.geff.GeffMetadata'] = None,
) -> dict[str, float | int]:
    """
    Evaluate predicted tracking graphs against ground-truth graphs.

    Computes the exact same score as Kaggle's official scorer using the vendored
    reference implementation:
    - Per-sample evaluation via tracksdata.evaluate()
    - Per-sample metrics via per_sample_metrics()
    - Aggregation via summarise() for correct weighted averaging of adjusted Jaccard
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
        Default: (1.625, 0.40625, 0.40625) micrometers -- this competition's real physical
        voxel scale, NOT the (4.0,1.0,1.0) anisotropy ratio.
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
        - 'adjusted_edge_jaccard': float — edge Jaccard with node-count penalty (weight-averaged per sample)
        - 'division_jaccard': float — micro-averaged division Jaccard (NaN if no divisions)
        - 'score': float — final submission score
        - 'num_pred_nodes_total': int — total predicted nodes across all datasets
        - 'num_gt_nodes_total': int — total ground-truth nodes
        - 'num_datasets': int — number of graph pairs evaluated

    Raises
    ------
    ValueError
        If pred_graphs and gt_graphs are misaligned, missing keys, or if either is empty.
    RuntimeError
        If evaluation of a sample fails (with sample ID for diagnosis).
    """
    # P0-4 (hardened): reject mixed container types explicitly, before any
    # dict/list-specific logic runs. Letting one branch treat the other
    # type's container as its own (e.g. calling .keys() on a list, or
    # indexing a dict with an int) produces a confusing AttributeError or
    # integer-key KeyError deep inside the function instead of a clear,
    # user-facing alignment failure.
    pred_is_dict = isinstance(pred_graphs, dict)
    gt_is_dict = isinstance(gt_graphs, dict)
    if pred_is_dict != gt_is_dict:
        raise ValueError(
            "pred_graphs and gt_graphs must both be dicts or both be lists, "
            f"not mixed types. Got pred_graphs: {type(pred_graphs).__name__}, "
            f"gt_graphs: {type(gt_graphs).__name__}."
        )
    if not pred_is_dict and isinstance(gt_metadata, dict):
        raise ValueError(
            "gt_metadata was provided as a dict, but pred_graphs/gt_graphs "
            "are lists -- metadata must be a list aligned by position when "
            "pred_graphs/gt_graphs are lists, or all three (pred_graphs, "
            "gt_graphs, gt_metadata) must be dicts keyed by the same sample IDs."
        )

    # Normalize to lists and extract dataset IDs for alignment checking
    if isinstance(pred_graphs, dict):
        pred_keys = set(pred_graphs.keys())
        gt_keys = set(gt_graphs.keys())

        # Require identical key sets in BOTH directions: a GT-only sample must
        # not be silently dropped just because iteration is pred-driven, and
        # a pred-only sample must not be silently evaluated against nothing.
        missing_gt = sorted(pred_keys - gt_keys)
        missing_pred = sorted(gt_keys - pred_keys)
        if missing_gt or missing_pred:
            raise ValueError(
                f"pred_graphs and gt_graphs key sets do not match. "
                f"Missing ground-truth graphs for dataset IDs: {missing_gt}. "
                f"Missing prediction graphs for dataset IDs: {missing_pred}. "
                f"Prediction IDs: {sorted(pred_keys)}. GT IDs: {sorted(gt_keys)}"
            )

        dataset_ids = list(pred_graphs.keys())
        pred_list = [pred_graphs[did] for did in dataset_ids]
        gt_list = [gt_graphs[did] for did in dataset_ids]

        if gt_metadata is not None and isinstance(gt_metadata, dict):
            metadata_keys = set(gt_metadata.keys())
            missing_meta = sorted(pred_keys - metadata_keys)
            unexpected_meta = sorted(metadata_keys - pred_keys)
            if missing_meta or unexpected_meta:
                raise ValueError(
                    f"Missing metadata for dataset IDs: {missing_meta}. "
                    f"Unexpected metadata for dataset IDs not in pred_graphs/gt_graphs: "
                    f"{unexpected_meta}. Provided metadata IDs: {sorted(metadata_keys)}"
                )
            metadata_list = [gt_metadata[did] for did in dataset_ids]
        else:
            metadata_list = None
    else:
        dataset_ids = None
        pred_list = pred_graphs
        gt_list = gt_graphs
        metadata_list = gt_metadata

    # Validation: list length and alignment
    if len(pred_list) == 0 or len(gt_list) == 0:
        raise ValueError("pred_graphs and gt_graphs must not be empty")
    if len(pred_list) != len(gt_list):
        raise ValueError(
            f"pred_graphs and gt_graphs must have the same length. "
            f"Got {len(pred_list)} pred and {len(gt_list)} gt."
        )
    if metadata_list is not None and len(metadata_list) != len(pred_list):
        raise ValueError(
            f"gt_metadata must have the same length as pred_graphs/gt_graphs. "
            f"Got {len(metadata_list)} metadata, {len(pred_list)} graphs."
        )

    # Evaluate each (pred, gt) pair and collect per-sample metrics
    per_sample_rows = []
    num_gt_nodes_total = 0

    for idx, (pred, gt) in enumerate(zip(pred_list, gt_list, strict=False)):
        sample_id = dataset_ids[idx] if dataset_ids else f"sample_{idx}"

        try:
            # Evaluate the pair (this mutates pred in-place with matching attributes)
            er = evaluate(pred, gt, scale=scale, max_distance=max_distance)
        except Exception as e:
            raise RuntimeError(
                f"Evaluation failed for sample '{sample_id}' (index {idx}): {e}"
            ) from e

        # Extract T_true from metadata if available; otherwise use GT node count
        if metadata_list is not None and metadata_list[idx] is not None:
            m = metadata_list[idx]
            if hasattr(m, 'extra') and m.extra and 'estimated_number_of_nodes' in m.extra:
                n_total = m.extra['estimated_number_of_nodes']
            else:
                n_total = gt.num_nodes()
        else:
            n_total = gt.num_nodes()

        # Compute node recall for this sample
        try:
            nr = node_recall(pred, gt)
        except Exception:
            # If node_recall fails, use NaN; this is non-fatal for the overall evaluation
            nr = float("nan")

        # Compute per-sample metrics
        row = per_sample_metrics(er, n_total, nr)
        per_sample_rows.append(row)

        num_gt_nodes_total += gt.num_nodes()

    # Aggregate per-sample metrics using the reference implementation's summarise()
    summary = summarise(per_sample_rows)

    # Extract results from summary
    # Note: summarise() returns 'adj_edge_jaccard'; map to 'adjusted_edge_jaccard' for public API
    edge_jaccard = summary['edge_jaccard']
    adjusted_edge_jaccard = summary['adj_edge_jaccard']
    division_jaccard = summary['division_jaccard']
    score = summary['score']

    # Compute total predicted nodes across all samples. per_sample_rows holds
    # the dicts returned by per_sample_metrics() (NOT EvaluationResult objects
    # -- a prior version of this line did `er.num_pred_nodes` against these
    # dicts, an AttributeError that fired on every single call reaching this
    # point; never caught because no test exercised evaluate_submission() to
    # completion with data present).
    num_pred_nodes_total = sum(row['num_pred_nodes'] for row in per_sample_rows)

    return {
        'edge_jaccard': edge_jaccard,
        'adjusted_edge_jaccard': adjusted_edge_jaccard,
        'division_jaccard': division_jaccard,
        'score': score,
        'num_pred_nodes_total': num_pred_nodes_total,
        'num_gt_nodes_total': num_gt_nodes_total,
        'num_datasets': len(pred_list),
    }
