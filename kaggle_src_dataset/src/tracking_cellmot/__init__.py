"""
tracking_cellmot: Vendored scoring and data loading from royerlab's kaggle-cell-tracking-competition reference implementation.

Key exports:
- evaluate(): Score a single (pred, gt) graph pair
- evaluate_datasets(): Micro-averaged scoring over multiple graph pairs
- evaluate_divisions(): Evaluate division-matching accuracy
- open_dataset(): Load a Zarr dataset with optional GPU resampling and normalization
"""

from .division_metrics import evaluate_divisions
from .io import DEFAULT_SCALE, open_dataset, save_graph, list_datasets
from .metrics import (
    ADJUSTMENT_ALPHA,
    SCORE_DIVISION_WEIGHT,
    DatasetsResult,
    EvaluationResult,
    evaluate,
    evaluate_datasets,
    per_sample_metrics,
    summarise,
)

__all__ = [
    # Constants
    "ADJUSTMENT_ALPHA",
    "SCORE_DIVISION_WEIGHT",
    "DEFAULT_SCALE",
    # Result types
    "EvaluationResult",
    "DatasetsResult",
    # Evaluation
    "evaluate",
    "evaluate_datasets",
    "evaluate_divisions",
    "per_sample_metrics",
    "summarise",
    # Data loading
    "open_dataset",
    "save_graph",
    "list_datasets",
]
