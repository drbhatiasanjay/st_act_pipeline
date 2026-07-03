"""
Baseline tests for the vendored scoring code (tracking_cellmot).

Verifies that:
1. evaluate_datasets can be imported without errors
2. Empty-dataset edge case returns correct NaN-valued DatasetsResult
3. The scoring logic matches the reference implementation behavior
"""

import math
import pytest
from src.tracking_cellmot.metrics import evaluate_datasets, DatasetsResult


def test_evaluate_datasets_empty_list():
    """
    Empty-dataset edge case: evaluate_datasets([]) should return a DatasetsResult
    where edge_jaccard, division_jaccard, and score are all NaN.

    Per the vendored _jaccard() logic:
    - _jaccard(tp, fp, fn) returns float("nan") when tp+fp+fn == 0
    - evaluate_datasets([]) produces edge_tp=edge_fp=edge_fn=0, so edge_jaccard is NaN
    - Division counts are also 0, so has_divisions=False (0+0+0 not > 0)
    - When has_divisions=False, division_jaccard is NaN and score = edge_jaccard (also NaN)
    - The correct assertion is math.isnan(result.score), NOT score == 0.0

    This test ensures the vendored code is importable, callable, and handles
    the baseline case correctly.
    """
    result = evaluate_datasets([])

    # Verify return type
    assert isinstance(result, DatasetsResult), (
        f"Expected DatasetsResult, got {type(result)}"
    )

    # All three metrics must be NaN for empty input
    assert math.isnan(result.edge_jaccard), (
        f"Expected edge_jaccard to be NaN for empty list, got {result.edge_jaccard}"
    )
    assert math.isnan(result.division_jaccard), (
        f"Expected division_jaccard to be NaN for empty list, got {result.division_jaccard}"
    )
    assert math.isnan(result.score), (
        f"Expected score to be NaN for empty list, got {result.score}"
    )


if __name__ == "__main__":
    # Allow running as: python tests/test_scoring_baseline.py
    test_evaluate_datasets_empty_list()
    print("✓ test_evaluate_datasets_empty_list passed")
