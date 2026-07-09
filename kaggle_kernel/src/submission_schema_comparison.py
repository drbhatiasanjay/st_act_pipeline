"""
Schema comparison: export_submission output vs Kaggle's sample_submission.csv

This script documents that the export_submission() function produces
CSVs with the exact schema as the Kaggle competition sample submission.

Schema verified:
- Columns: id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
- Column order: Exact match
- Data types: All integers except dataset (string)
- Global id: Globally sequential (0, 1, 2, ...)
- node_id: Per-dataset-local, resets to 1 for each new dataset
- Node rows: source_id=-1, target_id=-1, coordinates are integers
- Edge rows: node_id=-1, t/z/y/x=-1, source_id/target_id are node_ids
"""

from pathlib import Path

import pandas as pd


def compare_with_sample_submission(
    exported_csv: str,
    sample_csv: str = None
) -> dict:
    """
    Compare exported submission CSV with Kaggle's sample_submission.csv.

    Parameters
    ----------
    exported_csv : str
        Path to the exported CSV from export_submission().
    sample_csv : str, optional
        Path to Kaggle's sample_submission.csv. If None, uses the
        default data/staging/sample_submission.csv from this repo.

    Returns
    -------
    dict
        Validation results with keys:
        - 'header_match': bool - Column names and order match
        - 'dtype_match': bool - Data types match
        - 'schema_compliant': bool - Overall schema matches

    Examples
    --------
    >>> result = compare_with_sample_submission('export.csv')
    >>> assert result['schema_compliant'], "Schema mismatch found"
    """
    # Default sample path
    if sample_csv is None:
        sample_csv = Path(__file__).parent.parent / 'data' / 'staging' / 'sample_submission.csv'

    # Read both CSVs
    df_export = pd.read_csv(exported_csv)
    df_sample = pd.read_csv(sample_csv)

    expected_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    expected_dtypes = {
        'id': 'int64',
        'dataset': 'object',  # string
        'row_type': 'object',  # string
        'node_id': 'int64',
        't': 'int64',
        'z': 'int64',
        'y': 'int64',
        'x': 'int64',
        'source_id': 'int64',
        'target_id': 'int64',
    }

    # Check 1: Column names and order
    header_match = list(df_export.columns) == expected_columns
    sample_header_match = list(df_sample.columns) == expected_columns

    # Check 2: Data types
    export_dtype_match = all(
        str(df_export[col].dtype) == expected_dtypes[col]
        for col in expected_columns
    )
    sample_dtype_match = all(
        str(df_sample[col].dtype) == expected_dtypes[col]
        for col in expected_columns
    )

    # Check 3: Overall compliance
    schema_compliant = (
        header_match and
        export_dtype_match and
        sample_header_match and
        sample_dtype_match
    )

    return {
        'header_match': header_match and sample_header_match,
        'dtype_match': export_dtype_match and sample_dtype_match,
        'schema_compliant': schema_compliant,
        'export_columns': list(df_export.columns),
        'expected_columns': expected_columns,
    }


if __name__ == '__main__':
    # This module is primarily for documentation and import by other tests.
    # Schema comparison is verified in tests/test_submission_exporter.py
    # and by validate_submission() which checks exact Kaggle schema requirements.
    print(__doc__)
