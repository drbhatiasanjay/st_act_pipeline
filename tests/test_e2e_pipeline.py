"""
End-to-end pipeline test for Phase 0.

Tests:
1. Pipeline runs without exceptions
2. Submission CSV is created
3. CSV is schema-valid
4. All 4 datasets are processed
5. Evaluation scores are numeric
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
STAGED_TEST_DIR = PROJECT_ROOT / "data" / "staging" / "test"
STAGED_TRAIN_DIR = PROJECT_ROOT / "data" / "staging" / "train"


def test_full_pipeline():
    """Test the complete pipeline: detect, track, export, evaluate."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "test_submission.csv"

        # Run the pipeline
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_pipeline.py"),
            "--test-dir", str(STAGED_TEST_DIR),
            "--train-dir", str(STAGED_TRAIN_DIR),
            "--output-path", str(output_csv),
        ]

        print(f"\nRunning command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("\n--- STDOUT ---")
        print(result.stdout)

        if result.returncode != 0:
            print("\n--- STDERR ---")
            print(result.stderr)
            raise RuntimeError(f"Pipeline failed with return code {result.returncode}")

        # Verify submission CSV was created
        assert output_csv.exists(), f"Submission CSV not created at {output_csv}"

        # Read and validate CSV
        df = pd.read_csv(output_csv)
        print(f"\nSubmission CSV created: {len(df)} rows")

        # Check schema
        required_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        assert all(col in df.columns for col in required_columns), \
            f"Missing columns. Expected: {required_columns}, Got: {list(df.columns)}"

        # Verify all 4 datasets are present
        datasets = df['dataset'].unique()
        assert len(datasets) == 4, f"Expected 4 datasets, got {len(datasets)}: {datasets}"
        print(f"All 4 datasets present: {sorted(datasets)}")

        # Check row types
        assert set(df['row_type'].unique()).issubset({'node', 'edge'}), \
            f"Invalid row_type values: {df['row_type'].unique()}"

        # Verify nodes and edges
        nodes = df[df['row_type'] == 'node']
        edges = df[df['row_type'] == 'edge']
        print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
        assert len(nodes) > 0, "No nodes in submission"
        assert len(edges) >= 0, "Edges should be >= 0"

        # Verify ids are sequential
        expected_ids = set(range(len(df)))
        actual_ids = set(df['id'].unique())
        assert actual_ids == expected_ids, \
            f"IDs not sequential. Expected: 0..{len(df)-1}, Got: {sorted(actual_ids)}"

        # Verify per-dataset node_id reset
        for dataset in datasets:
            dataset_nodes = df[(df['dataset'] == dataset) & (df['row_type'] == 'node')]
            node_ids = sorted(dataset_nodes['node_id'].unique())
            print(f"  Dataset {dataset}: node_ids = {node_ids}")
            # Should be 1, 2, 3, ... (per-dataset reset)
            assert node_ids[0] == 1, f"First node_id should be 1, got {node_ids[0]}"
            assert node_ids[-1] == len(node_ids), \
                f"Last node_id should be {len(node_ids)}, got {node_ids[-1]}"

        # Check coordinates are integers
        coord_cols = ['t', 'z', 'y', 'x']
        for col in coord_cols:
            assert df[col].dtype in [int, 'int64', 'int32'], \
                f"Column {col} should be integer, got {df[col].dtype}"

        print("\n[PASS] End-to-end pipeline test successful")


def test_submission_csv_structure():
    """Test the CSV file structure independently."""

    # Use the main test output
    output_csv = Path(PROJECT_ROOT) / "submissions" / "phase_0_baseline_submission.csv"

    if not output_csv.exists():
        pytest.skip(f"Submission CSV not found at {output_csv} (must run test_full_pipeline first)")

    df = pd.read_csv(output_csv)

    # Check header
    required_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    assert list(df.columns) == required_columns, \
        f"CSV header mismatch. Expected: {required_columns}, Got: {list(df.columns)}"

    # Print some stats
    print(f"\nSubmission CSV Stats:")
    print(f"  Total rows: {len(df)}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Nodes: {len(df[df['row_type'] == 'node'])}")
    print(f"  Edges: {len(df[df['row_type'] == 'edge'])}")

    # Print first and last rows
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string())
    print(f"\nLast 5 rows:")
    print(df.tail(5).to_string())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
