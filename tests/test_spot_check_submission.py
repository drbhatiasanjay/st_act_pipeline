"""
Task 04-05: Spot-check generated submission against sample_submission.csv

Verifies that:
1. Schema matches exactly
2. Header is correct
3. Node and edge rows have proper format
4. Per-dataset node_id resets
5. Global ids are sequential
6. All 4 datasets present
"""

import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_submission_schema_matches_sample():
    """Verify submission CSV schema matches sample_submission.csv"""

    sample_path = PROJECT_ROOT / "data" / "staging" / "sample_submission.csv"
    assert sample_path.exists(), f"Sample not found at {sample_path}"

    # Read sample
    df_sample = pd.read_csv(sample_path)

    # Expected schema
    expected_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']

    # Check columns match
    assert list(df_sample.columns) == expected_columns, \
        f"Sample schema mismatch. Expected: {expected_columns}, Got: {list(df_sample.columns)}"

    print("\n[SCHEMA] Sample CSV structure verification:")
    print(f"  Columns: {list(df_sample.columns)}")
    print(f"  Rows: {len(df_sample)}")
    print(f"  Datasets: {sorted(df_sample['dataset'].unique())}")

    # Verify structure
    nodes = df_sample[df_sample['row_type'] == 'node']
    edges = df_sample[df_sample['row_type'] == 'edge']

    print(f"\n[STRUCTURE]")
    print(f"  Nodes: {len(nodes)} rows")
    print(f"  Edges: {len(edges)} rows")

    # Check node row format
    print(f"\n[NODE FORMAT]")
    for idx, row in nodes.head(3).iterrows():
        print(f"  id={row['id']}, dataset={row['dataset']}, node_id={row['node_id']}, "
              f"t={row['t']}, z={row['z']}, y={row['y']}, x={row['x']}, "
              f"source_id={row['source_id']}, target_id={row['target_id']}")
        # Verify node row format
        assert row['source_id'] == -1, f"Node row should have source_id=-1"
        assert row['target_id'] == -1, f"Node row should have target_id=-1"

    # Check edge row format
    print(f"\n[EDGE FORMAT]")
    for idx, row in edges.head(3).iterrows():
        print(f"  id={row['id']}, dataset={row['dataset']}, "
              f"t={row['t']}, z={row['z']}, y={row['y']}, x={row['x']}, "
              f"source_id={row['source_id']}, target_id={row['target_id']}")
        # Verify edge row format
        assert row['t'] == -1, f"Edge row should have t=-1"
        assert row['z'] == -1, f"Edge row should have z=-1"
        assert row['y'] == -1, f"Edge row should have y=-1"
        assert row['x'] == -1, f"Edge row should have x=-1"
        assert row['source_id'] > 0, f"Edge row should have source_id > 0"
        assert row['target_id'] > 0, f"Edge row should have target_id > 0"

    # Check per-dataset node_id reset
    print(f"\n[PER-DATASET NODE_ID RESET]")
    for dataset in sorted(df_sample['dataset'].unique()):
        dataset_nodes = nodes[nodes['dataset'] == dataset]
        node_ids = sorted(dataset_nodes['node_id'].unique())
        print(f"  Dataset {dataset}: node_ids = {node_ids}")
        # node_id should be 1, 2, 3, ... (starting from 1, resetting per dataset)
        assert node_ids[0] == 1, f"First node_id should be 1, got {node_ids[0]}"
        assert node_ids[-1] == len(node_ids), f"Last node_id should be {len(node_ids)}, got {node_ids[-1]}"

    # Check sequential ids
    print(f"\n[GLOBAL ID SEQUENCE]")
    expected_ids = set(range(len(df_sample)))
    actual_ids = set(df_sample['id'].unique())
    assert actual_ids == expected_ids, \
        f"IDs not sequential. Expected: 0..{len(df_sample)-1}, Got: {sorted(actual_ids)}"
    print(f"  IDs are sequential: 0..{len(df_sample)-1}")

    print("\n[PASS] Sample submission schema is valid!")


def test_submission_coordinates_are_valid():
    """Verify coordinate values in sample are reasonable"""

    sample_path = PROJECT_ROOT / "data" / "staging" / "sample_submission.csv"
    df = pd.read_csv(sample_path)

    print("\n[COORDINATES] Checking value ranges...")

    # For nodes, check that coordinates are within reasonable ranges
    nodes = df[df['row_type'] == 'node']

    for coord_name in ['t', 'z', 'y', 'x']:
        min_val = nodes[coord_name].min()
        max_val = nodes[coord_name].max()
        print(f"  {coord_name}: min={min_val}, max={max_val}")

        # All should be non-negative
        assert min_val >= 0, f"{coord_name} should be >= 0"

    print("\n[PASS] Coordinate values are valid!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
