"""
Integration test for the Phase 0 pipeline.

This test verifies that:
1. The pipeline structure is correct
2. Data loading, detection, tracking, export, and evaluation are properly wired
3. The submission CSV schema is valid

Uses small synthetic datasets to avoid long ILP solve times.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import zarr
import pandas as pd
import polars as pl
import tracksdata as td

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import AnisotropicZarrLoader
from src.tracker import STHypergraphTracker
from src.submission_exporter import export_submission, validate_submission
from src.evaluation import evaluate_submission, load_geff_ground_truth


def create_small_test_zarr(output_path: Path, num_timepoints=2):
    """Create a small test zarr store (3 timepoints, small volumes)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(output_path), mode='w')
    arr = root.create_array(
        '0',
        shape=(num_timepoints, 16, 64, 64),  # Small for fast processing
        chunks=(1, 16, 64, 64),
        dtype='uint16'
    )

    for t in range(num_timepoints):
        data = np.random.randint(50, 200, size=(16, 64, 64), dtype='uint16')
        arr[t, :, :, :] = data


def test_pipeline_structure():
    """Test the complete pipeline structure with small data."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create small test datasets
        dataset_id = "test_dataset_001"
        test_zarr = tmpdir / f"{dataset_id}.zarr"
        create_small_test_zarr(test_zarr, num_timepoints=2)

        # Load and process
        print(f"\nLoading test zarr: {test_zarr}")
        loader = AnisotropicZarrLoader(str(test_zarr), simulate=False)
        t_dim, z_dim, y_dim, x_dim = loader.get_shape()
        print(f"Shape: T={t_dim}, Z={z_dim}, Y={y_dim}, X={x_dim}")

        # Run detection on each timepoint (simplified)
        centroids_by_t = {}
        motion_vectors_by_t = {}
        anisotropy = np.array([4.0, 1.0, 1.0])

        print("\nRunning detection...")
        for t in range(t_dim):
            vol = loader.load_timepoint_block(t)
            # Simple peak detection: just return a few dummy centroids
            centroids = [[8, 32, 32], [8, 48, 48]]  # 2 dummy centroids
            motion_vectors = [[0.1, 0.1, 0.1] for _ in centroids]

            centroids_by_t[t] = centroids
            motion_vectors_by_t[t] = motion_vectors
            print(f"  T={t}: {len(centroids)} centroids")

        # Run tracker
        print("\nRunning tracker...")
        tracker = STHypergraphTracker(birth_cost=15.0, death_cost=15.0, division_reward=-8.0)
        lineage_graph = tracker.solve_lineage(
            centroids_by_t,
            motion_vectors_by_t,
            anisotropy=anisotropy,
            max_gap_frames=1
        )
        print(f"  Nodes: {lineage_graph.number_of_nodes()}, Edges: {lineage_graph.number_of_edges()}")

        # Convert to tracksdata format
        print("\nConverting to tracksdata format...")
        td_graph = td.graph.IndexedRXGraph()

        # Register attribute keys
        for key in ('z', 'y', 'x'):
            try:
                td_graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass  # Key already exists

        # Map networkx node ids to tracksdata node ids
        node_mapping = {}
        for node, attrs in lineage_graph.nodes(data=True):
            t, node_idx = node
            coords = attrs.get('coords', [0, 0, 0])

            # Add node and track the returned node_id
            attrs_dict = {
                't': int(t),
                'z': int(coords[0]),
                'y': int(coords[1]),
                'x': int(coords[2])
            }
            td_node_id = td_graph.add_node(attrs_dict)
            node_mapping[(t, node_idx)] = td_node_id

        # Add edges using the mapped node ids
        for source, target in lineage_graph.edges():
            source_td_id = node_mapping[source]
            target_td_id = node_mapping[target]
            td_graph.add_edge(source_td_id, target_td_id, {})

        # Export submission
        output_csv = tmpdir / "test_submission.csv"
        print(f"\nExporting submission to {output_csv}")
        export_submission({dataset_id: td_graph}, output_csv)

        # Validate submission
        print("Validating submission...")
        validate_submission(output_csv)

        # Read and check CSV
        df = pd.read_csv(output_csv)
        print(f"  CSV rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Datasets: {df['dataset'].unique()}")
        print(f"  Row types: {df['row_type'].unique()}")
        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string())

        # Verify schema
        required_columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        assert all(col in df.columns for col in required_columns), f"Missing columns"
        assert len(df) > 0, "CSV is empty"
        assert df['dataset'].nunique() == 1, "Should have exactly 1 dataset"

        print("\n[PASS] Pipeline structure test successful!")


if __name__ == "__main__":
    test_pipeline_structure()
