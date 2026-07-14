"""
Unit tests for submission exporter.

Tests export_submission() and validate_submission() functions
against synthetic tracksdata graphs.
"""

import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import tracksdata as td

from src.submission_exporter import export_submission, validate_submission


class TestExportSubmission:
    """Tests for export_submission() function."""

    @pytest.fixture
    def temp_csv(self):
        """Create a temporary directory for CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_synthetic_graph(self, nodes_data, edges_data=None):
        """
        Helper to create a synthetic tracksdata graph.

        Parameters
        ----------
        nodes_data : list[dict]
            List of node specs. Each dict should have keys:
            - name: node identifier (for reference in edges, not stored)
            - t, z, y, x: coordinates

        edges_data : list[tuple] or None
            List of (source_name, target_name) tuples where names match nodes_data.

        Returns
        -------
        tracksdata.graph.IndexedRXGraph
            A synthetic graph with the specified structure.
        """
        # Create empty graph with node attribute schema
        graph = td.graph.IndexedRXGraph()

        # Register node attributes if there are nodes
        if nodes_data:
            # Register z, y, x (t is already present by default in IndexedRXGraph)
            for key in ('z', 'y', 'x'):
                try:
                    graph.add_node_attr_key(key, pl.Int64, 0)
                except ValueError:
                    pass  # Key already exists

            # Add nodes and track mapping from names to internal node IDs
            name_to_id = {}
            for node_spec in nodes_data:
                node_name = node_spec['name']
                attrs = {k: v for k, v in node_spec.items() if k != 'name'}
                # tracksdata add_node takes attrs dict and returns the internal node ID
                node_id = graph.add_node(attrs)
                name_to_id[node_name] = node_id

        # Add edges
        if edges_data:
            for source_name, target_name in edges_data:
                source_id = name_to_id[source_name]
                target_id = name_to_id[target_name]
                # add_edge takes source, target, and attrs (empty dict for no edge attributes)
                graph.add_edge(source_id, target_id, {})

        return graph

    def test_export_rounds_fractional_coordinates_not_truncates(self, temp_csv):
        """REGRESSION-relevant: export_submission() must round(), not bare
        int(), when reading node attrs. int() truncates toward zero, so a
        coordinate landing infinitesimally below its true integer value
        (e.g. 29.999999999998, the exact class of float-precision noise a
        line-fit smoothing step can introduce -- see
        run_pipeline.py:convert_nx_to_tracksdata, fixed in ace1a60) would
        silently export as z=29 instead of z=30.

        Uses a Float64-typed node attr schema (not the Int64 schema
        create_synthetic_graph() uses elsewhere in this file) -- this is
        deliberate: an IndexedRXGraph with an Int64-typed attr key hard-
        crashes on read if a genuine float was ever inserted (verified
        directly), so the exact silent-truncation bug this test targets is
        only reachable via a Float64-typed schema, which a future caller
        populating this graph directly from raw (float) peak coordinates
        could plausibly choose without realizing the schema choice matters.
        This test intentionally does NOT go through
        run_pipeline.py:convert_nx_to_tracksdata -- the point is verifying
        export_submission() has its own defense, independent of what any
        upstream caller does."""
        graph = td.graph.IndexedRXGraph()
        for key in ('z', 'y', 'x'):
            try:
                graph.add_node_attr_key(key, pl.Float64, 0.0)
            except ValueError:
                pass

        graph.add_node({'t': 0, 'z': 29.999999999998, 'y': 5.0, 'x': 5.0})

        csv_path = temp_csv / 'test_fractional_coords.csv'
        export_submission({'dataset_A': graph}, csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]['z'] == 30, (
            f"z must round to 30, not truncate to 29 -- got {df.iloc[0]['z']}"
        )

    def test_export_single_node_no_edges(self, temp_csv):
        """Test export of a single node with no edges."""
        # Create synthetic graph with 1 node at (t=0, z=5, y=10, x=15)
        nodes = [{'name': 'n1', 't': 0, 'z': 5, 'y': 10, 'x': 15}]
        graph = self.create_synthetic_graph(nodes)

        graphs_dict = {'dataset_A': graph}
        csv_path = temp_csv / 'test_single_node.csv'

        result_path = export_submission(graphs_dict, csv_path)

        # Verify file was written
        assert Path(result_path).exists()

        # Read back and verify content
        df = pd.read_csv(result_path)
        assert len(df) == 1
        assert df.iloc[0]['id'] == 0
        assert df.iloc[0]['dataset'] == 'dataset_A'
        assert df.iloc[0]['row_type'] == 'node'
        assert df.iloc[0]['node_id'] == 1
        assert df.iloc[0]['t'] == 0
        assert df.iloc[0]['z'] == 5
        assert df.iloc[0]['y'] == 10
        assert df.iloc[0]['x'] == 15
        assert df.iloc[0]['source_id'] == -1
        assert df.iloc[0]['target_id'] == -1

    def test_export_nodes_and_edges(self, temp_csv):
        """Test export with 3 nodes and 2 edges."""
        nodes = [
            {'name': 'n1', 't': 0, 'z': 5, 'y': 10, 'x': 15},
            {'name': 'n2', 't': 1, 'z': 5, 'y': 11, 'x': 16},
            {'name': 'n3', 't': 2, 'z': 5, 'y': 12, 'x': 17},
        ]
        edges = [('n1', 'n2'), ('n2', 'n3')]
        graph = self.create_synthetic_graph(nodes, edges)

        graphs_dict = {'dataset_A': graph}
        csv_path = temp_csv / 'test_nodes_edges.csv'

        result_path = export_submission(graphs_dict, csv_path)
        df = pd.read_csv(result_path)

        # Should have 5 rows: 3 nodes + 2 edges
        assert len(df) == 5

        # Verify ids are sequential (0..4)
        assert df['id'].tolist() == [0, 1, 2, 3, 4]

        # Verify node rows
        node_rows = df[df['row_type'] == 'node']
        assert len(node_rows) == 3
        assert node_rows['node_id'].tolist() == [1, 2, 3]
        assert node_rows['source_id'].tolist() == [-1, -1, -1]
        assert node_rows['target_id'].tolist() == [-1, -1, -1]

        # Verify edge rows
        edge_rows = df[df['row_type'] == 'edge']
        assert len(edge_rows) == 2
        assert edge_rows['node_id'].tolist() == [-1, -1]
        assert edge_rows['t'].tolist() == [-1, -1]
        assert edge_rows['z'].tolist() == [-1, -1]
        assert edge_rows['y'].tolist() == [-1, -1]
        assert edge_rows['x'].tolist() == [-1, -1]
        # First edge: n1->n2 (node_id 1->2)
        assert edge_rows.iloc[0]['source_id'] == 1
        assert edge_rows.iloc[0]['target_id'] == 2
        # Second edge: n2->n3 (node_id 2->3)
        assert edge_rows.iloc[1]['source_id'] == 2
        assert edge_rows.iloc[1]['target_id'] == 3

    def test_export_multiple_datasets(self, temp_csv):
        """Test export with multiple datasets (node_id reset per dataset)."""
        # Dataset A: 2 nodes, 1 edge
        nodes_a = [
            {'name': 'a1', 't': 0, 'z': 5, 'y': 10, 'x': 15},
            {'name': 'a2', 't': 1, 'z': 5, 'y': 11, 'x': 16},
        ]
        edges_a = [('a1', 'a2')]
        graph_a = self.create_synthetic_graph(nodes_a, edges_a)

        # Dataset B: 2 nodes, 1 edge
        nodes_b = [
            {'name': 'b1', 't': 0, 'z': 5, 'y': 20, 'x': 25},
            {'name': 'b2', 't': 1, 'z': 5, 'y': 21, 'x': 26},
        ]
        edges_b = [('b1', 'b2')]
        graph_b = self.create_synthetic_graph(nodes_b, edges_b)

        graphs_dict = {'dataset_A': graph_a, 'dataset_B': graph_b}
        csv_path = temp_csv / 'test_multi_datasets.csv'

        result_path = export_submission(graphs_dict, csv_path)
        df = pd.read_csv(result_path)

        # Should have 6 rows: 2 nodes + 1 edge per dataset
        assert len(df) == 6

        # Verify global id is continuous
        assert df['id'].tolist() == [0, 1, 2, 3, 4, 5]

        # Verify dataset A rows
        df_a = df[df['dataset'] == 'dataset_A']
        assert len(df_a) == 3
        node_ids_a = df_a[df_a['row_type'] == 'node']['node_id'].tolist()
        assert node_ids_a == [1, 2]

        # Verify dataset B rows (node_id should reset)
        df_b = df[df['dataset'] == 'dataset_B']
        assert len(df_b) == 3
        node_ids_b = df_b[df_b['row_type'] == 'node']['node_id'].tolist()
        assert node_ids_b == [1, 2]  # Reset, not [3, 4]

    def test_export_all_datasets_zero_detections_produces_valid_empty_csv(self, temp_csv):
        """REGRESSION GUARD: a submission where every dataset has zero nodes/edges must
        still export a schema-correct, header-only CSV, not crash.

        Real bug, hit live on Kaggle (inference_kernel.py v4): pd.DataFrame(rows) on a
        fully-empty rows list produces a DataFrame with zero columns, so the subsequent
        df[column_order] raised KeyError since those column names didn't exist yet. This
        happens for real whenever the checkpoint under test produces no detections on any
        real test sample (confirmed with the known severely-undertrained sanity-check
        checkpoint) -- not just a hypothetical input.
        """
        empty_graph_a = self.create_synthetic_graph([])
        empty_graph_b = self.create_synthetic_graph([])
        graphs_dict = {'dataset_A': empty_graph_a, 'dataset_B': empty_graph_b}
        csv_path = temp_csv / 'test_all_empty.csv'

        result_path = export_submission(
            graphs_dict, csv_path, required_dataset_ids=['dataset_A', 'dataset_B']
        )

        df = pd.read_csv(result_path)
        assert len(df) == 0
        assert list(df.columns) == [
            'id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id'
        ]
        # validate_submission() must accept a genuinely empty-but-schema-correct submission
        assert validate_submission(result_path) is True

    def test_coordinates_are_integers(self, temp_csv):
        """Test that exported coordinates are integers (no floats)."""
        # Create synthetic graph with coordinates that might be floats
        nodes = [{'name': 'n1', 't': 0, 'z': 5, 'y': 10, 'x': 15}]
        graph = self.create_synthetic_graph(nodes)

        graphs_dict = {'dataset_A': graph}
        csv_path = temp_csv / 'test_coords_int.csv'

        export_submission(graphs_dict, csv_path)

        # Read CSV as strings to verify no decimal points
        with open(csv_path) as f:
            lines = f.readlines()
            # Header line
            assert lines[0].strip() == 'id,dataset,row_type,node_id,t,z,y,x,source_id,target_id'
            # First data line (the node)
            data_line = lines[1]
            # No decimal points in coordinates
            assert '.0' not in data_line


class TestValidateSubmission:
    """Tests for validate_submission() function."""

    @pytest.fixture
    def temp_csv(self):
        """Create a temporary directory for CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_test_csv(self, csv_path, data):
        """Helper to create a test CSV file."""
        df = pd.DataFrame(data)
        column_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        df = df[column_order]
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_validate_valid_submission(self, temp_csv):
        """Test validation of a valid submission."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 5, 'y': 11, 'x': 16, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]
        csv_path = self.create_test_csv(temp_csv / 'valid.csv', data)

        # Should not raise
        result = validate_submission(csv_path)
        assert result is True

    def test_validate_header_mismatch(self, temp_csv):
        """Test validation fails on header mismatch."""
        csv_path = temp_csv / 'bad_header.csv'
        df = pd.DataFrame({
            'id': [0],
            'dataset': ['ds_a'],
            'row_type': ['node'],
            'node_id': [1],
            't': [0],
            'z': [5],
            'y': [10],
            'x': [15],
            'source_id': [-1],
            'bad_column': [-1],  # Wrong column name
        })
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="CSV header mismatch"):
            validate_submission(csv_path)

    def test_validate_non_sequential_ids(self, temp_csv):
        """Test validation fails on non-sequential global ids."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 5, 'y': 11, 'x': 16, 'source_id': -1, 'target_id': -1},  # id=2 instead of 1
        ]
        csv_path = self.create_test_csv(temp_csv / 'non_seq_ids.csv', data)

        with pytest.raises(ValueError, match="id column not sequential"):
            validate_submission(csv_path)

    def test_validate_invalid_row_type(self, temp_csv):
        """Test validation fails on invalid row_type."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'invalid', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': -1, 'target_id': -1},
        ]
        csv_path = self.create_test_csv(temp_csv / 'bad_row_type.csv', data)

        with pytest.raises(ValueError, match="Invalid row_type values"):
            validate_submission(csv_path)

    def test_validate_node_row_source_not_minus_one(self, temp_csv):
        """Test validation fails when node row has source_id != -1."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': 0, 'target_id': -1},  # Bad: source_id should be -1
        ]
        csv_path = self.create_test_csv(temp_csv / 'bad_node_source.csv', data)

        with pytest.raises(ValueError, match="For 'node' rows, source_id must be -1"):
            validate_submission(csv_path)

    def test_validate_node_row_target_not_minus_one(self, temp_csv):
        """Test validation fails when node row has target_id != -1."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': -1, 'target_id': 1},  # Bad: target_id should be -1
        ]
        csv_path = self.create_test_csv(temp_csv / 'bad_node_target.csv', data)

        with pytest.raises(ValueError, match="For 'node' rows, target_id must be -1"):
            validate_submission(csv_path)

    def test_validate_edge_row_node_id_not_minus_one(self, temp_csv):
        """Test validation fails when edge row has node_id != -1."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': 1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},  # Bad: node_id should be -1
        ]
        csv_path = self.create_test_csv(temp_csv / 'bad_edge_node_id.csv', data)

        with pytest.raises(ValueError, match="For 'edge' rows, node_id must be -1"):
            validate_submission(csv_path)

    def test_validate_edge_row_coordinates_not_minus_one(self, temp_csv):
        """Test validation fails when edge row has coordinate != -1."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': 0, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},  # Bad: t should be -1
        ]
        csv_path = self.create_test_csv(temp_csv / 'bad_edge_coords.csv', data)

        with pytest.raises(ValueError, match="For 'edge' rows, column 't' must be -1"):
            validate_submission(csv_path)

    def test_validate_edge_row_source_negative(self, temp_csv):
        """Test validation fails when edge row has negative source_id."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': -1, 'target_id': 2},  # Bad: source_id should be positive
        ]
        csv_path = self.create_test_csv(temp_csv / 'bad_edge_source.csv', data)

        with pytest.raises(ValueError, match="For 'edge' rows, source_id must be positive integer"):
            validate_submission(csv_path)

    def test_validate_node_id_per_dataset_reset(self, temp_csv):
        """Test validation of node_id reset per dataset."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 5, 'y': 11, 'x': 16, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_b', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 20, 'x': 25, 'source_id': -1, 'target_id': -1},  # Reset to 1
            {'id': 3, 'dataset': 'ds_b', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 5, 'y': 21, 'x': 26, 'source_id': -1, 'target_id': -1},
        ]
        csv_path = self.create_test_csv(temp_csv / 'node_id_reset.csv', data)

        # Should not raise
        result = validate_submission(csv_path)
        assert result is True

    def test_validate_no_duplicate_node_ids_per_dataset(self, temp_csv):
        """Test validation fails on duplicate (dataset, node_id) pairs."""
        data = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 5, 'y': 10, 'x': 15, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 1, 'z': 5, 'y': 11, 'x': 16, 'source_id': -1, 'target_id': -1},  # Duplicate node_id=1 in ds_a
        ]
        csv_path = self.create_test_csv(temp_csv / 'dup_node_ids.csv', data)

        # The validation catches this as non-sequential (expecting [1, 2] but got [1] unique values)
        # This is because duplicates violate the sequential check
        with pytest.raises(ValueError, match="node_id is not sequential|Found duplicate"):
            validate_submission(csv_path)

    def test_validate_file_not_found(self, temp_csv):
        """Test validation fails when file doesn't exist."""
        csv_path = temp_csv / 'nonexistent.csv'

        with pytest.raises(FileNotFoundError):
            validate_submission(csv_path)


class TestIntegrationExportAndValidate:
    """Integration tests: export synthetic graph, then validate output."""

    def create_synthetic_graph(self, nodes_data, edges_data=None):
        """Helper to create a synthetic tracksdata graph."""
        graph = td.graph.IndexedRXGraph()

        if nodes_data:
            # 't' is already present by default, only add z, y, x
            for key in ('z', 'y', 'x'):
                try:
                    graph.add_node_attr_key(key, pl.Int64, 0)
                except ValueError:
                    pass  # Key already exists

            # Add nodes and track mapping from names to internal node IDs
            name_to_id = {}
            for node_spec in nodes_data:
                node_name = node_spec['name']
                attrs = {k: v for k, v in node_spec.items() if k != 'name'}
                node_id = graph.add_node(attrs)
                name_to_id[node_name] = node_id

        if edges_data:
            for source_name, target_name in edges_data:
                source_id = name_to_id[source_name]
                target_id = name_to_id[target_name]
                graph.add_edge(source_id, target_id, {})

        return graph

    def test_export_and_validate_cycle(self):
        """Test full cycle: export synthetic graph, then validate the CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create synthetic graph
            nodes = [
                {'name': 'n1', 't': 0, 'z': 5, 'y': 10, 'x': 15},
                {'name': 'n2', 't': 1, 'z': 5, 'y': 11, 'x': 16},
                {'name': 'n3', 't': 2, 'z': 5, 'y': 12, 'x': 17},
            ]
            edges = [('n1', 'n2'), ('n2', 'n3')]
            graph = self.create_synthetic_graph(nodes, edges)

            graphs_dict = {'dataset_A': graph}
            csv_path = tmpdir / 'test_export_validate.csv'

            # Export
            export_submission(graphs_dict, csv_path)

            # Validate
            result = validate_submission(csv_path)
            assert result is True

    def test_export_multiple_datasets_and_validate(self):
        """Test export of multiple datasets and validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Dataset A: 2 nodes, 1 edge
            nodes_a = [
                {'name': 'a1', 't': 0, 'z': 5, 'y': 10, 'x': 15},
                {'name': 'a2', 't': 1, 'z': 5, 'y': 11, 'x': 16},
            ]
            edges_a = [('a1', 'a2')]
            graph_a = self.create_synthetic_graph(nodes_a, edges_a)

            # Dataset B: 3 nodes, 2 edges
            nodes_b = [
                {'name': 'b1', 't': 0, 'z': 5, 'y': 20, 'x': 25},
                {'name': 'b2', 't': 1, 'z': 5, 'y': 21, 'x': 26},
                {'name': 'b3', 't': 2, 'z': 5, 'y': 22, 'x': 27},
            ]
            edges_b = [('b1', 'b2'), ('b2', 'b3')]
            graph_b = self.create_synthetic_graph(nodes_b, edges_b)

            graphs_dict = {'dataset_A': graph_a, 'dataset_B': graph_b}
            csv_path = tmpdir / 'test_multi_datasets_validate.csv'

            # Export
            export_submission(graphs_dict, csv_path)

            # Validate
            result = validate_submission(csv_path)
            assert result is True

            # Verify specifics
            df = pd.read_csv(csv_path)
            # Dataset A: 2 nodes + 1 edge = 3 rows
            # Dataset B: 3 nodes + 2 edges = 5 rows
            # Total: 8 rows
            assert len(df) == 8
            assert len(df[df['dataset'] == 'dataset_A']) == 3
            assert len(df[df['dataset'] == 'dataset_B']) == 5
