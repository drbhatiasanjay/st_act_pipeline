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

    def test_export_all_datasets_zero_detections_produces_valid_empty_csv_generic_mode(self, temp_csv):
        """REGRESSION GUARD: in GENERIC mode (required_dataset_ids=None), a
        submission where every dataset has zero nodes/edges must still
        export a schema-correct, header-only CSV, not crash.

        Real bug, hit live on Kaggle (inference_kernel.py v4): pd.DataFrame(rows) on a
        fully-empty rows list produces a DataFrame with zero columns, so the subsequent
        df[column_order] raised KeyError since those column names didn't exist yet. This
        happens for real whenever the checkpoint under test produces no detections on any
        real test sample (confirmed with the known severely-undertrained sanity-check
        checkpoint) -- not just a hypothetical input.

        P0-6: generic (required_dataset_ids=None) mode is the ONLY mode that
        still accepts this -- see
        test_export_all_datasets_zero_detections_raises_in_required_mode for
        the new fail-closed required-mode behavior.
        """
        empty_graph_a = self.create_synthetic_graph([])
        empty_graph_b = self.create_synthetic_graph([])
        graphs_dict = {'dataset_A': empty_graph_a, 'dataset_B': empty_graph_b}
        csv_path = temp_csv / 'test_all_empty.csv'

        result_path = export_submission(graphs_dict, csv_path)

        df = pd.read_csv(result_path)
        assert len(df) == 0
        assert list(df.columns) == [
            'id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id'
        ]
        # validate_submission() must accept a genuinely empty-but-schema-correct submission
        assert validate_submission(result_path) is True

    def test_export_all_datasets_zero_detections_raises_in_required_mode(self, temp_csv):
        """P0-6 (Part E1.6): required_dataset_ids mode must REJECT (not
        silently accept, not just warn) any required dataset with zero
        nodes -- fabricating an empty-but-"valid" submission for a real
        graded run would hide a genuinely broken checkpoint/pipeline."""
        empty_graph_a = self.create_synthetic_graph([])
        empty_graph_b = self.create_synthetic_graph([])
        graphs_dict = {'dataset_A': empty_graph_a, 'dataset_B': empty_graph_b}
        csv_path = temp_csv / 'test_all_empty_required.csv'

        with pytest.raises(ValueError, match="ZERO nodes"):
            export_submission(
                graphs_dict, csv_path, required_dataset_ids=['dataset_A', 'dataset_B']
            )

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


def _make_graph(nodes_data, edges_data=None):
    """Module-level synthetic-graph helper (P0-6, Part F3) -- identical
    pattern to the class-scoped create_synthetic_graph() helpers above,
    factored out for reuse across the new required_dataset_ids test classes."""
    graph = td.graph.IndexedRXGraph()
    if nodes_data:
        for key in ('z', 'y', 'x'):
            try:
                graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass
        name_to_id = {}
        for node_spec in nodes_data:
            node_name = node_spec['name']
            attrs = {k: v for k, v in node_spec.items() if k != 'name'}
            node_id = graph.add_node(attrs)
            name_to_id[node_name] = node_id
    if edges_data:
        for source_name, target_name in edges_data:
            graph.add_edge(name_to_id[source_name], name_to_id[target_name], {})
    return graph


class TestExportSubmissionRequiredDatasetIdsValidation:
    """P0-6 (Part F3): export_submission()'s required_dataset_ids preflight
    validation -- empty list, duplicates, non-string/empty entries."""

    @pytest.fixture
    def temp_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _one_node_graph(self):
        return _make_graph([{'name': 'n1', 't': 0, 'z': 1, 'y': 1, 'x': 1}])

    def test_empty_required_ids_raises(self, temp_csv):
        graphs = {'ds_a': self._one_node_graph()}
        with pytest.raises(ValueError, match="must not be an empty list"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=[])

    def test_empty_string_required_id_raises(self, temp_csv):
        graphs = {'ds_a': self._one_node_graph()}
        with pytest.raises(ValueError, match="non-empty strings"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a', ''])

    def test_duplicate_required_id_raises(self, temp_csv):
        graphs = {'ds_a': self._one_node_graph()}
        with pytest.raises(ValueError, match="duplicate ID"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a', 'ds_a'])

    def test_missing_required_dataset_raises(self, temp_csv):
        """F3.1: a dataset required but absent from graphs_dict must raise."""
        graphs = {'ds_a': self._one_node_graph()}
        with pytest.raises(ValueError, match="Missing"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a', 'ds_b'])

    def test_unexpected_dataset_raises(self, temp_csv):
        """F3.2: a dataset present in graphs_dict but not required must raise."""
        graphs = {'ds_a': self._one_node_graph(), 'ds_b': self._one_node_graph()}
        with pytest.raises(ValueError, match="Unexpected"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a'])

    def test_zero_node_required_sample_raises(self, temp_csv):
        """F3.6: any required dataset with zero nodes raises, even when
        other required datasets have real detections."""
        graphs = {'ds_a': self._one_node_graph(), 'ds_b': _make_graph([])}
        with pytest.raises(ValueError, match="ZERO nodes"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a', 'ds_b'])

    def test_zero_total_edges_raises(self, temp_csv):
        """F3.8: every required dataset has >=1 node but zero edges anywhere
        -- must still fail in required mode (a real submission with isolated
        singleton detections and no real tracked edges is not acceptable)."""
        graphs = {'ds_a': self._one_node_graph(), 'ds_b': self._one_node_graph()}
        with pytest.raises(ValueError, match="edge row count"):
            export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a', 'ds_b'])

    def test_valid_multi_dataset_submission_passes(self, temp_csv):
        """F3.22: a genuinely valid multi-dataset submission (each required
        dataset has nodes, at least one dataset has a real edge) exports and
        validates cleanly."""
        graph_a = _make_graph(
            [{'name': 'a1', 't': 0, 'z': 1, 'y': 1, 'x': 1}, {'name': 'a2', 't': 1, 'z': 1, 'y': 1, 'x': 2}],
            edges_data=[('a1', 'a2')],
        )
        graph_b = self._one_node_graph()
        graphs = {'ds_a': graph_a, 'ds_b': graph_b}
        csv_path = export_submission(graphs, temp_csv / 'out.csv', required_dataset_ids=['ds_a', 'ds_b'])
        assert validate_submission(csv_path, required_dataset_ids=['ds_a', 'ds_b']) is True


class TestValidateSubmissionRequiredMode:
    """P0-6 (Part F3/E2): validate_submission()'s required_dataset_ids
    structural checks -- exact dataset equality, per-required-dataset node
    presence, edge structural validity (endpoints, self-edges, consecutive
    time, duplicates, degree limits, non-negative/positive/integer values)."""

    @pytest.fixture
    def temp_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _write_csv(self, path, rows):
        df = pd.DataFrame(rows)
        column_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        df = df[column_order]
        df.to_csv(path, index=False)
        return path

    def _base_valid_rows(self):
        """One dataset, 2 nodes (t=0 -> t=1), 1 valid edge between them."""
        return [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]

    def test_valid_submission_passes_required_mode(self, temp_csv):
        csv_path = self._write_csv(temp_csv / 'valid.csv', self._base_valid_rows())
        assert validate_submission(csv_path, required_dataset_ids=['ds_a']) is True

    def test_header_only_fails_required_mode(self, temp_csv):
        """F3.7: header-only passes generic mode but must fail required mode."""
        csv_path = temp_csv / 'empty.csv'
        pd.DataFrame(columns=['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']).to_csv(csv_path, index=False)
        assert validate_submission(csv_path) is True  # generic mode still accepts it
        with pytest.raises(ValueError, match="header-only"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_missing_required_dataset_raises(self, temp_csv):
        csv_path = self._write_csv(temp_csv / 'missing.csv', self._base_valid_rows())
        with pytest.raises(ValueError, match="Missing"):
            validate_submission(csv_path, required_dataset_ids=['ds_a', 'ds_b'])

    def test_unexpected_dataset_raises(self, temp_csv):
        csv_path = self._write_csv(temp_csv / 'unexpected.csv', self._base_valid_rows())
        with pytest.raises(ValueError, match="Unexpected"):
            validate_submission(csv_path, required_dataset_ids=['ds_z'])

    def test_zero_node_required_dataset_raises(self, temp_csv):
        """F3.5/F3.6: a required dataset with zero node rows can only occur
        as total absence from the CSV (an edge row can never reference a
        dataset with no node rows of its own) -- both are caught by the
        same exact-dataset-equality check."""
        rows = self._base_valid_rows()
        csv_path = self._write_csv(temp_csv / 'zero_node.csv', rows)
        with pytest.raises(ValueError, match="Missing"):
            validate_submission(csv_path, required_dataset_ids=['ds_a', 'ds_b'])

    def test_missing_source_node_raises(self, temp_csv):
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 1, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 99, 'target_id': 1},
        ]
        csv_path = self._write_csv(temp_csv / 'missing_source.csv', rows)
        with pytest.raises(ValueError, match="missing source"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_missing_target_node_raises(self, temp_csv):
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 99},
        ]
        csv_path = self._write_csv(temp_csv / 'missing_target.csv', rows)
        with pytest.raises(ValueError, match="missing target"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_cross_dataset_endpoint_raises(self, temp_csv):
        """F3.11: an edge in ds_a referencing node_id=2 must be treated as a
        missing endpoint even though ds_b legitimately has a node_id=2 --
        node_id numbering is per-dataset-local, so lookups must be keyed by
        (dataset, node_id), never by node_id alone."""
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_b', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_b', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 3, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]
        csv_path = self._write_csv(temp_csv / 'cross_dataset.csv', rows)
        with pytest.raises(ValueError, match="missing target"):
            validate_submission(csv_path, required_dataset_ids=['ds_a', 'ds_b'])

    def test_self_edge_raises(self, temp_csv):
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 1},
        ]
        csv_path = self._write_csv(temp_csv / 'self_edge.csv', rows)
        with pytest.raises(ValueError, match="self-edge"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_non_consecutive_time_raises(self, temp_csv):
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 2, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]
        csv_path = self._write_csv(temp_csv / 'non_consecutive.csv', rows)
        with pytest.raises(ValueError, match="target_t == source_t \\+ 1"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_duplicate_edge_raises(self, temp_csv):
        rows = self._base_valid_rows() + [
            {'id': 3, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]
        csv_path = self._write_csv(temp_csv / 'dup_edge.csv', rows)
        with pytest.raises(ValueError, match="Duplicate edge"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_out_degree_exceeds_two_raises(self, temp_csv):
        """F3.15: a node with 3 outgoing edges (only 0/1/2 children are
        physically valid) must raise."""
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 3, 't': 1, 'z': 1, 'y': 2, 'x': 5, 'source_id': -1, 'target_id': -1},
            {'id': 3, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 4, 't': 1, 'z': 1, 'y': 2, 'x': 6, 'source_id': -1, 'target_id': -1},
            {'id': 4, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
            {'id': 5, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 3},
            {'id': 6, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 4},
        ]
        csv_path = self._write_csv(temp_csv / 'out_degree.csv', rows)
        with pytest.raises(ValueError, match="out-degree"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_in_degree_exceeds_one_raises(self, temp_csv):
        """F3.16: a node with 2 incoming edges (a real cell has exactly one
        parent) must raise."""
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 0, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 3, 't': 1, 'z': 1, 'y': 2, 'x': 5, 'source_id': -1, 'target_id': -1},
            {'id': 3, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 3},
            {'id': 4, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 2, 'target_id': 3},
        ]
        csv_path = self._write_csv(temp_csv / 'in_degree.csv', rows)
        with pytest.raises(ValueError, match="in-degree"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_negative_node_coordinate_raises(self, temp_csv):
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': -1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]
        csv_path = self._write_csv(temp_csv / 'neg_coord.csv', rows)
        with pytest.raises(ValueError, match="negative time/coordinate"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_fractional_coordinate_raises(self, temp_csv):
        """F3.18: a fractional (non-integer) coordinate value fails the
        generic integer-value check, which required mode still runs."""
        rows = self._base_valid_rows()
        csv_path = temp_csv / 'fractional.csv'
        df = pd.DataFrame(rows)
        column_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        df = df[column_order]
        df['z'] = df['z'].astype(float)
        df.loc[0, 'z'] = 1.5
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="integer values"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_zero_source_id_raises(self, temp_csv):
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 0, 'target_id': 1},
        ]
        csv_path = self._write_csv(temp_csv / 'zero_source.csv', rows)
        with pytest.raises(ValueError, match="positive"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_zero_target_id_raises(self, temp_csv):
        """A literal target_id=0 slips past the generic '< 0' check (0 is
        not negative) but must still be rejected in required mode: node_ids
        are 1-indexed, so 0 can never be a genuine node reference."""
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 0},
        ]
        csv_path = self._write_csv(temp_csv / 'zero_target.csv', rows)
        with pytest.raises(ValueError, match="positive"):
            validate_submission(csv_path, required_dataset_ids=['ds_a'])

    def test_valid_division_two_children_passes(self, temp_csv):
        """F3.21: a genuine division (one parent node -> 2 children at
        t+1) is a legitimate, physically valid pattern and must PASS."""
        rows = [
            {'id': 0, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 1, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 2, 'dataset': 'ds_a', 'row_type': 'node', 'node_id': 3, 't': 1, 'z': 1, 'y': 2, 'x': 5, 'source_id': -1, 'target_id': -1},
            {'id': 3, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
            {'id': 4, 'dataset': 'ds_a', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 3},
        ]
        csv_path = self._write_csv(temp_csv / 'division.csv', rows)
        assert validate_submission(csv_path, required_dataset_ids=['ds_a']) is True

    def test_valid_multi_dataset_submission_passes(self, temp_csv):
        """F3.22: independent verification via validate_submission() alone
        (not just the export_submission round-trip test above)."""
        rows = self._base_valid_rows() + [
            {'id': 3, 'dataset': 'ds_b', 'row_type': 'node', 'node_id': 1, 't': 0, 'z': 1, 'y': 2, 'x': 3, 'source_id': -1, 'target_id': -1},
            {'id': 4, 'dataset': 'ds_b', 'row_type': 'node', 'node_id': 2, 't': 1, 'z': 1, 'y': 2, 'x': 4, 'source_id': -1, 'target_id': -1},
            {'id': 5, 'dataset': 'ds_b', 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': 1, 'target_id': 2},
        ]
        csv_path = self._write_csv(temp_csv / 'multi.csv', rows)
        assert validate_submission(csv_path, required_dataset_ids=['ds_a', 'ds_b']) is True
