"""
Unit tests for the evaluation harness (src/evaluation.py).

Tests:
1. load_geff_ground_truth() on real staged .geff files
2. evaluate_submission() with identical pred/gt graphs (perfect match)
3. evaluate_submission() with empty prediction graphs
4. Micro-averaging across multiple datasets
"""

import os
import pytest
import math
from src.evaluation import (
    load_geff_ground_truth,
    load_gt_for_dataset,
    evaluate_submission,
)


# Real staged data paths
DATA_STAGING_TRAIN = "data/staging/train"

# Sample dataset IDs from staged data
SAMPLE_DATASETS = [
    "44b6_0113de3b",  # 52 nodes, 50 edges
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]


class TestLoadGeffGroundTruth:
    """Test loading .geff files into tracksdata graphs."""

    def test_load_geff_real_staged_file(self):
        """Load real staged .geff file and verify graph structure."""
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")

        # This should not raise
        graph, metadata = load_geff_ground_truth(geff_path)

        # Verify graph has nodes
        assert graph is not None, "Loaded graph should not be None"
        assert graph.num_nodes() >= 1, f"Expected >= 1 node, got {graph.num_nodes()}"

        # Verify metadata has estimated_number_of_nodes
        assert hasattr(metadata, 'extra'), "Metadata should have 'extra' attribute"
        assert isinstance(metadata.extra, dict), "metadata.extra should be a dict"
        assert 'estimated_number_of_nodes' in metadata.extra, \
            "metadata.extra should contain 'estimated_number_of_nodes'"

        # Verify T_true is a positive integer
        t_true = metadata.extra['estimated_number_of_nodes']
        assert isinstance(t_true, int), f"T_true should be int, got {type(t_true)}"
        assert t_true > 0, f"T_true should be positive, got {t_true}"

    def test_load_geff_multiple_samples(self):
        """Load multiple .geff files to verify consistent behavior."""
        for dataset_id in SAMPLE_DATASETS[:2]:  # Test first 2 samples
            geff_path = os.path.join(DATA_STAGING_TRAIN, f"{dataset_id}.geff")
            graph, metadata = load_geff_ground_truth(geff_path)

            assert graph.num_nodes() > 0, f"Dataset {dataset_id} graph is empty"
            assert 'estimated_number_of_nodes' in metadata.extra, \
                f"Dataset {dataset_id} missing estimated_number_of_nodes"

    def test_load_geff_file_not_found(self):
        """Verify that missing .geff file raises FileNotFoundError."""
        geff_path = os.path.join(DATA_STAGING_TRAIN, "nonexistent_dataset.geff")

        with pytest.raises(FileNotFoundError):
            load_geff_ground_truth(geff_path)


class TestLoadGtForDataset:
    """Test the dataset-specific GT loader."""

    def test_load_gt_for_dataset(self):
        """Load GT graph for a specific dataset by ID."""
        dataset_id = SAMPLE_DATASETS[0]

        graph = load_gt_for_dataset(dataset_id, DATA_STAGING_TRAIN)

        assert graph is not None, "Loaded GT graph should not be None"
        assert graph.num_nodes() > 0, "GT graph should have nodes"

    def test_load_gt_for_missing_dataset(self):
        """Verify FileNotFoundError for missing dataset."""
        with pytest.raises(FileNotFoundError):
            load_gt_for_dataset("nonexistent_id", DATA_STAGING_TRAIN)


class TestEvaluateSubmission:
    """Test the main evaluate_submission() function."""

    def test_evaluate_identical_pred_and_gt(self):
        """
        Test: provide GT graph as both pred and GT.
        Expected: edge_jaccard should be 1.0 (perfect match).
        """
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, metadata = load_geff_ground_truth(geff_path)

        # Use an independent copy as pred so tracksdata's in-place graph.match() during
        # evaluate_datasets() doesn't mutate the same object being used as GT.
        pred_graph = gt_graph.copy()
        result = evaluate_submission(
            [pred_graph],
            [gt_graph],
            gt_metadata=[metadata],
        )

        # Verify return type
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

        # Verify required keys exist
        required_keys = [
            'edge_jaccard',
            'adjusted_edge_jaccard',
            'division_jaccard',
            'score',
            'num_pred_nodes_total',
            'num_gt_nodes_total',
            'num_datasets',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Verify edge_jaccard is 1.0 (perfect match)
        assert result['edge_jaccard'] == 1.0, \
            f"Edge Jaccard for identical graphs should be 1.0, got {result['edge_jaccard']}"

        # NOTE: adjusted_edge_jaccard is NOT expected to equal edge_jaccard here, even for a
        # perfect match. T_true is the GEFF's `estimated_number_of_nodes` (the full-embryo cell
        # count estimate, e.g. 25755) not the sparse labeled node count (52) -- ground truth in
        # this competition is deliberately sparse (see data/staging/README.md). Since
        # T_pred == T_true == 52 (both sides are the same sparse graph here) is nowhere near the
        # full 25755 estimate, the adjustment formula's node_ratio is strongly negative and boosts
        # the score above 1.0. This is correct behavior of the real formula, not a bug -- verified
        # directly against the formula: adjusted = edge_jaccard * (1 - 0.1*(T_pred-T_true)/T_true).
        expected_node_ratio = (result['num_pred_nodes_total'] - metadata.extra['estimated_number_of_nodes']) / metadata.extra['estimated_number_of_nodes']
        expected_adjusted = max(0.0, result['edge_jaccard'] * (1.0 - 0.1 * expected_node_ratio))
        assert abs(result['adjusted_edge_jaccard'] - expected_adjusted) < 1e-9, \
            f"adjusted_edge_jaccard should match the documented formula exactly: expected {expected_adjusted}, got {result['adjusted_edge_jaccard']}"

        # Verify score is at least edge_jaccard (it may include division_jaccard, or the
        # node-count bonus described above)
        assert result['score'] >= 1.0, \
            f"Score should be >= 1.0 for perfect edge match, got {result['score']}"

        # Verify node counts
        assert result['num_pred_nodes_total'] > 0, "num_pred_nodes_total should be > 0"
        assert result['num_gt_nodes_total'] > 0, "num_gt_nodes_total should be > 0"
        assert result['num_datasets'] == 1, "num_datasets should be 1"

    def test_evaluate_empty_prediction(self):
        """
        Test: provide empty prediction graph, real GT.
        Expected: edge_jaccard should be 0.0, score >= 0.
        """
        import tracksdata as td
        import polars as pl

        # Load real GT
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, metadata = load_geff_ground_truth(geff_path)

        # Create empty prediction graph. A bare IndexedRXGraph() only registers 't' as a node
        # attribute key by default -- 'x','y','z' schemas are established lazily when nodes are
        # first added. Passing a schema-less empty graph into evaluate_datasets() crashes deep in
        # tracksdata's node-matching internals with KeyError('z'), since it looks up the schema
        # for all four keys unconditionally. Pre-registering the keys (with zero actual nodes)
        # avoids this -- this is the correct way to represent "detector found nothing", a
        # realistic scenario once Phase 1's placeholder detector runs against real embryos.
        empty_pred = td.graph.IndexedRXGraph()
        for key in ('z', 'y', 'x'):
            empty_pred.add_node_attr_key(key, pl.Float64, 0.0)

        result = evaluate_submission(
            [empty_pred],
            [gt_graph],
            gt_metadata=[metadata],
        )

        # Verify edge_jaccard is 0.0 (no predicted edges)
        assert result['edge_jaccard'] == 0.0, \
            f"Edge Jaccard for empty pred should be 0.0, got {result['edge_jaccard']}"

        # Verify score is >= 0
        assert result['score'] >= 0.0, \
            f"Score should be >= 0, got {result['score']}"

    def test_evaluate_multiple_datasets(self):
        """
        Test: micro-averaging across 2+ datasets.
        Verify that metrics are summed before ratios.
        """
        # Load 2 GT graphs
        datasets = SAMPLE_DATASETS[:2]
        gt_graphs = []
        metadata_list = []

        for dataset_id in datasets:
            geff_path = os.path.join(DATA_STAGING_TRAIN, f"{dataset_id}.geff")
            graph, metadata = load_geff_ground_truth(geff_path)
            gt_graphs.append(graph)
            metadata_list.append(metadata)

        # Use same graphs as pred (perfect match for each)
        result = evaluate_submission(
            gt_graphs,
            gt_graphs,
            gt_metadata=metadata_list,
        )

        # Verify that num_datasets is correct
        assert result['num_datasets'] == 2, \
            f"num_datasets should be 2, got {result['num_datasets']}"

        # Verify edge_jaccard is 1.0 (perfect match across all datasets)
        assert result['edge_jaccard'] == 1.0, \
            f"Edge Jaccard for identical graphs should be 1.0, got {result['edge_jaccard']}"

    def test_evaluate_mismatched_list_lengths(self):
        """Verify that mismatched lengths raise ValueError."""
        import tracksdata as td

        gt_graph, _ = load_geff_ground_truth(
            os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        )
        empty_graph = td.graph.IndexedRXGraph()

        # Provide different number of pred and gt graphs
        with pytest.raises(ValueError, match="same length"):
            evaluate_submission(
                [empty_graph],  # 1 pred
                [gt_graph, gt_graph],  # 2 gt
            )

    def test_evaluate_empty_inputs(self):
        """Verify that empty input lists raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            evaluate_submission([], [])

    def test_evaluate_with_dict_inputs(self):
        """Test evaluate_submission with dict-based inputs."""
        # Load one sample as dict
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, metadata = load_geff_ground_truth(geff_path)

        pred_dict = {SAMPLE_DATASETS[0]: gt_graph}
        gt_dict = {SAMPLE_DATASETS[0]: gt_graph}
        metadata_dict = {SAMPLE_DATASETS[0]: metadata}

        result = evaluate_submission(
            pred_dict,
            gt_dict,
            gt_metadata=metadata_dict,
        )

        # Verify perfect match
        assert result['edge_jaccard'] == 1.0, \
            f"Dict-based input should work correctly, got edge_jaccard {result['edge_jaccard']}"


class TestAdjustmentFormula:
    """Test the node-count adjustment formula."""

    def test_adjustment_with_excess_prediction(self):
        """
        Test: predict more nodes than a baseline prediction, holding GT fixed.
        Expected: adjusted_edge_jaccard strictly decreases as T_pred grows (more of the
        node-count penalty applies), while edge_jaccard itself is unaffected (the extra
        nodes are spurious/unmatched, and unmatched predicted nodes are structurally excluded
        from the edge FP count -- see REFERENCE_IMPLEMENTATION.md).

        Note: T_true here is metadata.extra['estimated_number_of_nodes'] (~25755, the
        full-embryo estimate), not the sparse label count (52) -- see the comment in
        test_evaluate_identical_pred_and_gt. Actually exceeding T_true would need tens of
        thousands of extra nodes, which isn't a practical unit test; instead this verifies the
        monotonic direction of the penalty (more excess nodes -> lower adjusted score), which is
        what the docstring's original intent was checking for.
        """
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, metadata = load_geff_ground_truth(geff_path)

        baseline_pred = gt_graph.copy()
        baseline = evaluate_submission([baseline_pred], [gt_graph], gt_metadata=[metadata])

        # Add spurious extra nodes far from any real cell (won't match anything in GT within
        # the 7um gate) -- coordinates must be int to match the real geff schema (voxel-space
        # int64 per data/staging/README.md).
        excess_pred = gt_graph.copy()
        for i in range(500):
            excess_pred.add_node({'t': 0, 'z': 1000 + i, 'y': 1000, 'x': 1000})
        excess = evaluate_submission([excess_pred], [gt_graph], gt_metadata=[metadata])

        assert excess['num_pred_nodes_total'] == baseline['num_pred_nodes_total'] + 500

        assert excess['edge_jaccard'] == baseline['edge_jaccard'], \
            "Spurious unmatched nodes must not change edge_jaccard (they're excluded from FP)"

        assert excess['adjusted_edge_jaccard'] < baseline['adjusted_edge_jaccard'], \
            (f"More excess predicted nodes should strictly lower the adjusted score: "
             f"baseline={baseline['adjusted_edge_jaccard']}, excess={excess['adjusted_edge_jaccard']}")


if __name__ == "__main__":
    # Allow running with: python tests/test_evaluation_harness.py
    pytest.main([__file__, "-v"])
