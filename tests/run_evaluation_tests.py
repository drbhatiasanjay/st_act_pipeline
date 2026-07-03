#!/usr/bin/env python3
"""
Standalone test runner for evaluation harness (no pytest required).
Run with: python tests/run_evaluation_tests.py
"""

import os
import sys
import traceback
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    load_geff_ground_truth,
    load_gt_for_dataset,
    evaluate_submission,
)


# Real staged data paths
DATA_STAGING_TRAIN = "data/staging/train"

# Sample dataset IDs from staged data
SAMPLE_DATASETS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]

# Test results tracking
test_results = {"passed": 0, "failed": 0, "errors": []}


def assert_true(condition, message):
    """Helper to make assertions."""
    if not condition:
        raise AssertionError(message)


def test_load_geff_real_staged_file():
    """Test: load_geff_ground_truth on real staged .geff file."""
    print("\n[TEST] load_geff_ground_truth on real staged file...")
    try:
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")

        graph, metadata = load_geff_ground_truth(geff_path)

        assert_true(graph is not None, "Loaded graph should not be None")
        assert_true(graph.num_nodes() >= 1, f"Expected >= 1 node, got {graph.num_nodes()}")
        assert_true(hasattr(metadata, 'extra'), "Metadata should have 'extra' attribute")
        assert_true(isinstance(metadata.extra, dict), "metadata.extra should be a dict")
        assert_true('estimated_number_of_nodes' in metadata.extra,
                   "metadata.extra should contain 'estimated_number_of_nodes'")

        t_true = metadata.extra['estimated_number_of_nodes']
        assert_true(isinstance(t_true, int), f"T_true should be int, got {type(t_true)}")
        assert_true(t_true > 0, f"T_true should be positive, got {t_true}")

        print(f"  ✓ Loaded graph with {graph.num_nodes()} nodes, T_true={t_true}")
        test_results["passed"] += 1
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_load_geff_real_staged_file: {e}")
        traceback.print_exc()
        return False


def test_load_gt_for_dataset():
    """Test: load_gt_for_dataset helper."""
    print("\n[TEST] load_gt_for_dataset...")
    try:
        dataset_id = SAMPLE_DATASETS[0]
        graph = load_gt_for_dataset(dataset_id, DATA_STAGING_TRAIN)

        assert_true(graph is not None, "Loaded GT graph should not be None")
        assert_true(graph.num_nodes() > 0, "GT graph should have nodes")

        print(f"  ✓ Loaded GT graph for {dataset_id} with {graph.num_nodes()} nodes")
        test_results["passed"] += 1
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_load_gt_for_dataset: {e}")
        traceback.print_exc()
        return False


def test_evaluate_identical_pred_and_gt():
    """Test: provide GT graph as both pred and GT - should get perfect match."""
    print("\n[TEST] evaluate_submission with identical pred/gt graphs...")
    try:
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, metadata = load_geff_ground_truth(geff_path)

        result = evaluate_submission(
            [gt_graph],
            [gt_graph],
            gt_metadata=[metadata],
        )

        # Verify return type and keys
        assert_true(isinstance(result, dict), f"Expected dict, got {type(result)}")
        required_keys = [
            'edge_jaccard', 'adjusted_edge_jaccard', 'division_jaccard',
            'score', 'num_pred_nodes_total', 'num_gt_nodes_total', 'num_datasets'
        ]
        for key in required_keys:
            assert_true(key in result, f"Missing key: {key}")

        # Verify edge_jaccard is 1.0 (perfect match)
        assert_true(
            result['edge_jaccard'] == 1.0,
            f"Edge Jaccard should be 1.0 for identical graphs, got {result['edge_jaccard']}"
        )

        # Verify adjusted_edge_jaccard is 1.0 (same nodes, no penalty)
        assert_true(
            result['adjusted_edge_jaccard'] == 1.0,
            f"Adjusted should be 1.0, got {result['adjusted_edge_jaccard']}"
        )

        # Verify score >= 1.0
        assert_true(result['score'] >= 1.0, f"Score should be >= 1.0, got {result['score']}")

        # Verify node counts
        assert_true(result['num_pred_nodes_total'] > 0, "num_pred_nodes_total should be > 0")
        assert_true(result['num_gt_nodes_total'] > 0, "num_gt_nodes_total should be > 0")
        assert_true(result['num_datasets'] == 1, "num_datasets should be 1")

        print(f"  ✓ Perfect match: edge_jaccard={result['edge_jaccard']}, "
              f"score={result['score']:.4f}")
        test_results["passed"] += 1
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_evaluate_identical_pred_and_gt: {e}")
        traceback.print_exc()
        return False


def test_evaluate_empty_prediction():
    """Test: empty prediction graph against real GT."""
    print("\n[TEST] evaluate_submission with empty prediction...")
    try:
        import tracksdata as td

        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, metadata = load_geff_ground_truth(geff_path)

        # Create empty prediction graph
        empty_pred = td.graph.IndexedRXGraph()

        result = evaluate_submission(
            [empty_pred],
            [gt_graph],
            gt_metadata=[metadata],
        )

        # Verify edge_jaccard is 0.0
        assert_true(
            result['edge_jaccard'] == 0.0,
            f"Edge Jaccard for empty pred should be 0.0, got {result['edge_jaccard']}"
        )

        # Verify score is >= 0
        assert_true(result['score'] >= 0.0, f"Score should be >= 0, got {result['score']}")

        print(f"  ✓ Empty prediction: edge_jaccard={result['edge_jaccard']}, "
              f"score={result['score']:.4f}")
        test_results["passed"] += 1
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_evaluate_empty_prediction: {e}")
        traceback.print_exc()
        return False


def test_evaluate_multiple_datasets():
    """Test: micro-averaging across multiple datasets."""
    print("\n[TEST] evaluate_submission with multiple datasets...")
    try:
        datasets = SAMPLE_DATASETS[:2]  # Use 2 samples
        gt_graphs = []
        metadata_list = []

        for dataset_id in datasets:
            geff_path = os.path.join(DATA_STAGING_TRAIN, f"{dataset_id}.geff")
            graph, metadata = load_geff_ground_truth(geff_path)
            gt_graphs.append(graph)
            metadata_list.append(metadata)

        # Use same graphs as pred (perfect match)
        result = evaluate_submission(
            gt_graphs,
            gt_graphs,
            gt_metadata=metadata_list,
        )

        # Verify num_datasets
        assert_true(result['num_datasets'] == 2, f"num_datasets should be 2, got {result['num_datasets']}")

        # Verify edge_jaccard is 1.0
        assert_true(
            result['edge_jaccard'] == 1.0,
            f"Edge Jaccard should be 1.0 for identical graphs, got {result['edge_jaccard']}"
        )

        print(f"  ✓ Multi-dataset (2 samples): edge_jaccard={result['edge_jaccard']}, "
              f"score={result['score']:.4f}")
        test_results["passed"] += 1
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_evaluate_multiple_datasets: {e}")
        traceback.print_exc()
        return False


def test_evaluate_dict_inputs():
    """Test: evaluate_submission with dict-based inputs."""
    print("\n[TEST] evaluate_submission with dict inputs...")
    try:
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
        assert_true(
            result['edge_jaccard'] == 1.0,
            f"Dict-based input should work, got edge_jaccard {result['edge_jaccard']}"
        )

        print(f"  ✓ Dict-based input: edge_jaccard={result['edge_jaccard']}")
        test_results["passed"] += 1
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_evaluate_dict_inputs: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("EVALUATION HARNESS TEST SUITE")
    print("=" * 70)

    # Run all tests
    test_load_geff_real_staged_file()
    test_load_gt_for_dataset()
    test_evaluate_identical_pred_and_gt()
    test_evaluate_empty_prediction()
    test_evaluate_multiple_datasets()
    test_evaluate_dict_inputs()

    # Print summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {test_results['passed']} passed, {test_results['failed']} failed")
    print("=" * 70)

    if test_results["errors"]:
        print("\nFAILURES:")
        for error in test_results["errors"]:
            print(f"  - {error}")

    if test_results["failed"] > 0:
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
