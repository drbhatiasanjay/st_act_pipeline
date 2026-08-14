"""
Unit tests for oracle score decomposition (src/oracle_evaluation.py).

Tests the four oracle modes:
1. GT nodes + GT edges (control - should score ~1.0)
2. GT nodes + model-predicted edges (linker ceiling)
3. Model-predicted nodes + oracle edges (detector ceiling)
4. Model-predicted nodes + model-predicted edges (actual end-to-end)
"""

import logging
import os

import numpy as np
import polars as pl
import pytest
from tracksdata.graph import IndexedRXGraph

from src.evaluation import (
    load_geff_ground_truth,
)
from src.oracle_evaluation import (
    _build_oracle_graph,
    _compute_physical_distance,
    _find_nearest_gt_node,
    compute_detection_metrics,
    evaluate_oracle_modes,
)

logger = logging.getLogger(__name__)

# Real staged data paths
DATA_STAGING_TRAIN = "data/staging/train"

# Sample dataset IDs from staged data
SAMPLE_DATASETS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
]


class TestPhysicalDistance:
    """Test physical distance computation."""

    def test_compute_physical_distance_zero(self):
        """Distance from a point to itself should be 0."""
        coord = (10.0, 20.0, 30.0)
        dist = _compute_physical_distance(coord, coord)
        assert abs(dist) < 1e-9

    def test_compute_physical_distance_anisotropic(self):
        """Test distance with anisotropic scale."""
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (1.0, 1.0, 1.0)
        # With scale (1.625, 0.40625, 0.40625), distance should be:
        # sqrt((1*1.625)^2 + (1*0.40625)^2 + (1*0.40625)^2)
        dist = _compute_physical_distance(coord1, coord2)
        expected = np.sqrt(1.625**2 + 0.40625**2 + 0.40625**2)
        assert abs(dist - expected) < 1e-9


class TestFindNearestGtNode:
    """Test nearest GT node matching within distance threshold."""

    def test_find_nearest_gt_node_exact_match(self):
        """Find GT node at exact same coordinates."""
        gt_nodes = {
            0: {'t': 0, 'z': 10.0, 'y': 20.0, 'x': 30.0},
        }
        pred_coord = (10.0, 20.0, 30.0)
        gt_id, dist = _find_nearest_gt_node(pred_coord, gt_nodes)
        assert gt_id == 0
        assert abs(dist) < 1e-9

    def test_find_nearest_gt_node_multiple_candidates(self):
        """Find nearest among multiple GT nodes."""
        gt_nodes = {
            0: {'t': 0, 'z': 10.0, 'y': 20.0, 'x': 30.0},
            1: {'t': 0, 'z': 15.0, 'y': 25.0, 'x': 35.0},
            2: {'t': 0, 'z': 5.0, 'y': 15.0, 'x': 25.0},
        }
        pred_coord = (10.0, 20.0, 30.0)
        gt_id, dist = _find_nearest_gt_node(pred_coord, gt_nodes)
        assert gt_id == 0  # Exact match should be nearest

    def test_find_nearest_gt_node_outside_threshold(self):
        """Return None if nearest GT node is beyond max_distance."""
        gt_nodes = {
            0: {'t': 0, 'z': 100.0, 'y': 200.0, 'x': 300.0},
        }
        pred_coord = (10.0, 20.0, 30.0)
        gt_id, dist = _find_nearest_gt_node(pred_coord, gt_nodes, max_distance_um=1.0)
        assert gt_id is None
        assert np.isinf(dist)


class TestOracleGraphBuilding:
    """Test oracle graph construction by GT matching."""

    def test_build_oracle_graph_with_matches(self):
        """Build oracle graph when all predicted nodes match GT nodes."""
        # Create simple GT graph: 2 nodes at t=0, 2 at t=1, one edge
        gt_graph = IndexedRXGraph()
        # Add remaining required keys (t is pre-registered)
        for key in ('x', 'y', 'z'):
            try:
                gt_graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass  # Already exists

        # Add GT nodes at t=0 and t=1
        gt_n1 = gt_graph.add_node({'t': 0, 'z': 0, 'y': 0, 'x': 0})
        gt_n2 = gt_graph.add_node({'t': 1, 'z': 0, 'y': 0, 'x': 0})
        gt_graph.add_node({'t': 1, 'z': 10, 'y': 10, 'x': 10})

        # Add edge t=0 -> t=1
        gt_graph.add_edge(gt_n1, gt_n2, {})

        # Predicted nodes exactly match GT nodes
        pred_nodes_by_frame = {
            0: [(0.0, 0.0, 0.0)],
            1: [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)],
        }

        oracle_graph, localization_errors = _build_oracle_graph(gt_graph, pred_nodes_by_frame)

        # Oracle graph should have nodes and an edge
        assert oracle_graph.num_nodes() >= 2
        assert len(localization_errors) >= 1  # At least some nodes matched
        assert all(err < 0.1 for err in localization_errors)  # Errors should be tiny


class TestOracleModes:
    """Test the four oracle modes on real data."""

    def test_oracle_modes_on_real_data(self):
        """Run all four oracle modes on a real staged sample."""
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, gt_metadata = load_geff_ground_truth(geff_path)

        # Create a synthetic predicted graph (just copy GT for now)
        pred_graph = gt_graph.copy()

        # Synthetic predicted nodes exactly matching GT
        pred_nodes_by_frame = {}
        all_attrs = gt_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
        for row in all_attrs.to_dicts():
            t = int(row['t'])
            if t not in pred_nodes_by_frame:
                pred_nodes_by_frame[t] = []
            pred_nodes_by_frame[t].append((
                float(row['z']),
                float(row['y']),
                float(row['x']),
            ))

        # Evaluate all modes
        results = evaluate_oracle_modes(
            sample_id=SAMPLE_DATASETS[0],
            pred_graph=pred_graph,
            pred_nodes_by_frame=pred_nodes_by_frame,
            gt_graph=gt_graph,
            gt_metadata=gt_metadata,
        )

        # Verify all modes present
        assert 'gt_nodes_gt_edges' in results
        assert 'gt_nodes_model_edges' in results
        assert 'model_nodes_oracle_edges' in results
        assert 'model_nodes_model_edges' in results

        # Mode 1 (GT+GT control) must reproduce the scoring path exactly:
        # edge_jaccard must be exactly 1.0 (identical graph vs itself), not
        # just "high" -- a >=0.5 threshold previously let a real node-ID bug
        # (local enumerate() index compared against real graph node IDs,
        # silently zeroing every match) pass undetected because mode 1 alone
        # doesn't exercise that code path. See git history for the fix.
        assert results['gt_nodes_gt_edges']['edge_jaccard'] == 1.0, (
            f"GT+GT control edge_jaccard must be exactly 1.0, got "
            f"{results['gt_nodes_gt_edges']['edge_jaccard']}"
        )

        # Mode 3 (model_nodes_oracle_edges) is the one this test's synthetic
        # setup can actually pin exactly: predicted nodes are set to the
        # literal GT coordinates above, so oracle-edge matching must recover
        # edge_jaccard == 1.0 too. This is the specific assertion that would
        # have caught the node-ID confusion bug (it previously returned 0.0
        # silently, with a passing >=0.0 assertion, no crash).
        assert results['model_nodes_oracle_edges']['edge_jaccard'] == 1.0, (
            f"model_nodes_oracle_edges edge_jaccard must be exactly 1.0 when "
            f"predicted nodes exactly match GT coordinates, got "
            f"{results['model_nodes_oracle_edges']['edge_jaccard']} -- this is "
            f"the exact symptom of the node-ID (local index vs real graph ID) "
            f"matching bug; do not weaken this assertion to make it pass."
        )

        gt_gt_score = results['gt_nodes_gt_edges']['score']
        logger.info(f"GT+GT score: {gt_gt_score:.4f}")

        # All modes should have valid score >= 0
        for mode_name, mode_result in results.items():
            assert mode_result['score'] >= 0.0, f"{mode_name} score is negative"
            assert mode_result['edge_jaccard'] >= 0.0, f"{mode_name} edge_jaccard is negative"
            assert mode_result['edge_jaccard'] <= 1.0, f"{mode_name} edge_jaccard > 1.0"

    def test_oracle_modes_structure(self):
        """Verify oracle modes return expected structure."""
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, gt_metadata = load_geff_ground_truth(geff_path)

        # Empty prediction (detector found nothing)
        empty_pred = IndexedRXGraph()
        for key in ('x', 'y', 'z'):
            try:
                empty_pred.add_node_attr_key(key, pl.Float64, 0.0)
            except ValueError:
                pass  # Key already exists

        pred_nodes_by_frame = {}

        results = evaluate_oracle_modes(
            sample_id=SAMPLE_DATASETS[0],
            pred_graph=empty_pred,
            pred_nodes_by_frame=pred_nodes_by_frame,
            gt_graph=gt_graph,
            gt_metadata=gt_metadata,
        )

        # All modes should return dicts with required keys
        required_keys = {'edge_jaccard', 'adjusted_edge_jaccard', 'division_jaccard', 'score', 'predicted_nodes_total', 'gt_nodes_total'}
        for mode_name, mode_result in results.items():
            assert isinstance(mode_result, dict)
            for key in required_keys:
                assert key in mode_result, f"{mode_name} missing key {key}"


class TestDetectionMetrics:
    """Test detection precision/recall computation."""

    def test_detection_metrics_perfect_predictions(self):
        """Compute metrics when all GT nodes have matching predictions."""
        # Create simple GT graph
        gt_graph = IndexedRXGraph()
        for key in ('x', 'y', 'z'):
            try:
                gt_graph.add_node_attr_key(key, pl.Float64, 0.0)
            except ValueError:
                pass

        gt_graph.add_node({'t': 0, 'z': 0.0, 'y': 0.0, 'x': 0.0})
        gt_graph.add_node({'t': 1, 'z': 10.0, 'y': 10.0, 'x': 10.0})

        # Predicted nodes exactly match
        pred_nodes_by_frame = {
            0: [(0.0, 0.0, 0.0)],
            1: [(10.0, 10.0, 10.0)],
        }

        precision, recall, localization_errors = compute_detection_metrics(
            pred_nodes_by_frame,
            gt_graph,
        )

        # Both should be 1.0 (perfect match)
        assert precision == 1.0, f"Expected precision=1.0, got {precision}"
        assert recall == 1.0, f"Expected recall=1.0, got {recall}"

    def test_detection_metrics_no_predictions(self):
        """Compute metrics when detector finds nothing."""
        # Create simple GT graph with nodes
        gt_graph = IndexedRXGraph()
        for key in ('x', 'y', 'z'):
            try:
                gt_graph.add_node_attr_key(key, pl.Float64, 0.0)
            except ValueError:
                pass

        gt_graph.add_node({'t': 0, 'z': 0.0, 'y': 0.0, 'x': 0.0})
        gt_graph.add_node({'t': 1, 'z': 10.0, 'y': 10.0, 'x': 10.0})

        # No predicted nodes
        pred_nodes_by_frame = {}

        precision, recall, localization_errors = compute_detection_metrics(
            pred_nodes_by_frame,
            gt_graph,
        )

        # Precision should be 0/0 = 0 or undefined, recall = 0
        assert recall == 0.0, f"Expected recall=0.0 when no predictions, got {recall}"
        assert len(localization_errors) == 0

    def test_detection_metrics_false_positives(self):
        """Compute metrics with false-positive predictions."""
        # Create simple GT graph with 1 node
        gt_graph = IndexedRXGraph()
        for key in ('x', 'y', 'z'):
            try:
                gt_graph.add_node_attr_key(key, pl.Float64, 0.0)
            except ValueError:
                pass

        gt_graph.add_node({'t': 0, 'z': 0.0, 'y': 0.0, 'x': 0.0})

        # Predict 2 nodes: 1 correct + 1 false positive far away
        pred_nodes_by_frame = {
            0: [
                (0.0, 0.0, 0.0),  # Matches GT
                (100.0, 100.0, 100.0),  # False positive
            ],
        }

        precision, recall, localization_errors = compute_detection_metrics(
            pred_nodes_by_frame,
            gt_graph,
        )

        # Precision = 1 match / 2 predictions = 0.5
        assert precision == 0.5, f"Expected precision=0.5, got {precision}"
        # Recall = 1 matched / 1 GT = 1.0
        assert recall == 1.0, f"Expected recall=1.0, got {recall}"


class TestOracleModeComparisons:
    """Test logical relationships between oracle modes."""

    def test_modes_form_sensible_hierarchy(self):
        """
        Verify that oracle modes form a sensible scoring hierarchy:
        - Mode 1 (GT+GT) should be >= Mode 2 (GT+Model edges) in most cases
        - Mode 3 (Model+Oracle edges) represents detector ceiling
        - Mode 4 (Model+Model) is the actual system
        """
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, gt_metadata = load_geff_ground_truth(geff_path)

        # Use GT as predictions (perfect detector + linker)
        pred_graph = gt_graph.copy()

        pred_nodes_by_frame = {}
        all_attrs = gt_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
        for row in all_attrs.to_dicts():
            t = int(row['t'])
            if t not in pred_nodes_by_frame:
                pred_nodes_by_frame[t] = []
            pred_nodes_by_frame[t].append((
                float(row['z']),
                float(row['y']),
                float(row['x']),
            ))

        results = evaluate_oracle_modes(
            sample_id=SAMPLE_DATASETS[0],
            pred_graph=pred_graph,
            pred_nodes_by_frame=pred_nodes_by_frame,
            gt_graph=gt_graph,
            gt_metadata=gt_metadata,
        )

        # When predictions = GT, all modes should score similarly
        mode_scores = {k: v['score'] for k, v in results.items()}

        logger.info("Mode scores (pred=GT):")
        for mode, score in mode_scores.items():
            logger.info(f"  {mode}: {score:.4f}")

        # When pred == GT exactly, all four modes should agree exactly on
        # edge_jaccard (== 1.0) since they all reduce to the same identical
        # graph comparison. A >=0.5 threshold previously hid a real bug where
        # mode 3 alone silently returned 0.0 (node-ID matching bug) while
        # this assertion still passed -- pin exact equality, not a loose bound.
        edge_jaccards = {k: v['edge_jaccard'] for k, v in results.items()}
        for mode, jaccard in edge_jaccards.items():
            assert jaccard == 1.0, (
                f"{mode} edge_jaccard should be exactly 1.0 when pred==GT, "
                f"got {jaccard}"
            )


class TestMode2IndependentPredGraph:
    """Regression test: mode 2 (gt_nodes_model_edges) must not assume
    pred_graph's node IDs are valid keys into gt_graph's ID space.

    Every existing test above builds pred_graph via gt_graph.copy(), which
    trivially shares the same real node IDs and previously masked this exact
    bug -- confirmed 2026-08-14 when running evaluate_oracle_modes() against
    a genuinely independent prediction graph (from
    scripts/oracle_check_probe_checkpoint.py, PredictionGraphAssembler's own
    internal node numbering) crashed with KeyError inside tracksdata's
    add_edge(), the same node-ID confusion class as the mode 3 bug already
    fixed in _build_oracle_graph.
    """

    def test_mode2_survives_independently_numbered_pred_graph(self):
        """pred_graph built from scratch (own node-ID space, not a GT copy)
        with nodes at the real GT coordinates must not crash mode 2, and
        should recover edge_jaccard==1.0 for that mode since every predicted
        node exactly matches a real GT node."""
        geff_path = os.path.join(DATA_STAGING_TRAIN, f"{SAMPLE_DATASETS[0]}.geff")
        gt_graph, gt_metadata = load_geff_ground_truth(geff_path)

        gt_attrs = gt_graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
        gt_edge_list = gt_graph.edge_list()
        gt_real_ids = gt_graph.node_ids()
        gt_id_to_row = dict(zip(gt_real_ids, gt_attrs.to_dicts(), strict=True))

        # Build a genuinely independent graph: fresh schema, tracksdata
        # assigns its own internal IDs on add_node(), unrelated to gt_graph's.
        pred_graph = IndexedRXGraph()
        for key in ('t', 'x', 'y', 'z'):
            try:
                pred_graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass

        gt_id_to_pred_id = {}
        for gt_id, row in gt_id_to_row.items():
            pred_id = pred_graph.add_node({
                't': int(row['t']), 'z': int(round(row['z'])),
                'y': int(round(row['y'])), 'x': int(round(row['x'])),
            })
            gt_id_to_pred_id[gt_id] = pred_id

        # Reproduce every real GT edge, but under pred_graph's own new IDs --
        # this is exactly the shape a real, independently-assembled
        # prediction graph has: correct linking, unrelated ID numbering.
        for gt_src, gt_tgt in gt_edge_list:
            pred_graph.add_edge(gt_id_to_pred_id[gt_src], gt_id_to_pred_id[gt_tgt], {})

        pred_nodes_by_frame = {}
        for row in gt_attrs.to_dicts():
            t = int(row['t'])
            pred_nodes_by_frame.setdefault(t, []).append(
                (float(row['z']), float(row['y']), float(row['x']))
            )

        # Must not raise -- this crashed with KeyError before the mode 2 fix.
        results = evaluate_oracle_modes(
            sample_id=SAMPLE_DATASETS[0],
            pred_graph=pred_graph,
            pred_nodes_by_frame=pred_nodes_by_frame,
            gt_graph=gt_graph,
            gt_metadata=gt_metadata,
        )

        assert results['gt_nodes_model_edges']['edge_jaccard'] == 1.0, (
            f"mode 2 should recover edge_jaccard=1.0 when the independently-"
            f"numbered pred_graph's edges exactly match GT edges under "
            f"coordinate matching, got {results['gt_nodes_model_edges']['edge_jaccard']}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
