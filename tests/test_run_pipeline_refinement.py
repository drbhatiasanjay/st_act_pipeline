"""
Unit tests for run_pipeline.py's post-solve graph refinement functions
(prune_short_tracks, linefit_smooth_coordinates) -- the two low-risk,
competitor-validated improvements from COMPETITOR_RESEARCH_2026-07-13.md
(items 3 and 4), both buildable/testable with synthetic graphs, no real
checkpoint or GPU needed. See DEFERRED_IMPROVEMENTS.md's priority matrix.

Run: py -m pytest tests/test_run_pipeline_refinement.py -v
"""
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_pipeline import linefit_smooth_coordinates, prune_short_tracks


def make_linear_track(node_ids: list[tuple[int, int]], coords: list) -> nx.DiGraph:
    """A simple straight chain: node_ids[0] -> node_ids[1] -> ..."""
    g = nx.DiGraph()
    for n, c in zip(node_ids, coords, strict=False):
        g.add_node(n, coords=np.array(c, dtype=np.float64))
    for a, b in zip(node_ids, node_ids[1:], strict=False):
        g.add_edge(a, b)
    return g


class TestPruneShortTracks:
    def test_short_component_without_division_is_dropped(self):
        # A 2-node track, min_track_len=4 -> should be pruned entirely
        g = make_linear_track([(0, 0), (1, 0)], [[0, 0, 0], [0, 0, 1]])

        result = prune_short_tracks(g, min_track_len=4, keep_division_components=True)

        assert result.number_of_nodes() == 0

    def test_component_at_or_above_min_len_is_kept(self):
        nodes = [(t, 0) for t in range(4)]
        coords = [[t, 0, 0] for t in range(4)]
        g = make_linear_track(nodes, coords)

        result = prune_short_tracks(g, min_track_len=4, keep_division_components=True)

        assert result.number_of_nodes() == 4
        assert set(result.nodes()) == set(nodes)

    def test_short_component_WITH_division_is_protected(self):
        """REGRESSION-relevant: this is the exact safety check
        DEFERRED_IMPROVEMENTS.md's old item 1 flagged as the real blocker for
        naive pruning -- a genuine division event must survive even if the
        component is short."""
        g = nx.DiGraph()
        parent = (0, 0)
        child_a = (1, 0)
        child_b = (1, 1)
        g.add_node(parent, coords=np.array([0.0, 0.0, 0.0]))
        g.add_node(child_a, coords=np.array([1.0, 0.0, 0.0]))
        g.add_node(child_b, coords=np.array([1.0, 1.0, 0.0]))
        g.add_edge(parent, child_a)
        g.add_edge(parent, child_b)  # parent has out-degree 2 -> division

        result = prune_short_tracks(g, min_track_len=10, keep_division_components=True)

        assert result.number_of_nodes() == 3, "division component must survive despite being far below min_track_len"

    def test_division_protection_can_be_disabled(self):
        g = nx.DiGraph()
        parent = (0, 0)
        child_a = (1, 0)
        child_b = (1, 1)
        g.add_node(parent, coords=np.array([0.0, 0.0, 0.0]))
        g.add_node(child_a, coords=np.array([1.0, 0.0, 0.0]))
        g.add_node(child_b, coords=np.array([1.0, 1.0, 0.0]))
        g.add_edge(parent, child_a)
        g.add_edge(parent, child_b)

        result = prune_short_tracks(g, min_track_len=10, keep_division_components=False)

        assert result.number_of_nodes() == 0

    def test_does_not_mutate_input_graph(self):
        g = make_linear_track([(0, 0), (1, 0)], [[0, 0, 0], [0, 0, 1]])
        original_node_count = g.number_of_nodes()

        prune_short_tracks(g, min_track_len=4)

        assert g.number_of_nodes() == original_node_count

    def test_empty_graph_returns_empty(self):
        g = nx.DiGraph()
        result = prune_short_tracks(g, min_track_len=4)
        assert result.number_of_nodes() == 0


class TestLinefitSmoothCoordinates:
    def test_perfectly_linear_track_is_unchanged_by_smoothing(self):
        """A track that's ALREADY a perfect line should be a no-op (within
        floating point tolerance) regardless of the blend weight -- the fit
        exactly reproduces the raw coordinates."""
        nodes = [(t, 0) for t in range(5)]
        coords = [[float(t), float(2 * t), 0.0] for t in range(5)]
        g = make_linear_track(nodes, coords)

        result = linefit_smooth_coordinates(g, weight=0.76, window=2)

        for n, expected in zip(nodes, coords, strict=False):
            actual = result.nodes[n]["coords"]
            assert np.allclose(actual, expected, atol=1e-6), f"node {n}: {actual} != {expected}"

    def test_jittered_midpoint_is_pulled_toward_the_line(self):
        """REGRESSION-relevant: this is the actual point of the feature -- a
        single jittered node in an otherwise-linear track must move measurably
        closer to the fitted line, not stay at its raw (jittered) position."""
        nodes = [(t, 0) for t in range(5)]
        coords = [[float(t), 0.0, 0.0] for t in range(5)]
        coords[2] = [2.0, 5.0, 0.0]  # inject a real jitter at the middle node
        g = make_linear_track(nodes, coords)

        result = linefit_smooth_coordinates(g, weight=0.76, window=2)

        smoothed_y = result.nodes[(2, 0)]["coords"][1]
        assert smoothed_y < 5.0, "jittered node must move toward the fitted line, not stay at raw position"
        assert smoothed_y > 0.0, "with weight=0.76 (not 1.0) some raw jitter should still remain"

    def test_weight_zero_is_a_true_no_op(self):
        nodes = [(t, 0) for t in range(5)]
        coords = [[float(t), 0.0, 0.0] for t in range(5)]
        coords[2] = [2.0, 99.0, 0.0]
        g = make_linear_track(nodes, coords)

        result = linefit_smooth_coordinates(g, weight=0.0, window=2)

        assert np.allclose(result.nodes[(2, 0)]["coords"], [2.0, 99.0, 0.0])

    def test_gap_closing_edge_does_not_corrupt_the_neighborhood(self):
        """A t -> t+2 gap-closing edge must NOT be treated as part of the
        linear chain (it would corrupt the dt assumption in the fit) -- a node
        with only a gap-edge neighbor should be treated as isolated (skipped,
        len(neighborhood) < 3), not smoothed using a wrong dt=2 step."""
        g = nx.DiGraph()
        g.add_node((0, 0), coords=np.array([0.0, 0.0, 0.0]))
        g.add_node((2, 0), coords=np.array([10.0, 10.0, 10.0]))
        g.add_edge((0, 0), (2, 0))  # gap-closing edge, skips t=1

        result = linefit_smooth_coordinates(g, weight=0.76, window=2)

        # neighborhood for each node is just itself (len=1 < 3) -- untouched
        assert np.allclose(result.nodes[(0, 0)]["coords"], [0.0, 0.0, 0.0])
        assert np.allclose(result.nodes[(2, 0)]["coords"], [10.0, 10.0, 10.0])

    def test_branch_point_stops_the_chain_walk(self):
        """A node with 2 successors (division) must not be treated as having a
        single well-defined 'next' position -- the walk must stop there."""
        g = nx.DiGraph()
        g.add_node((0, 0), coords=np.array([0.0, 0.0, 0.0]))
        g.add_node((1, 0), coords=np.array([1.0, 0.0, 0.0]))
        g.add_node((2, 0), coords=np.array([2.0, 0.0, 0.0]))
        g.add_node((2, 1), coords=np.array([2.0, 5.0, 0.0]))
        g.add_edge((0, 0), (1, 0))
        g.add_edge((1, 0), (2, 0))
        g.add_edge((1, 0), (2, 1))  # division at (1,0): 2 successors

        result = linefit_smooth_coordinates(g, weight=0.76, window=2)

        # (1,0)'s forward walk must stop immediately (2 successors) -- its
        # neighborhood is only [(-1,(0,0)), (0,(1,0))], len=2 < 3, untouched
        assert np.allclose(result.nodes[(1, 0)]["coords"], [1.0, 0.0, 0.0])

    def test_does_not_add_or_remove_nodes_or_edges(self):
        nodes = [(t, 0) for t in range(5)]
        coords = [[float(t), 0.0, 0.0] for t in range(5)]
        g = make_linear_track(nodes, coords)
        n_before, e_before = g.number_of_nodes(), g.number_of_edges()

        result = linefit_smooth_coordinates(g, weight=0.76, window=2)

        assert result.number_of_nodes() == n_before
        assert result.number_of_edges() == e_before

    def test_empty_graph_is_a_no_op(self):
        g = nx.DiGraph()
        result = linefit_smooth_coordinates(g, weight=0.76, window=2)
        assert result.number_of_nodes() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
