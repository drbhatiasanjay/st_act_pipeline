"""
Unit tests for src/prediction_graph.py's PredictionGraphAssembler -- the
P0-3 fix (2026-07-17) for overlapping-window duplicate-node creation.

Confirmed defect (P0-3 audit): validate_epoch()/evaluate_checkpoint.py/
verify_eval_fixed.py each unconditionally add_node()'d BOTH channels every
batch, so frame t_idx+1 (produced by window A's channel 1 AND window B's
channel 0) got two distinct graph node IDs for one real detection --
verified via a reproducer using the real IndexedRXGraph/add_node code:
4 nodes / frame counts {0:1, 1:2, 2:1} for a 3-frame sample instead of the
correct 3 nodes / {0:1, 1:1, 2:1}.

These tests exercise the real PredictionGraphAssembler class directly (the
same class the three production callers now use) -- not a reimplementation
of graph-assembly logic.

Run: py -m pytest tests/test_prediction_graph.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prediction_graph import PredictionGraphAssembler


def frame_counts(assembler: PredictionGraphAssembler, sample_id: str) -> dict:
    graph = assembler.pred_graphs()[sample_id]
    attrs = graph.node_attrs(attr_keys=["t"])
    counts: dict = {}
    for row in attrs.to_dicts():
        counts[row["t"]] = counts.get(row["t"], 0) + 1
    return counts


class TestThreeFrameContinuity:
    """Test A: the exact P0-3 reproducer shape -- window A's channel-1
    output for frame 1 and window B's channel-0 output for frame 1 are
    IDENTICAL coordinates, and must still collapse to one canonical node
    (not because of coordinate matching -- see TestNonIdenticalOverlap for
    proof it's not coordinate-based)."""

    def test_produces_exactly_three_nodes_and_two_edges(self):
        assembler = PredictionGraphAssembler()
        sample_id = "sample_a"

        assembler.validate_window_order(sample_id, 0)
        src_ids_a, _src_coords_a, tgt_ids_a, _tgt_coords_a = assembler.process_window(
            sample_id, t_idx=0, peaks_t=[(1, 1, 1)], peaks_t1=[(2, 2, 2)],
        )
        assembler.add_edges(sample_id, src_ids_a, tgt_ids_a, edges=[(0, 0, 1.0)])

        assembler.validate_window_order(sample_id, 1)
        src_ids_b, _src_coords_b, tgt_ids_b, _tgt_coords_b = assembler.process_window(
            sample_id, t_idx=1, peaks_t=[(2, 2, 2)], peaks_t1=[(3, 3, 3)],
        )
        assembler.add_edges(sample_id, src_ids_b, tgt_ids_b, edges=[(0, 0, 1.0)])

        graph = assembler.pred_graphs()[sample_id]
        assert len(graph.node_ids()) == 3
        assert len(graph.edge_list()) == 2
        assert frame_counts(assembler, sample_id) == {0: 1, 1: 1, 2: 1}

    def test_edges_form_one_continuous_chain(self):
        assembler = PredictionGraphAssembler()
        sample_id = "sample_a"

        assembler.validate_window_order(sample_id, 0)
        src_a, _, tgt_a, _ = assembler.process_window(
            sample_id, t_idx=0, peaks_t=[(1, 1, 1)], peaks_t1=[(2, 2, 2)],
        )
        assembler.add_edges(sample_id, src_a, tgt_a, edges=[(0, 0, 1.0)])

        assembler.validate_window_order(sample_id, 1)
        src_b, _, tgt_b, _ = assembler.process_window(
            sample_id, t_idx=1, peaks_t=[(2, 2, 2)], peaks_t1=[(3, 3, 3)],
        )
        assembler.add_edges(sample_id, src_b, tgt_b, edges=[(0, 0, 1.0)])

        # window B's source node (frame 1) must be the SAME node ID as
        # window A's target node (frame 1) -- proving reuse, not a second node.
        assert src_b == tgt_a

        graph = assembler.pred_graphs()[sample_id]
        edge_list = graph.edge_list()
        frame0_id, frame1_id, frame2_id = src_a[0], tgt_a[0], tgt_b[0]
        assert sorted(edge_list) == sorted([[frame0_id, frame1_id], [frame1_id, frame2_id]])


class TestNonIdenticalOverlap:
    """Test B: proves canonical-frame reuse is NOT coordinate-based. Window
    B's channel-0 "observation" of frame 1 has DIFFERENT coordinates (and
    could have a different count) than window A's channel-1 canonical
    node -- the real model is not guaranteed to return identical overlapping
    predictions. The canonical frame-1 node must still be reused, and the
    differing channel-0 peak must never be compared/merged/create a node."""

    def test_canonical_node_reused_despite_differing_channel0_observation(self):
        assembler = PredictionGraphAssembler()
        sample_id = "sample_a"

        assembler.validate_window_order(sample_id, 0)
        src_a, _, tgt_a, _ = assembler.process_window(
            sample_id, t_idx=0, peaks_t=[(1, 1, 1)], peaks_t1=[(2, 2, 2)],
        )
        assembler.add_edges(sample_id, src_a, tgt_a, edges=[(0, 0, 1.0)])

        # Window B's channel-0 "observation" of frame 1 differs in
        # coordinates from window A's canonical (2,2,2) -- and even differs
        # in count (two channel-0 peaks here vs. one canonical node).
        assembler.validate_window_order(sample_id, 1)
        src_b, src_coords_b, tgt_b, _ = assembler.process_window(
            sample_id, t_idx=1, peaks_t=[(2, 2, 3), (9, 9, 9)], peaks_t1=[(3, 3, 3)],
        )

        # The canonical source returned for frame 1 must be window A's
        # single canonical node -- NOT a new node built from window B's
        # differing/plural channel-0 peaks.
        assert src_b == tgt_a
        assert len(src_b) == 1
        assert src_coords_b == [(2, 2, 2)]  # canonical coords, not (2,2,3)/(9,9,9)

        graph = assembler.pred_graphs()[sample_id]
        assert len(graph.node_ids()) == 3  # not 4 or 5 -- the differing peaks never became nodes
        assert frame_counts(assembler, sample_id) == {0: 1, 1: 1, 2: 1}

        # Diagnostic-only: the differing channel-0 peaks are still counted,
        # just never turned into graph nodes.
        diag = assembler.diagnostics()
        assert diag["raw_channel0_peaks_total"] == 1 + 2  # window A's + window B's
        assert diag["predicted_nodes_total"] == 3


class TestGenuineNearbyCells:
    """Test C: two distinct channel-1 detections within the SAME newly
    owned frame must both remain as separate nodes -- no distance-based or
    rounded-coordinate merge may occur within a frame."""

    def test_two_nearby_channel1_detections_both_kept(self):
        assembler = PredictionGraphAssembler()
        sample_id = "sample_a"

        assembler.validate_window_order(sample_id, 0)
        src, _, tgt, tgt_coords = assembler.process_window(
            sample_id, t_idx=0, peaks_t=[(1, 1, 1)],
            peaks_t1=[(5, 5, 5), (5, 5, 6)],  # two genuinely distinct, adjacent cells
        )

        assert len(tgt) == 2
        assert tgt_coords == [(5, 5, 5), (5, 5, 6)]
        graph = assembler.pred_graphs()[sample_id]
        assert len(graph.node_ids()) == 3
        assert frame_counts(assembler, sample_id) == {0: 1, 1: 2}


class TestEdgeIndexing:
    """Test D: with multiple cells per frame, persistent node IDs must
    correctly map transformer/greedy-assignment edge indices to the right
    canonical source and target graph nodes."""

    def test_multiple_cells_map_to_correct_canonical_nodes(self):
        assembler = PredictionGraphAssembler()
        sample_id = "sample_a"

        assembler.validate_window_order(sample_id, 0)
        src, src_coords, tgt, tgt_coords = assembler.process_window(
            sample_id, t_idx=0,
            peaks_t=[(1, 1, 1), (10, 10, 10)],
            peaks_t1=[(2, 2, 2), (20, 20, 20)],
        )
        # index 0 (coords (1,1,1)) -> index 1 (coords (20,20,20))
        # index 1 (coords (10,10,10)) -> index 0 (coords (2,2,2))
        assembler.add_edges(sample_id, src, tgt, edges=[(0, 1, 0.9), (1, 0, 0.8)])

        graph = assembler.pred_graphs()[sample_id]
        edge_list = graph.edge_list()
        assert sorted(edge_list) == sorted([[src[0], tgt[1]], [src[1], tgt[0]]])

        # Confirm coordinates line up with the IDs used.
        attrs = graph.node_attrs(attr_keys=["t", "x", "y", "z"])
        by_id = {nid: row for nid, row in zip(graph.node_ids(), attrs.to_dicts(), strict=True)}
        assert (by_id[src[0]]["z"], by_id[src[0]]["y"], by_id[src[0]]["x"]) == (1, 1, 1)
        assert (by_id[tgt[1]]["z"], by_id[tgt[1]]["y"], by_id[tgt[1]]["x"]) == (20, 20, 20)


class TestChronologicalContract:
    """Test E."""

    def test_normal_contiguous_windows_pass(self):
        assembler = PredictionGraphAssembler()
        assembler.validate_window_order("s1", 0)
        assembler.validate_window_order("s1", 1)
        assembler.validate_window_order("s1", 2)  # must not raise

    def test_duplicate_window_raises(self):
        assembler = PredictionGraphAssembler()
        assembler.validate_window_order("s1", 0)
        assembler.validate_window_order("s1", 1)
        with pytest.raises(RuntimeError, match="chronological window-order contract"):
            assembler.validate_window_order("s1", 1)

    def test_backward_window_raises(self):
        assembler = PredictionGraphAssembler()
        assembler.validate_window_order("s1", 3)
        with pytest.raises(RuntimeError, match="chronological window-order contract"):
            assembler.validate_window_order("s1", 2)

    def test_gapped_window_raises(self):
        assembler = PredictionGraphAssembler()
        assembler.validate_window_order("s1", 0)
        with pytest.raises(RuntimeError, match="chronological window-order contract"):
            assembler.validate_window_order("s1", 2)

    def test_error_identifies_sample_and_expected_actual_t_idx(self):
        assembler = PredictionGraphAssembler()
        assembler.validate_window_order("sample_xyz", 5)
        with pytest.raises(RuntimeError, match=r"sample_id=sample_xyz.*expected t_idx=6.*got t_idx=9"):
            assembler.validate_window_order("sample_xyz", 9)

    def test_first_window_may_start_at_any_t_idx(self):
        assembler = PredictionGraphAssembler()
        assembler.validate_window_order("s1", 17)  # must not raise
        assembler.validate_window_order("s1", 18)  # must not raise


class TestMultipleSamples:
    """Test F: interleaving two samples' windows must not create
    cross-sample node reuse or shared state, as long as each sample's OWN
    t_idx sequence stays contiguous."""

    def test_interleaved_samples_have_independent_graphs(self):
        assembler = PredictionGraphAssembler()

        assembler.validate_window_order("sample_a", 0)
        src_a0, _, tgt_a0, _ = assembler.process_window(
            "sample_a", t_idx=0, peaks_t=[(1, 1, 1)], peaks_t1=[(2, 2, 2)],
        )
        assembler.add_edges("sample_a", src_a0, tgt_a0, edges=[(0, 0, 1.0)])

        assembler.validate_window_order("sample_b", 0)
        src_b0, _, tgt_b0, _ = assembler.process_window(
            "sample_b", t_idx=0, peaks_t=[(1, 1, 1)], peaks_t1=[(2, 2, 2)],  # SAME coords as sample_a
        )
        assembler.add_edges("sample_b", src_b0, tgt_b0, edges=[(0, 0, 1.0)])

        # Interleave back to sample_a -- its own sequence (0 -> 1) stays contiguous.
        assembler.validate_window_order("sample_a", 1)
        src_a1, _, tgt_a1, _ = assembler.process_window(
            "sample_a", t_idx=1, peaks_t=[(2, 2, 2)], peaks_t1=[(3, 3, 3)],
        )
        assembler.add_edges("sample_a", src_a1, tgt_a1, edges=[(0, 0, 1.0)])

        graphs = assembler.pred_graphs()
        assert set(graphs.keys()) == {"sample_a", "sample_b"}
        assert len(graphs["sample_a"].node_ids()) == 3  # frames 0,1,2
        assert len(graphs["sample_b"].node_ids()) == 2  # frames 0,1 only

        # Independent graph OBJECTS (not just independent ID numbering --
        # each sample gets its own IndexedRXGraph instance, so processing
        # sample_b's windows can never mutate sample_a's graph).
        assert graphs["sample_a"] is not graphs["sample_b"]

        # sample_a's frame-1 canonical node must be reused for ITS OWN
        # window t_idx=1, not accidentally pulled from sample_b's identical
        # coordinates -- confirmed by sample_a ending up with the correct
        # 3-node chain despite sample_b sharing the same (1,1,1)/(2,2,2)
        # coordinates.
        frame_counts_a = {}
        for row in graphs["sample_a"].node_attrs(attr_keys=["t"]).to_dicts():
            frame_counts_a[row["t"]] = frame_counts_a.get(row["t"], 0) + 1
        assert frame_counts_a == {0: 1, 1: 1, 2: 1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
