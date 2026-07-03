"""
Positive/negative test suite for STHypergraphTracker (src/tracker.py).

Written against the CURRENT tracker.py (post-fix: flow-conservation equalities kept,
b_n+d_n<=1 constraint deliberately absent - see PROJECT.md Key Decisions and the
comment block in solve_lineage). These tests exist to:

1. Lock in correct behavior on realistic small scenarios (linear tracks, division,
   gap-closing, anisotropic pruning) - PRD FR-3 / TRACK-01..03.
2. Prevent regression of the ILP-infeasibility bug already found and fixed once
   this session (isolated/orphan nodes with no plausible neighbor must resolve as
   legitimate one-frame singletons, not make the whole solve infeasible - which
   silently produces ZERO edges anywhere in the graph, not just for the orphan).

Run: py -m pytest tests/test_tracker.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tracker import STHypergraphTracker

ANISOTROPY = np.array([4.0, 1.0, 1.0])  # real competition ratio, not the stale (5,1,1)


def zero_motion(centroids_by_t):
    """Convenience: build a motion_vectors_by_t dict of all-zero vectors matching centroids_by_t."""
    return {t: [[0.0, 0.0, 0.0] for _ in cs] for t, cs in centroids_by_t.items()}


# ---------------------------------------------------------------------------
# POSITIVE CASES - the tracker should link/split/skip correctly
# ---------------------------------------------------------------------------

class TestPositiveLinking:
    def test_simple_linear_track_links_across_two_frames(self):
        """One cell, barely moving, at t=0 and t=1 -> exactly one edge, no divisions."""
        centroids = {
            0: [[3.0, 45.0, 45.0]],
            1: [[3.1, 45.5, 45.5]],
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph.has_edge((0, 0), (1, 0))

    def test_two_independent_tracks_stay_separate(self):
        """Two well-separated cells at t=0/t=1 -> each links to its own nearest neighbor,
        never cross-linked."""
        centroids = {
            0: [[3.0, 10.0, 10.0], [3.0, 100.0, 100.0]],
            1: [[3.1, 11.0, 11.0], [3.1, 101.0, 101.0]],
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_edges() == 2
        assert graph.has_edge((0, 0), (1, 0))
        assert graph.has_edge((0, 1), (1, 1))
        # never cross-linked to the far cell
        assert not graph.has_edge((0, 0), (1, 1))
        assert not graph.has_edge((0, 1), (1, 0))

    def test_division_produces_split_with_two_daughters(self):
        """One parent cell at t=0 whose two daughters appear VERY close by at t=1 ->
        parent should have out-degree 2 (a division), not be forced to pick one.

        Distance matters here: the ILP's edge cost is distance^2 * gap_penalty, so
        cheap linking requires small displacements - see
        test_cost_scale_caps_realistic_link_distance for the general threshold.
        With birth=10, death=10 (defaults) and division_reward=-8, dividing into two
        daughters 2um away each is well inside the break-even distance (~4.4um for
        these costs), so the solver should prefer it over 3 isolated singletons."""
        centroids = {
            0: [[3.0, 50.0, 50.0]],
            1: [[3.0, 48.0, 50.0], [3.0, 52.0, 50.0]],  # 2um from parent each way
        }
        tracker = STHypergraphTracker(division_reward=-8.0)
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        parent = (0, 0)
        successors = list(graph.successors(parent))
        assert len(successors) == 2, f"expected a division (2 daughters), got {successors}"
        assert set(successors) == {(1, 0), (1, 1)}

    def test_gap_closing_skips_a_missing_frame(self):
        """Cell present at t=0 and t=2 but undetected at t=1 (e.g. dye fading) ->
        with max_gap_frames>=1, t=0 should link directly to t=2."""
        centroids = {
            0: [[3.0, 45.0, 45.0]],
            2: [[3.2, 46.0, 46.0]],
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=2)

        assert graph.has_edge((0, 0), (2, 0)), "gap-closing should bridge the missing t=1 frame"

    def test_motion_compensation_links_a_fast_moving_cell(self):
        """A cell with a real 20um raw displacement is too expensive to link on raw
        distance alone (edge cost 20^2=400 >> birth+death=20 for default costs), so
        with zero motion prediction the solver correctly prefers NOT linking. Once its
        predicted motion vector warps the coordinate down to ~2um from the t+1
        detection (edge cost 4 << 20), the same pair becomes cheap enough to link.

        NOTE: `prune_unphysical_edges` gates on RAW (unwarped) coordinates, not the
        motion-warped ones - confirmed from reading solve_lineage directly. This 20um
        raw jump must stay under the default prune gates (max_xy_micron=30 for
        gap=1) or it would be rejected before motion compensation ever gets a chance
        to help; that's a real, separate limitation worth knowing about once real
        motion vectors are wired in (TRACK-01) - a genuinely fast-moving real cell
        whose raw jump exceeds the prune gate can never be rescued by motion
        compensation, no matter how accurate the predicted vector is.
        """
        centroids = {
            0: [[3.0, 0.0, 0.0]],
            1: [[3.0, 20.0, 0.0]],  # 20um raw Y displacement (within the 30um prune gate)
        }

        # Without motion: raw distance 20um is too expensive to link (cost 400 vs 20 to stay isolated)
        tracker = STHypergraphTracker()
        graph_no_motion = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)
        assert not graph_no_motion.has_edge((0, 0), (1, 0)), (
            "without motion compensation, a 20um raw jump should be too costly to link "
            "given default birth/death costs - if this now links, the cost weighting changed"
        )

        # With an accurate motion vector: warped distance ~2um is cheap enough to link
        motion = {0: [[0.0, 18.0, 0.0]]}  # warps 0 -> 18, landing 2um from the detection at 20
        graph_with_motion = tracker.solve_lineage(centroids, motion, ANISOTROPY, max_gap_frames=0)
        assert graph_with_motion.has_edge((0, 0), (1, 0)), (
            "motion-warped coordinate (2um residual) should bring the pair into economical linking range"
        )


# ---------------------------------------------------------------------------
# NEGATIVE CASES - the tracker should correctly REJECT/handle bad or edge-case input
# ---------------------------------------------------------------------------

class TestNegativeCasesAndRegressions:
    def test_isolated_orphan_node_does_not_make_solve_infeasible(self):
        """REGRESSION GUARD for the ILP-infeasibility bug already found+fixed this session.

        Cell A at t=0 has no plausible neighbor anywhere (placed far from everything,
        beyond both the search radius and the anisotropic pruning gates). Cell B/B' at
        t=0/t=1 are a normal, easily-linkable pair.

        If the b_n+d_n<=1 constraint were ever reintroduced (the bug), the ENTIRE ILP
        becomes infeasible - CBC returns no solution, every y_var.varValue is None, and
        the resulting graph has ZERO edges everywhere, not just for A. So this test
        checks that B->B' still links, which is only possible if the solve succeeded
        globally despite A being unlinkable.
        """
        centroids = {
            0: [[3.0, 45.0, 45.0], [3.0, 5000.0, 5000.0]],  # B, then far-away orphan A
            1: [[3.1, 45.5, 45.5]],                            # only B' - nothing near A
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() >= 1, (
            "solve went infeasible (zero edges anywhere) - the b_n+d_n<=1 regression is back"
        )
        assert graph.has_edge((0, 0), (1, 0)), "the normal, easily-linkable pair must still link"
        # the orphan must be present as a node but with no edges (a legitimate one-frame singleton)
        orphan = (0, 1)
        assert graph.in_degree(orphan) == 0
        assert graph.out_degree(orphan) == 0

    def test_all_orphans_still_solves_to_zero_edges_not_a_crash(self):
        """Every node isolated (no candidates within range at all) -> valid result is
        just disconnected singleton nodes, and solve_lineage must not raise."""
        centroids = {
            0: [[3.0, 0.0, 0.0]],
            1: [[3.0, 9000.0, 9000.0]],
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 0

    def test_unphysical_z_jump_is_pruned_not_linked(self):
        """Two detections that are spatially close in XY but separated by an unphysical
        Z jump (exceeds max_z_micron * gap under the real anisotropy) must NOT be linked -
        prune_unphysical_edges should reject the candidate edge entirely."""
        # anisotropy z-scale = 4.0; max_z_micron default = 15.0 -> allowed_z = 15.0 for gap=1
        # a 10-voxel Z delta * 4.0 physical scale = 40 physical microns >> 15 allowed
        centroids = {
            0: [[0.0, 45.0, 45.0]],
            1: [[10.0, 45.0, 45.0]],
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert not graph.has_edge((0, 0), (1, 0)), "unphysical Z-jump should have been pruned"
        # both nodes should fall back to birth/death singletons, not force a bad link
        assert graph.number_of_edges() == 0

    def test_prune_unphysical_edges_respects_gap_scaling(self):
        """Direct unit test of the pruning helper: the same absolute Z displacement that
        is pruned at gap=1 must be ALLOWED at a larger gap (limits scale with gap)."""
        tracker = STHypergraphTracker()
        u = np.array([0.0, 45.0, 45.0])
        v = np.array([10.0, 45.0, 45.0])  # 10 voxels * 4.0 anisotropy = 40 physical microns Z

        assert tracker.prune_unphysical_edges(u, v, gap=1, anisotropy=ANISOTROPY) is True, (
            "40um Z jump in a single frame gap should be pruned (limit=15um for gap=1)"
        )
        assert tracker.prune_unphysical_edges(u, v, gap=3, anisotropy=ANISOTROPY) is False, (
            "the same jump over gap=3 frames (limit=45um) should be allowed"
        )

    def test_distance_beyond_search_radius_forces_birth_death_not_a_link(self):
        """Two detections within physical-distance pruning gates but beyond the 40um
        global search radius must not be linked."""
        centroids = {
            0: [[3.0, 0.0, 0.0]],
            1: [[3.0, 45.0, 0.0]],  # 45 physical microns > 40um radius
        }
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_edges() == 0

    def test_no_division_and_death_at_the_same_node(self):
        """Structural invariant: a node can never simultaneously be marked as dividing
        and dying (s_n + d_n <= 1). Build a scenario with an economically-favored
        division (see test_division_produces_split_with_two_daughters for the distance
        math) and verify the resulting graph is self-consistent (a dividing node has
        out-degree >= 2, meaning it did NOT also take the death path)."""
        centroids = {
            0: [[3.0, 50.0, 50.0]],
            1: [[3.0, 48.0, 50.0], [3.0, 52.0, 50.0]],  # 2um from parent each way
        }
        tracker = STHypergraphTracker(division_reward=-8.0)
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        parent = (0, 0)
        # if the solver had (incorrectly) also called this node "dead", it would have
        # zero successors instead of two - so out-degree==2 is itself the invariant check
        assert graph.out_degree(parent) == 2

    def test_empty_input_returns_empty_graph_without_crashing(self):
        """No detections at all -> solve_lineage must return a valid empty graph, not raise."""
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage({}, {}, ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_cost_scale_caps_realistic_link_distance(self):
        """DOCUMENTS A REAL CALIBRATION FINDING (relevant to Phase 4 tracker-cost tuning
        and to anyone wiring real motion vectors in Phase 2/TRACK-01):

        A single edge only "breaks even" against 2 isolated birth+death singletons when
        distance^2 * gap_penalty < birth_cost + death_cost (derived from the flow-
        conservation constraints: linking saves one birth_cost payment on the downstream
        node but adds the edge cost, while the upstream node pays birth regardless since
        it's the first frame). With the ACTUAL production costs used in
        run_pipeline.py's `STHypergraphTracker(birth_cost=15.0, death_cost=15.0, ...)`,
        that threshold is distance < sqrt(30) = 5.48 microns per frame (gap=1).

        Real embryonic cell nuclei can plausibly move faster than 5.48um/frame during
        active developmental stages - if so, the CURRENT default costs would make the
        ILP systematically prefer birth+death singletons over correct links for any
        such cell, silently capping edge_jaccard recall regardless of how good detection
        is. This should be an explicit, measured calibration target in Phase 4 (or
        earlier, if Phase 1's baseline-parity score comes in surprisingly low and this
        is a plausible cause), not left as an implicit default.
        """
        production_birth = 15.0
        production_death = 15.0
        breakeven_distance = np.sqrt(production_birth + production_death)  # ~5.477

        centroids_just_under = {
            0: [[3.0, 0.0, 0.0]],
            1: [[3.0, breakeven_distance - 0.5, 0.0]],
        }
        centroids_just_over = {
            0: [[3.0, 0.0, 0.0]],
            1: [[3.0, breakeven_distance + 0.5, 0.0]],
        }
        tracker = STHypergraphTracker(birth_cost=production_birth, death_cost=production_death)

        graph_under = tracker.solve_lineage(
            centroids_just_under, zero_motion(centroids_just_under), ANISOTROPY, max_gap_frames=0
        )
        graph_over = tracker.solve_lineage(
            centroids_just_over, zero_motion(centroids_just_over), ANISOTROPY, max_gap_frames=0
        )

        assert graph_under.has_edge((0, 0), (1, 0)), (
            f"a {breakeven_distance - 0.5:.2f}um jump (just under the break-even "
            f"distance {breakeven_distance:.2f}um) should still link"
        )
        assert not graph_over.has_edge((0, 0), (1, 0)), (
            f"a {breakeven_distance + 0.5:.2f}um jump (just over the break-even "
            f"distance {breakeven_distance:.2f}um) should NOT link under production costs - "
            f"if this now links, either the cost function or birth/death defaults changed"
        )

    def test_single_timepoint_all_nodes_are_singletons(self):
        """Only one timepoint exists (nothing to link to, ever) -> every node must
        resolve as an isolated birth+death singleton, and solve_lineage must not crash
        or go infeasible just because every node is simultaneously unexplainable in
        both directions."""
        centroids = {0: [[3.0, 10.0, 10.0], [3.0, 90.0, 90.0]]}
        tracker = STHypergraphTracker()
        graph = tracker.solve_lineage(centroids, zero_motion(centroids), ANISOTROPY, max_gap_frames=0)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
