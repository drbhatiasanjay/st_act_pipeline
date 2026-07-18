"""
P0-7A: metric-parity regression tests.

Proves two things, and only two things:

A. Our vendored src/tracking_cellmot/{metrics,division_metrics}.py now match
   the pinned upstream commit royerlab/kaggle-cell-tracking-competition
   @ 075fc5f5a52d11077f9dc2b074644618f26939e2 (PR #2, "Updating metric to
   patch weakly connected component exploit") -- via ported upstream test
   cases (Section A) and the exact drift cases the read-only audit
   demonstrated (Section B).
B. The P0-4 wrapper (src/evaluation.py's evaluate_submission()) is
   unaffected -- it consumes only the stable EvaluationResult/DivisionCounts
   shapes, never score_divisions()'s DivisionScores return type directly
   (Section C proves this by repository-search evidence plus a live
   end-to-end smoke check; Section D re-runs a P0-4 aggregation scenario
   unchanged).

All graphs are synthetic, in-memory, offline -- no network access, no live
GitHub imports. Upstream behavior is vendored as literal expected values in
this file (derived once, this session, from a read-only differential
reproducer run against real fetched upstream source -- see the P0-7A patch
report for that derivation), not fetched at test time.
"""
import ast
import copy
from pathlib import Path

import polars as pl
import tracksdata as td

from src.tracking_cellmot.division_metrics import (
    DivisionCounts,
    DivisionScores,
    _is_strongly_connected_division,
    evaluate_divisions,
    extract_divisions,
    score_divisions,
)
from src.tracking_cellmot.metrics import evaluate


def _build_graph(nodes: dict, edges: list) -> tuple[td.graph.InMemoryGraph, dict[str, int]]:
    """Build an InMemoryGraph, returning (graph, name->node_id mapping).

    Matches the upstream test suite's own helper exactly (same construction
    pattern) so ported cases need no adaptation beyond the import path.
    """
    g = td.graph.InMemoryGraph()
    g.add_node_attr_key("z", pl.Float64, 0.0)
    g.add_node_attr_key("y", pl.Float64, 0.0)
    g.add_node_attr_key("x", pl.Float64, 0.0)
    ids = {}
    for name, attrs in nodes.items():
        ids[name] = g.add_node(attrs=copy.deepcopy(attrs))
    for src, tgt in edges:
        g.add_edge(ids[src], ids[tgt], {})
    return g, ids


# ---------------------------------------------------------------------------
# score_divisions() caller scanner -- pure filesystem + AST, independent of
# git tracked/staged/committed/untracked state (no git commands used at all,
# see TestDivisionScoresReturnTypeCompatibility for why this matters: a
# git-grep-based version would only see TRACKED files, so this test file's
# own score_divisions( occurrences would be invisible pre-commit and
# suddenly "found" post-commit, flipping a passing test to failing purely
# from a commit happening -- a verification-state bug, not a real caller
# change).
# ---------------------------------------------------------------------------

_ALLOWED_SCORE_DIVISIONS_CALLER_FILES = frozenset({
    "src/tracking_cellmot/division_metrics.py",
    "kaggle_src_dataset/src/tracking_cellmot/division_metrics.py",
})

_SCORE_DIVISIONS_SCAN_EXCLUDED_DIR_NAMES = frozenset({
    ".git", "__pycache__", "tests", "data", ".pytest_cache",
    "checkpoints_smoke_test", "checkpoint_dataset", "data_splits",
})


def _find_score_divisions_calls_in_source(source: str, filename: str = "<source>") -> list[int]:
    """Parse source with ast and return the line number of every ast.Call
    node whose callee is named (bare or attribute-qualified)
    score_divisions. A bare `def score_divisions(...)` is not an ast.Call
    and is therefore never matched -- only actual invocations are."""
    tree = ast.parse(source, filename=filename)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "score_divisions":
            lines.append(node.lineno)
    return lines


def _find_score_divisions_callers_in_repo(repo_root: Path) -> dict[str, list[int]]:
    """Walk every *.py file under repo_root on disk (rglob -- plain
    filesystem traversal, not git-aware in any way) and AST-scan each for
    score_divisions( calls. Excludes tests/ and common cache/build/data
    directories. Result is identical regardless of whether any file is
    tracked, staged, committed, or untracked -- it never invokes git."""
    results: dict[str, list[int]] = {}
    for path in sorted(repo_root.rglob("*.py")):
        rel_parts = path.relative_to(repo_root).parts
        if any(part in _SCORE_DIVISIONS_SCAN_EXCLUDED_DIR_NAMES for part in rel_parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            lines = _find_score_divisions_calls_in_source(source, filename=str(path))
        except SyntaxError:
            continue
        if lines:
            results["/".join(rel_parts)] = lines
    return results


_GT_NODES = {
    "P":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
    "D":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
    "C1": {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
    "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
    "G1": {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
    "G2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
}
_GT_EDGES = [("P", "D"), ("D", "C1"), ("D", "C2"), ("C1", "G1"), ("C2", "G2")]


def _make_gt():
    return _build_graph(_GT_NODES, _GT_EDGES)


# ---------------------------------------------------------------------------
# A. Upstream-ported cases (royerlab/kaggle-cell-tracking-competition
#    @ 075fc5f, tests/test_division_metrics.py). Assertions preserved
#    verbatim from upstream -- not weakened.
# ---------------------------------------------------------------------------

class TestUpstreamPortedExtractDivisions:
    def test_no_divisions(self):
        g, _ = _build_graph(
            {"A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0}},
            [("A", "B"), ("B", "C")],
        )
        assert extract_divisions(g) == {}

    def test_single_division(self):
        g, ids = _build_graph(
            {"A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "B":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
             "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
             "D":  {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
             "E":  {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0}},
            [("A", "B"), ("B", "C1"), ("B", "C2"), ("C2", "D"), ("C1", "E")],
        )
        divs = extract_divisions(g)
        assert list(divs.keys()) == [ids["B"]]
        sub = divs[ids["B"]]
        assert sub.num_nodes() == 6
        assert sub.num_edges() == 5


class TestUpstreamPortedStronglyConnectedDivision:
    """These functions did not exist in our old vendored file at all --
    their mere presence and pass is itself parity evidence."""

    def test_accepts_local_division_window(self):
        pred, ids = _build_graph(
            {"GP": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "P":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1": {"t": 2, "z": 0.0, "y": 1.0, "x": 0.0},
             "C2": {"t": 2, "z": 0.0, "y": -1.0, "x": 0.0},
             "G2": {"t": 3, "z": 0.0, "y": -1.0, "x": 0.0}},
            [("GP", "P"), ("P", "C1"), ("P", "C2"), ("C2", "G2")],
        )
        assert _is_strongly_connected_division(pred, ids["P"], {ids["GP"]}, [{ids["C1"]}, {ids["G2"]}])

    def test_rejects_great_grandchild_match(self):
        pred, ids = _build_graph(
            {"P":   {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1":  {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
             "C2":  {"t": 1, "z": 0.0, "y": -1.0, "x": 0.0},
             "G2":  {"t": 2, "z": 0.0, "y": -1.0, "x": 0.0},
             "GG2": {"t": 3, "z": 0.0, "y": -1.0, "x": 0.0}},
            [("P", "C1"), ("P", "C2"), ("C2", "G2"), ("G2", "GG2")],
        )
        assert not _is_strongly_connected_division(pred, ids["P"], {ids["P"]}, [{ids["C1"]}, {ids["GG2"]}])

    def test_requires_distinct_pred_daughter_lineages(self):
        pred, ids = _build_graph(
            {"P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "C": {"t": 1, "z": 0.0, "y": 1.0, "x": 0.0},
             "G": {"t": 2, "z": 0.0, "y": 1.0, "x": 0.0},
             "X": {"t": 1, "z": 0.0, "y": 20.0, "x": 0.0}},
            [("P", "C"), ("P", "X"), ("C", "G")],
        )
        assert not _is_strongly_connected_division(pred, ids["P"], {ids["P"]}, [{ids["C"]}, {ids["G"]}])


class TestUpstreamPortedScoreDivisions:
    def test_perfect_prediction(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        result = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 1 for v in result.scores.values())
        assert len(result.tp_forks) == 1
        assert result.fp_forks == set()

    def test_rejected_local_fork_is_false_positive(self):
        """The exact exploit case named in the upstream PR: a dummy branch
        plus a wrong-parent grandchild match must NOT be credited as TP."""
        gt, gt_ids = _make_gt()
        pred, pred_ids = _build_graph(
            {"P":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "D":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
             "X":  {"t": 2, "z": 0.0, "y": 50.0, "x": 0.0},
             "G2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0}},
            [("P", "D"), ("D", "C1"), ("D", "X"), ("C1", "G2")],
        )
        result = score_divisions(pred, gt, max_distance=1.0)
        assert result.scores[gt_ids["D"]] == 0
        assert result.tp_forks == set()
        assert result.fp_forks == {pred_ids["D"]}
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=0, fn=1, fp=1)


# ---------------------------------------------------------------------------
# B. Required P0-7A differential regressions (11 cases). Each asserts the
#    CURRENT (post-sync) local implementation reproduces the upstream
#    result the audit's reproducer already established by direct execution.
# ---------------------------------------------------------------------------

class TestP07ARequiredDifferentials:
    def test_01_no_divisions(self):
        nodes = {"A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                 "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                 "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0}}
        edges = [("A", "B"), ("B", "C")]
        pred, _ = _build_graph(nodes, edges)
        gt, _ = _build_graph(nodes, edges)
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=0, fn=0, fp=0)

    def test_02_one_correct_division(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=1, fn=0, fp=0)

    def test_03_one_missed_division(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {"P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "D": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
             "G1": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0}},
            [("P", "D"), ("D", "C1"), ("C1", "G1")],
        )
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=0, fn=1, fp=0)

    def test_04_false_positive_division_exploit_case(self):
        """THE confirmed drift case. Old local behavior (pre-P0-7A) was
        effectively tp=1,fp=0,fn=0 (exploited). Post-sync must be
        tp=0,fp=1,fn=1 -- exact upstream-correct values, not just a
        floating-score comparison."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {"P":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "D":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
             "X":  {"t": 2, "z": 0.0, "y": 50.0, "x": 0.0},
             "G2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0}},
            [("P", "D"), ("D", "C1"), ("D", "X"), ("C1", "G2")],
        )
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert counts == DivisionCounts(tp=0, fn=1, fp=1)
        denom = counts.tp + counts.fp + counts.fn
        assert counts.tp / denom == 0.0  # division_jaccard for this sample

    def test_05_correct_parent_wrong_daughters(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {"P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
             "D": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
             "Y1": {"t": 2, "z": 0.0, "y": 50.0, "x": 0.0},
             "Y2": {"t": 2, "z": 0.0, "y": -50.0, "x": 0.0}},
            [("P", "D"), ("D", "Y1"), ("D", "Y2")],
        )
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=0, fn=1, fp=1)

    def test_06_wrong_parent_correct_looking_daughters(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {"Q":  {"t": 0, "z": 0.0, "y": 100.0, "x": 0.0},
             "D2": {"t": 1, "z": 0.0, "y": 100.0, "x": 0.0},
             "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
             "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0}},
            [("Q", "D2"), ("D2", "C1"), ("D2", "C2")],
        )
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=0, fn=1, fp=0)

    def test_07_multiple_divisions_both_correct(self):
        nodes = {
            "P1": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "D1": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            "C1a": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
            "C1b": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
            "P2": {"t": 0, "z": 0.0, "y": 100.0, "x": 0.0},
            "D2": {"t": 1, "z": 0.0, "y": 100.0, "x": 0.0},
            "C2a": {"t": 2, "z": 0.0, "y": 105.0, "x": 0.0},
            "C2b": {"t": 2, "z": 0.0, "y": 95.0, "x": 0.0},
        }
        edges = [("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                 ("P2", "D2"), ("D2", "C2a"), ("D2", "C2b")]
        gt, _ = _build_graph(nodes, edges)
        pred, _ = _build_graph(nodes, edges)
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=2, fn=0, fp=0)

    def test_08_zero_node_zero_edge_prediction(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph({}, [])
        assert evaluate_divisions(pred, gt, max_distance=1.0) == DivisionCounts(tp=0, fn=1, fp=0)

    def test_09_three_predicted_children_gt_has_two(self):
        """Out-degree cap: OLD local edge_jaccard ~0.6667 (extra edge
        penalized as FP). NEW (post-sync) must be exactly 1.0 -- the
        biologically-invalid 3rd edge is dropped before FP counting."""
        gt_nodes = {"P": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                    "C1": {"t": 1, "z": 0.0, "y": 5.0, "x": 0.0},
                    "C2": {"t": 1, "z": 0.0, "y": -5.0, "x": 0.0}}
        gt_edges = [("P", "C1"), ("P", "C2")]
        pred_nodes = dict(gt_nodes)
        pred_nodes["C3"] = {"t": 1, "z": 0.0, "y": 15.0, "x": 0.0}
        pred_edges = gt_edges + [("P", "C3")]

        gt, _ = _build_graph(gt_nodes, gt_edges)
        pred, _ = _build_graph(pred_nodes, pred_edges)
        result = evaluate(pred, gt, max_distance=1.0)
        assert (result.edge_tp, result.edge_fp, result.edge_fn) == (2, 0, 0)
        assert result.edge_tp / (result.edge_tp + result.edge_fp + result.edge_fn) == 1.0

    def test_10_non_consecutive_frame_predicted_edge_dropped(self):
        """An edge spanning more than one timestep (t=0 -> t=2, skipping
        t=1) must be filtered out entirely before TP/FP counting -- neither
        rewarded nor penalized, simply excluded."""
        gt_nodes = {"A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                    "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                    "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0}}
        gt_edges = [("A", "B"), ("B", "C")]
        # Pred matches A and C exactly but links them directly, skipping B's
        # timestep -- a genuine (t_target - t_source == 2) edge.
        pred_nodes = {"A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                      "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0}}
        pred_edges = [("A", "C")]

        gt, _ = _build_graph(gt_nodes, gt_edges)
        pred, _ = _build_graph(pred_nodes, pred_edges)
        result = evaluate(pred, gt, max_distance=1.0)
        # The non-consecutive edge must not be counted as a TP (there is no
        # matching single-step GT edge for it) and must not inflate FP either
        # -- it is dropped by the consecutive-frame filter before counting.
        assert result.edge_tp == 0
        assert result.edge_fp == 0

    def test_11_duplicate_predicted_edges_collapse_to_one_gt_edge(self):
        """Two predicted nodes both matching the SAME GT node, each with an
        outgoing edge that maps onto the same single GT edge, must collapse
        to at most one counted match -- not double-count the intersection."""
        gt_nodes = {"A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                    "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0}}
        gt_edges = [("A", "B")]
        # Two near-duplicate predicted source nodes, both within match
        # radius of GT's A, each linking to a node matching GT's B.
        pred_nodes = {
            "A1": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
            "A2": {"t": 0, "z": 0.0, "y": 0.05, "x": 0.0},
            "B1": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        }
        pred_edges = [("A1", "B1"), ("A2", "B1")]

        gt, _ = _build_graph(gt_nodes, gt_edges)
        pred, _ = _build_graph(pred_nodes, pred_edges)
        result = evaluate(pred, gt, max_distance=1.0)
        # At most 1 TP -- the duplicate mapping onto the same GT edge must
        # not both count as true positives.
        assert result.edge_tp <= 1


# ---------------------------------------------------------------------------
# C. score_divisions() return-type compatibility.
# ---------------------------------------------------------------------------

class TestDivisionScoresReturnTypeCompatibility:
    def test_return_type_is_division_scores_namedtuple(self):
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        result = score_divisions(pred, gt, max_distance=1.0)
        assert isinstance(result, DivisionScores)
        assert hasattr(result, "scores") and isinstance(result.scores, dict)
        assert hasattr(result, "tp_forks") and isinstance(result.tp_forks, set)
        assert hasattr(result, "fp_forks") and isinstance(result.fp_forks, set)

    def test_evaluate_divisions_still_returns_division_counts(self):
        """The stable public shape consumed by metrics.evaluate() and, one
        layer further out, src/evaluation.py -- must be unchanged."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        result = evaluate_divisions(pred, gt, max_distance=1.0)
        assert isinstance(result, DivisionCounts)
        assert result._fields == ("tp", "fn", "fp")

    def test_score_divisions_call_scanner_detects_external_call(self):
        """Negative control, required by the P0-7A v2 review: prove the AST
        scanner actually detects a score_divisions( call when one exists,
        using a synthetic in-memory source string. Never touches a real
        production file -- the "external caller" here is fabricated text
        parsed directly by ast.parse()."""
        synthetic_source = (
            "from src.tracking_cellmot.division_metrics import score_divisions\n"
            "\n"
            "def not_allowed_caller(pred, gt):\n"
            "    return score_divisions(pred, gt)\n"
        )
        lines = _find_score_divisions_calls_in_source(synthetic_source, filename="<synthetic>")
        assert lines == [4]

    def test_score_divisions_call_scanner_ignores_bare_definition(self):
        """The function definition line itself must never be mistaken for a
        call -- def statements are not ast.Call nodes."""
        synthetic_source = (
            "def score_divisions(pred_graph, gt_graph, scale=None, max_distance=7.0):\n"
            "    return None\n"
        )
        lines = _find_score_divisions_calls_in_source(synthetic_source, filename="<synthetic>")
        assert lines == []

    def test_no_external_caller_of_score_divisions_other_than_evaluate_divisions(self):
        """Pure filesystem + AST repository scan (see
        _find_score_divisions_callers_in_repo) -- deliberately independent
        of git tracked/staged/committed/untracked state, no git commands
        used. Confirms every real Call to score_divisions() anywhere in
        production source lives in exactly the two allowed
        division_metrics.py mirrors (the one expected internal call inside
        evaluate_divisions() in each), so the DivisionScores return-type
        change cannot break any external caller -- and this result is
        identical whether this very test file is untracked (fresh
        `git apply`), staged, or committed, since tests/ is excluded from
        the scan by directory name, not by git state."""
        repo_root = Path(__file__).resolve().parents[1]
        callers = _find_score_divisions_callers_in_repo(repo_root)
        assert callers, "expected to find score_divisions( call sites in the repo"

        offenders = {
            path: lines for path, lines in callers.items()
            if path not in _ALLOWED_SCORE_DIVISIONS_CALLER_FILES
        }
        assert offenders == {}, f"unexpected external caller(s) of score_divisions(): {offenders}"

        for allowed_path in _ALLOWED_SCORE_DIVISIONS_CALLER_FILES:
            found = callers.get(allowed_path, [])
            assert len(found) == 1, (
                f"{allowed_path}: expected exactly 1 internal score_divisions( call "
                f"(inside evaluate_divisions()), found {found}"
            )

    def test_evaluate_end_to_end_still_produces_evaluation_result(self):
        """metrics.evaluate() (the function src/evaluation.py actually
        imports and calls) still returns the unchanged EvaluationResult
        shape after the sync."""
        from src.tracking_cellmot.metrics import EvaluationResult

        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        result = evaluate(pred, gt, max_distance=1.0)
        assert isinstance(result, EvaluationResult)
        assert result._fields == (
            "edge_tp", "edge_fp", "edge_fn",
            "division_tp", "division_fp", "division_fn", "num_pred_nodes",
        )


# ---------------------------------------------------------------------------
# D. P0-4 wrapper smoke check (full P0-4 suite run separately; this is a
#    minimal in-file confirmation that evaluate_submission() still works
#    end-to-end against the newly-synced primitive).
# ---------------------------------------------------------------------------

class TestP04WrapperSmokeUnaffected:
    def test_evaluate_submission_end_to_end_with_synced_metric(self):
        from src.evaluation import evaluate_submission

        class _FakeMetadata:
            def __init__(self, n):
                self.extra = {"estimated_number_of_nodes": n}

        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        result = evaluate_submission(
            pred_graphs={"s1": pred}, gt_graphs={"s1": gt},
            scale=None, max_distance=1.0,
            gt_metadata={"s1": _FakeMetadata(6)},
        )
        assert result["num_datasets"] == 1
        assert result["division_jaccard"] == 1.0
        assert 0.0 <= result["score"] <= 1.1
