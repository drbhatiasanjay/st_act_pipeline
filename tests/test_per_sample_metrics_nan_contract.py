"""
Regression test: per_sample_metrics() NaN contract.

dry_run_investigation.py calls per_sample_metrics(er, float("nan"), float("nan"))
to signal that no valid estimated_number_of_nodes or node_recall denominator is
available for a partial-window evaluation. This test guards the contract:
  - adj_edge_jaccard must be NaN (not raise, not silently return 0 or a
    spurious value derived from a NaN arithmetic shortcut).

Without this guard, a metrics.py refactor that changes NaN propagation could
silently break dry_run_investigation.py's adj_J=N/A guarantee.
"""
import math

import polars as pl
import tracksdata as td

from src.tracking_cellmot import evaluate, per_sample_metrics


def _two_node_graph():
    g = td.graph.IndexedRXGraph()
    for key in ("z", "y", "x"):
        try:
            g.add_node_attr_key(key, pl.Float64, 0.0)
        except ValueError:
            pass
    g.add_node({"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    g.add_node({"t": 1, "z": 0.0, "y": 0.0, "x": 0.0})
    g.add_edge(0, 1, {})
    return g


class TestPerSampleMetricsNanContract:
    def test_nan_n_total_yields_nan_adj_edge_jaccard(self):
        """n_total=float("nan") -> adj_edge_jaccard must be NaN.

        This is the API contract used by dry_run_investigation.py for any
        evaluation window where estimated_number_of_nodes is unavailable.
        Passing NaN must propagate cleanly rather than raising or returning
        a spurious numeric value."""
        g = _two_node_graph()
        er = evaluate(g, g, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)
        result = per_sample_metrics(er, float("nan"), float("nan"))
        assert math.isnan(result["adj_edge_jaccard"]), (
            f"Expected adj_edge_jaccard=NaN for n_total=nan, "
            f"got {result['adj_edge_jaccard']!r}"
        )

    def test_valid_n_total_yields_finite_adj_edge_jaccard(self):
        """Sanity check: a real n_total must produce a finite, non-NaN value."""
        g = _two_node_graph()
        er = evaluate(g, g, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)
        result = per_sample_metrics(er, 1000, float("nan"))
        assert not math.isnan(result["adj_edge_jaccard"]), (
            f"Expected finite adj_edge_jaccard for n_total=1000, "
            f"got {result['adj_edge_jaccard']!r}"
        )
        assert result["adj_edge_jaccard"] >= 0.0
