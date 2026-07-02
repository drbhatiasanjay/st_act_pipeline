# Host Reference Implementation — royerlab/kaggle-cell-tracking-competition

**Found:** 2026-07-03, while researching PRD.md §11's open question ("does the host publish a
reference metric implementation?"). Answer: **yes** — linked directly from the competition's
`/overview/evaluation` page: `https://github.com/royerlab/kaggle-cell-tracking-competition`.

**This resolves PRD.md §11 completely and should change how FR-1 and FR-5 get implemented** — do
not hand-roll a `.geff` reader or a from-scratch Jaccard scorer. Depend on `tracksdata` (the
library this reference repo is built on) and vendor/wrap its actual `metrics.py` /
`division_metrics.py` / `io.py`, so our local validation is *guaranteed* to match what Kaggle
computes, instead of risking a subtly-wrong reimplementation.

Repo layout:
```
royerlab/kaggle-cell-tracking-competition/
├── metrics.md                          # human-readable metric spec (exact formulas below)
├── scripts/
│   ├── evaluate.py                     # CLI evaluation entrypoint
│   ├── train_unet_transformer.py       # host's own reference model - U-Net + Transformer, not a plain CNN
│   └── predict_unet_transformer.py
└── src/tracking_cellmot/
    ├── __init__.py
    ├── metrics.py                      # edge_jaccard, adjusted_edge_jaccard, evaluate(), evaluate_datasets(), summarise()
    ├── division_metrics.py             # evaluate_divisions() - the real division-matching algorithm
    ├── img_proc.py                     # (not yet fetched)
    ├── io.py                           # open_dataset(), save_graph(), list_datasets() - real data loading pattern
    └── models/                         # (not yet fetched)
```

## 1. Exact metric formulas (from `metrics.md`, confirmed against `metrics.py` source)

**Edge Jaccard:** `TP / (TP + FP + FN)`, via a two-stage process:
1. Node matching: optimal bipartite assignment between predicted and GT nodes, max centroid
   distance **7 µm**.
2. Edge matching: a predicted edge is TP if both endpoints match GT nodes connected by a GT edge.
   FP if (a) target matches a GT node connected to a *different* source, or (b) source matches a
   GT node connected to a *different* target.

**Adjusted Edge Jaccard** (the actual scored quantity):
```
adjusted_jaccard = max(0, jaccard · (1 − α · (T_pred − T_true) / T_true))
```
`α = 0.1` (`ADJUSTMENT_ALPHA` in code). `T_true` = the `.geff` metadata's
`extra.estimated_number_of_nodes` field (confirmed this is exactly what that field is for — see
`data/staging/README.md`). `T_pred` = total predicted nodes, *regardless of whether they matched*.

**Division Jaccard:** `TP / (TP + FP + FN)` for division events, computed by
`evaluate_divisions()` — see §2, it's not a simple out-degree check.

**Final score:** `adjusted_edge_jaccard + w · division_jaccard`, `w = 0.1`
(`SCORE_DIVISION_WEIGHT`), **micro-averaged**: per-sample TP/FP/FN are summed across every
video/dataset *before* computing the ratio (see `evaluate_datasets()` / `summarise()` below) — not
a simple mean of per-sample scores. If a split has zero divisions anywhere, the division term is
dropped entirely and `score = edge_jaccard` (not `+ 0`).

**Sparse ground truth handling (confirmed, resolves the concern in `data/staging/README.md`):**
predicted nodes that don't match any GT node are **not** counted as false positives, and predicted
divisions in unannotated regions are ignored. This is enforced structurally in the matching code
(only GT-adjacent regions ever enter the FP tally), not via a fudge factor.

## 2. `src/tracking_cellmot/metrics.py` (full source, fetched verbatim)

Key entry points:
- `evaluate(graph, gt_graph, scale=None, max_distance=7.0) -> EvaluationResult` — scores one
  (prediction, ground-truth) pair, returns raw TP/FP/FN counts for edges and divisions plus
  `num_pred_nodes`.
- `evaluate_datasets(graph_pairs, scale=None, max_distance=7.0) -> DatasetsResult` — runs
  `evaluate()` over every (pred, gt) pair, **sums counts across all pairs first**, then computes
  micro-averaged `edge_jaccard`, `division_jaccard`, and the combined `score`.
- `per_sample_metrics()` / `summarise()` — the exact weighting scheme: `adj_edge_jaccard` is
  weight-averaged per sample by `w_i = TP_i + FP_i + FN_i` (matches the PRD's "weighted by
  (TP+FP+FN)" description exactly).

```python
import warnings
from typing import Literal, NamedTuple

import polars as pl
import tracksdata as td


class EvaluationResult(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int


class DatasetsResult(NamedTuple):
    edge_jaccard: float
    division_jaccard: float
    score: float


ADJUSTMENT_ALPHA: float = 0.1        # J_adj = max(0, J * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
SCORE_DIVISION_WEIGHT: float = 0.1   # score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard

COUNT_COLUMNS: tuple[str, ...] = (
    "edge_tp", "edge_fp", "edge_fn",
    "division_tp", "division_fp", "division_fn",
    "num_pred_nodes",
)
METRIC_COLUMNS: tuple[str, ...] = COUNT_COLUMNS + (
    "node_recall", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard",
)


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def evaluate(
    graph: "td.graph.BaseGraph",
    gt_graph: "td.graph.BaseGraph",
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> EvaluationResult:
    """Evaluate a predicted graph against ground truth using centroid-distance node matching.
    Uses tracksdata's DistanceMatching for the 7um-gated bipartite node match, then compares
    edge sets and delegates division scoring to division_metrics.evaluate_divisions()."""
    from .division_metrics import evaluate_divisions

    _evaluate(graph, gt_graph, "jaccard", scale, max_distance)  # performs graph.match() in place

    if graph.num_edges() == 0:
        edge_tp, edge_fp, edge_fn = 0, 0, gt_graph.num_edges()
    else:
        edge_attrs = _evaluate_matched_graph(graph, gt_graph)
        edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
        edge_valid_pred = int(edge_attrs["pred_valid"].sum())
        edge_fp = edge_valid_pred - edge_tp
        edge_fn = gt_graph.num_edges() - edge_tp

    div = evaluate_divisions(graph, gt_graph, scale=scale, max_distance=max_distance)

    return EvaluationResult(
        edge_tp=edge_tp, edge_fp=edge_fp, edge_fn=edge_fn,
        division_tp=div.tp, division_fp=div.fp, division_fn=div.fn,
        num_pred_nodes=graph.num_nodes(),
    )


def evaluate_datasets(
    graph_pairs: list[tuple["td.graph.BaseGraph", "td.graph.BaseGraph"]],
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DatasetsResult:
    """Micro-averaged: sums TP/FP/FN across ALL pairs before computing the Jaccard ratio."""
    edge_tp = edge_fp = edge_fn = 0
    div_tp = div_fp = div_fn = 0
    for pred, gt in graph_pairs:
        r = evaluate(pred, gt, scale=scale, max_distance=max_distance)
        edge_tp += r.edge_tp; edge_fp += r.edge_fp; edge_fn += r.edge_fn
        div_tp += r.division_tp; div_fp += r.division_fp; div_fn += r.division_fn

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    has_divisions = (div_tp + div_fp + div_fn) > 0
    division_jaccard = _jaccard(div_tp, div_fp, div_fn) if has_divisions else float("nan")
    score = edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard if has_divisions else edge_jaccard

    return DatasetsResult(edge_jaccard=edge_jaccard, division_jaccard=division_jaccard, score=score)


def per_sample_metrics(er: EvaluationResult, n_total: float, node_recall: float) -> dict:
    """n_total = the GEFF `estimated_number_of_nodes` metadata value for this sample."""
    if n_total > 0:
        total_node_ratio = (er.num_pred_nodes - n_total) / n_total
    else:
        total_node_ratio = float("nan")

    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float("nan")
    if edge_jaccard == edge_jaccard and total_node_ratio == total_node_ratio:
        adj_edge_jaccard = max(0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
    else:
        adj_edge_jaccard = float("nan")

    return {
        "edge_tp": er.edge_tp, "edge_fp": er.edge_fp, "edge_fn": er.edge_fn,
        "division_tp": er.division_tp, "division_fp": er.division_fp, "division_fn": er.division_fn,
        "num_pred_nodes": er.num_pred_nodes, "node_recall": node_recall,
        "total_node_ratio": total_node_ratio,
        "edge_jaccard": edge_jaccard, "adj_edge_jaccard": adj_edge_jaccard,
    }


def summarise(rows: list[dict]) -> dict:
    """Run-level aggregation: edge/division Jaccard micro-averaged (counts summed first);
    adj_edge_jaccard is weight-averaged per sample by w_i = TP_i+FP_i+FN_i."""
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    if not valid:
        return {"n": 0, "edge_jaccard": float("nan"), "division_jaccard": float("nan"),
                "adj_edge_jaccard": float("nan"), "n_adj": 0, "score": float("nan")}
    totals = {c: sum(r[c] for r in valid) for c in COUNT_COLUMNS}

    adj_rows = [r for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    weights = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    total_w = sum(weights)
    adj_edge_jaccard = (
        sum(w * r["adj_edge_jaccard"] for w, r in zip(weights, adj_rows)) / total_w
        if total_w > 0 else float("nan")
    )

    division_total = totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    if division_total == 0:
        division_jaccard = float("nan")
        score = adj_edge_jaccard
    else:
        division_jaccard = _jaccard(totals["division_tp"], totals["division_fp"], totals["division_fn"])
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard

    return {
        "n": len(valid),
        "edge_jaccard": _jaccard(totals["edge_tp"], totals["edge_fp"], totals["edge_fn"]),
        "division_jaccard": division_jaccard,
        "adj_edge_jaccard": adj_edge_jaccard, "n_adj": len(adj_rows),
        "score": score,
    }
```
*(Helper internals `_evaluate`, `_evaluate_matched_graph`, `_compute_score`, `node_recall`,
`nan_metrics_row` omitted here for brevity — re-fetch
`raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/main/src/tracking_cellmot/metrics.py`
for the complete file if implementing directly, or just `pip install tracksdata` and import the
package if it ships this module.)*

## 3. Division scoring is genuinely sophisticated — don't approximate it

`division_metrics.evaluate_divisions()` does **not** just check "does a node have out-degree ≥ 2
near a GT division." The real algorithm:

1. `extract_divisions(gt_graph)` — pulls each GT division into its own subgraph: parent → divider
   → {child1, child2} → grandchildren.
2. For each GT division, match the *full* predicted graph against just that division's subgraph
   (7 µm gated, same `DistanceMatching`).
3. `_has_stage_coverage()` — a candidate match must have ≥1 matched node in the GT's pre-split
   "one-node stage" AND matched nodes covering ≥2 distinct daughter lineages (checked via
   descendant-set membership, not raw distance).
4. Candidates are grouped into weakly-connected components of the *predicted* graph, and only
   components containing an actual predicted dividing node (out-degree ≥ 2) are kept.
5. **Global bipartite maximum-matching** between candidate predicted dividing nodes and GT
   divisions — this is the part easy to miss: it prevents one predicted fork from being credited
   to multiple GT divisions, and prevents one GT division from being satisfied by multiple
   predicted forks. A GT division only counts as TP if it's paired in this matching.
6. FP predicted divisions are counted only among nodes that matched a GT node that itself has ≥1
   child in the ground truth (i.e., not at the edge of a sparse annotation window) — this is the
   concrete mechanism behind "predicted divisions in unannotated regions are ignored."

Re-implementing this from a one-paragraph description (as PRD.md originally assumed) would almost
certainly get the edge cases wrong (double-crediting forks, mis-scoring divisions at annotation
boundaries). **Vendor or depend on the real `division_metrics.py`.**

## 4. `io.py` — the real data-loading pattern (and a gap in our current code)

```python
DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)  # confirms PRD anisotropy exactly

def open_dataset(ds_path, target_scale=None, normalize=True, gamma=1.0, device="cuda",
                  require_tracks=False, load_image=True, downsample=None) -> Dataset:
    """Opens `{ds_path}.zarr` (+ `.geff` if require_tracks). Reads image via zarr + dask,
    tracks via td.graph.IndexedRXGraph.from_geff(tracks_path) - the real .geff reader,
    no hand-rolled parsing needed. Optionally quantile-normalizes intensities and/or
    resamples to isotropic voxel spacing on GPU (torch, trilinear interpolation)."""
```

**The gap this exposes in our repo:** the host's own pipeline normalizes intensities via
precomputed quantiles (`image_statistics.quantiles` from the zarr attrs — confirmed present on
every staged sample) before any downstream thresholding:
```python
tensor = (tensor - q_low) / (q_high - q_low + 1e-6)
tensor = tensor.clamp(min=0.0)
```
Our current `extract_peaks_from_volume` (in `run_pipeline.py`) thresholds raw uint16 voxel values
against constants `0.4` / `0.45` that only make sense against a normalized `[0, ~1]` range — this
was already flagged as a placeholder in the PRD, but now we know exactly what real normalization
should look like and where the quantiles come from.

`td.graph.IndexedRXGraph.from_geff()` / `.to_geff()` (write side) and `save_graph()` /
`list_datasets()` in this same file are direct, ready-made replacements for FR-1's planned
hand-rolled `.geff` reader and FR-4's planned submission writer's node/edge serialization — worth
checking whether `to_geff()` output can be reshaped into the competition's CSV schema directly,
before writing a CSV serializer from scratch.

## 5. Also worth following up (not yet fetched, lower priority)

- `scripts/train_unet_transformer.py` / `predict_unet_transformer.py` — the host's own reference
  *model* (a U-Net + Transformer, not a plain CNN). Directly relevant to PRD Phase 2's model
  architecture decision — our current `STACTCentroidPredictor` is a 2-conv-layer shallow FCN;
  worth comparing against what the host itself considers a reasonable baseline architecture.
- `src/tracking_cellmot/img_proc.py` and `models/` — not yet pulled.

## 6. Action items this unblocks

- **PRD.md §11 open question is resolved** — update it to point here instead of "not yet confirmed."
- **FR-1 (data ingestion):** prefer `tracksdata`'s `IndexedRXGraph.from_geff()` over a hand-rolled
  `.geff` parser.
- **FR-5 (local metric):** vendor/import `metrics.py` + `division_metrics.py` directly rather than
  reimplementing from prose — this removes most of the risk in that task.
- **FR-2 (training) / Phase 2:** revisit model architecture against the host's own
  U-Net+Transformer reference before committing to deepening the current 2-conv-layer FCN.
- Check whether `tracksdata` is pip-installable (`pip install tracksdata`) and add it to
  `requirements.txt` once confirmed.
