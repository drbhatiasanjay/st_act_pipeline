# PRD: ST-ACT — Spatio-Temporal Anisotropic Cell Tracker
### Target: Biohub — Cell Tracking During Development (Kaggle)

**Status:** Draft v1 · **Owner:** (unassigned) · **Last updated:** 2026-07-03

---

## 1. Executive Summary

`st_act_pipeline` is our entry for the Kaggle competition
[**Biohub – Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development),
hosted by the Chan Zuckerberg Biohub / Royer Group. The task: detect every cell nucleus in a
3D+time light-sheet microscopy volume of a developing zebrafish embryo, link detections into
per-cell tracks over time, and correctly resolve mitosis (cell division) events.

The repo already contains a well-conceived architecture — an anisotropy-aware 3D CNN, and a
genuinely sophisticated global ILP (integer linear program) hypergraph tracker with gap-closing
and mitosis smoothing. **However, an audit of the current code against the actual competition
spec found five blocking gaps that mean a submission generated today would score ~0** (wrong
CSV schema, no real data ever loaded, an untrained/unwired detection model, fabricated motion
vectors, and an ILP solver that won't scale to real data volume). This PRD defines what "done"
looks like, closes those gaps in priority order, and lays out the roadmap to compete for and
hold the #1 leaderboard position.

---

## 2. Background

- **Host:** Biohub (Royer Group, San Francisco). Builds on **Ultrack**, Loïc Royer's AI cell-tracking
  algorithm published in *Nature Methods* (2025), which "dramatically improved the accuracy and
  scalability of cell segmentation and tracking" but left tracking tens of thousands of cells
  "computationally demanding and ripe for innovation."
  ([FEBS Network](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking))
- **Data provenance:** Light-sheet microscopy videos of zebrafish embryos, captured by the Royer
  Group; annotated as (by annotation count) the largest publicly available cell-tracking dataset,
  released CC0.
- **Goal stated by host:** improve tracking performance *and* establish transparent,
  community-driven benchmarking standards for cell-tracking methods.
- **Prize pool:** $60,000. **Entry deadline:** 2026-09-22 (per Kaggle listing, ~3 months from
  competition launch on 2026-06-29).
- **Why this matters for us:** correct 3D+time lineage reconstruction is a generically valuable
  capability (embryonic development, immune response, tissue regeneration, disease progression),
  and this competition is the fastest way to benchmark our tracker against a global field.

Sources: [Kaggle competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) ·
[image.sc forum announcement](https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671) ·
[FEBS Network article](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking)

---

## 3. Competition Specification (source of truth)

### 3.1 Task
For every test embryo movie, output a **tracking graph**: one node per detected cell per
timepoint (3D centroid), and one edge per temporal link between a cell and its position in the
next timepoint it's tracked to. A cell with **≥2 outgoing edges** represents a division
(mitosis) event.

### 3.2 Data
| Item | Spec |
|---|---|
| Image volumes | Zarr **v3**, single array at path `0/`, shape **`(T, Z, Y, X)`**, typically `(100, 64, 256, 256)`, **`uint16`** |
| Chunking | `(1, 64, 256, 256)` per timepoint, blosc/zstd compressed |
| Physical voxel size | **z = 1.625 µm, y = x = 0.40625 µm** → anisotropy ratio ≈ **4.0 : 1 : 1** (Z:Y:X) |
| Ground truth (train only) | `.geff` (graph exchange format, Zarr-v3-based): `nodes/ids`, `nodes/props/{t,z,y,x}/values` (voxel-space centroids), `edges/ids` shape `(N,2)` = (source, target); sparse annotations; `estimated_number_of_nodes` metadata per sample |
| Folder layout | `train/` = paired `.zarr` + `.geff`; `test/` = `.zarr` only (public copies — **real hidden test set is swapped in at scoring time**); naming `{embryo_id}_{field_of_view}` |
| Split discipline | **Train and test are embryo-disjoint** — no embryo appears in both |
| Reference | `sample_submission.csv` |

### 3.3 Evaluation Metric
```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```
- **Edge Jaccard:** per timeframe, predicted nodes are matched to ground-truth nodes via optimal
  bipartite assignment on centroid distance (**max gating threshold: 7.0 µm**, physical space).
  An edge is a true positive only if *both* endpoints matched to GT nodes *and* that GT pair is
  itself linked by a GT edge. `TP / (TP + FP + FN)`, with a penalty adjustment for over-predicting
  node count. Per-sample results are weighted by `(TP + FP + FN)` before averaging across test
  samples.
- **Division Jaccard:** a division = a node with out-degree ≥ 2. For each GT division event, the
  metric checks whether the *predicted* graph contains a connected component spanning the
  pre-division stage and reaching **both** daughter lineages. Micro-averaged across samples.
- Score can exceed 1.0 by construction — this is a known/accepted property of the formula, not a bug.

### 3.4 Submission Format
CSV header (exact): `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`

| `row_type` | Fields used | Fields set to `-1` |
|---|---|---|
| `node` | `t,z,y,x` (integer voxel coords) | `source_id`, `target_id` |
| `edge` | `source_id`, `target_id` (reference `node_id`s) | `z,y,x`, `node_id` |

`dataset` must equal the test folder name **without** the `.zarr` suffix, and **every** test
dataset folder must appear in the submission (missing datasets presumably score 0 for that sample).

### 3.5 Leaderboard (snapshot, ~29% of test data revealed, ~3 months remaining)
| Rank | User | Score |
|---|---|---|
| 1 | Kaushik Ramayya Chikkala | 0.875 |
| 2 | doheon114 | 0.874 |
| 3 | Kevin | 0.872 |
| 4–5 | Rahul Parmeshwar / Pathik Patel | 0.872 |
| 6 | Matt Goldfield | 0.864 |
| 7 | Scott Willis | 0.860 |
| 8 | Juliewww | 0.857 |
| 9 | Soumyajyoti Biswas | 0.856 |
| 10 | hikarimaru | 0.852 |

~49 teams currently on the board. **Important:** the public leaderboard is computed on ~29% of
test data — final ranking uses the full hidden set, so overfitting to the visible slice is a
real risk (see §8 Risks).

### 3.6 Public Baselines
| Notebook | Approach | Public score |
|---|---|---|
| [`inversion/cell-tracking-getting-started-w-nearest-neighbor`](https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor) | Naive nearest-neighbor linking | 0.143 |
| [`xiaoleilian/biohub-cell-tracking-classical-baseline`](https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline) | Single-scale local-maxima detection + Hungarian assignment + µm-gated distance + motion-aware linking + conservative division detection | 0.763 |

These bound our floor: **any credible submission must clear 0.763** (the classical, non-learned
baseline) before a learned model is worth the added complexity. The current top score of 0.875
is the moving target.

---

## 4. Current State Audit (this repo, as of 2026-07-03)

A graphify structural scan (`graphify-out/GRAPH_REPORT.md`) plus a direct read of every source
file surfaced the following. Ranked by severity — **all five of §4.1 are launch-blocking.**

### 4.1 Critical blockers (submission would score ~0 today)

1. **Submission schema is wrong.**
   [`run_pipeline.py:143-155`](run_pipeline.py) writes `Time,TrackID,ParentTrackID,Z,Y,X` to
   `output/submission.csv`. The competition requires
   `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` with **separate node and edge rows**
   per §3.4. There is no `dataset` column at all, so multi-embryo test sets can't even be
   represented. **This alone guarantees a failing submission.**

2. **The pipeline has never touched real competition data.**
   `AnisotropicZarrLoader._init_store()` ([`src/data_loader.py:34`](src/data_loader.py)) silently
   **fabricates a fake Zarr store** whenever the configured path is missing — which it always is,
   because `config/hyperparams.yaml` points at a local demo path
   (`data/cell_tracking_volume.zarr`), never the Kaggle `train/`/`test/` directories. The fake
   store is `(20, 10, 128, 128) float32`; real data is `(100, 64, 256, 256) uint16` at path `0/`.
   There is **no code anywhere in the repo that reads `.geff` ground truth** — meaning there is
   currently no way to train against, or locally validate against, real annotations.

3. **The detection model is defined but never used, and never trained.**
   `STACTCentroidPredictor` ([`src/model.py`](src/model.py)) is a real (if shallow, 2-conv-layer)
   3D FCN with heatmap + motion-vector heads — but `run_pipeline.py` never imports or calls it.
   Instead, `extract_peaks_from_volume()` ([`run_pipeline.py:56`](run_pipeline.py)) does a naive
   fixed-stride (`step=8`) threshold scan and is called **twice with different thresholds** to
   fake "CNN" and "U-Net" outputs for `ensemble_consensus_centroids()`'s DBSCAN merge. There is no
   training loop, loss function, dataset/dataloader, augmentation, or checkpoint anywhere;
   `models/` is empty.

4. **Motion vectors fed to the tracker are hardcoded constants.**
   `run_pipeline.py:100-103` assigns every single cell the same vector `[0.05, 0.2, 0.3]` instead
   of the model's predicted motion field. This nullifies the tracker's motion-compensated
   gap-closing (`STHypergraphTracker.solve_lineage`'s `warped_u` term,
   [`src/tracker.py:100`](src/tracker.py)) — a core piece of the architecture's design intent.

5. **The ILP solver will not scale to competition-size data.**
   `STHypergraphTracker.solve_lineage` ([`src/tracker.py:38`](src/tracker.py)) builds one global
   PuLP/CBC ILP over *all* nodes and candidate edges across *all* 100 timepoints at once. Real
   volumes plausibly contain thousands of cells; with gap-closing (`max_gap_frames=2`, so 3 lookahead
   offsets) the candidate-edge count grows combinatorially. CBC is not built for
   tens-of-thousands-of-binary-variables ILPs under a Kaggle notebook's runtime budget — this
   will time out or exhaust memory on real data, unmodified.

### 4.2 Secondary gaps
- **Anisotropy ratio is wrong.** Hardcoded `(5.0, 1.0, 1.0)` throughout
  (`data_loader.py`, `run_pipeline.py`, `hyperparams.yaml`); the real physical ratio from
  §3.2 is **`1.625 / 0.40625 = 4.0`**, i.e. `(4.0, 1.0, 1.0)`. A 25% error here directly corrupts
  every physical-distance gate in the tracker (7 µm matching threshold, edge-pruning limits,
  search radius).
- **No local implementation of the competition metric.** We cannot know our real score without
  submitting to Kaggle. There is no `edge_jaccard` / `division_jaccard` scorer in the repo.
- **`ensemble_consensus_centroids`'s "ensemble" isn't one.** Both inputs come from the same
  placeholder function; there's only one real model architecture in the codebase (`model.py`
  has no second network), so DBSCAN is deduplicating a signal against a shifted copy of itself.
- **Mitosis smoothing uses a mock intensity proxy**, not real image intensity
  (`smooth_mitosis_edges`, [`src/tracker.py:172`](src/tracker.py), comment: *"Mock biological
  intensity sum for simulation demo"*).
- **Single hardcoded dataset path** — no iteration over multiple test embryo folders as §3.2
  requires.
- **No experiment tracking / no checkpointing** — `models/` and reproducibility infra absent.
- `system_diagnostics.py` and `notebooks/exploratory_analysis.ipynb` are fine as-is and out of
  scope for this PRD (hardware/RAM diagnostics utility; stub notebook).

### 4.3 What's actually good and worth keeping
- **`STHypergraphTracker`'s core ILP formulation** (flow conservation with explicit birth/death/
  split variables, temporal gap-closing with exponential lookahead penalty, anisotropic edge
  pruning) is a legitimately strong design that maps directly onto what the `division_jaccard`
  term rewards — this is a real differentiator over nearest-neighbor and even the Hungarian
  classical baseline, *if* it's fed real data and made to scale.
- **`AnisotropicCoordinateTransformer`** correctly separates voxel-space from physical-space
  reasoning, which several baselines skip.

---

## 5. Goals

**G1 (Unblock).** Produce a submission that is schema-valid and scores **above the classical
baseline (0.763)** using the existing architecture wired to real data.

**G2 (Compete).** Train the detection model on real annotations and reach parity with or exceed
the current leaderboard #1 (0.875 at time of writing — expect this number to keep climbing).

**G3 (Lead & hold).** Establish a repeatable weekly iteration loop (local metric → experiment →
submit → compare) so we can **track and beat the leaderboard continuously as more of the hidden
test set is revealed**, not just optimize against the current ~29% public slice.

### Non-goals
- Building a general-purpose cell-tracking product/UI (this is a competition entry).
- Supporting microscopy modalities other than the competition's light-sheet zebrafish data.
- Real-time/streaming inference — batch offline inference against Kaggle test folders is sufficient.

---

## 6. Functional Requirements

### FR-1 — Correct data ingestion
- Read real Kaggle `train/{embryo}_{fov}/` and `test/{embryo}_{fov}/` Zarr v3 stores at array
  path `0/`, shape `(T,Z,Y,X)`, `uint16`. Remove (or gate behind an explicit `--simulate` flag)
  the silent fake-data fallback in `_init_store` — it must never activate against a real
  competition path.
- Fix anisotropy ratio to `(4.0, 1.0, 1.0)` (or read `z=1.625, y=x=0.40625` from config and
  compute the ratio) everywhere it's currently hardcoded to `(5.0, 1.0, 1.0)`.
- Implement a `.geff` reader: load `nodes/ids`, `nodes/props/{t,z,y,x}/values`, `edges/ids` into
  the same in-memory representation (`centroids_by_t`, ground-truth edge list) the tracker already
  consumes, so it can be used for both training targets and local validation.
- Iterate over **all** dataset folders present at inference time and emit one `dataset` block per
  folder in the submission — not a single hardcoded path.

### FR-2 — Trained detection model wired into the pipeline
- Build a training script: sample volumes from `train/`, generate heatmap targets (Gaussian blobs
  at `.geff` centroids, anisotropy-aware sigma) and motion-vector targets (displacement to the
  same cell's position at `t+1`, derived from GT edges), train `STACTCentroidPredictor` (or a
  deepened successor — current 2-conv-layer depth is likely under-capacity for 256×256×64 volumes)
  with a proper loss (focal/weighted-MSE for heatmap + smooth-L1 for motion), train/val split by
  **embryo** (never split within an embryo — mirrors the competition's embryo-disjoint train/test
  discipline), checkpointing into `models/`.
- Replace `extract_peaks_from_volume`'s naive stride-scan with real inference: run
  `STACTCentroidPredictor` on (chunked, via the existing `stream_chunks_3d`) volumes, take local
  maxima of the predicted heatmap above a tuned threshold as centroids, and use the **predicted
  motion-vector field** (not a hardcoded constant) as input to the tracker.
- If ensembling is retained, it must be a real ensemble (e.g. two independently trained
  seeds/architectures), not two calls to the same placeholder function.

### FR-3 — Tracker at competition scale
- Feed real predicted motion vectors into `STHypergraphTracker.solve_lineage`.
- Replace mitosis smoothing's mock intensity proxy with real local image-intensity evaluation
  around candidate split frames.
- Make the ILP tractable at real scale: windowed/rolling-horizon solving (e.g. solve over
  overlapping N-frame windows and stitch tracks across window boundaries), and/or swap CBC for a
  purpose-built min-cost-flow formulation (the flow-conservation constraints already resemble
  one) so runtime stays within Kaggle notebook limits for `(100, 64, 256, 256)` volumes with
  thousands of cells.

### FR-4 — Correct submission generation
- Rewrite the export step to emit `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
  exactly, with separate `node`/`edge` rows, integer voxel coordinates, `-1` sentinels per §3.4,
  and one `dataset` block per test folder actually processed.
- Validate the generated file against `sample_submission.csv`'s structure before treating a run
  as "submit-ready."

### FR-5 — Local evaluation harness
- Implement `edge_jaccard` and `division_jaccard` exactly per §3.3 (7.0 µm gating, TP/(TP+FP+FN)
  with the over-prediction penalty, sample-weighted averaging; division = out-degree ≥ 2,
  connected-component check spanning both daughter lineages) against held-out **train** embryos
  with known `.geff` ground truth.
- Every model/tracker change must be evaluated against this harness before a Kaggle submission is
  spent — submissions are a scarce, rate-limited resource.

---

## 7. Proposed Architecture (target state)

```
train/*.zarr, train/*.geff  ─┐
                              ├─▶ AnisotropicZarrLoader + GeffReader (FR-1)
test/*.zarr (per embryo)    ─┘        │
                                       ▼
                         STACTCentroidPredictor (trained, FR-2)
                         heatmap → local-max peaks   motion → per-cell displacement
                                       │                        │
                                       ▼                        ▼
                          STHypergraphTracker.solve_lineage (windowed ILP, FR-3)
                          + smooth_mitosis_edges (real intensity)
                                       │
                                       ▼
                    Submission writer → id,dataset,row_type,node_id,t,z,y,x,source_id,target_id (FR-4)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        Local eval harness (edge/division jaccard, FR-5)   Kaggle submission
```

`STHypergraphTracker`'s ILP core and `AnisotropicCoordinateTransformer` are retained essentially
as-is (§4.3); everything else on the critical path (§4.1) is rebuilt or rewired.

---

## 8. Phased Roadmap

| Phase | Scope | Exit criterion |
|---|---|---|
| **0 — Unblock** | Fix anisotropy constant; point loader at real `train/`/`test/`; implement `.geff` reader; implement FR-4 submission writer; implement FR-5 local metric | A schema-valid submission generated end-to-end from real data, scored locally |
| **1 — Baseline parity** | Wire existing (untrained) placeholder detection through the corrected pipeline; validate against local metric | Local score ≥ 0.763 (beats classical baseline) — proves the pipeline itself is sound before investing in training |
| **2 — Learned detection** | Train `STACTCentroidPredictor` (or deepened variant) on real `.geff` targets; replace naive peak extraction; feed real motion vectors into tracker | Local score materially exceeds Phase 1; first real Kaggle submission |
| **3 — Scale & correctness** | Windowed/rolling-horizon ILP or min-cost-flow reformulation; real-intensity mitosis smoothing | Full-size `(100,64,256,256)` volumes process within Kaggle runtime limits without truncation |
| **4 — Metric-directed tuning** | Calibrate 7 µm gating sensitivity, precision/recall trade-off (Jaccard penalizes FP and FN symmetrically), division-recall specifically (only 0.1-weighted but likely differentiates top scorers) | Local score ≥ current leaderboard #1 |
| **5 — Competitive iteration loop** | Weekly cycle: local eval → targeted change → submit → compare against leaderboard delta; monitor new public notebooks/discussion for technique shifts; guard against overfitting to the ~29%-revealed public slice | Sustained top-of-leaderboard position as more hidden test data is revealed |

---

## 9. Success Metrics

- **Primary:** competition leaderboard rank and score (target: rank #1, tracked weekly through
  2026-09-22).
- **Leading indicator:** local `edge_jaccard + 0.1 × division_jaccard` on held-out train embryos,
  computed via FR-5 — must correlate with public leaderboard movement before it's trusted as a
  decision signal.
- **Guardrail:** local metric on held-out embryos must not diverge from public leaderboard score
  by more than a small margin — large divergence signals overfitting to the visible 29% slice
  (see Risks).

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Public leaderboard reflects only ~29% of test data; optimizing against it risks overfitting and rank collapse when the full hidden set is revealed | Always validate primarily against local held-out **train** embryos with real `.geff` ground truth (FR-5), not just public LB feedback |
| ILP solve time explodes on full-scale volumes (§4.1.5) | Windowed/rolling-horizon solving from Phase 3; benchmark solve time on realistic synthetic scale before first real submission |
| Kaggle notebook compute/runtime limits | Use existing `stream_chunks_3d` for memory-bounded inference; profile end-to-end runtime against Kaggle's limits before submission |
| Detection model under-capacity (current 2-conv-layer FCN) for 256×256×64 volumes | Treat current architecture as a placeholder to validate the pipeline (Phase 1), and budget explicit capacity/architecture experimentation in Phase 2 |
| Division events are rare relative to survivals/deaths — model may under-predict them despite only being 0.1-weighted, since top-of-leaderboard separation may hinge on it | Track division recall/precision as its own metric in the local harness, not just blended score |
| Competition rules on team size, external data, compute may constrain approach | Re-fetch and confirm `/rules` page (not fully retrievable at PRD-authoring time — see §11) before Phase 2 investment |

---

## 11. Open Questions / Follow-ups

- Full `/rules` page (timeline detail beyond the entry deadline, team size limits, external-data
  policy, compute/runtime caps) did not fully render during research for this PRD and should be
  re-confirmed directly on Kaggle before committing to Phase 2+ compute investment.
- Discussion board content wasn't retrievable at authoring time — worth a manual pass for any
  host-clarified metric edge cases (e.g. exact over-prediction penalty formula, tie-breaking in
  bipartite matching) before FR-5 implementation is finalized against real host code if/when
  released.
- Confirm whether the host provides a reference metric implementation (common in Kaggle bio
  competitions) to validate FR-5 against, rather than reimplementing purely from prose.
