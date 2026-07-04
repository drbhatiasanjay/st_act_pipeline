# Roadmap: ST-ACT — Spatio-Temporal Anisotropic Cell Tracker

**Defined:** 2026-07-03  
**Depth:** standard  
**Total phases:** 6 (Phases 0–5)  
**Coverage:** 19/19 v1 requirements mapped

---

## Overview

ST-ACT is a competition entry for the Biohub–Cell Tracking During Development Kaggle competition. The project has a sophisticated architecture in place (3D CNN for detection, ILP-based hypergraph tracker with gap-closing and mitosis smoothing), but five critical blockers prevent any valid submission today. This roadmap closes those gaps in priority order, validates the pipeline end-to-end, trains a real detection model, scales it to production, and drives competitive iteration toward top-of-leaderboard finish.

The phases are derived from the PRD's § 8 phased roadmap, with v1 requirements mapped to the phases that unblock and deliver them.

---

## Phase Details

### Phase 0: Unblock

**Status: ✓ COMPLETE (2026-07-04)** — verified by gsd-verifier, 5/5 must-haves, 60/60 tests
passing. See `.planning/phases/00-unblock/00-UNBLOCK-VERIFICATION.md` for the full report.

**Goal:** Generate a schema-valid submission end-to-end from real competition data, scored locally.

**Exit Criterion:** A submission CSV with correct schema (`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`), separate node/edge rows, generated from real Zarr stores and ground-truth `.geff` annotations, scored locally using the exact competition metric (edge Jaccard + 0.1 × division Jaccard).

**Real result (2026-07-04):** Full 8-dataset run completed in ~8.3 minutes.
`output/phase0_submission.csv` — 18,735 rows, schema-valid, all 4 real test datasets present.
Local score: `edge_jaccard=0.0084, division_jaccard=0.0, combined=0.0092` (vs. 0.763 baseline —
**expected, not a defect**: the placeholder detector emits raw stride-8 grid points, not real
peaks, so near-zero node matches within the 7µm gate is the mathematically expected outcome, not
a pipeline bug. Carried forward as Phase 1's central risk — see its entry in the Progress table
above and `PLAN-04-WIRE-PIPELINE-AND-TEST-SUMMARY.md`'s "Key Finding" section.

Also confirmed empirically this phase: the ILP tracker is **70.2% of total runtime** even at a
heavily-capped 30 candidates/timepoint — direct, current evidence for Phase 3's scaling concern
below, not just a theoretical risk.

**Dependencies:** None (first phase, greenfield)

**Scope change (2026-07-03, materially lowers this phase's risk):** the host publishes a full
reference implementation (`royerlab/kaggle-cell-tracking-competition`, built on `tracksdata`),
vendored/documented in `REFERENCE_IMPLEMENTATION.md`. DATA-04 and EVAL-01..03 are now
"integrate the real library" tasks, not "reimplement a bespoke spec from prose." A small
real-data subset is also already staged at `data/staging/` (4 embryos, train+test, see its
README) — Phase 0 work can start against real data immediately, no extraction step needed first.

**Requirements Mapped:**
- DATA-01: Pipeline reads real Kaggle `train/{embryo}_{fov}/` and `test/{embryo}_{fov}/` Zarr v3 stores at array path `0/`, shape `(T,Z,Y,X)`, `uint16`
- DATA-02: Simulated-data fallback never silently activates against a real competition path
- DATA-03: Anisotropy ratio is `(4.0, 1.0, 1.0)` everywhere (corrected from hardcoded `(5.0, 1.0, 1.0)`)
- DATA-04: `.geff` ground truth loaded via `tracksdata.graph.IndexedRXGraph.from_geff()` (host's own reader), not hand-rolled parsing
- DATA-05: Pipeline iterates over all dataset folders present at inference time, emitting one `dataset` block per folder
- DATA-06: Raw uint16 intensities normalized via each sample's `image_statistics.quantiles` before thresholding — without this, even Phase 1's placeholder detector produces meaningless output on real data
- SUB-01: Export emits `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` exactly, separate node/edge rows, integer voxel coords, `-1` sentinels. Confirmed from the real `sample_submission.csv`: `id` is globally sequential across the whole file; `node_id` is per-`dataset`-local and resets to 1 for each new dataset block — do not treat `node_id` as globally unique
- SUB-02: One `dataset` block per test folder actually processed, matching real test basenames exactly
- SUB-03: Generated file validated against `sample_submission.csv`'s structure before a run is treated as submit-ready
- EVAL-01: `edge_jaccard` computed via `tracksdata`'s `evaluate()`/`evaluate_datasets()` (7.0µm gated `DistanceMatching`, micro-averaged)
- EVAL-02: `division_jaccard` computed via the vendored `division_metrics.evaluate_divisions()` — real algorithm (GT-subgraph extraction, stage coverage, global bipartite max-matching), not a simple out-degree≥2 check
- EVAL-03: Combined score `adjusted_edge_jaccard + 0.1 × division_jaccard` per the exact confirmed formula (§1 of `REFERENCE_IMPLEMENTATION.md`); division term dropped (not +0) when a sample has zero GT divisions
- EVAL-04: `traccuracy` retained only as an optional secondary sanity check; `tracksdata` is primary

**Success Criteria (Observable):**

1. **Real data loads without fallback:** `AnisotropicZarrLoader` successfully opens Zarr stores from `train/` and `test/` directories; `_init_store()` never fabricates a fake store; shape verified as `(T, Z, Y, X)` uint16; metadata confirms physical voxel size (z = 1.625 µm, y = x = 0.40625 µm).

2. **Anisotropy applied consistently:** Distance calculations, coordinate transforms, and Gaussian target generation use anisotropy ratio `(4.0, 1.0, 1.0)` (Z:Y:X); verified by spot-checking 7.0 µm gating threshold applied to sample node pairs.

3. **Ground-truth annotations loaded:** `.geff` reader successfully extracts `nodes/ids`, `nodes/props/{t,z,y,x}/values` (voxel-space centroids), `edges/ids` for sample train embryo; data loaded into the same in-memory graph representation the tracker consumes (nodes as dict, edges as list of tuples).

4. **Submission file schema validated:** Generated CSV has exact header `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`; rows with `row_type=node` have populated `t,z,y,x` (integer voxels), `-1` for `source_id,target_id`; rows with `row_type=edge` have populated `source_id,target_id` (reference `node_id`s), `-1` for `z,y,x,node_id`; all datasets from test folders appear as separate blocks; file passes validation against `sample_submission.csv` schema.

5. **Local metric computes and correlates:** `edge_jaccard + 0.1 × division_jaccard` computed on 2–3 held-out train embryos with real `.geff` ground truth; score displayed in logs; verified to respond to changes in tracker output (sanity check: perturbing predictions changes score).

---

### Phase 1: Baseline Parity

**Goal:** Prove the pipeline is sound before investing in model training, by wiring the existing untrained placeholder detection model and validating it reaches the classical baseline (0.763).

**Exit Criterion:** Local metric on held-out train embryos reaches ≥ 0.763 using the existing placeholder detection model (no new training), proving the entire pipeline (data loading, detection wiring, tracking, submission generation, local evaluation) is correct.

**Dependencies:** Phase 0 complete

**Requirements Mapped:** None (Phase 1 is validation and wiring of Phase 0's infrastructure; no new v1 requirements)

**Success Criteria (Observable):**

1. **Placeholder detection integrated end-to-end:** `STACTCentroidPredictor` (untrained) loaded and called in `run_pipeline.py`; heatmap predictions passed to local-maxima peak extraction; motion-vector field passed to tracker (even if just zeros or synthetic motion).

2. **Local score reaches ≥ 0.763 on held-out embryos:** Metric computed on 2–3 held-out train embryos using placeholder detection; score printed and logged; passes baseline threshold (classical non-learned approach scores 0.763).

3. **Full-size volumes run without truncation:** End-to-end pipeline executes on at least one full `(100, 64, 256, 256)` volume without crashes, memory exhaustion, or silent truncation; final submission CSV contains all timepoints and all cells.

4. **Tracker output is sensible:** Spot-check that predicted edges and division events are plausible (e.g., cells don't jump across unphysical distances in one timeframe; divisions occur at non-zero out-degree nodes); comparison to local ground truth shows qualitative agreement (not quantitatively perfect, but structurally sound).

---

### Phase 2: Learned Detection

**Goal:** Train a real detection model on `.geff` ground-truth targets and wire it end-to-end, beating the classical baseline and generating the first real Kaggle submission.

**Exit Criterion:** Local score materially exceeds Phase 1 (target: ≥0.80+), achieved by training the model on real targets, replacing naive peak extraction and hardcoded motion vectors with real predictions, and successfully submitting to Kaggle leaderboard.

**Dependencies:** Phase 1 complete; Kaggle API + remote-GPU-kernel setup (parallel work in `../st_act_pipeline-kaggle-setup` branch) complete

**Requirements Mapped:**
- MODEL-01: Training script generates heatmap targets (anisotropy-aware Gaussian blobs at `.geff` centroids) and motion-vector targets (displacement to `t+1`, from GT edges)
- MODEL-02: Detection model trained with train/val split by embryo (never split within one embryo, matching competition's embryo-disjoint train/test discipline)
- MODEL-03: `extract_peaks_from_volume`'s naive stride-scan replaced with real model inference (local maxima of predicted heatmap above tuned threshold)
- MODEL-04: Tracker receives the model's predicted motion-vector field, not hardcoded constant
- TRACK-01: Tracker receives real predicted motion vectors (depends on MODEL-04)

**Success Criteria (Observable):**

1. **Heatmap and motion-vector targets generated:** Training script loads `.geff` ground truth for all train embryos; for each GT centroid, generates anisotropy-aware Gaussian blob heatmap target (3D, responsive to (4,1,1) voxel spacing); for each GT edge, computes motion-vector target as displacement to `t+1` position; targets visualized and spot-checked against raw volumes.

2. **Embryo-disjoint train/val split enforced:** Data loader splits by embryo ID, never within an embryo; train set contains embryos {A, B, C, ...}, validation set contains held-out embryo(s) {X, Y, ...}; no data leakage between train and val; split sizes logged.

3. **Model replaces naive peak extraction:** `extract_peaks_from_volume` retired or gated behind `use_naive=False`; trained model inference produces heatmap volume; local maxima detected above threshold (e.g., 0.5); threshold calibrated against local metric; comparison on sample frame shows learned peaks differ from naive peaks (i.e., genuine learning, not just re-implementing stride-scan).

4. **Motion vectors wired to tracker:** Model's motion-vector field (shape matching heatmap, 3 channels for z, y, x displacement) extracted and passed to `STHypergraphTracker.solve_lineage`; spot-check shows motion vectors are non-zero and responsive to ground truth (e.g., cells moving in-frame have non-zero vectors).

5. **Local score materially exceeds Phase 1:** Metric computed on held-out train embryos with trained model; score reported (target ≥0.80 or meaningful +0.05 over Phase 1); Kaggle submission successful (file uploaded, scoring begins).

---

### Phase 3: Scale & Correctness

**Goal:** Make the tracking pipeline tractable at real production scale (thousands of cells, 100 timepoints, gap-closing), ensuring it runs within Kaggle's 12-hour notebook limit and handles full-size volumes without truncation or memory issues.

**Exit Criterion:** Full-size `(100, 64, 256, 256)` volumes process end-to-end within Kaggle's runtime budget, with real-intensity mitosis smoothing and no memory or truncation errors.

**Dependencies:** Phase 2 complete

**Requirements Mapped:**
- TRACK-02: Mitosis smoothing uses real local image-intensity evaluation, not the mock proxy
- TRACK-03: ILP is tractable at real scale (thousands of cells × 100 timepoints × gap-closing) via windowed/rolling-horizon solving or a min-cost-flow reformulation — candidate: OR-Tools `SimpleMinCostFlow` for the bulk assignment

**Success Criteria (Observable):**

1. **ILP solve time stays within budget:** Windowed/rolling-horizon ILP solver implemented (or min-cost-flow reformulation via OR-Tools `SimpleMinCostFlow` verified to fit flow-conservation constraints); end-to-end solve on realistic synthetic scale (e.g., 5000 cells, 100 frames, 5% gap-closing demand) completes in <2 hours locally; profiling shows solver dominates runtime, other steps negligible; scaling extrapolation confirms Kaggle 12-hour budget is safe.

2. **Real-intensity mitosis smoothing active:** `smooth_mitosis_edges()` now evaluates local image intensity (real voxel values from volume) around candidate split frames, not mock proxy; logic verified: high intensity → high confidence in true mitosis; low intensity → possible false positive mitosis.

3. **Full-size volumes process without truncation:** Pipeline runs on at least one full `(100, 64, 256, 256)` volume (or larger if available); final submission CSV contains all T frames, all detected cells, all edges (no silent truncation); memory peak logged; no out-of-memory errors.

4. **Divide-and-conquer integrity maintained:** If windowed approach: tracks stitched across window boundaries correctly (spot-check that a cell tracked across window N and N+1 has consistent identity and no duplicate edges); if min-cost-flow: flow conservation verified via unit test (every node in, same out, except sources/sinks).

---

### Phase 4: Metric-Directed Tuning

**Goal:** Calibrate the tracker's parameters and thresholds systematically against the local evaluation harness, targeting a score ≥0.875 (current leaderboard #1) and maximizing division recall.

**Exit Criterion:** Local score on held-out train embryos reaches ≥0.875, with division recall tracked and optimized as an independent signal (since it's only 0.1-weighted but likely differentiates top-tier competition entries).

**Dependencies:** Phase 3 complete

**Requirements Mapped:** v2's ITER-01 (systematic calibration of tracker costs; reframed as v1-critical given that top-of-leaderboard separation is the goal and division events are rare but high-leverage)

**Success Criteria (Observable):**

1. **7 µm gating threshold calibrated:** Grid search or Bayesian optimization over gating threshold (e.g., 5–10 µm range); score plotted against threshold; optimal threshold identified and locked in; score improvement documented (e.g., +0.02 from initial Phase 3 value).

2. **Tracker costs tuned:** Birth, death, division, and gap-closing costs (in the ILP objective) systematically varied; local metric re-run after each change; best parameter set locked in; sensitivity documented (e.g., division cost trade-off: +0.01 edge Jaccard but -0.02 division Jaccard).

3. **Division recall tracked separately:** Local harness augmented to report division_recall = TP_divisions / (TP_divisions + FN_divisions) independently; plotted alongside overall score; optimization targets division_recall ≥ 0.90 (or best achievable) even if it trades off a small amount of overall Jaccard.

4. **Local score reaches ≥0.875:** Final calibrated pipeline achieves local score ≥0.875 on held-out train embryos; score logged and compared to current leaderboard #1 (0.875); if gap remains, documented as overfitting risk or architectural limitation (triggers Phase 5 investigation).

---

### Phase 5: Competitive Iteration Loop

**Goal:** Establish a rapid weekly cycle of local evaluation, targeted optimization, Kaggle submission, and leaderboard tracking, while guarding against overfitting to the ~29% visible test slice.

**Exit Criterion:** Sustained top-of-leaderboard position through final entry deadline (2026-09-22), with weekly local eval → change → submit rhythm established and monitored for divergence between local metric and public leaderboard score.

**Dependencies:** Phase 4 complete; ongoing competition feedback

**Requirements Mapped:** v2's ITER-02, ITER-03 (model ensembling, node-embedding preprocessing); VIZ-01, VIZ-02 (napari QC tooling, Neuroglancer web UI for results visualization)

**Success Criteria (Observable):**

1. **Weekly eval → change → submit cycle in place:** Every 7 days, run local eval on held-out train embryos, identify bottleneck (edge Jaccard, division recall, runtime, etc.), make targeted code change, validate locally, upload Kaggle submission, log public leaderboard rank and score; cycle tracked in a simple `ITERATION_LOG.md`.

2. **Local metric correlates with public leaderboard:** Weekly local scores and Kaggle leaderboard scores plotted side-by-side; correlation coefficient measured; if divergence > 0.05 points, investigation triggered (possible signs of public-slice overfitting or hidden test shift).

3. **napari QC tooling enables visual inspection:** `napari` + `napari-geff` (or equivalent) installed and workflow documented; for each large score change, pull a sample embryo, load predictions and ground truth side-by-side in napari, visually inspect agreement (e.g., "division events look realistic", "no phantom tracks"); at least 2 embryos inspected per phase.

4. **Leaderboard rank tracked and stable:** Weekly leaderboard rank logged; target ≥ rank 5 by end of Phase 4, rank 1–3 by entry deadline; if rank drops > 2 positions between submissions, revert to previous best submission and investigate regressor.

---

## Progress

| Phase | Status | Goal | Exit Criterion |
|---|---|---|---|
| **0 — Unblock** | ✓ Complete (2026-07-04) | Schema-valid submission from real data, scored locally | Submission CSV + local score computed -- MET (18,735-row valid submission, score 0.0092) |
| **1 — Baseline parity** | Ready to plan | Prove pipeline is sound | Local score ≥ 0.763 (classical baseline) -- **carry-forward risk**: Phase 0's real run shows the placeholder detector produces raw stride-8 grid points, not real peaks; likely needs actual peak-finding/NMS (not training) before 0.763 is reachable, see PLAN-04 SUMMARY.md |
| **2 — Learned detection** | Blocked by Phase 1 | Train model, beat baseline, submit to Kaggle | Local score ≥ 0.80, Kaggle submission live |
| **3 — Scale & correctness** | Blocked by Phase 2 | Make ILP tractable, real-intensity mitosis | Full volumes in <12 hours, no truncation |
| **4 — Metric-directed tuning** | Blocked by Phase 3 | Systematic parameter calibration | Local score ≥ 0.875 (leaderboard #1) |
| **5 — Competitive iteration loop** | Blocked by Phase 4 | Weekly cycle, leaderboard climb, visualizations | Sustained rank ≥ 3, no overfitting divergence |

---

## Requirement Traceability

| Requirement | Phase | Status |
|---|---|---|
| DATA-01 | Phase 0 | Complete |
| DATA-02 | Phase 0 | Complete |
| DATA-03 | Phase 0 | Complete |
| DATA-04 | Phase 0 | Complete |
| DATA-05 | Phase 0 | Complete |
| DATA-06 | Phase 0 | Complete |
| SUB-01 | Phase 0 | Complete |
| SUB-02 | Phase 0 | Complete |
| SUB-03 | Phase 0 | Complete |
| EVAL-01 | Phase 0 | Complete |
| EVAL-02 | Phase 0 | Complete |
| EVAL-03 | Phase 0 | Complete |
| EVAL-04 | Phase 0 | Complete |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| MODEL-04 | Phase 2 | Pending |
| TRACK-01 | Phase 2 | Pending |
| TRACK-02 | Phase 3 | Pending |
| TRACK-03 | Phase 3 | Pending |

**Coverage:** 20/20 v1 requirements mapped, 0 orphaned. Phase 0 complete (2026-07-04). Phase 1
(Baseline parity) has no dedicated new requirements—it validates Phase 0's infrastructure with
the existing placeholder detector, but carries a real risk from Phase 0's finding: that detector
needs real peak-finding (not just wiring) to have a realistic shot at the 0.763 baseline.

---

**Roadmap defined:** 2026-07-03  
**Ready for planning:** Approve roadmap, then run `/gsd:plan-phase 0` to break Phase 0 into executable plans.
