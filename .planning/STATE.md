# Project State: ST-ACT — Spatio-Temporal Anisotropic Cell Tracker

**Last Updated:** 2026-07-04

---

## Project Reference

**Core Value:** A Kaggle competition submission that is schema-valid, scores above the classical baseline (0.763), and is engineered to compete for the top of the leaderboard.

**Target Leaderboard:** Biohub – Cell Tracking During Development (Kaggle)
- **Current #1 Score:** 0.875 (Kaushik Ramayya Chikkala)
- **Entry Deadline:** 2026-09-22 (~11 weeks from 2026-07-03)
- **Public Test Data Coverage:** ~29% revealed; full hidden test set used for final ranking

**Key Constraints:**
- Kaggle notebook runtime limit: 12 hours
- PyTorch + MONAI (not Keras 3/JAX) for 3D biomedical volume handling
- Competition is embryo-disjoint train/test (no data leakage)
- Submissions are rate-limited and precious

**Core Dependencies/Context:**
- Kaggle API + remote-GPU-kernel setup in parallel worktree (`../st_act_pipeline-kaggle-setup`, branch `kaggle-setup`) — Phase 2 training depends on this; Phase 0/1 do not
- ILP tracker flow-conservation bug already fixed this session; isolated nodes now resolve correctly instead of causing Infeasible

---

## Current Position

**Roadmap Status:** Phase 0 (Unblock) — ✓ COMPLETE (2026-07-04, verified by gsd-verifier, 5/5
must-haves, 60/60 tests passing). Phase 1 (Baseline Parity) — ATTEMPTED, exit criterion NOT
MET (2026-07-08): real peak-finding shipped and verified working, local score reached
0.0259 (up from 0.0092), still far below the 0.763 baseline. Root cause quantified, not
hidden — see `01-SUMMARY.md`. Ready to plan Phase 2.

**Phase Structure:**
```
Phase 0 (Unblock) -- ✓ COMPLETE 2026-07-04
  ├─ ✓ Plan 00: Vendor Scoring Code (COMPLETE)
  ├─ ✓ Plan 01: Data Loading Pipeline (COMPLETE)
  ├─ ✓ Plan 02: Evaluation Harness (COMPLETE)
  ├─ ✓ Plan 03: Submission Exporter (COMPLETE)
  └─ ✓ Plan 04: Pipeline Wiring + real E2E run (COMPLETE -- 8.3min, score 0.0092)
  └─ Phase 1 (Baseline parity) -- ATTEMPTED, EXIT CRITERION NOT MET 2026-07-08
       ├─ Real peak-finding (max_pool3d/maximum_filter NMS) replacing grid scan -- DONE
       ├─ NMS tie-explosion bug found+fixed (282k false peaks -> sane counts) -- DONE
       ├─ Threshold recalibration (0.85/0.9) -- DONE
       ├─ Motion-vector bug fixed (was nonzero constant, now correctly zero) -- DONE
       ├─ MAX_CANDIDATES_PER_TIMEPOINT raised 30->75 (profiled safe with SCIP) -- DONE
       └─ Score: 0.0092 -> 0.0153 -> 0.0259 -- still far below 0.763 baseline
  └─ Phase 2 (Learned detection) -- READY TO PLAN (carries forward Phase 1's finding:
       real detection precision, not just peak-finding, is required)
       └─ Phase 3 (Scale & correctness) -- carries forward real ILP scaling profile data
            └─ Phase 4 (Metric-directed tuning)
                 └─ Phase 5 (Competitive iteration loop)
```

**Current Sprint:** Phase 1 attempted and honestly closed out (see `01-SUMMARY.md`). Next:
`/gsd:discuss-phase 2` (or `/gsd:plan-phase 2` directly).

**v1 Requirement Coverage:**
- Total v1 requirements: 20
- Phase 0: 13 requirements (DATA-01..06, SUB-01..03, EVAL-01..04) -- ALL COMPLETE
- Phase 1: 0 requirements (validation only)
- Phase 2: 5 requirements (MODEL-01..04, TRACK-01)
- Phase 3: 2 requirements (TRACK-02, TRACK-03)
- Coverage: 19/19 ✓

---

## Performance Metrics

**Primary Success Signal (Competition):**
- Leaderboard rank and score, updated weekly through 2026-09-22
- Target: Rank #1 by final deadline

**Leading Indicator (Local):**
- `edge_jaccard + 0.1 × division_jaccard` on held-out train embryos (with real `.geff` ground truth)
- Must correlate with public leaderboard before trusted as decision signal
- Current baseline (classical non-learned): 0.763
- Phase 2 target: ≥0.80
- Phase 4 target: ≥0.875

**Guardrails:**
- Local metric on held-out embryos must not diverge from public leaderboard by >0.05 points (signals overfitting to visible 29% slice)
- Division recall tracked separately (0.1-weighted but high-leverage for top-tier separation)
- Full-size `(100, 64, 256, 256)` volumes must process end-to-end in <12 hours by Phase 3

---

## Accumulated Context

### Decisions

1. **Phase structure mirrors PRD § 8 exactly.** PRD's 6-phase roadmap is well-formed and aligns with requirements; no reordering or restructuring needed. All phases 0–5 included in v1 roadmap.

2. **PyTorch + MONAI chosen over Keras 3/JAX.** Framework selection rationale: ecosystem maturity for 3D biomedical volumes (sliding-window inference, anisotropic augmentations) outweighs JAX's framework-purity; GPU (not TPU) is the compute plan. Noted in REQUIREMENTS.md Out of Scope section.

3. **Local evaluation harness is critical to competition success.** Every model/tracker change must be validated against local harness (edge Jaccard + division Jaccard on held-out embryos) before a Kaggle submission is spent; submissions are scarce and rate-limited.

4. **Overfitting to ~29% public slice is a known risk.** Mitigated by always validating primarily against held-out train embryos with real `.geff` ground truth (not just public leaderboard feedback). Phase 5 includes monitoring for local-vs-public divergence.

5. **Kaggle setup (API + GPU kernel) is parallel work.** Phase 0 and Phase 1 do NOT depend on this (local development). Phase 2 (model training) DOES depend on it being complete. This is tracked separately in `../st_act_pipeline-kaggle-setup` branch; no blocking dependency for immediate Phase 0 work.

6. **Phase 0, Plan 00 executed 2026-07-03 (COMPLETE).** Successfully vendored `metrics.py`, `division_metrics.py`, and `io.py` from `royerlab/kaggle-cell-tracking-competition` via raw GitHub URLs (git clone unavailable in execution environment; fallback to HTTP fetch). Three commits created (1e3fabd, c6249bb, b710b1d); SUMMARY.md written. Scoring code now trusted dependency, ready for downstream plans (02, 04, 05).

7. **Phase 0, Plan 01 executed 2026-07-03 (COMPLETE).** Fixed AnisotropicZarrLoader to correctly load Zarr v3 OME-NGFF format and apply correct physical anisotropy (4.0, 1.0, 1.0) Z:Y:X (not hardcoded 5.0). This bug affected both real data paths and simulated fallback. Real data tests now pass on staged training data (44b6_0113de3b.zarr and siblings). Data loader ready for tracker and model pipeline in Phase 1.

8. **Phase 0, Plan 02 executed 2026-07-03 (COMPLETE).** Implemented local evaluation harness (`src/evaluation.py`, 226 lines) with clean API: `evaluate_submission(pred_graphs, gt_graphs, scale, max_distance, gt_metadata)`. Provides: micro-averaged edge Jaccard (via `tracksdata.evaluate_datasets()`), division Jaccard (via vendored `evaluate_divisions()`), node-count adjustment formula (`J_adj = max(0, J * (1 - 0.1 * (T_pred - T_true) / T_true))`), and combined score. Correctly drops division term when no GT divisions exist (not `+0`). Helpers: `load_geff_ground_truth()` and `load_gt_for_dataset()`. Unit tests (pytest + standalone runner) created; use real staged .geff data (52+ nodes per sample). Ready for Plan 04 (pipeline wiring) and all downstream submission validation.

### Pending Todos

- [x] **Phase 0, Plan 00 (Vendor Scoring Code) — COMPLETE 2026-07-03:** Executed all tasks; vendored metrics/division_metrics/io; __init__.py created; baseline test written. SUMMARY.md and commit artifacts created.
- [x] **Phase 0, Plan 01 (Data Loading Pipeline) — COMPLETE 2026-07-03:** Fixed AnisotropicZarrLoader for Zarr v3 OME-NGFF with correct anisotropy (4.0, 1.0, 1.0). Real data tests pass on staged .zarr files.
- [x] **Phase 0, Plan 02 (Evaluation Harness) — COMPLETE 2026-07-03:** Implemented `evaluate_submission()` API with `load_geff_ground_truth()` and `load_gt_for_dataset()` helpers. Unit tests written for real staged .geff data. Adjustment formula and division-term handling verified.
- [x] **Phase 0, Plan 03 (Submission Exporter) — COMPLETE 2026-07-03:** Implemented `export_submission()` and `validate_submission()` with 9 schema validation checks. 18 unit tests pass. Schema verified exact match with Kaggle sample_submission.csv. Per-dataset node_id reset (critical requirement) implemented correctly.
- [ ] **Phase 0, Plan 04:** Pipeline wiring
- [ ] **Phase 1 setup:** Reserve 2–3 held-out train embryos for validation (never touched during Phase 2 training)
- [ ] **Confirm Kaggle rules:** Full `/rules` page (team size, external-data policy, compute/runtime caps) not fully retrievable at PRD time; re-verify before Phase 2 investment
- [x] **Confirm reference metric implementation — RESOLVED 2026-07-03:** host publishes a full
  reference implementation, `royerlab/kaggle-cell-tracking-competition` (linked directly from
  `/overview/evaluation`), built on a real `tracksdata` library (`.from_geff()`/`.to_geff()`/
  `DistanceMatching`). Exact formulas, the division-matching algorithm, and vendored source are in
  `REFERENCE_IMPLEMENTATION.md` at repo root. **Scope change for Phase 0 planning:** EVAL-01..04
  should vendor/wrap `tracksdata`'s `metrics.py` + `division_metrics.py` directly rather than
  reimplementing from prose — this removes most of the risk in that task. Also confirmed: the
  `adjusted_edge_jaccard` penalty formula is `max(0, jaccard·(1 − 0.1·(T_pred−T_true)/T_true))`
  where `T_true` = the `.geff`'s `estimated_number_of_nodes` field, and unmatched predicted nodes
  are structurally excluded from the FP count (sparse-GT fairness confirmed, not just implied).
  **Reconcile with this session's separate geff/traccuracy decision below:** `tracksdata` wraps
  `geff` and is the same-family tool that implements this competition's **exact bespoke** score;
  `traccuracy` computes generic CTC metrics (TRA/DET) which are a **different** formula — treat
  `tracksdata` as primary for FR-5, keep `traccuracy` only as an optional secondary sanity check.

### Blockers/Concerns

1. **ILP solve time at scale (Phase 3).** Current CBC-based ILP may not scale to thousands of cells × 100 timepoints × gap-closing within Kaggle's 12-hour budget. Mitigation in Phase 3: windowed/rolling-horizon solving or OR-Tools `SimpleMinCostFlow` reformulation (already verified to fit flow-conservation constraints). Must benchmark on realistic synthetic scale early in Phase 2.

2. **Detection model under-capacity (Phase 2).** Current 2-conv-layer FCN may be too shallow for `256×256×64` volumes. Treat as placeholder for Phase 1 validation; budget architecture experimentation in Phase 2 (deeper ResNet-style backbone, 3D convolutions, anisotropy-aware augmentations).

3. **Division events are rare (Phase 4).** Micro-averaged over samples; only 0.1-weighted in overall score, but top-tier separation may hinge on division recall. Phase 4 includes dedicated tracking of division_recall as independent metric.

4. **Hidden test set will reveal different patterns (Phase 5).** Leaderboard is ~29% public; final ranking uses full hidden set. Weekly cycle in Phase 5 monitors for overfitting; if local-vs-public divergence > 0.05, revert to last best submission and investigate.

5. **NEW 2026-07-04 -- Phase 0's real E2E run confirms the placeholder detector cannot realistically
   reach 0.763 with wiring alone (Phase 1 risk, upgraded from theoretical to confirmed).** Real run:
   score 0.0092. Root cause verified directly (not assumed): submission coordinates fall on a rigid
   stride-8 grid (e.g. x=52,60,68,...), exactly matching `extract_peaks_from_volume`'s sampling --
   there's no peak-finding/NMS step, so "detections" are raw thresholded grid points, not centroids.
   Real ground truth is sparse; the odds of a grid point landing within the 7um match tolerance of a
   true labeled cell are near-zero regardless of threshold value. **Action for Phase 1 planning:**
   budget for at least a real peak-finding step (local maxima / non-max suppression) -- the host's
   own NMS approach is already documented in `REFERENCE_IMPLEMENTATION.md` Sec5 (`max_pool3d`-based,
   kernel sized from physical um) and can be adopted without training a model. Without this, Phase 1
   cannot succeed regardless of threshold tuning effort.

6. **NEW 2026-07-04 -- ILP is confirmed (not just flagged) as the dominant runtime cost.** Real run:
   70.2% of total pipeline time even at a hard-capped 30 candidates/timepoint (down from an
   unbounded original that caused a 2.5+ hour stuck run at higher candidate density). This is
   Blocker #1 above, now with direct empirical grounding from real data rather than only synthetic
   profiling -- raises confidence that Phase 3's windowed/min-cost-flow mitigation is necessary, not
   precautionary.

5. **NEW 2026-07-03 — ILP cost-scale caps realistic per-frame movement, confirmed via a new
   positive/negative test suite (`tests/test_tracker.py`, 14 tests, all passing against current
   `src/tracker.py`).** The edge cost is `distance² × gap_penalty`; linking only beats paying two
   isolated birth+death costs when `distance² < birth_cost + death_cost`. With the actual
   production costs (`birth_cost=15, death_cost=15` in `run_pipeline.py`), that break-even is
   **~5.48 microns of movement per frame** — verified with an exact test asserting a link forms
   just under that threshold and doesn't just over it. If real embryonic nuclei move faster than
   that between frames (plausible during active development), the ILP will systematically prefer
   birth+death singletons over correct links regardless of detection quality, silently capping
   `edge_jaccard` recall. **Also confirmed:** `prune_unphysical_edges` gates on raw (unwarped)
   coordinates, not motion-compensated ones — a genuinely fast-moving cell whose raw per-frame
   jump exceeds the prune gate (`max_xy_micron=30`/`max_z_micron=15`, scaled by frame-gap) can
   never be rescued by an accurate motion vector, no matter how good MODEL-04's prediction is.
   **Action:** if Phase 1's baseline-parity score comes in surprisingly low, check this before
   assuming detection/data is broken — it may just be birth/death costs needing recalibration
   earlier than Phase 4. Consider moving cost calibration (currently v2 ITER-01/Phase 4) earlier
   if Phase 1 numbers look off for no other explainable reason.

   **REFINED 2026-07-03 — measured against real ground truth, not just theoretical.** Computed
   actual inter-frame displacement from all 4 staged `.geff` files (2127 real consecutive-frame
   edges, physical units via the real anisotropy `(1.625,0.40625,0.40625)`): median 0.91–2.88µm,
   mean 1.33–3.12µm per sample, **only 3.1% of real edges overall exceed the 5.48µm break-even**
   (per-sample range 0%–8%). So this is a real but *narrow* risk, not a systemic one — most true
   links are well inside the safe zone where current costs correctly favor linking. Revised
   guidance: don't preemptively recalibrate costs before Phase 1 (the earlier "consider moving
   this earlier" suggestion is downgraded) — but if division recall specifically underperforms in
   Phase 4, check whether the affected cells are disproportionately in that fast-moving tail
   first, since rapid movement plausibly correlates with mitosis (dividing cells can move/deform
   quickly right around the split), which would make this a real, if secondary, driver of the
   division_jaccard gap rather than the mitosis-smoothing logic itself.

---

## Session Continuity

**Session Start:** 2026-07-03 (roadmap creation)  
**Session Update:** 2026-07-08 (Phase 1 attempted and honestly closed out, exit criterion not met, ready for Phase 2)

**Completed 2026-07-05 to 2026-07-08 (Phase 1, plus carried-over tooling from 2026-07-04):**
- CBC->SCIP ILP solver swap in `src/tracker.py` (11.7x real-data speedup, verified score-identical)
- ruff/mypy/pre-commit tooling set up (`pyproject.toml`, `.pre-commit-config.yaml`)
- `/gsd:discuss-phase 1` run; 4 decisions locked in `01-CONTEXT.md` (real NMS peak-finding,
  threshold sweep, zero motion vectors, use-all-4-samples validation set)
- Real peak-finding implemented in `run_pipeline.py` (`pool_kernel_from_um()` + rewritten
  `extract_peaks_from_volume()`), replacing the stride-8 grid scan
- **Found and fixed a real NMS bug**: naive `vol == pooled` comparison on raw (non-logit)
  intensity ties across flat plateau regions, producing ~282,000 false "peaks"/timepoint
  (6.7% of all voxels) at every threshold tested -- fixed via connected-component
  centroid collapsing (`scipy.ndimage.label`)
- **Found and fixed a real performance bug**: `torch.max_pool3d` took ~13s/timepoint for
  this project's kernel size on CPU; switched to `scipy.ndimage.maximum_filter`, verified
  ~22x faster with identical output
- **Found and fixed a real motion-vector bug**: code was NOT using zero vectors as assumed
  -- a hardcoded non-zero `[0.05, 0.2, 0.3]` constant was still in place; fixed to `[0.0, 0.0, 0.0]`
- Threshold recalibration via `scripts/sweep_threshold.py`: `CNN_THRESHOLD=0.85`, `UNET_THRESHOLD=0.9`
- `MAX_CANDIDATES_PER_TIMEPOINT` raised 30->75, validated via direct profiling (dense-tail
  solve-time scaling measured, not guessed) before committing to a full run
- Full pipeline run on all 4 staged train datasets: **local score 0.0259** (up from 0.0092,
  but still far below the 0.763 baseline) -- see `01-SUMMARY.md` for full root-cause analysis
- `/graphify` run on the full project (803 nodes, 1316 edges, 47 communities) -- confirmed
  the vendored-scoring-code integration and orchestration structure match what the code
  actually does; no code-level action items surfaced, used for navigation/verification only
- **Operational note**: two multi-hour wall-clock gaps this session were caused by the
  laptop's default sleep timeout suspending long-running background pipeline processes
  (confirmed via CPU-time-vs-wall-clock analysis), not code or performance defects.
  Disabled via `powercfg /change standby-timeout-ac 0` / `-dc 0` mid-session --
  **still needs to be restored to the user's normal setting**, not yet done.

**Session Update (2026-07-04):** Phase 0 fully complete, verified, ready for Phase 1

**Completed This Session (2026-07-03 to 2026-07-04):**
- Roadmap initialized (6-phase structure); REFERENCE_IMPLEMENTATION.md created and extended
  multiple times with real host-repo findings (metrics.py/division_metrics.py/io.py source,
  quantile_normalize exact percentiles, NMS peak detector, greedy-vs-ILP edge assignment)
- Phase 0, Plans 00-03 executed and independently verified (each caught real bugs post-execution
  rather than trusting "PLAN COMPLETE" claims -- see individual SUMMARY.md files for detail)
- Phase 0, Plan 04 executed: `run_pipeline.py` wired end-to-end (two-pass design: submission from
  test/, scoring from train/), plus:
  - Root-caused and fixed a 2.5+ hour stuck run (detector threshold miscalibration against real
    quantile-normalized data -> ~18,000 false candidates/timepoint -> ILP combinatorial blowup)
  - Built `src/run_tracker.py`: JSONL run logging, live progress/ETA, detection-result caching,
    per-dataset checkpointing; `scripts/compare_runs.py` for cross-run comparison
  - Fixed a real `tracksdata` API bug in the submission exporter (`.nodes()`/`.edges()` aren't
    callable on this version; `.node_ids()`/`.edge_list()` are)
  - **Real end-to-end run completed**: 8.3 minutes, schema-valid submission (18,735 rows), local
    score 0.0092 (expected given the detector-quality finding, not a defect)
- `gsd-verifier` confirmed Phase 0 PASSED: 5/5 must-haves, 60/60 tests passing (verified directly,
  not from agent self-report) -- see `.planning/phases/00-unblock/00-UNBLOCK-VERIFICATION.md`
- ROADMAP.md/REQUIREMENTS.md/STATE.md updated to reflect Phase 0 completion (this update)

**What Happens Next:**
1. `/gsd:discuss-phase 2` (or `/gsd:plan-phase 2` directly) -- Learned Detection
2. Phase 2 planning must account for Phase 1's confirmed finding: real peak-finding alone
   (no learned classification) reaches only 0.0259, ~30x short of baseline -- a trained
   detector with real precision (distinguishing true cells from bright noise/glow) is
   required, not optional. See `01-SUMMARY.md`'s "Why the baseline wasn't reached."
3. Restore normal power/sleep settings (`powercfg /change standby-timeout-ac 15` /
   `-dc 10` or the user's actual prior values) -- disabled mid-Phase-1 to prevent
   multi-hour background-run interruptions, not yet restored.
4. Decide whether to commit the accumulated uncommitted changes (SCIP swap, tooling
   setup, Phase 1's peak-finding rewrite) -- all still sitting uncommitted as of this update.

**Risk Status:**
- ✓ Scoring code parity risk RESOLVED: reference implementation vendored exactly
- ✓ Phase 0 fully complete and verified
- ✓ Phase 1 risk CONFIRMED AND QUANTIFIED (was theoretical, now measured): real
  peak-finding alone reaches only 0.0259 local score. Root cause is now a well-defined
  two-part gap (detection precision + ILP candidate capacity), not an open question --
  see `01-SUMMARY.md`. Phase 2 and Phase 3 scope is now grounded in real measurements
  instead of Phase 0's synthetic estimates.
- ⚠ ILP scaling (Phase 3) risk CONFIRMED with real profiling data: solve time scales
  super-linearly with candidates/timepoint even under SCIP (cap 30/50/75/100 ->
  1.97s/4.99s/13.44s/27.09s for a 15-timepoint dense slice) -- real numbers now available
  to size Phase 3's windowed/min-cost-flow work, not just Phase 0's synthetic profile.
- Kaggle setup (parallel work) still needed before Phase 2 training starts
- ILP scale-up must be validated early (Phase 2/Phase 3 boundary)

---

**Last update:** 2026-07-08 — Phase 1 (Baseline Parity) attempted and honestly closed out;
exit criterion not met (score 0.0259 vs 0.763 baseline), root cause quantified, ready for
Phase 2 planning.
**Next step:** `/gsd:discuss-phase 2` (Learned Detection)
