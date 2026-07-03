# Project State: ST-ACT — Spatio-Temporal Anisotropic Cell Tracker

**Last Updated:** 2026-07-03

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

**Roadmap Status:** Phase 0 (Unblock) — In Progress (Plans 00-03 complete, Plans 04 pending)

**Phase Structure:**
```
Phase 0 (Unblock)
  ├─ ✓ Plan 00: Vendor Scoring Code (COMPLETE)
  ├─ ✓ Plan 01: Data Loading Pipeline (COMPLETE)
  ├─ ✓ Plan 02: Evaluation Harness (COMPLETE)
  ├─ ✓ Plan 03: Submission Exporter (COMPLETE)
  └─ Plan 04: Pipeline Wiring
  └─ Phase 1 (Baseline parity)
       └─ Phase 2 (Learned detection)
            └─ Phase 3 (Scale & correctness)
                 └─ Phase 4 (Metric-directed tuning)
                      └─ Phase 5 (Competitive iteration loop)
```

**Current Sprint:** Phase 0, Plan 03 (Submission Exporter) executed and completed successfully

**v1 Requirement Coverage:**
- Total v1 requirements: 19
- Phase 0: 12 requirements (DATA-01..05, SUB-01..03, EVAL-01..04)
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
**Session Update:** 2026-07-03 (Phase 0, Plans 00-03 execution complete)

**Completed This Session:**
- Roadmap initialized (6-phase structure)
- REFERENCE_IMPLEMENTATION.md created (host reference analysis)
- Phase 0, Plan 00 executed: vendored metrics.py, division_metrics.py, io.py
  - 3 atomic commits (1e3fabd, c6249bb, b710b1d)
  - SUMMARY.md written (.planning/phases/00-unblock/PLAN-00-VENDOR-SCORING-CODE-SUMMARY.md)
  - Test file created (tests/test_scoring_baseline.py)
- Phase 0, Plan 01 executed: data loading pipeline for Zarr v3 + GEFF
- Phase 0, Plan 02 executed: evaluation harness with micro-averaged metrics
- Phase 0, Plan 03 executed: submission exporter (export_submission + validate_submission)
  - 4 atomic commits (9d57f63, 196e649, 3ad4ce0 + docs)
  - SUMMARY.md written (.planning/phases/00-unblock/PLAN-03-IMPLEMENT-SUBMISSION-EXPORTER-SUMMARY.md)
  - 18 unit tests (all passing)
  - Schema verified against real sample_submission.csv
- STATE.md updated (Phase 0 progress tracked)

**What Happens Next:**
1. Execute Phase 0, Plan 04 (pipeline wiring) — depends on Plan 03 ✓
2. Validate Phase 1 execution (no new work; just wiring/testing Phase 0's infrastructure)
3. Depending on Phase 1 outcome: proceed to Phase 2 planning or loop back to debug Phase 0

**Risk Status:**
- ✓ Scoring code parity risk RESOLVED: reference implementation vendored exactly
- Remaining Phase 0 work (Plans 01–04) unblocked by Plan 00 completion
- Kaggle setup (parallel work) still needed before Phase 2 training starts
- ILP scale-up must be validated early (Phase 2/Phase 3 boundary)

---

**Last update:** 2026-07-03 (19:30 UTC) — Phase 0, Plan 03 (Submission Exporter) complete  
**Next step:** Execute Phase 0, Plan 04 (Pipeline Wiring)
