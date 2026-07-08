---
phase: 1
plan: ad-hoc (plan-mode approved, not gsd-planner/gsd-executor)
subsystem: Detection (peak-finding) + ILP candidate scaling
tags: [peak-finding, nms, threshold-calibration, candidate-cap, ilp-scaling]
completed: 2026-07-08
duration: multi-session (interrupted twice by real system sleep gaps, see Deviations)
dependencies:
  requires: [Phase 0 complete]
  provides:
    - Real 3D peak-finding (scipy.ndimage.maximum_filter-based NMS) replacing the
      stride-8 grid scan in run_pipeline.py's extract_peaks_from_volume()
    - Recalibrated CNN_THRESHOLD=0.85 / UNET_THRESHOLD=0.9 (via scripts/sweep_threshold.py)
    - Zero motion vectors (was a hardcoded non-zero constant, a real bug)
    - MAX_CANDIDATES_PER_TIMEPOINT raised 30->75, profiled safe with SCIP
  affects:
    - Phase 2 (Learned detection) -- planning input: real peak-finding alone does
      NOT reach the 0.763 baseline; a trained detector with real precision (not
      just real peak-finding on raw intensity) is required, not optional
    - Phase 3 (Scale & correctness) -- direct empirical confirmation that even
      SCIP-accelerated ILP solving cannot absorb the full real candidate density
      (up to ~1150/timepoint in dense regions) without further scaling work
---

# Phase 1: Baseline Parity — SUMMARY

**Objective:** Prove the pipeline is sound by wiring the existing untrained placeholder
detector and validating it reaches the classical baseline (local score >= 0.763), per
`ROADMAP.md`'s Phase 1 exit criterion.

**Status:** NOT MET — local score reached 0.0259, far below the 0.763 baseline. This is
real, honestly-reported Phase 1 signal, not a defect in this phase's execution: real
peak-finding is now genuinely working (verified directly, not assumed), and the score
still falls dramatically short. See "Why the baseline wasn't reached" below.

---

## What Was Built

- **Real peak-finding**: `run_pipeline.py`'s `extract_peaks_from_volume()` rewritten from
  a raw stride-8 grid scan (Phase 0's placeholder, confirmed to produce zero real cell
  matches) to genuine 3D non-max suppression, replicating the host's own documented
  approach (`REFERENCE_IMPLEMENTATION.md` §5): a physical-micron-sized pooling kernel via
  new `pool_kernel_from_um()`, applied to the raw quantile-normalized intensity volume.
- **NMS tie-explosion bug found and fixed**: the host's reference approach assumes sharp,
  point-like logits from a trained model. Applied naively (`vol == pooled`) to raw real
  microscopy intensity — which has broad, flat-topped bright regions, not sharp peaks —
  every voxel in a plateau ties for "peak," producing ~282,000 candidates per timepoint
  (6.7% of all voxels) at every threshold from 0.5 to 0.98, virtually
  threshold-independent. Fixed by collapsing each connected tied region to one centroid
  via `scipy.ndimage.label`/`center_of_mass`. Verified directly on real data (one
  timepoint, one dataset): 282,000 -> 289-395 peaks depending on threshold, a sane order
  of magnitude.
- **Performance fix**: the initial NMS implementation used `torch.nn.functional.max_pool3d`,
  which took ~13s/timepoint for the kernel size this project needs (3,13,13) on CPU.
  Switched to `scipy.ndimage.maximum_filter` (separable, highly optimized) — verified
  ~22x faster (0.6s/timepoint) with identical output.
- **Threshold recalibration**: built `scripts/sweep_threshold.py`, ran across all 4 staged
  train datasets x 5 sample timepoints x 7 threshold values. The old `0.92`/`0.94`
  constants (tuned for the retired grid scan) were meaningless for the new algorithm.
  Locked in `CNN_THRESHOLD=0.85`, `UNET_THRESHOLD=0.9` — candidate counts proved fairly
  threshold-insensitive in the 0.8-0.95 range; real signal is dominated by per-timepoint
  cell density, not threshold choice.
- **Motion vector bug fixed**: Phase 1's locked decision (`01-CONTEXT.md`) was zero motion
  vectors. Direct grep of the actual code found it was NOT already zero — every centroid
  at every timepoint got a hardcoded non-zero `[0.05, 0.2, 0.3]` constant, a leftover from
  simulated-data testing. Fixed to `[0.0, 0.0, 0.0]`.
- **`MAX_CANDIDATES_PER_TIMEPOINT` raised 30 -> 75**: the original cap-30 run scored only
  0.0153, with 786/786 detection-timepoints across a full 4-dataset run hitting the cap —
  i.e. the cap, not detection quality, was the dominant limiter once real peak-finding was
  in place. Profiled directly (not guessed) before raising: on the densest staged
  dataset's dense tail (t=85-99, avg ~1110 real candidates/timepoint), SCIP solve time at
  cap=30/50/75/100 was 1.97s/4.99s/13.44s/27.09s for a 15-timepoint slice — confirmed
  super-linear but tractable. A single full-dataset validation run at cap=75 completed in
  7.2 minutes (145.8s detection + 287s tracking, 6884 nodes, 1996 edges) before committing
  to the full 4-dataset run.

## Real Results (verified directly, not from agent self-report)

- **Score progression**: `0.0092` (Phase 0, grid-scan placeholder) -> `0.0153` (real NMS,
  cap=30) -> `0.0259` (real NMS, cap=75). ~2.8x improvement over Phase 0, but the 0.763
  baseline needs another ~30x from here.
- **Final metrics** (cap=75, all 4 staged train datasets): `edge_jaccard=0.0241`,
  `adjusted_edge_jaccard=0.0259`, `division_jaccard=0.0`, `combined=0.0259`.
- **Truncation is still the dominant limiter even at cap=75**: 770/800 detection-timepoints
  (96%) across the full run still exceeded the cap and were truncated. Raising the cap
  clearly helps (30->75 nearly doubled the score) but has not come close to closing the
  gap, and further raises hit real, profiled ILP scaling limits (super-linear growth
  confirmed above).
- All 61 unit/integration tests pass after every change in this phase (peak-finding
  rewrite, NMS fix, threshold recalibration, motion-vector fix, cap raise).

## Why the baseline wasn't reached

This is a genuine two-part gap, not a wiring problem:

1. **Detection precision, not just peak-finding.** The placeholder detector now does real
   NMS on raw intensity — but "real peak-finding" only finds genuine local maxima; it
   cannot distinguish a real cell from any other locally-brightest blob (out-of-focus
   glow, tissue autofluorescence, noise). Real candidate counts of ~1000+/timepoint in
   dense/late-development regions likely include many false positives a trained
   classifier would reject. This is exactly Phase 2's job (`MODEL-01..04`), not
   something further Phase 1 tuning can fix.
2. **ILP candidate capacity, even with SCIP.** `STHypergraphTracker.solve_lineage()`
   solves one global ILP per dataset; solve time scales super-linearly with
   candidates/timepoint. SCIP (already swapped in from CBC, 11.7x faster) bought real
   headroom, but not unlimited headroom — the profiled data above shows cap=100 already
   approaching a per-15-timepoint cost that would not comfortably scale to cap=500+ across
   a full 100-timepoint, multi-dataset run. This is Phase 3's `TRACK-03` scope
   (windowed/rolling-horizon solving or a min-cost-flow reformulation), now with direct
   empirical grounding for exactly how much more headroom is needed.

## Deviations from Plan

- This phase was executed under an ad hoc plan-mode-approved plan (`c-users-hemas-downloads-st-act-pipeline-snoopy-kite.md`)
  rather than the normal `/gsd:plan-phase 1` -> `/gsd:execute-phase 1` cycle, because the
  harness entered plan mode mid-`/gsd:discuss-phase 1` and restricted edits to the plan
  file until approved. `01-CONTEXT.md` was written normally (captures the 4 discussed
  decisions); no `01-PLAN.md` exists, this `01-SUMMARY.md` is written directly instead.
- Two separate multi-hour wall-clock gaps occurred mid-run from the laptop's default sleep
  timeout suspending the background pipeline process (verified via CPU-time-vs-wall-clock
  analysis, not assumed) — not a code or performance defect. Resolved by disabling sleep
  (`powercfg /change standby-timeout-ac 0` / `-dc 0`) partway through; the final full run
  completed cleanly once sleep was disabled. **Sleep settings should be restored** after
  this phase's work concludes (not yet done as of this summary).
- `MAX_CANDIDATES_PER_TIMEPOINT` was not one of the 4 decisions locked in `01-CONTEXT.md`
  discussion, but was raised as a direct, profiled response to the cap-30 run's own
  finding (786/786 timepoints truncated) — squarely within Phase 1's "investigate before
  declaring done" mandate, not scope creep.

## Verification Checklist

- ✓ `py -m pytest tests/` — 61 passed, 1 skipped, after every change in this phase.
- ✓ Threshold sweep run on all 4 staged train datasets, candidate counts sane (9-8145,
  correctly threshold- and timepoint-sensitive, no explosion) — see `scripts/sweep_threshold.py` output.
- ✓ Single-dataset full-run validation at cap=75 before committing to the full 4-dataset run.
- ✓ Full pipeline run completed on all 4 staged train datasets, real score computed:
  `combined=0.0259`.
- ✗ **Local score >= 0.763 NOT achieved** (0.0259). Phase 1's exit criterion is not met.
  Root cause identified and documented above, not silently patched or hidden.

## Recommendation

Do not attempt further Phase 1-scoped tuning (e.g. raising the cap again) as the primary
lever — the two-part gap above (detection precision + ILP capacity) is now well
quantified, and both root causes point to work already scoped for Phase 2 and Phase 3.
Proceed to Phase 2 (Learned Detection) planning; carry forward Phase 3's `TRACK-03` ILP
scaling requirement with this phase's real profiling data as a starting point instead of
Phase 0's synthetic estimates.
