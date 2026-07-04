---
phase: 0
plan: 04
subsystem: Pipeline Integration & E2E Validation
tags: [integration, e2e-test, ilp-performance, run-tracking, detector-quality]
completed: 2026-07-04
duration: multi-session (2 executor stalls recovered manually; see Deviations)
dependencies:
  requires: [Plan 00, Plan 01, Plan 02, Plan 03]
  provides:
    - Real end-to-end pipeline run (run_pipeline.py) on all 8 staged datasets
    - output/phase0_submission.csv (schema-valid, 4 test datasets)
    - src/run_tracker.py (progress/ETA, detection cache, dataset checkpointing)
    - scripts/compare_runs.py (cross-run comparison)
  affects:
    - Phase 1 (Baseline parity) -- planning input: current placeholder detector
      cannot realistically reach 0.763 without real peak-finding, not just
      threshold tuning (see Key Finding below)
    - Phase 3 (Scale & correctness) -- direct empirical confirmation the ILP
      is the dominant cost (70.2% of runtime) even at heavily-capped candidate
      density
---

# Phase 0, Plan 04: Wire Pipeline and Test — SUMMARY

**Objective:** Integrate the data loader, tracker, exporter, and evaluation harness into
`run_pipeline.py`, then execute an end-to-end test on real staged data to generate a
schema-valid, scoreable submission.

**Status:** COMPLETE ✓ (Phase 0 exit criterion met; see Key Finding for what it reveals about Phase 1)

---

## What Was Built

- `run_pipeline.py` refactored into two explicit passes sharing the same `run_dataset()` code:
  **submission pass** (`--test-dir`, 4 real test datasets, no ground truth) and **local scoring
  pass** (`--train-dir`, same 4 embryo IDs' train/ copies, has `.geff` ground truth) — kept
  deliberately separate so the same embryo IDs appearing in both folders don't get double-counted
  into 8 submission blocks instead of 4.
- `src/run_tracker.py`: `RunTracker` (structured JSONL event log, streamed incrementally so a
  killed run leaves a diagnosable partial record; live progress + ETA), detection-result caching
  (keyed on zarr path+mtime+thresholds, independent of tracker costs), and per-dataset
  checkpointing (keyed on the full config) so a rerun after a partial failure only reprocesses
  what changed or failed.
- `scripts/compare_runs.py`: reads `logs/runs/*.jsonl` into a comparison table or full per-run detail.
- Real end-to-end run completed: `py run_pipeline.py --test-dir data/staging/test/ --train-dir
  data/staging/train/ --output-path output/phase0_submission.csv`

## Real Results (verified directly, not from agent self-report)

- **Submission**: `output/phase0_submission.csv`, 18,735 rows, 11,807 node rows + 6,927 edge
  rows, exactly the 4 required test datasets, passes `validate_submission()`.
- **Timing**: Pass 1 (submission) 199.8s, Pass 2 (scoring) 295.7s — **~8.3 minutes total for all
  8 dataset-runs**. ILP tracking is **70.2% of total time** (205s + 222s of ~500s combined) —
  confirms Phase 3's "ILP solve time at scale" concern is real, not hypothetical, even at a
  heavily-capped 30 candidates/timepoint.
- **Local score**: `edge_jaccard=0.0084, adjusted_edge_jaccard=0.0092, division_jaccard=0.0,
  score=0.0092`. Far below the 0.763 baseline.

## Key Finding: why the score is near-zero, and what it means for Phase 1

Inspecting the submission's actual coordinates shows a rigid pattern: `(z=0,y=4,x=52)`,
`(z=0,y=4,x=60)`, `(z=0,y=4,x=68)` — steps of exactly 8, matching `extract_peaks_from_volume`'s
stride-8 grid sampling exactly. **The placeholder detector's output is raw grid points that
exceeded an intensity threshold, not cell centroids** -- there's no peak-finding/NMS step at all.
Real ground truth is sparse (tens to low-thousands of labeled cells per sample); the odds of a
rigid grid point landing within the 7µm match tolerance of one of them are naturally tiny,
independent of threshold value.

**This means Phase 1's roadmap assumption should be revisited**: "wire the existing (untrained)
placeholder detection model... validate it reaches the classical baseline (0.763)" assumes the
current detector is capable of that with correct wiring alone. This run is strong empirical
evidence it is not -- it needs genuine peak-finding (local maxima / non-max suppression) before
threshold tuning can do anything useful, independent of any model training (that's Phase 2's
job). `REFERENCE_IMPLEMENTATION.md` §5 already has the host's real NMS approach
(`max_pool3d`-based, kernel sized from physical µm) documented from earlier research -- adopting
at least that much (still without training a model) is likely necessary, not optional, for Phase
1 to have a realistic shot at 0.763.

## Also Fixed This Plan (discovered via direct verification, not assumed)

1. **Detector threshold miscalibration** caused a 2.5+ hour stuck run (30GB+ RAM, no output)
   before this plan's real run: `extract_peaks_from_volume`'s original thresholds (0.4/0.45,
   tuned against old simulated `[0,1]`-uniform data) produced ~18,000 candidates in a single real
   timepoint against quantile-normalized real data, which the ILP tracker could not handle.
   Raised thresholds (0.92/0.94) plus a hard `MAX_CANDIDATES_PER_TIMEPOINT` safety cap (empirically
   tuned via direct profiling of `solve_lineage`'s super-linear scaling, not guessed) resolved it.
2. `export_submission()` used `graph.nodes()`/`graph.edges()` (networkx-style call syntax);
   tracksdata's actual API on `IndexedRXGraph` only supports `.node_ids()`/`.edge_list()` as
   properties, not callables. Fixed and verified against all 18 exporter tests plus a hand-built
   2-dataset CSV inspection.

## Deviations from Plan

- The plan's own default `scale=(4.0,1.0,1.0)` in the eval harness spec (Plan 02) was the
  anisotropy ratio, not the real physical micron scale -- fixed during Plan 02 (see its SUMMARY).
- Two executor-agent runs stalled mid-stream on this plan specifically (long-running: one at ~26
  min/67 tool calls, one at ~49 min/67 tool calls). Both times, real progress had been made and
  committed or left as recoverable uncommitted diffs -- recovered by direct git/file inspection
  rather than re-running from scratch. Root cause of the stalls themselves not diagnosed (agent
  infrastructure, not this codebase); noted as an operational pattern to watch for future
  long-running plans.
- Task 04-06 (timing instrumentation) and 04-05 (spot-check) were completed as part of recovering
  from the second stall, alongside the core 04-01..04-03 wiring.

## Verification Checklist

- ✓ `run_pipeline.py --help` works, accepts `--test-dir`/`--train-dir`/`--output-path`/`--force-rerun`
- ✓ Submission pass produces exactly 4 dataset blocks (not 8)
- ✓ `output/phase0_submission.csv` exists, schema-valid, passes `validate_submission()`
- ✓ Local scoring pass runs against `train/` + matching `.geff`, separately from the submission file
- ✓ Full real run completed end-to-end: 8.3 minutes, real score computed (0.0092, below baseline --
  expected given the detector-quality finding above, not a pipeline defect)
- ✓ Run-tracker JSONL log + live progress/ETA verified working on the real run (checkpoint cache
  hit confirmed for a previously-processed dataset: 0.01s instead of ~61s)
