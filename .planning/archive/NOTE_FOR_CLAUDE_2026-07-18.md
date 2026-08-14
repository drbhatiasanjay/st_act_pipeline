# Note for the Claude session driving GPU_SANITY_GATE_DESIGN_2026-07-18.md

Read your design doc in full (read-only — did not modify it, per instruction). One grounding
fact in your §0 has changed since you wrote it; everything else checks out as-is.

## P0-7A (metric parity) has now landed — update your §0/§2 accordingly

Your §0 says: *"P0-7A ('metric parity') is stated to be in progress separately — this design
treats current `evaluate_submission()` as **not yet the corrected metric** until P0-7A lands."*
Your §2 prerequisite 1 lists P0-7A as a hard blocker not yet satisfied.

**That has changed as of this session.** P0-7A is integrated and pushed:

- Commit: `8eeace8ab090aecb1cfad336f82865f752c825c4`
- Pushed to `origin/master`, normal fast-forward from `4901d45` (your quoted HEAD)
- Exactly 5 files: `src/tracking_cellmot/division_metrics.py`, `src/tracking_cellmot/metrics.py`,
  their Kaggle mirrors, and `tests/test_p07a_metric_parity.py` (new)
- `src/evaluation.py` (and its Kaggle mirror) — **unchanged**, confirmed via
  `git log -1 --format=%H -- src/evaluation.py` still showing `f139eec...`, not the new commit.
  Your §0 reference to `src/evaluation.py:36`'s `DEFAULT_SCALE` line is unaffected.
- What actually changed: `division_metrics.py`/`metrics.py` synced to the pinned upstream
  commit `royerlab/kaggle-cell-tracking-competition@075fc5f5a52d11077f9dc2b074644618f26939e2`
  (their PR #2, "patch weakly connected component exploit") — real scoring-semantics fixes to
  division matching and edge out-degree/merge-duplicate/consecutive-frame handling, not a
  cosmetic change. `evaluate_submission()` now runs the corrected metric.

**Practical effect on your design:** your §2 prerequisite 1 is now satisfied. Prerequisite 2
(P0-7 training-integrity freeze) is **still not integrated** — I have not started it and was
explicitly told not to. Your gate still cannot run until both land; only one of the two blockers
has cleared.

## Your stale HEAD reference

`4901d450cb12ed6577633b71d4f50bc1bc3c6` (your §0, already self-flagged as 3 hex chars short of
the real `4901d450cb12ed6577634633b71d4f50bc1bc3c6`) is now doubly stale — `master` has moved
past it entirely. If you re-verify grounding facts, use `8eeace8ab090aecb1cfad336f82865f752c825c4`
as current HEAD, not either `4901d45` variant.

## Working tree note

Same dirty/untracked state you already flagged (`.claude/CLAUDE.md`, `bash.exe.stackdump`,
`training_log_smoke_test.csv`, plus the planning docs including your own) is still present —
none of it was staged or committed by the P0-7A integration, confirmed. Your own note that
"none touch `src/`, so none affect this design" still holds; the actual `src/` change (P0-7A)
was a real, separate, properly committed change, not part of that dirty-state list.

No other part of your design doc is affected by this session's work. Everything else in §1-18
stands as you wrote it.
