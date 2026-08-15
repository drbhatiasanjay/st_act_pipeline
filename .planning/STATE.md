# Project State: ST-ACT — Spatio-Temporal Anisotropic Cell Tracker

**Last Updated:** 2026-08-14 (full rewrite — previous version stale since 2026-07-09, superseded
by everything below; see `.planning/archive/` for the full prior planning-doc history if needed)

---

## Where things actually stand right now

**Master HEAD:** `5267853138a26b0cc5db23e8f264b4d6ba14b80a` (PR #9 merge). All P0-1 through P0-7
infrastructure phases are merged. Active branch for the current investigation:
`gpu-sanity-gate-wave2-v2`.

**The single most important fact discovered today (2026-08-14):** the historical training
collapse (`max_sigmoid≈2.2e-6`, from a 2026-07-14 real Kaggle run, predating the 2026-07-15
`DetectionLoss` reweighting fix) is **confirmed reversed** on real Kaggle T4 GPU hardware. Ran
the bounded GPU learning probe (PR #7, `kaggle_kernel_learning_probe`, 512 batches) end-to-end —
verdict PASS, `max_sigmoid=0.379`, real gradients throughout, `predicted_nodes_total=8557` (not
structural zero). This was the "single highest-leverage open question" flagged in the old
(now-archived) `SESSION_HANDOFF_2026-07-19.md` and had never been checked until today. Full
report and every §4 criterion independently verified — see `KAGGLE_EXECUTION_RUNBOOK_GPU_LEARNING_PROBE_2026-07-20.md`
for the procedure; raw outputs in `../kaggle_probe_output_v2/` and the Kaggle worktree
`../st_act_pipeline_kaggle_exec/`.

**Second major finding today:** an independent Codex gap-analysis review (`gap_analysis_codex.md`,
repo root, reviewed at `417aa73` on `gpu-sanity-gate-wave2-v2`) proposed a 5-wave rework (P0.1
masked/positive-unlabeled detection loss, P0.2 oracle score decomposition, P0.3 cross-Z model
kernels, P0.4 predicted-node linker training, P0.5 calibration+scale). Real primary-source
research (fetching and reading `royerlab/kaggle-cell-tracking-competition`'s actual reference
training script, not paraphrasing) found:
- The reference implementation does **dense-negative BCE**, not masked/PU supervision, for
  detection loss — the same *shape* our code already uses. Codex's P0.1 "confirmed defect"
  framing does not hold up against the one real working precedent this competition provides.
- A local controlled diagnostic (`scripts/diagnose_detection_loss_recipe.py`) showed our current
  recipe (Gaussian target + adaptive weighting) learning cleanly (probability separation growing
  to +0.376 over 300 steps), while replicating the reference's simpler formula exactly
  (hard point target + fixed non-adaptive weight) **collapsed to 0.0 probability everywhere** —
  the current adaptive mechanism is load-bearing at our resolution, not a bug to remove.
- Real sparsity measured directly from staged `.geff` data: 0.16%–13.5% of real cells are
  annotated depending on sample (`estimated_number_of_nodes` lives under `GeffMetadata.extra`,
  not as a direct attribute — tripped up the first measurement attempt). Severe, but shared
  identically by the reference implementation, so not sufficient alone to explain the historical
  collapse.

**Net effect:** most of Codex's proposed P0.1/P0.3/P0.4 rework is now lower-priority than
originally scoped. P0.2 (oracle decomposition) was still worth building — implemented in
`src/oracle_evaluation.py` + `scripts/oracle_decomposition.py`, with a real node-ID-matching bug
(local per-timepoint index vs real tracksdata graph ID) found and fixed via direct verification,
not trusted from a subagent's self-report. Currently running that oracle-ceiling check against
the real (non-collapsed) probe checkpoint — first time this project has had a genuinely working
checkpoint to test detector-vs-linker ceiling against (`scripts/oracle_check_probe_checkpoint.py`
in `../st_act_pipeline_kaggle_exec/`).

---

## Real training run COMPLETE (2026-08-14 16:11 UTC → 2026-08-15 ~00:29 UTC, 20483s / 5.69h)

Kernel `drbhatiasanjay/st-act-gpu-smoke-test` (misleadingly titled, actually runs the real
`train_kernel.py`), deployed SHA `bc989ed` (includes the normalization fix, confirmed matching
in the real checkpoint manifest — not just assumed). 5000/12392 train pairs, 1 epoch, full
71-sample validation fold (no cap, unlike the bounded probe's 2 samples).

**Result: `val_score=0.001986` — the first non-zero real validation score this project has ever
gotten** (both prior probe runs scored exactly 0.0). `is_structural_zero=False`,
`predicted_nodes_total=554366`/`predicted_edges_total=388859` across all 71 samples, zero
fallback failures, no traceback, `health_status="healthy"`. Still a very low score relative to
the 0.763 baseline — expected, this is 5000/12392 pairs of ONE epoch, not a full uncapped run —
but it's real, non-degenerate signal, consistent with (not contradicting) everything found
earlier today.

Real timing breakdown matters for future runs: `training_log.csv`'s `epoch_wall_clock_seconds`
(7086.8s) is train-phase only; the real total (`training_progress.json`'s `elapsed_seconds`,
20483.1s) includes the full 71-sample validation pass, which took ~3.7h on its own — validation
is NOT cheap just because it doesn't backprop. Budget for this in any future run's ETA.

All three real Kaggle runs (2 probes + this training run) now logged in `kaggle_runs.db` via
`scripts/run_registry.py` (`list` / `compare` / `show` subcommands) for structured comparison —
built today specifically because this comparison had no structured home before.

**Next decision:** whether to commit to a longer/uncapped real training run given this first
non-zero (if tiny) result, or investigate further before spending more GPU budget. Not yet
decided — surface to user.

## Active investigation plan

Full context and reasoning: `C:\Users\hemas\.claude\plans\adaptive-soaring-cook.md` (plan-mode
output from today's session). Summary:

1. ~~Fix normalization bug (`q0.1/q0.9`→`[0,1]` should be `q0.001/q0.999`→`[0,4]`) in
   `src/data_loader.py`~~ — **still outstanding**, confirmed live as of 2026-08-14, not yet fixed.
   Must land before any full training run (both `kaggle_kernel/train_kernel.py` and
   `kaggle_kernel_inference/inference_kernel.py` share this loader).
2. ✓ Oracle decomposition built and bug-fixed (P0.2).
3. ✓ GPU learning probe run — collapse confirmed reversed.
4. **In progress:** oracle-ceiling check against the real probe checkpoint — determines whether
   detector or linker is the actual remaining bottleneck, with real (not collapsed) model output
   for the first time.
5. Not yet started: decide next real training investment based on (4)'s result, rather than
   guessing between P0.1/P0.3/P0.4.

---

## What NOT to re-litigate (already resolved this session, don't re-derive)

- Don't re-propose masked/PU-loss detection supervision without new evidence — checked against
  the real reference implementation and it doesn't use that approach.
- Don't assume the model is still collapsed — confirmed reversed via real Kaggle GPU run today.
- Don't trust `gt_metadata.estimated_number_of_nodes` as a direct attribute — it's under
  `gt_metadata.extra['estimated_number_of_nodes']`.

## Housekeeping

Root-level stale/superseded planning docs (GPU_SANITY_GATE_DESIGN v1-v4, FRESH_ARCHITECTURE_PLAN,
NOTE_FOR_CLAUDE/CODEX, TOOLING_ASSETS_PLAN, LOOP_ENGINEERING_APPROACH, P07_FROZEN_SPEC,
SESSION_HANDOFF_2026-07-19, COMPETITOR_RESEARCH, ISSUES_AND_FIXES, TASK_3_4_EVALUATION_RESULTS,
SEGFAULT_INVESTIGATION) moved to `.planning/archive/` on 2026-08-14 to reduce context drift —
their substance is either superseded by findings above or already condensed into
`.claude/CLAUDE.md`. `gap_analysis_codex.md` and `KAGGLE_EXECUTION_RUNBOOK_GPU_LEARNING_PROBE_2026-07-20.md`
kept at root — actively referenced. `VSCode_Extn_LLM_Metric.md` appears to be an unrelated stray
file (different project topic) — not touched, flag to user if it's not supposed to be here.
