# Session Handoff — written just before /compact (2026-07-19)

Purpose: let a fresh session (this one, post-compact, or another) resume with zero
re-derivation. This is the authoritative record of where things actually stand.

## Current authoritative state

- **Primary repo HEAD (`master`, pushed to `origin/master`):**
  `8eeace8ab090aecb1cfad336f82865f752c825c4`
- **P0-7A (metric parity) is CLOSED / INTEGRATED.** Nothing further to do on it.
- **Remain stopped on P0-7 (training-integrity) until the frozen P0-7 specification is
  explicitly supplied by the user.** Do not start it speculatively. Do not deploy, train,
  or run the GPU sanity gate until then either.

## What P0-7A actually did (for context, not to be reopened)

Read-only audit (`REFERENCE_IMPLEMENTATION.md`'s vendored scoring code had drifted from
live upstream) found `src/tracking_cellmot/{metrics,division_metrics}.py` (+ Kaggle
mirror) were stale against `royerlab/kaggle-cell-tracking-competition@075fc5f5a52d1107
7f9dc2b074644618f26939e2` (their PR #2, "patch weakly connected component exploit").
Confirmed via direct differential reproduction (synthetic graphs run through both old
and new code) that this was a real, score-affecting bug — an old exploit case scored
`division_jaccard=1.0` (a fabricated perfect division score) where the corrected metric
scores `0.0`. Fixed by syncing both files verbatim to the pinned upstream commit,
verified via a caller-matrix audit (zero external callers of the changed
`score_divisions()` return type outside `division_metrics.py` itself), 23 new tests in
`tests/test_p07a_metric_parity.py`, and a v1→v2 fix (an AST-based, git-state-independent
caller scanner replacing a `git grep`-based one that had a real pre/post-commit
tracked-file blind spot). Two full adversarial review rounds (v1 approved with one
blocker, v2 approved clean) before integration. `src/evaluation.py` (P0-4 wrapper) was
never touched — confirmed structurally isolated from the primitive that changed.

**Known pre-existing Ruff findings** in `metrics.py` (8 total, both `src/` and mirror):
verbatim-identical lines from the original vendored file, unrelated to the PR#2 diff,
not introduced by P0-7A. Documented, not fixed (out of scope).

## Known unrelated local working-tree state — do not touch without explicit authorization

`git status --short` on the primary repo currently shows (as of this writing):
```
 M .claude/CLAUDE.md
 M bash.exe.stackdump
 M training_log_smoke_test.csv
?? GPU_SANITY_GATE_DESIGN_2026-07-18.md
?? LOOP_ENGINEERING_APPROACH_2026-07-18.md
?? NOTE_FOR_CLAUDE_2026-07-18.md
?? NOTE_FOR_CODEX_2026-07-18.md
?? TOOLING_ASSETS_PLAN_2026-07-18.md
```
All explicitly authorized as known-unrelated by the user across this session. Do not
stage, commit, delete, restore, or modify any of them without fresh explicit
authorization, even if they look stale or in-the-way.

## Parallel coordination threads (other sessions, same repo)

This project is being worked by multiple concurrent Claude/Codex sessions coordinating
via plain markdown notes in the repo root (all in the "known unrelated" list above,
except `TOOLING_ASSETS_PLAN_2026-07-18.md` which is this session's own earlier output):
- **`TOOLING_ASSETS_PLAN_2026-07-18.md`** — this session's plan for 4 reusable-asset
  proposals (CLAUDE.md operational-lesson entry, patch-generation guard script,
  native-crash differential-isolation script, cross-project memory entry). A separate
  "Codex" session picked this up; `NOTE_FOR_CODEX_2026-07-18.md` (also this session's
  output) corrected two things for it: use an isolated worktree, not master directly,
  and reuse the already-written CLAUDE.md text verbatim instead of duplicating it.
- **`GPU_SANITY_GATE_DESIGN_2026-07-18.md`** — a separate "Claude Session 3" wrote this
  (planning-only, no code touched) defining a fail-closed GPU sanity gate to run *after*
  both P0-7A and P0-7 land, checking for real dual-branch (detector + transformer)
  learning before any LOEO-scale spend. `NOTE_FOR_CLAUDE_2026-07-18.md` (this session's
  output) told that session P0-7A has now landed (their design treated it as pending),
  gave the correct current HEAD, and confirmed their prerequisite-2 (P0-7) is still open.
- If resuming and any of these threads' state seems relevant, read the note files
  directly rather than re-deriving — they're current as of this session's end.

## Key findings from this session worth carrying forward (not yet acted on)

1. **P1 architecture audit** (read-only, evidence-gathered against real fetched
   upstream source, not memory) found several confirmed architecture divergences from
   the reference implementation — independent self-attention instead of real
   cross-attention in `SimpleNodeTransformer`, a positional-encoding bug (array-index-
   based, not coordinate-based — a real correctness bug, not just a design choice),
   `UNet3D` has zero cross-Z-slice convolution and no temporal attention, the
   transformer is trained via GT-node teacher-forcing rather than the model's own
   detected nodes (train/inference distribution mismatch), and normalization uses a
   different quantile window/codomain than upstream. None of these have been acted on —
   they're candidate P1 work, explicitly out of scope until P1 is authorized.
2. **Detection-branch overfit diagnostic** (small, local, CPU, non-deployment,
   explicitly NOT the formal GPU sanity gate): on one real GT-annotated pair
   (`sample_id='6bba_05b6850b'`, `t_idx=0`), the corrected `DetectionLoss` showed
   **CLEARLY POSITIVE** learning signal — loss 3.26→0.65, GT-center/background
   separation ratio climbing from ~1x to a stable ~17-18x plateau, real finite nonzero
   gradients throughout. Does NOT prove transformer/edge learning, generalization, or
   LOEO performance — explicitly scoped to the detection branch only.
3. **Two native-crash classes observed this session, both confirmed pre-existing and
   environmental, not code regressions:** (a) PyTorch's own C++ autograd backward
   engine segfaults non-deterministically under sustained load on this Windows machine
   (confirmed via a rigorous 10-run alternating baseline/v4 differential isolation —
   both sides crash at the same ~80% rate); (b) real Zarr/blosc2 decompression can also
   segfault under similar load. Both are machine/environment fragility, not P0-6/P0-7A
   bugs — if either recurs, retry once with `-u` (unbuffered) for visibility, don't
   chase further.
4. **Competition progress introspection (honest, evidence-based, as of ~2026-07-18):**
   the last confirmed real Kaggle GPU training run (2026-07-14, pre-dating the
   DetectionLoss reweighting fix) ended in total collapse (`max_sigmoid≈2.2e-6`). Since
   then, real correctness fixes have landed (reweighting, embryo-disjoint splits,
   edge-logits/BCEWithLogitsLoss, P0-6 deployment safety, now P0-7A metric parity) but
   **no new real Kaggle GPU run has confirmed the collapse is actually reversed** — the
   local CPU overfit diagnostic (finding 2 above) is a positive but narrow signal, not
   that confirmation. This remains the single highest-leverage open question for the
   actual competition score, separate from and not resolved by P0-7A.

## Operational reminders specific to this repo (already in `.claude/CLAUDE.md`, restated
because they bit this session more than once)

- **Always symlink `data/staging` into any new worktree** before running data-dependent
  tests (`ln -s /c/Users/hemas/Downloads/st_act_pipeline/data/staging data/staging`,
  relative to worktree root; may need `mkdir -p data` first).
- **`training_log_smoke_test.csv` gets re-polluted by any test run that exercises
  `test_train_smoke.py`, including via the diagnostic suite** — clean with
  `git stash push -- training_log_smoke_test.csv` if it interferes with a scope check;
  never `git checkout --`.
- **`test_e2e_pipeline.py::test_full_pipeline` can hang for 85+ minutes** (confirmed
  this session via live process inspection — a real ILP combinatorial blowup, matching
  an already-documented CLAUDE.md precedent), unrelated to metric/P0-7A work. If running
  the full diagnostic suite, budget for this or exclude it
  (`--ignore=tests/test_e2e_pipeline.py`) and note the exclusion honestly in any report.
- **This machine's local torch is `2.8.0+cpu`** — no CUDA support at all, regardless of
  whether a physical GPU or Kaggle GPU quota is available. Any "GPU is available" signal
  from the user refers to Kaggle quota, not this machine, unless stated otherwise.
- Background shell tasks in this harness can lose their tracking record across a session
  restart (happened twice this session) — the underlying OS process usually keeps
  running fine; check `tasklist`/live process state directly rather than assuming
  failure when a completion notification never arrives.

## If resuming after /compact

Read this file first. Current state: P0-7A closed at `8eeace8ab090aecb1cfad336f82865f
752c825c4`, explicitly stopped pending the frozen P0-7 specification. Do not start P0-7,
deploy, train, or run the GPU sanity gate until that spec is supplied by the user.
