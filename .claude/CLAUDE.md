# st_act_pipeline — project instructions

Kaggle competition entry for **biohub-cell-tracking-during-development** (3D+time light-sheet
microscopy cell tracking, zebrafish embryos). Full spec, current-state audit, and phased roadmap
live in **`PRD.md`** — read it first, it is the source of truth for the competition spec and
scope, not this file.

## Facts that have already caused real bugs — do not re-derive, just use these

- **Physical anisotropy is `(4.0, 1.0, 1.0)`** (Z:Y:X), from real voxel scale z=1.625µm,
  y=x=0.40625µm. It is hardcoded to the wrong `(5.0, 1.0, 1.0)` in several places
  (`data_loader.py`, `run_pipeline.py`, `hyperparams.yaml`) — fixing this is part of PRD Phase 0.
- **`AnisotropicZarrLoader`'s simulated-data fallback must never silently activate** against a
  real competition path. It exists for local/offline testing only; if it fires against
  `data/train/` or `data/test/` it means the real Zarr wasn't found, and that should be loud, not
  silent.
- **Real data is Zarr format 3** (OME-NGFF, array at `<sample>.zarr/0`), not the Zarr v2 layout
  the simulated store currently produces. `.geff` ground truth (train only) should be read via the
  `geff` package (reads straight to `networkx`), not hand-parsed.
- **Competition score:** `edge_jaccard + 0.1 × division_jaccard`. Floor to beat: the public
  classical baseline at **0.763**. Current leaderboard #1 at time of PRD authoring: **0.875**.
- **`STHypergraphTracker`'s flow constraints:** the equalities (`b_n + incoming == 1`,
  `outgoing + d_n == 1 + s_n`) are required for the tracker to do anything — reverting them to
  `<=` silently produces zero tracked edges (verified, don't reintroduce). There is deliberately
  **no** `b_n + d_n <= 1` constraint — that combination is a legitimate one-frame singleton
  (isolated detection with no plausible neighbor), not a contradiction; adding it back makes the
  ILP infeasible on real (sparse, noisy) data. See `src/tracker.py` comments at the constraint
  block.

## Workflow

- This project is executed via **GSD** (`/gsd:*` skills) against `PRD.md`'s §8 phased roadmap —
  use `/gsd:plan-phase` / `/gsd:execute-phase` per phase rather than large ad hoc sessions.
- Submissions are a scarce, rate-limited resource — validate every change against the local
  evaluation harness (`src/evaluation.py`, once built) before spending a Kaggle submission.
