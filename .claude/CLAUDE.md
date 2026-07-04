# st_act_pipeline — project instructions

Kaggle competition entry for **biohub-cell-tracking-during-development** (3D+time light-sheet
microscopy cell tracking, zebrafish embryos). Full spec, current-state audit, and phased roadmap
live in **`PRD.md`** — read it first, it is the source of truth for the competition spec and
scope, not this file.

## Facts that have already caused real bugs — do not re-derive, just use these

- **Physical anisotropy is `(4.0, 1.0, 1.0)`** (Z:Y:X), from real voxel scale z=1.625µm,
  y=x=0.40625µm — **fixed as of Phase 0** in `data_loader.py`, `run_pipeline.py`,
  `hyperparams.yaml`. If you find `(5.0, 1.0, 1.0)` anywhere, it's a regression, not intentional.
  Separately, `src/evaluation.py`'s `DEFAULT_SCALE` must be the real **physical micron** value
  `(1.625, 0.40625, 0.40625)`, not the `(4.0,1.0,1.0)` **ratio** — using the ratio there inflates
  every distance by ~2.46x and silently corrupts the 7.0µm match gate (this exact bug shipped
  once and was only caught by manually re-deriving the math, not by tests).
- **`AnisotropicZarrLoader`'s simulated-data fallback must never silently activate** against a
  real competition path (`simulate=False` is the loader's actual guard now). It exists for
  local/offline testing only; if it fires against `data/train/` or `data/test/` it means the real
  Zarr wasn't found, and that should be loud, not silent.
- **Real data is Zarr format 3** (OME-NGFF, array at `<sample>.zarr/0`, metadata file is
  `zarr.json`, NOT the Zarr v2 `.zattrs`/`.zarray` layout). Real staged layout is
  **flat**: `data/staging/train/{id}.zarr` + `{id}.geff`, `data/staging/test/{id}.zarr` (no
  `.geff`) — NOT nested per-id folders; a planner hallucinated a nested structure once, verify
  against `find`/`ls` before trusting a plan's stated paths.
- **`.geff` ground truth is read via `tracksdata.graph.IndexedRXGraph.from_geff()`**, not the
  bare `geff` package and not hand-parsed. Returns a `(graph, GeffMetadata)` **tuple**, not a bare
  graph. The host's actual scoring code (`tracking_cellmot`, vendored into
  `src/tracking_cellmot/`) is **not on PyPI** — `tracksdata` itself is a normal pip package, the
  scoring logic had to be fetched from `github.com/royerlab/kaggle-cell-tracking-competition`.
- **`tracksdata.graph.IndexedRXGraph` — `.nodes`/`.edges` are accessor properties, not
  callables.** `graph.nodes()` raises `TypeError: 'NodesAccessor' object is not callable` on this
  version (`0.1.0rc6`, pinned exactly since it's pre-1.0). Use `.node_ids()` / `.edge_list()`.
  A bare `IndexedRXGraph()` only auto-registers a `t` node-attr key; `x`/`y`/`z` schemas are
  established lazily on first `add_node()` — passing a genuinely-empty graph into the vendored
  scorer crashes with `KeyError('z')` unless you pre-register those keys via `add_node_attr_key()`.
- **Competition score:** `adjusted_edge_jaccard + 0.1 × division_jaccard`, where
  `adjusted = max(0, jaccard · (1 − 0.1·(T_pred−T_true)/T_true))` and `T_true` is the `.geff`'s
  `estimated_number_of_nodes` (the full-embryo cell estimate, NOT the sparse labeled-node count —
  even a "perfect" match against sparse labels won't equal 1.0 unless `T_pred` also approximates
  the full estimate). Division term is dropped entirely (not `+0`) when a sample has zero GT
  divisions. Floor to beat: the public classical baseline at **0.763**. Leaderboard #1 at PRD
  time: **0.875**. Exact vendored source in `REFERENCE_IMPLEMENTATION.md`.
- **`STHypergraphTracker`'s flow constraints:** the equalities (`b_n + incoming == 1`,
  `outgoing + d_n == 1 + s_n`) are required for the tracker to do anything — reverting them to
  `<=` silently produces zero tracked edges (verified, don't reintroduce). There is deliberately
  **no** `b_n + d_n <= 1` constraint — that combination is a legitimate one-frame singleton
  (isolated detection with no plausible neighbor), not a contradiction; adding it back makes the
  ILP infeasible on real (sparse, noisy) data. See `src/tracker.py` comments at the constraint
  block. `tests/test_tracker.py` regression-tests this exact scenario — if it starts failing,
  someone reintroduced the bug.
- **The ILP is the dominant runtime cost, confirmed on real data (~70% of total pipeline time
  even at a hard-capped 30 candidates/timepoint)**, not just a theoretical Phase 3 concern.
  Solve time scales super-linearly with candidates/timepoint — profile before raising
  `MAX_CANDIDATES_PER_TIMEPOINT` in `run_pipeline.py`, don't just guess a bigger number.
- **The placeholder detector (`extract_peaks_from_volume`) has no real peak-finding** — it's a
  raw stride-8 grid threshold scan, so predictions land on a rigid grid, not cell centroids. This
  makes the local score near-zero (~0.009) regardless of threshold tuning; reaching the 0.763
  baseline needs actual peak-finding (local maxima/NMS), not just wiring or threshold changes.
  The host's real NMS approach is already documented in `REFERENCE_IMPLEMENTATION.md` §5.
- **Detection thresholds must be recalibrated whenever the underlying data distribution changes**
  (e.g. simulated `[0,1]`-uniform data vs. real quantile-normalized data). A threshold tuned
  against one distribution can produce catastrophically wrong candidate counts against another
  with no error or crash — just silent, exponential downstream cost. This exact mistake caused a
  2.5+ hour stuck run (~18,000 false candidates/timepoint → ILP combinatorial blowup) the first
  time real data replaced simulated data here. Sanity-check candidate counts (order of magnitude,
  not exact) before running anything at full scale.

## Operational lessons — read before running long jobs or delegating to sub-agents

- **Never trust a sub-agent's "PLAN COMPLETE" / "all tests pass" claim without independently
  re-running the actual command yourself.** This session caught 6+ real bugs an executor claimed
  were done/passing: a physical-scale bug, two tests that aliased the same mutable graph object
  instead of copying it, a schema-less-empty-graph crash, a wrong `tracksdata` API call, and a
  hallucinated directory structure in a *plan* (before execution even started, missed by the
  plan-checker too). Verification means: run the test command, read the actual file, inspect the
  actual output — not reading the agent's summary of what it did.
- **Long-running executor/Task calls (30-50+ min) have stalled mid-stream twice this session**
  ("API Error: Response stalled mid-stream"). When this happens: check `git log`/`git status`/
  the actual filesystem directly for what really landed before assuming failure or re-running
  from scratch — real, committable progress was recoverable both times.
- **Before launching another background command, check for and clean up stray processes from
  earlier attempts** (`tasklist`/`taskkill` on Windows) — redundant concurrent runs waste CPU/RAM
  and muddy timing measurements. This happened once (5 stray `python.exe` pytest processes
  running simultaneously from unconfirmed earlier launches).
- **For any long/expensive per-unit batch job (multiple datasets, multiple timepoints, etc.),
  build in from the start**: incremental crash-safe progress logging (not just end-of-run
  summaries), live ETA, per-unit result caching keyed by the config that produced it, and
  per-unit checkpointing so a partial failure doesn't require reprocessing everything. This
  pattern is implemented in `src/run_tracker.py` — reuse it (or the equivalent
  `long-running-pipeline-tracking` skill) rather than re-inventing it for Phase 2/3 work.

## Workflow

- This project is executed via **GSD** (`/gsd:*` skills) against `PRD.md`'s §8 phased roadmap —
  use `/gsd:plan-phase` / `/gsd:execute-phase` per phase rather than large ad hoc sessions.
- Submissions are a scarce, rate-limited resource — validate every change against the local
  evaluation harness (`src/evaluation.py`) before spending a Kaggle submission.
