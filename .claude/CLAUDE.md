# st_act_pipeline — project instructions

Kaggle competition entry for **biohub-cell-tracking-during-development** (3D+time light-sheet
microscopy cell tracking, zebrafish embryos). Full spec, current-state audit, and phased roadmap
live in **`PRD.md`** — read it first, it is the source of truth for the competition spec and
scope, not this file.

## Facts that have already caused real bugs — do not re-derive, just use these

- **Physical anisotropy is `(4.0, 1.0, 1.0)`** (Z:Y:X), from real voxel scale z=1.625µm,
  y=x=0.40625µm — **fixed as of Phase 0** in `data_loader.py`. If you find `(5.0, 1.0, 1.0)`
  anywhere, it's a regression, not intentional. (`config/hyperparams.yaml` also lists this
  ratio, but that file is confirmed dead/unread by any `.py` in the repo — don't trust "fixed
  in hyperparams.yaml" as meaning anything actually executes differently.)
  Separately, `src/evaluation.py`'s `DEFAULT_SCALE` must be the real **physical micron** value
  `(1.625, 0.40625, 0.40625)`, not the `(4.0,1.0,1.0)` **ratio** — using the ratio there inflates
  every distance by ~2.46x and silently corrupts the 7.0µm match gate (this exact bug shipped
  once and was only caught by manually re-deriving the math, not by tests). **A second instance
  of the exact same ratio-vs-microns conflation shipped in `run_pipeline.py`'s `anisotropy`
  variable** (fed into `ensemble_consensus_centroids()`/`tracker.py`'s `solve_lineage()`/
  `prune_unphysical_edges()`, all of which compare against real-micron thresholds) — fixed by
  passing `DEFAULT_SCALE` there too. If you add a new call site that gates a distance against a
  micron threshold, use `DEFAULT_SCALE`, never the bare ratio.
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
- **`kaggle_kernel/train_kernel.py`'s Kaggle dependency install uses `pip install --no-deps`
  deliberately** (protects Kaggle's pre-installed numpy/scipy from a transitive-resolution
  corruption bug hit earlier), but `--no-deps` also blocks a package's own required companion
  packages, not just optional transitive extras. Confirmed real case: modern `polars` ships as a
  thin Python package that separately `Requires: polars-runtime-32` (the actual compiled Rust
  extension, `polars._plr`) — `--no-deps` silently skipped installing it, and polars' own source
  swallows that failure (`with contextlib.suppress(ImportError): from polars._plr import
  PySeries` in `polars/series/series.py`), so every `graph.node_attrs()` call in
  `src/targets.py`/`src/train.py` raised a caught `NameError` and silently fell back to
  all-zero GT targets for an entire ~75-minute Kaggle GPU run with no crash. Fix: install such
  companion packages explicitly by name alongside the main package (same pattern already used
  for `ilpy`→`pyscipopt`). A PyPI metadata audit of all 16 currently-pinned Kaggle packages found
  only polars has this split-runtime pattern (checked `Requires-Dist` directly, not guessed) — but
  re-check any *newly added* pinned package the same way before trusting `--no-deps` with it.
  **This exact bug recurred a second time**, in `kaggle_kernel_inference/inference_kernel.py`
  (the separate no-internet Code Competition submission kernel written in Phase 3.5): the
  `--force-reinstall` fix already existed in `kaggle_kernel/train_kernel.py`, but wasn't carried
  over when the new script reimplemented the same install step, and the first real run reproduced
  the identical "Polars binary is missing!" failure. A fix living in one kernel script does not
  automatically apply to a sibling script — when writing a new Kaggle kernel that reuses an
  install/setup pattern from an existing one, diff the block against the known-good original
  instead of rewriting it from memory.
- **`src/train.py`'s `TrainingLoop.train_epoch()` hard-fails if any single fallback type
  (`heatmap_generation_failure`, `edge_target_generation_failure`, etc.) fires on >50% of
  batches in an epoch** — added after the polars bug above proved silent per-batch fallbacks can
  run to completion producing a checkpoint trained on garbage with no error, ever. If you see this
  `RuntimeError`, it means the pipeline is actually broken (not just occasional missing data);
  diagnose the root cause before retrying, don't raise the threshold to make it go away.

## Operational lessons — read before running long jobs or delegating to sub-agents

- **Never trust a sub-agent's "PLAN COMPLETE" / "all tests pass" claim without independently
  re-running the actual command yourself.** Bugs caught this way across the project so far: a
  physical-scale bug, two tests that aliased the same mutable graph object instead of copying it,
  a schema-less-empty-graph crash, a wrong `tracksdata` API call, a hallucinated directory
  structure in a *plan* (before execution even started, missed by the plan-checker too), and —
  Phase 2 Wave 1 — `CompetitionDataset.__getitem__()` claimed "shape/dtype/metadata correctness
  confirmed" while actually slicing away 63 of 64 Z-slices (`frame_t[0:1, :, :]` on a (Z,Y,X)
  array slices axis 0 instead of adding a channel axis); the executor's own test only asserted
  `ndim == 3`, which passed for both the bug's wrong output and the correct shape, so "all tests
  passed" was true and still hid a silent-data-corruption bug that would have broken UNet3D
  training in the very next wave with no crash. A weak assertion (ndim/type-only, not exact shape
  or exact value) is *not* real verification — write assertions specific enough that the actual
  bug you're worried about would fail them. Verification means: run the test command, read the
  actual file, inspect the actual output — not reading the agent's summary of what it did, and not
  trusting a test that technically ran but wasn't specific enough to catch the failure mode.
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
- **Kaggle kernel runs cannot be monitored live via CLI/API — this is a confirmed, permanent
  platform limitation, not a tooling gap to keep trying to close.** `kaggle kernels output`
  only ever returns the *last completed* version's log; while a kernel is `RUNNING`, polling it
  repeatedly returns byte-identical stale data (confirmed for over an hour straight). There is no
  CLI/API method to cancel a running kernel or stream its live log (checked
  `kaggle_api_extended.py`'s actual client methods directly; Kaggle GitHub issues #653 and #388
  are open, unresolved feature requests for exactly this). Given that, the real procedure for a
  Kaggle GPU run is:
  1. Run the training script locally first if at all feasible, to catch import/dependency
     breakage before spending Kaggle GPU quota.
  2. After triggering, poll `kaggle kernels status` for RUNNING/COMPLETE/ERROR — that's the only
     thing the CLI can tell you mid-run; don't re-poll `kernels output` expecting new data.
  3. To see real early progress, ask a human (or a browser/cowork session) to check the website's
     Logs tab directly for the first couple minutes — specifically for the dependency-install
     fail-loud checks (e.g. `polars X.Y.Z extension verified OK`) and any hard-fail
     `RuntimeError`, so breakage is caught in minutes, not after a full run.
  4. After COMPLETE/ERROR, pull the CSV log via CLI and check it directly (it now includes
     `num_batches` and `epoch_wall_clock_seconds` alongside the fallback-count columns) before
     trusting any checkpoint — independent verification from the real artifact, never from a run
     summary alone.

  If step 3 is done via browser automation (a Cowork session or similar) rather than a human
  manually looking, these failure modes are confirmed real, not hypothetical:
  - **The notebook's bare URL (or `/log`, `/versions`) while logged in as owner does not open a
    passive read view** — it redirects into the same shared live editor/draft-session shell as
    whatever tab is actually being worked in. Two tabs on that shell at once (human + automated,
    or two automated) race on autosave and throw `ConcurrencyViolation` — once this **silently
    discarded a pasted fix entirely** (looked applied in the tab, but reloading showed the old
    code, because the other tab's save won the race). The only genuinely read-only URL is a
    specific completed version's `.../log?scriptVersionId=<id>` (from that version's own "..."
    menu) — never navigate the bare notebook URL for monitoring.
  - **The small icon left of each "Active Events" entry is a live Stop/Cancel button for a
    still-running version, with no confirmation dialog** — not a neutral "open" control. Clicking
    it cancelled a run that had already cleared a bug under test and was mid-training. Use that
    entry's "..." menu → "Open Logs in Viewer" instead.
  - **That "..." menu's "Open in Viewer" option can itself spin up another live `/edit/run/<id>`
    editor session in the background** (not a static log page), confirmed twice — close any such
    extra tab immediately.
  - **The Kaggle log viewer is a heavy virtualized widget**: scrolling several lines at once, or
    screenshotting immediately after navigation, frequently renders a blank viewport for a few
    seconds even though the content exists — wait ~3-5s and re-screenshot before concluding a
    script or log was wiped.
  - **Paste code fixes via real OS clipboard (copy, `Ctrl+A`, `Ctrl+V`), never simulated
    keystroke-by-keystroke typing of multi-line code** — the editor's autoindent re-indents on
    every typed newline and corrupts pasted indentation; a real paste event inserts text verbatim.
  - **Full page-text extraction of a long-running log (a few thousand+ lines) can exceed a text
    tool's output limit** — grep the saved dump for `ERROR|Traceback|WARNING` instead of
    re-reading it in full; prefer the small "Running for Xs" counter for a quick liveness check.
  - **Never let more than one path (a local CLI push, a teammate's browser click, another agent
    or Cowork session) trigger a new run against the same kernel at the same time** — two
    independent triggers raced once and produced two back-to-back failed versions that were
    briefly indistinguishable from a genuinely new bug. Agree on one "who triggers the next run"
    owner per iteration before pushing or clicking Save & Run All.
- **`kaggle kernels push` can hit a persistent `409 Client Error: Conflict` that does NOT resolve
  via the usual fixes** (closing the specific tab, closing *all* tabs on that kernel, waiting up
  to ~90s, changing the kernel title to remove punctuation) — confirmed on the inference kernel
  push in Phase 3.5, distinct from the earlier (successfully resolved) training-kernel instance of
  this same error class. When CLI push is genuinely stuck like this, don't keep blindly retrying —
  fall back to the manual-paste pattern: give the user the full corrected file content, have them
  open the kernel editor (one tab only), Ctrl+A + delete the existing code, paste the new content
  in via real clipboard paste (never simulated typing — see the autoindent-corruption note above),
  then trigger **Save Version → Save & Run All (Commit)** from the website directly.
- **This competition is a Code Competition** (Notebook-only submission, internet disabled, 12h
  runtime cap — see `PRD.md`/the real `/rules` page for full details). A critical, easy-to-miss
  consequence: **a plain `Save Version` / `Save & Run All (Commit)` run is NOT the same as a real
  graded `Submit` run.** Confirmed via a real log (`inference_kernel.py`, v3): during an ordinary
  Commit, `/kaggle/input/competitions/<slug>/test/` contains only the small **public example test
  set** (4 samples in this competition) — the real ~149-sample **hidden** test set the rules
  describe ("approximately the same size as the training dataset") is swapped in only when the
  notebook is actually rerun as part of clicking the real **Submit** button. Don't extrapolate a
  Commit run's wall-clock time directly onto the real submission's runtime budget without scaling
  by this ratio (in this case ~37x more pairs, 149/4), and don't assume a fast/successful Commit
  run alone proves the real Submit run will finish inside the 12h cap — it's strong evidence, not
  a guarantee, since the real run processes far more data.

## Model & effort policy

- **Mechanical/verification tasks** (grep a log for ERROR/Traceback, confirm a file/line exists,
  run pytest and report pass/fail, check git status) → `/model haiku`. Cheap, fast, no judgment
  required.
- **Normal iteration** (fix the bug that just surfaced, edit a call site, wire up an existing
  helper, rerun and check the log) → `/model sonnet`, default/high effort. This is most of the
  day-to-day work on this repo.
- **Hard judgment calls** (anisotropy/unit-scale math, ILP cost-function tuning, "does this
  architecture even support X," diagnosing an ambiguous crash across multiple files) → `/model
  opus` or `/model fable`, `/effort high` or `/effort xhigh`. Reserve for the handful of
  decisions where being wrong costs a wasted GPU run.
- **Decide model + effort once, at the start of a task, and don't switch mid-task.** Every switch
  forces a full-price, uncached re-read of everything already in the conversation — bouncing
  between levels mid-debug is itself a cost driver, not just a settings choice.
- Subagent frontmatter's `model:` field is currently broken upstream (ignored — subagents inherit
  the parent model regardless of what's set in the `.md` file, per
  `anthropics/claude-code#44385`). Until that's fixed, don't rely on frontmatter for this — either
  run the haiku-tier checks directly in the main session with `/model haiku` before switching
  back, or pass the model explicitly if dispatching via the Task tool.

## Workflow

- This project is executed via **GSD** (`/gsd:*` skills) against `PRD.md`'s §8 phased roadmap —
  use `/gsd:plan-phase` / `/gsd:execute-phase` per phase rather than large ad hoc sessions.
- Submissions are a scarce, rate-limited resource — validate every change against the local
  evaluation harness (`src/evaluation.py`) before spending a Kaggle submission.
