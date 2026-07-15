# ST-ACT Pipeline — Functional & Non-Functional Issues Review

**Date:** 2026-07-12 · **Scope:** Full review of `src/`, `run_pipeline.py`, `kaggle_kernel/train_kernel.py`, `tests/`, `config/hyperparams.yaml`, `.claude/CLAUDE.md`, `PRD.md`, plus a real crashed-run log found locally (`kaggle_sanity_outputs/st-act-gpu-smoke-test.log`).

**Status:** Review only. **No Kaggle run has been triggered.** Everything below is for your review before deciding what to fix and when to run again, per your instruction.

Kaggle's competition page itself renders client-side and returned no usable spec text via fetch, so competition criteria below are sourced from this repo's own `PRD.md` (marked authoritative by `CLAUDE.md`) — scoring formula, baselines, and deadlines were already captured there and cross-checked against the code that implements them (`src/evaluation.py`).

---

## 0. Recap: what "accurate" means for this competition

- Score = `adjusted_edge_jaccard + 0.1 × division_jaccard`, `adjusted = max(0, jaccard · (1 − 0.1·(T_pred−T_true)/T_true))`. `T_true` is the `.geff`'s full-embryo node estimate, not the sparse labeled count.
- Floor to beat: classical baseline **0.763**. Current public #1: **0.875**. Deadline: **2026-09-22**.
- Every issue below either (a) directly lowers the score components (edge_jaccard, division_jaccard, T_pred accuracy), (b) blocks training from running at all, or (c) makes it hard to tell (a)/(b) apart from the logs — that's the lens used to prioritize.

---

## 1. Functional issues (correctness bugs)

### 1.1 — CRITICAL — GPU compute-capability mismatch crashes training immediately on non-T4 hardware

**Evidence:** `kaggle_sanity_outputs/st-act-gpu-smoke-test.log` (synced locally, timestamped 2026-07-12 06:10 UTC — today):

```
Found GPU0 Tesla P100-PCIE-16GB which is of cuda capability 6.0.
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
...
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

The run loaded Zarr data for two timepoints, then died inside `UNet3D.forward()`'s first `conv3d` call — before any of the bugs below even get exercised. `requirements.txt` pins `torch>=2.1.0` with no CUDA-build constraint, and Kaggle's current base image ships a PyTorch build that has dropped Pascal-generation (`sm_60`) support entirely. Whether this specific log is from version #24 (which you reported running 2+ hrs, i.e. probably got allocated a T4 and is fine) or an earlier version isn't fully certain from the file alone — but it proves the failure mode is real and will recur on any future run that Kaggle happens to allocate a P100 (or other pre-Volta GPU) to, since accelerator allocation isn't fully controllable via the API.

**Best-fit solution:**
1. **Immediate/operational:** always explicitly pick the T4 accelerator in the website's "Save & Run All" dialog before triggering a run — matches what `CLAUDE.md`'s commit history (`d02ab0c`, `ed574f5`) already found necessary. Don't rely on default/auto accelerator selection.
2. **Code-level guard (recommended addition to `train_kernel.py`):** immediately after `device = torch.device(...)`, run a trivial GPU op (`torch.zeros(1,1,4,4,4, device=device); torch.nn.Conv3d(1,1,1).to(device)(x)`) inside a `try/except`. If it raises, fail loudly in <5 seconds with an explicit message ("GPU compute capability incompatible with installed PyTorch — reselect T4 accelerator") instead of failing ~1 second into training after Zarr loading has already started. This turns a confusing mid-run crash into an instant, actionable one — directly helps the "monitor it better" goal, since you'd see this in the first line of a log check rather than after however long data loading took.

---

### 1.2 — HIGH — `.geff` caching infrastructure exists but is never actually wired up (dead code)

**Evidence:** `src/train.py`:
- `TrainingLoop.__init__` (line 196) creates `self._geff_cache: dict = {}` with a comment explaining it exists to fix "~600 redundant re-parses per epoch."
- `src/targets.py` has the matching `load_geff_cached(geff_path, geff_cache)` helper and both `generate_heatmap_targets()` and `generate_edge_targets()` accept a `geff_cache` parameter for exactly this purpose.
- But **none of the three actual call sites in `train_epoch()` pass it**:
  - `_get_gt_nodes()` (line 260) calls `tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))` directly — doesn't call `load_geff_cached` at all, and is called twice per batch (t and t+1).
  - `generate_heatmap_targets(...)` (line 325) — no `geff_cache=self._geff_cache` argument.
  - `generate_edge_targets(...)` (line 370) — no `geff_cache=self._geff_cache` argument.

Net effect: `self._geff_cache` is allocated and never read from or written to. Every one of the ~600 redundant re-parses/epoch that this was built to eliminate is still happening. This is the single biggest reason training would still be slow even after the caching feature was "added."

**Best-fit solution:**
```python
# _get_gt_nodes:
graph, _ = load_geff_cached(str(geff_path), self._geff_cache)   # instead of from_geff(...) directly

# both call sites in train_epoch:
generate_heatmap_targets(..., geff_cache=self._geff_cache)
generate_edge_targets(..., geff_cache=self._geff_cache)
```
Small, mechanical fix — three call sites. Worth adding a one-line log of `len(self._geff_cache)` at epoch end so the fix is verifiable in the training log (see §3.2, monitoring).

---

### 1.3 — HIGH — Anisotropy *ratio* used where real physical microns are required, in `tracker.py` / `run_pipeline.py`

**Evidence:** `run_pipeline.py:402` — `anisotropy = np.array([4.0, 1.0, 1.0])` (the Z:Y:X **ratio**) is passed into:
- `STHypergraphTracker.solve_lineage()` / `prune_unphysical_edges()` (`src/tracker.py`), which compares the scaled distance against **micron**-valued thresholds: `max_z_micron=15.0`, `max_xy_micron=30.0`, and a `distance < 40.0` (µm) search radius.
- `ensemble_consensus_centroids()` (`run_pipeline.py:93`), which DBSCAN-clusters with `eps_microns=6.0`.

`extract_peaks_from_volume()` returns centroids in raw **voxel-index** units (confirmed: `ndimage.center_of_mass` on a label array, no unit conversion). Multiplying voxel-index deltas by the ratio `(4.0, 1.0, 1.0)` instead of the real physical scale `(1.625, 0.40625, 0.40625)` µm/voxel inflates every computed distance by `1/0.40625 ≈ 2.46x` relative to true microns — the exact same bug class `CLAUDE.md` already documents as fixed once in `src/evaluation.py`'s `DEFAULT_SCALE`. It recurred here, unfixed.

Concretely, comparing ratio-scaled distances against micron thresholds means the *real* effective gates are:
- Search radius: intended 40µm → actually ~16.3µm
- Z-jump limit: intended 15µm → actually ~6.1µm
- XY-jump limit: intended 30µm → actually ~12.2µm
- DBSCAN consensus-clustering radius: intended 6µm → actually ~2.4µm

All four are tighter than intended. Effect: the ILP tracker sees fewer valid candidate edges than it should (more forced births/deaths → fragmented tracks → lower `edge_jaccard`), and DBSCAN's over-tight clustering silently drops real CNN/UNet consensus detections as noise (`labels == -1` points are discarded at `run_pipeline.py:116`), hurting recall and `T_pred` accuracy.

`tests/test_tracker.py` doesn't catch this because its own fixtures use the same wrong convention consistently — its comment at line 209 literally states *"10 voxels × 4.0 anisotropy = 40 physical microns Z"*, which is the same ratio/physical-scale conflation baked into a test assertion, not just the implementation.

**Best-fit solution:** Pass `src.evaluation.DEFAULT_SCALE` (the real `(1.625, 0.40625, 0.40625)` µm/voxel) into `solve_lineage()`/`prune_unphysical_edges()`/`ensemble_consensus_centroids()` wherever a distance is being gated against a micron threshold, instead of the `(4.0,1.0,1.0)` ratio. **Caveat:** `BIRTH_COST`/`DEATH_COST`/`MAX_CANDIDATES_PER_TIMEPOINT`/the 40µm search radius may have been *empirically tuned* against the current (wrong-scale) distances — swapping the scale in isolation will change the ILP's effective behavior non-trivially and probably needs a re-tune pass afterward, not a blind swap. Flag for discussion before changing, given it touches the tracker's core cost function.

---

### 1.4 — MEDIUM — `validate_epoch()`'s `peaks_t`/`peaks_t1` are computed from the identical volume

**Evidence:** `src/train.py:523-534`:
```python
peaks_t = extract_peaks_from_volume(vol_np, threshold=threshold, voxel_size=DEFAULT_SCALE, nms_radius_um=...)
peaks_t1 = extract_peaks_from_volume(vol_np, threshold=threshold, voxel_size=DEFAULT_SCALE, nms_radius_um=...)
```
Both calls use the exact same `vol_np` (the single sigmoid output of one `UNet3D` forward pass on the concatenated `(frame_t, frame_t1)` pair). `UNet3D` (confirmed via `src/model.py`) architecturally emits only **one** combined `(B,1,Z,Y,X)` detection head — there is no per-frame-specific output to split apart. So `peaks_t` and `peaks_t1` are deterministically identical every validation batch, which corrupts the val score used for early stopping and checkpoint selection (edges get built between two literally-identical point sets rather than real inter-frame motion).

**Best-fit solution:** Since `val_loader` has `shuffle=False` and pairs are consecutive within a sample, restructure `validate_epoch()` to reuse the *previous* batch's detection output as this batch's `peaks_t` (a one-batch sliding cache keyed by `sample_id`, reset whenever `sample_id` changes or at `t_idx == 0`), rather than deriving both frames from a single forward pass. This is a real restructuring of the validation loop, not a one-line fix — flag for its own small task rather than bundling with 1.1-1.3.

---

## 2. Non-functional issues (performance, config, test coverage)

### 2.1 — MEDIUM — `CompetitionDataset.__getitem__()` re-opens the Zarr store from scratch on every item

**Evidence:** `src/dataset.py:139` — `loader = AnisotropicZarrLoader(str(zarr_path))` is constructed fresh inside `__getitem__`, and `_init_store()` (`src/data_loader.py:37`) does a real `zarr.open()` + quantile-attrs extraction + several `logger.info()` calls every time. Since `train_loader`/`val_loader` use `shuffle=False` and a sample has ~100 consecutive pairs, the same Zarr store gets reopened ~100x instead of once. Real log evidence: repeated `"Opening real Zarr v3 store..."` for the same `sample_id` at closely-spaced timestamps.

**Best-fit solution:** cache one `AnisotropicZarrLoader` per `sample_id` on the `CompetitionDataset` instance (simple dict keyed by `sample_id`, built lazily in `__getitem__` or eagerly in `_build_pair_index`, since that already opens each sample once anyway just to read `get_shape()`). Lower severity than §1.2's `.geff` cache — this only reopens metadata/headers, not full pixel data — but the same easy win.

### 2.2 — LOW — `config/hyperparams.yaml` is dead, unread, and inconsistent with real code

**Evidence:** `grep` across the whole repo finds no `yaml.safe_load`/`yaml.load`/any reference to this file from Python. It's never loaded. It also disagrees with the real pipeline in ways that would mislead anyone editing it expecting effect: `model.in_channels: 1` (real `UNet3D` uses `in_channels=2`), and hyperparameter key names (`learning_rate`, `batch_size`, `epochs`) don't match `TrainingLoop`'s actual hyperparams dict shape at all.

**Best-fit solution:** either wire it up for real (load it in `train_kernel.py` instead of the inline `HYPERPARAMS` dict) or delete it / mark it clearly as legacy/unused in a header comment, so it stops being a plausible-looking trap.

### 2.3 — LOW — `SimpleNodeTransformer`'s edge scoring is an unvectorized Python double loop

**Evidence:** `src/model.py:240-257` — pairwise edge scoring over all `(n_t, n_t1)` candidates runs as a Python-level nested loop rather than a vectorized batch op. Not yet measured against real node counts (the ILP profiling in `run_pipeline.py`'s comments covers the tracker, not this), so severity is unconfirmed — flagging as a scaling risk worth a quick profile once real per-frame node counts are known (per `run_pipeline.py`'s own note, dense timepoints see ~1110 candidates), not a confirmed bottleneck yet.

### 2.4 — LOW — `AugmentedCompetitionDataset` augmentations are a no-op stub

**Evidence:** `src/dataset.py:247` — `# TODO: Wave 3 augmentations` (elastic deformation, Y/X rotation, intensity jitter, patch dropout) — `augment=True` currently changes nothing. This is a legitimate accuracy lever left on the table (more training diversity generally helps generalization on a competition with only ~149-199 labeled samples), not a bug — listed here because it's a concrete, scoped way to "build a more accurate execution" per your ask.

### 2.5 — MEDIUM — Zero test coverage for the four modules where the confirmed bugs live

**Evidence:** `tests/` contains `test_data_loader_real.py`, `test_e2e_pipeline.py`, `test_evaluation_harness.py`, `test_pipeline_integration.py`, `test_scoring_baseline.py`, `test_spot_check_submission.py`, `test_submission_exporter.py`, `test_tracker.py` — no `test_train.py`, `test_targets.py`, `test_model.py`, `test_dataset.py`, or `test_inference.py`. None of §1.2, §1.4, or §2.1 would have been caught by CI, and none will be caught by a future regression if reintroduced.

**Best-fit solution:** at minimum, a regression test for §1.2 (assert `self._geff_cache` grows across batches sharing a sample, or mock `load_geff_cached`/`from_geff` and assert call count) and §1.4 (assert `peaks_t != peaks_t1` isn't just a happy accident of two different-but-correlated volumes) would have caught both real bugs found in this review. Consistent with the project's own stated lesson in `CLAUDE.md`: weak/absent tests are exactly how the Phase 2 Wave 1 slicing bug slipped through once already.

---

## 3. Monitoring improvements

### 3.1 — Kaggle run visibility remains fundamentally limited (already documented)

`CLAUDE.md` and `LESSONS_LEARNED_KAGGLE_MONITORING.md` already cover this at length (no live-log API, `ConcurrencyViolation` on shared draft sessions, the Stop-button trap, etc.) — not re-litigated here, just cross-referenced.

### 3.2 — Recommended additions to the CSV/log output for this specific set of bugs

- **Geff cache hit rate** — once §1.2 is fixed, log `len(self._geff_cache)` (distinct files parsed) alongside a running count of cache hits vs. misses per epoch. Turns "did the caching fix actually work" into a one-line grep instead of a timing inference.
- **Early GPU self-test output** — §1.1's proposed guard doubles as a monitoring win: the very first lines of any future log will say either "GPU self-test passed" or the exact incompatibility reason, instead of that information being implicit in *how far* the run got before dying.
- **Per-epoch `positive_fraction`/adaptive-threshold trigger count** — `validate_epoch()` already computes `positive_fraction` (line 512) to detect an undertrained model flooding the peak-finder; currently this only logs a warning per-batch. Rolling it into the per-epoch CSV row would make "is detection still miscalibrated" visible in the log summary, not just buried in per-batch warnings.

---

## 4. Already fixed (for context — not re-flagging)

Cross-checked against `git log` and re-read directly, these are confirmed resolved and don't need action:
- `.cpu()` before `.numpy()` in `generate_edge_targets`'s `match_to_gt` (`4d1e72b`)
- polars `--no-deps` / missing `polars-runtime-32` companion package (`67b3810`, with a fail-loud verification check now in `train_kernel.py`)
- `src/evaluation.py`'s `DEFAULT_SCALE` correctly uses real physical microns, not the ratio (confirmed by direct re-read — this is the "did it right" counterpart to §1.3's regression)
- `train_epoch()`'s majority-fallback hard-fail guard + richer CSV diagnostics (`ae0d6aa`)
- Gaussian heatmap generation vectorization (`016d168`)
- `CompetitionDataset.__getitem__()`'s Z-slicing bug (channel-axis vs Z-axis) — Phase 2 Wave 1, already fixed with an explanatory comment in place
- Kaggle `src`-dataset mount-path discovery via `os.walk` (my own earlier fix this session, verified live through Version #20+)
- `STHypergraphTracker`'s flow-conservation equalities and the deliberate absence of a `b_n + d_n <= 1` constraint — correct as-is, regression-tested

`config/hyperparams.yaml` note in `CLAUDE.md` ("physical anisotropy... fixed... in `hyperparams.yaml`") is now stale in one respect: that file is confirmed dead/unread (§2.2), so "fixed in hyperparams.yaml" no longer means anything executes differently — worth a small doc correction whenever convenient, not urgent.

---

## 5. Suggested fix order

1. §1.1 (GPU guard) — cheapest, highest-leverage, prevents wasted GPU-hours on a doomed run.
2. §1.2 (dead geff cache) — mechanical, 3-line fix, directly addresses training speed.
3. §2.1 (Zarr loader reuse) — mechanical, same category as #2.
4. §1.3 (anisotropy scale in tracker) — needs a re-tune pass after the fix; do this deliberately, not last-minute before a submission.
5. §1.4 (validate_epoch identical peaks) — needs a small design decision (sliding-window restructure); do after 1-3 so validation scores you're comparing against are trustworthy.
6. §2.5 (tests) — ideally alongside 1.2/1.4 so the fixes are regression-proof, not after.
7. §2.2, §2.3, §2.4 — lower priority, pick up opportunistically.

No run is queued. Let me know which of these you'd like fixed (all, or a subset) before the next Kaggle submission.
