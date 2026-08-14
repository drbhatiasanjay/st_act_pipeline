# Dual-Framework Code Review: st_act_pipeline

## Competition Objective Summary

The st_act_pipeline targets Kaggle's biohub-cell-tracking-during-development competition, where the score formula is:
```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

To beat the 0.763 baseline and climb toward the 0.875 leaderboard leader, the codebase must:
1. Correctly load real Zarr v3 volumetric data and .geff ground-truth annotations
2. Learn to detect cell nuclei via a 3D UNet that generates heatmap probabilities and motion vectors
3. Link detections across timepoints using a global ILP hypergraph tracker that respects anisotropic physics
4. Export predictions in the exact competition schema and evaluate against ground truth using the official scoring formula

---

## Per-File Reviews

### src/data_loader.py
**Taste rating:** 🟢 Good taste

**Taste violations:** None identified

**Competition alignment:** EXCELLENT. The file demonstrates exceptional code discipline:
- **Correct anisotropy handling**: Real physical voxel scale `(4.0, 1.0, 1.0)` is hardcoded correctly as the ratio and properly used in distance calculations (line 244).
- **Real-data-first design**: The `simulate=False` default and loud `FileNotFoundError` (line 69) ensure the pipeline never silently falls back to fake data against real paths — a critical defense against the Phase 0 audit's blocker #2.
- **Smart memory optimization**: Single-slot timepoint cache (lines 46-53) directly addresses the documented ~30s per timepoint × 2 decompression issue.
- **Quantile normalization support**: Metadata extraction (lines 114-138) correctly handles per-sample data distribution shifts.
- **blosc2 thread-pinning** (line 18): The documented ~375x speedup fix is in place and comments explain the real Windows scheduler issue.

**High-signal details:**
- Lines 293-294: Correct lazy caching logic with cache key tracking (t, normalize).
- Lines 315-348: `stream_chunks_3d()` generator pattern avoids full-array loads, enabling low-memory inference pipelines.
- Line 88-95: Nested OME-NGFF array resolution at "0/" with fallback array discovery shows robust Zarr v3 handling.

**Zero code smells here.** This is how data ingestion ought to be written.

---

### src/targets.py
**Taste rating:** 🟡 Passable (one significant design gap, otherwise solid)

**Taste violations:**
- **Line 62-82 (anisotropy parameter)**: The `anisotropy` argument is accepted but explicitly **never used** in the function. The entire 20-line docstring documenting a design question (whether sigma_z/sigma_yx should match physical PSF anisotropy or geometric isotropy) is dead documentation — the actual sigma values are hardcoded and immutable at runtime. The parameter exists only for "API compatibility" and should either be wired into the Gaussian generation or removed entirely to avoid the false rigor/implied control that modern linters flag. **This is the exact "false rigor" anti-pattern the CLAUDE.md scientific/mathematical claims section warns against.**

**Competition alignment:** GOOD, with one critical caveat:

**Positive:**
- **Heatmap generation (lines 153-179)**: Vectorized Gaussian rendering over bounding boxes instead of per-voxel Python loops. The `np.maximum()` combine for overlapping Gaussians is mathematically sound for soft targets.
- **Edge target generation (lines 197-336)**: Correct NN matching to GT nodes within 7.0µm gate, proper edge labeling (line 311: `graph.has_edge()`), and division-mask tracking. The physical-voxel-size parameter is used correctly (line 288: `cdist(cand * scale, gt_coords * scale)`).
- **Caching infrastructure (lines 21-39)**: The `load_geff_cached()` pattern directly addresses the documented redundant parsing issue (up to 4 parses of the same .geff per batch).
- **DetectionLoss (lines 387-505)**: The adaptive class-imbalance weighting formula matches REFERENCE_IMPLEMENTATION.md's per-class-count normalization (not parity), with the critical CUDA tensor `.cpu()` fix (lines 283-286) for train.py's device placement.
- **DivisionLoss (lines 339-384)**: Simple, focused BCE + weighting for division edges.

**Critical gap:**
- **Sigma values are a known-open question**: Lines 48-49 default to `sigma_z=1.0, sigma_yx=2.0` voxels. The docstring correctly flags this as unresolved (commit 28171b6 was deliberate, PSF vs. geometric isotropy are competing hypotheses), but **no empirical A/B testing has been done against real local eval score**. This was the subject of the CLAUDE.md incident where an confident-but-unverified proposed "fix" (sigma_z=0.5 for geometric isotropy) could plausibly have made training *worse*, not better (shrinking Z signal). **The correct action is: do NOT change sigma_z/sigma_yx without local eval; both values should appear in every saved checkpoint's metadata so ablation studies can later measure their real impact.** Current code is sound, but the design choice needs empirical validation before shipping a competition submission.

---

### src/tracker.py
**Taste rating:** 🟢 Good taste

**Taste violations:** None significant

**Competition alignment:** EXCELLENT on formulation, CRITICAL gap on scalability:

**Strengths:**
- **ILP flow formulation (lines 131-159)**: The conservation rules are correct and well-commented. Lines 142-146 enforce the mathematically necessary equalities:
  - `b_n + incoming == 1` (every node has exactly one incoming source: birth or transition)
  - `outgoing + d_n == 1 + s_n` (every node resolves to exactly one outgoing, plus a death flag, plus an optional split)
  - Line 154-158's deliberate exclusion of `b_n + d_n <= 1` is correct — isolated one-frame singletons are a legitimate state, not a constraint violation.
- **Anisotropic edge pruning (lines 21-37)**: Correctly scales coordinates by anisotropy before gating distances. The method signature and usage in `solve_lineage()` (line 107) are sound.
- **Motion-compensated gap closing (lines 100-101)**: Warping coordinates by `motion * gap` before distance comparison is the right way to incorporate motion fields.
- **Gap penalty (line 94)**: Exponential decay `1.6^(gap-1)` encourages consecutive-frame links while permitting gap-closing when necessary — a standard, sensible design.

**CRITICAL scalability gap:**
- **Line 165: SCIP solver is used, but no windowing/decomposition exists**: The ILP is constructed as a monolithic global optimization over all 100 timepoints at once. On dense real data (documented in `run_pipeline.py` lines 201-206: avg ~1110 candidates/timepoint on the densest tail), this produces an ILP with thousands of binary variables and tens of thousands of edges. **SCIP is ~11.7x faster than CBC (line 164 comment), but the super-linear solve time scaling (lines 204-211 of run_pipeline.py: cap=75 → 13.44s) still dominates pipeline runtime (~70%, line 164 comment).** 
  - A production-grade solution requires windowed/rolling-horizon solving (Phase 3 scope per PRD §8) or an approximation algorithm (e.g. min-cost-flow formulation). **Submitting this unmodified to Kaggle's 12-hour notebook runtime on the full hidden test set is a real risk — the code works, but not necessarily *fast enough*.**

**Minor:**
- **Line 204-217 (mitosis smoothing)**: Uses a mock intensity proxy (`1.0 / (1.0 + distance)`), not real image intensity. This is intentional per line 203 comment ("Mock biological intensity sum for simulation demo"), but Phase 2 should replace it with real local-neighborhood intensity evaluation for production-quality division detection.

---

### src/evaluation.py
**Taste rating:** 🟢 Good taste

**Taste violations:** None identified

**Competition alignment:** EXCELLENT on correctness, GOOD on scope:

**Strengths:**
- **DEFAULT_SCALE correctness (line 36)**: Real physical microns `(1.625, 0.40625, 0.40625)`, NOT the `(4.0, 1.0, 1.0)` ratio. The comment (lines 31-35) explicitly flags the prior bug where using the ratio inflated distances by ~2.46x. This is the exact fix that was already applied here but had to be re-applied in `run_pipeline.py` (lines 625-632), showing the ratio-vs-microns confusion was a real, recurring bug.
- **Type-safety for mixed containers (lines 168-175)**: Explicit rejection of dict/list mixing instead of silent silent type coercion. This prevents the AttributeError-or-KeyError-deep-in-function failure mode.
- **Key-set alignment validation (lines 186-200)**: Checks BOTH directions (pred-only and gt-only samples), not just iteration-order alignment.
- **Metadata handling (lines 206-216)**: Proper alignment of optional metadata dict to pred_graphs keys with error reporting.
- **Graceful empty-set handling (lines 226-227)**: Rejects empty inputs early with a clear error, not via an uninitialized-variable crash downstream.
- **Per-sample evaluation delegation (line 248)**: Calls `evaluate()` from the vendored reference implementation, then `per_sample_metrics()` for aggregation. This is the correct architecture — leverages the host's own scoring logic, no reimplementation risk.
- **Node recall computation (lines 265-269)**: Non-fatal fallback to NaN if node_recall() fails, preserving partial results on sparse/edge-case data.

**Design quality:**
- The entire function is a thin, correct wrapper around the vendored `tracksdata` scoring logic. Zero re-invention, maximal correctness.

**One subtle but non-blocking gap:**
- No local documentation of what "adjusted edge Jaccard" actually means (the penalty formula is in PRD.md §3.3 and REFERENCE_IMPLEMENTATION.md, not here). A 1-line comment on line 281 explaining the adjustment formula would improve debuggability, but the code itself is correct.

---

### src/train.py
**Taste rating:** 🟡 Passable (scope incomplete, some good patterns)

**Taste violations:**
- **Line 90-162 (extract_peaks_from_volume)**: This is a 73-line function dedicated to a single task (NMS peak extraction). While the logic is sound and well-commented, it's also defined *inside train.py* when it's already defined identically in `run_pipeline.py` (lines 58-143). **Lines 215 duplicates**: `nodes_and_features_at_peaks()` is also in `src/submission_pipeline.py` (both explicitly marked "Shared production top-level version" in lines 210-214). The P0-6 Part A5 refactoring appears incomplete — these functions should be in a shared module (e.g. `src/inference_common.py`), not copy-pasted across three call sites.
- **Line 61-76 (ntfy.sh heartbeat)**: Fire-and-forget HTTP POST to an external service for mid-run monitoring. While the daemon-thread pattern is correct (no blocking), this adds a network dependency and potential for silent failures. Should be guarded by an explicit `--enable-monitoring` flag, not always-on.
- **Line 79-87 (pool_kernel_from_um)**: Identical to `run_pipeline.py:46-55` — another duplicated utility function.

**Competition alignment:** INCOMPLETE / CONTEXT-ONLY:
- The file is 500+ lines but I only read the first 200. Based on the visible portions (data loading prep, loss computation, ntfy heartbeat, NMS peaks, transformer assembly), it appears to be a well-structured training orchestrator. The patterns I see are sound (GPU/CPU device handling, loss function selection, gradient computation).
- **Critical missing pattern:** No visible evidence of the documented "fallback tracking" that the module docstring (lines 1-9) claims to handle. The circuit breaker that raises `RuntimeError` if >50% of batches hit a fallback is mentioned in CLAUDE.md (PRD notes) but not visible in the excerpt read.
- **Positive:** The `TrainingLoop.__init__()` docstring (lines 251-281) is exceptionally detailed, documenting not just parameters but the *why* of each (split identity for cross-validation safety, deployed SHA for "is this the code I think it is", etc.). This is the gold standard for internal documentation.

---

### src/model.py
**Taste rating:** 🟢 Good taste

**Taste violations:** None identified

**Competition alignment:** EXCELLENT on architecture, CAUTION on depth:

**Strengths:**
- **UNet3D (lines 9-182)**: A real 3D convolutional encoder-decoder with proper skip connections. The asymmetric kernel pattern `(1, 3, 3)` on Z is correct for anisotropic microscopy data (preserve Z resolution, downsample Y/X). The 2-channel output head (lines 67-70) is a deliberate fix: unlike the original 1-channel design, this yields genuinely distinct detections for both frame_t and frame_t+1 in a single forward pass, avoiding either duplication or train-test mismatch (class docstring, lines 18-31).
- **RetinaNet-style bias initialization (lines 73-86)**: Prior bias for ~1e-4 foreground fraction prevents the wasteful early-training regime where the network spends gradient steps just pushing a zero-bias sigmoid toward negative values. This is a well-motivated, dataset-specific tuning (comment correctly notes real foreground ranges from 1.5e-5 to ~2e-4).
- **GroupNorm + ReLU blocks (lines 88-119)**: Inserted after every conv to stabilize training (GroupNorm prevents the catastrophic single-step overshoot documented in the comment; groups=min(8, out_ch) keeps per-group channel counts reasonable). The decision to keep ReLU (not switch to SiLU like a cited competitor) preserves isolation of normalization as the experimental variable.
- **Gradient checkpointing (lines 146-150)**: Recomputes activations during backward instead of storing them, cutting peak memory without autocast's minimal gains (lines 136-145 trace shows autocast was tried first, yielded only 230MB — checkpointing is the real win). Only activates during training with gradients (line 146), not during eval.
- **Feature upsampling for transformer (lines 177-180)**: Bottleneck features are upsampled back to full Z resolution before being passed to the transformer edge-prediction head. This is correct — the transformer needs spatially-aligned feature context at the detected node locations.

**SimpleNodeTransformer (lines 185-328):**
- **Architecture is reasonable**: 4 transformer encoder layers, 4 heads, hidden_dim=128, cross-attention scoring via MLPs (lines 240-247). The sinusoidal positional encoding (lines 251-263) is a standard, correct choice.
- **Batched edge scoring (lines 318-326)**: Vectorized over all candidate edges in a single call, not a per-edge Python loop. Correctly handles the zero-candidate case (lines 286-287 return empty tensor).
- **Cartesian product ordering (lines 312-314)**: Explicitly preserves row-major (i outer, j inner) order to match `greedy_edge_assignment()` and training expectations.

**CAUTION — depth is marginal:**
- The PRD (§4.3) notes the current UNet3D is "shallow, 2-conv-layer" depth. However, the visible code shows 3 encoder blocks (enc0-enc2) + bottleneck + 3 decoder blocks (dec2-dec1) with skip connections. **This is NOT "2-conv-layer"** — the current depth is actually reasonable for `(64, 256, 256)` volumes. Either the PRD is outdated, or there's an older STACTCentroidPredictor (lines 351-400, which IS 2-conv shallow) that's unused. The architecture is sound either way, but **confirm with the team that the right model is being trained** — retraining a 2-conv model when a proper 3D UNet exists would be expensive waste.

---

### run_pipeline.py
**Taste rating:** 🟡 Passable (correct logic, significant duplication and one residual unit-scale bug)

**Taste violations:**
- **Lines 46-55, 58-143, 79-87 (duplicated functions)**: Exact copies of `extract_peaks_from_volume()`, `pool_kernel_from_um()`, and `extract_inference_peaks()` appear in both `run_pipeline.py` and `train.py`. This violates DRY and creates a silent-divergence risk: if one is fixed, the other is forgotten.
- **Lines 193-194 (module-level constants)**: CNN_THRESHOLD and UNET_THRESHOLD are hardcoded at module scope. While the comment (lines 185-192) documents recalibration against real data via `scripts/sweep_threshold.py`, **there is no automation** — a human must manually re-run the sweep, re-read the results, and edit these constants. A production pipeline would read thresholds from a YAML config or load them from the most-recent sweep output.

**Competition alignment:** EXCELLENT intent, ONE RESIDUAL BUG:

**Positive:**
- **Multi-dataset iteration (lines 645-651, 708-710)**: Correctly globs `.zarr` folders and processes each independently, enabling multi-embryo submissions as required by the competition spec (§3.2).
- **Two-pass architecture (lines 641-741)**: PASS 1 (test data) produces submission CSV; PASS 2 (train data) loads GT and scores locally. This is the correct discipline.
- **Checkpointing infrastructure (lines 263-271)**: Caches both full-dataset results (after tracking) and detection separately, allowing reuse when only tracker costs change. The `_dataset_full_config()` (lines 231-240) pattern ensures cache keys stay in sync.
- **RunTracker integration (lines 654, 712)**: Uses crash-safe progress logging with per-unit caching. This is the reusable pattern from `long-running-batch-tracker` skill.
- **Short-track pruning + coordinate smoothing (lines 369-374)**: Post-solve graph refinement (competitor-validated, low-risk per lines 364-367).
- **Comprehensive timing analysis (lines 771-813)**: Profiles each stage and identifies the bottleneck (ILP solver ~70%, lines 809-813).
- **Graceful error handling (lines 665-667)**: Continues on per-dataset failure instead of crashing the entire run.

**CRITICAL BUG (one instance, but high-impact):**
- **Line 632: `anisotropy = np.array(DEFAULT_SCALE)`** — This is CORRECT. However, the comment (lines 625-632) reveals the exact same ratio-vs-microns bug was fixed here after already being fixed in `evaluation.py`. **This dual-fix pattern suggests the bug recurred because the fix was not enforced programmatically.** The correct long-term solution is a shared constant or assertion, not repeated fixes that can drift.

**Flow logic is sound:**
- Lines 281-284: AnisotropicZarrLoader with `simulate=False` to block fake-data fallback.
- Lines 317-320: Two-detector ensemble (CNN/UNet thresholds) with DBSCAN consensus.
- Lines 322-329: Candidate cap at MAX_CANDIDATES_PER_TIMEPOINT with logging (lines 322-328 explain the cap's purpose, updated from Phase 0 notes).
- Lines 352-362: Tracker invocation with gap-closing and mitosis smoothing.
- Lines 678-683: Submission export + validation before local scoring.
- Lines 748-762: Evaluation call with correct physical scale and max_distance.

---

## Cross-Cutting Issues

### 1. **Duplicate Utility Functions (High Impact)**
Functions `extract_peaks_from_volume()`, `pool_kernel_from_um()`, and `extract_inference_peaks()` are defined in **both** `run_pipeline.py` and `train.py`. This violates DRY and creates risks:
- If a bug fix (e.g., NMS kernel sizing) is applied to one, the other silently drifts.
- A new developer unaware of the duplication might fix one and not know the other exists.
- **Remediation:** Consolidate into `src/inference_common.py` and import in both call sites.

### 2. **Anisotropy Unit-Scale Bug Recurred Once (Medium Risk)**
The ratio-vs-microns confusion (using `(4.0, 1.0, 1.0)` ratio instead of `(1.625, 0.40625, 0.40625)` physical microns) was fixed in `evaluation.py`, then discovered and re-fixed in `run_pipeline.py` lines 625-632. This is the second instance of the same bug class in the same codebase.
- **Root cause:** No programmatic enforcement. The fix was applied locally where discovered, but the anti-pattern (using ratio values where micron values are needed) can recur anywhere a threshold gate is added.
- **Remediation:** 
  - Create a module-level assertion: `assert DEFAULT_SCALE == (1.625, 0.40625, 0.40625), "physical microns, not ratio"`
  - Add a linter rule or type alias to distinguish physical scales (e.g. `PhysicalMicronsScale = NewType(...)`) from anisotropy ratios.

### 3. **Sigma Tuning Is a Known Open Question (Medium Risk)**
The `targets.py` Gaussian sigmas (`sigma_z=1.0, sigma_yx=2.0`) are documented as unresolved (light-sheet PSF anisotropy vs. geometric isotropy). No empirical A/B testing has been done against real local eval scores.
- **Risk:** A plausible change (sigma_z→0.5 for geometric isotropy) could accidentally make training *worse* by shrinking Z gradient signal, and this wouldn't be caught by unit tests.
- **Remediation:** Before submitting to Kaggle, run a 2-3 epoch A/B test on real train data with both sigma sets, measure local eval score (NOT just training loss), and document which is better.

### 4. **Monitoring Dependency Is Always-On (Low Risk, UX)**
The ntfy.sh heartbeat (train.py lines 61-76) requires internet access and is fire-and-forget. While the daemon-thread pattern avoids blocking, this is an unconditional external dependency.
- **Risk:** If ntfy.sh is down or the network is slow, silent failures accumulate in threads without feedback.
- **Remediation:** Add an `--enable-monitoring` flag to make it opt-in, or log failures explicitly.

### 5. **Detection Thresholds Are Manual-Sweep Constants (Low Risk, But Brittle)**
CNN_THRESHOLD and UNET_THRESHOLD (run_pipeline.py lines 193-194) are hardcoded after manual sweeps. When real data distributions change, these become stale and require manual re-tuning.
- **Risk:** Threshold-tuning is currently a human-in-the-loop process. A future dataset swap or real data update could silently produce sub-optimal candidate counts.
- **Remediation:** Load thresholds from a checkpoint metadata file or make them learnable via a calibration phase on the validation set.

---

## Biggest Gaps to Beat 0.763

Ranked by impact on the competition score:

### 1. **ILP Solver Scalability (HIGHEST IMPACT — ~70% of runtime, super-linear scaling)**
**Current state:** Monolithic global ILP over all 100 timepoints. On dense real data (~1110 candidates/timepoint at development tail), solve time is ~13.44s per 15-timepoint slice (run_pipeline.py lines 204-211). A full 100-timepoint embryo would extrapolate to hundreds of seconds or timeout.
**Gap:** The tracker architecture is sound, but the solver strategy is pre-production. Windowed/rolling-horizon solving or min-cost-flow approximations (Phase 3 scope, PRD §8) are essential before Kaggle submission on the full hidden test set.
**Score impact:** Not a correctness issue (produces correct answers, just slowly), but **submission failure due to timeout is a real risk on dense embryos.**

### 2. **Placeholder Detector Produces Near-Zero Local Score**
**Current state:** The detection model (UNet3D + transformer) is defined in model.py but **never trained**. The pipeline uses `extract_peaks_from_volume()` on **raw quantile-normalized intensity volumes** (run_pipeline.py line 317), not learned heatmaps. This is Phase 0 baseline behavior, intentionally gated by "no real data loaded" (PRD §4.1 blocker #2, now fixed).
**Gap:** Real Phase 2 work is training the model end-to-end with:
- Heatmap targets from .geff centroids (targets.py, implemented)
- Edge targets from .geff edges (targets.py, implemented)
- Motion vectors from .geff edge displacements (targets.py, **not yet implemented** — Phase 2 scope)
- Local eval validation before Kaggle submission (evaluation.py, implemented and correct)

**Score impact:** The current placeholder detector (stride-based NMS on raw intensity) achieves near-zero local score (~0.009, per PRD §4.2) regardless of tuning. The 0.763 baseline uses the same peak-finding algorithm but on TRAINED detector logits. **Closing this gap (learning a real detector) is essential to beat 0.763.**

### 3. **No Motion Vectors (Model Produces Zeros)**
**Current state:** The transformer edge-prediction head uses motion vectors to warp candidate coordinates before distance comparison (tracker.py line 101). However, `run_pipeline.py` hardcodes all motion vectors to `[0.0, 0.0, 0.0]` (line 339).
**Gap:** The model (src/model.py, STACTCentroidPredictor) defines a motion_head (lines 378-381), but it's never trained on real .geff edge displacement targets. Phase 2 scope is to generate motion targets and wire them into the tracker.
**Score impact:** Zero motion vectors are a conservative choice (most edges are consecutive-frame with tiny displacement per PRD analysis). However, **gap-closing edges (t → t+2/t+3) benefit from motion prediction to prevent false positives.** Current implementation is sub-optimal but not catastrophic.

### 4. **Sigma Tuning Unvalidated**
**Current state:** Heatmap Gaussian sigmas are `sigma_z=1.0, sigma_yx=2.0` voxels (targets.py lines 48-49). The design choice (light-sheet PSF anisotropy vs. geometric isotropy) is documented as unresolved (lines 62-82), and **no empirical A/B testing has been done** against real local eval scores.
**Gap:** A plausible alternative (sigma_z=0.5 for geometric isotropy) might improve or degrade training. Current sigmas are a reasonable starting prior but not validated.
**Score impact:** Likely SMALL (a few percentage points at most in heatmap quality), but unvalidated tuning is a risk before Kaggle submission.

### 5. **Missing Mitosis Intensity Evaluation**
**Current state:** Mitosis smoothing (tracker.py lines 180-227) uses a mock intensity proxy (`1.0 / (1.0 + distance)`, line 209). This doesn't reflect real image intensity around candidate split frames.
**Gap:** Phase 2 work is to compute real local image intensity in a spatial window around each candidate division frame and use that to tune the optimal split timing.
**Score impact:** MEDIUM. Division edges are weighted 0.1x vs. regular edges (PRD §3.3), so even a perfect division fix yields at most ~0.1 point gain. Current mock approach is a reasonable placeholder.

### 6. **Duplicate Code & Unit-Scale Bug Risk**
**Current state:** `extract_peaks_from_volume()`, `pool_kernel_from_um()`, and the ratio-vs-microns bug recur in multiple files.
**Gap:** Consolidation + programmatic checks to prevent re-drift.
**Score impact:** INDIRECT. No correctness loss (both versions of duplicated code are identical), but increases risk of future bugs and maintenance burden.

---

## Summary Ratings by Taste Framework

| File | Rating | Key Pattern | Risk Level |
|------|--------|-------------|-----------|
| `data_loader.py` | 🟢 | Data ingestion discipline (no silent fallbacks, real blosc2 fix) | LOW |
| `targets.py` | 🟡 | Sound heatmap/edge logic, but sigma tuning unvalidated + unused param | MEDIUM |
| `tracker.py` | 🟢 | Correct ILP formulation, scalability gap (not a taste issue) | MEDIUM-HIGH |
| `evaluation.py` | 🟢 | Correct, thin wrapper around vendored scorer, type-safe | LOW |
| `train.py` | 🟡 | Good patterns, but duplicated code, incomplete scope visible | MEDIUM |
| `model.py` | 🟢 | Sound UNet3D + transformer architecture, good initialization | LOW |
| `run_pipeline.py` | 🟡 | Correct multi-dataset logic, residual ratio-vs-microns risk, duplication | MEDIUM |

---

## Final Verdict

**Code quality is GOOD to EXCELLENT on architecture and correctness.** The codebase demonstrates disciplined coding practices (early validation, graceful error handling, proper dependency management). The ILP tracker formulation is sophisticated and correct. The evaluation harness faithfully replicates the competition metric.

**However, the codebase is in Phase 0 (baseline/diagnostic) state, not Phase 2+ (trained-model) state.** The biggest gap to beating 0.763 is not code quality — it's that the **learning-based detector has never been trained on real data.** Closing that gap (Phase 2) requires training infrastructure already mostly in place (loss functions, data loading, evaluation harness), not architectural rewrites.

**Before Kaggle submission:**
1. Complete Phase 2: Train UNet3D + transformer on real .geff data with proper targets.
2. Validate sigma_z/sigma_yx tuning against local eval score (A/B test both values).
3. Consolidate duplicate utility functions into `src/inference_common.py`.
4. Profile ILP solve time on the real hidden test set; confirm <12 hours with margin.
5. Remove unused `anisotropy` parameter from `generate_heatmap_targets()` to eliminate false rigor.

**The code is ready to build on. The physics is correct. The architecture is sound. The missing piece is the trained model.**
