# Phase 2: Learned Detection — Planning Complete ✓

**Status:** PLANNING COMPLETE  
**Date:** 2026-07-08 (phase planner: Haiku 4.5)  
**Quality Gate:** ALL REQUIREMENTS SATISFIED  

---

## Quality Gate Checklist (Required before execution)

### ✓ PLAN.md Files Created

| File | Wave | Depends | Autonomous | Size |
|------|------|---------|-----------|------|
| `01-PLAN.md` | 1 | None | true | 9.8 KB |
| `02-PLAN.md` | 2 | Wave 1 | true | 15.5 KB |
| `03-PLAN.md` | 3 | Wave 2 | false | 13.9 KB |
| `04-PLAN.md` | 4 | Wave 3 + user sign-off | false | 15.7 KB |

All plans located in: `.planning/phases/02-learned-detection/`

### ✓ Valid Frontmatter

Each PLAN.md includes required frontmatter (YAML):
- **wave:** 1, 2, 3, or 4 (correct sequencing)
- **depends_on:** List of dependencies (empty for Wave 1, explicit Wave N references for others)
- **files_modified:** List of files created/modified in each wave
- **autonomous:** true/false (False for user-interaction gates in Waves 3–4)

**Examples from plans:**
```yaml
# 01-PLAN.md
wave: 1
depends_on: []
files_modified: [src/dataset.py, ...]
autonomous: true

# 03-PLAN.md
wave: 3
depends_on:
  - 02-PLAN.md (Wave 2 complete)
files_modified: [src/train.py, ...]
autonomous: false  # User trigger for Kaggle job
```

### ✓ Tasks Specific & Actionable

Each plan contains 4–6 tasks with:
- **Execution steps:** numbered, concrete actions (not vague "implement X")
- **Verification criteria:** measurable outcomes ([✓] checklist format)
- **Dependencies:** inter-task ordering when needed
- **Owner:** Executor, User, or both
- **Autonomous flag:** indicates if task requires user intervention

**Example (Task 1.1 — Normalization Decision):**
- Execution: Load 2-3 samples via both approaches, measure histograms, benchmark empirically
- Verification: Decision documented in 01-SUMMARY.md with explicit reasoning
- Owner: Executor (analysis) → User (decision)
- Autonomous: false (decision gate)

### ✓ Dependencies Correctly Identified

```
Wave 1 (startup)
  ├─ Task 1.1: Benchmark normalization → 01-SUMMARY.md
  ├─ Task 1.2: Extract dataset (depends 1.1)
  ├─ Task 1.3: Build split (depends 1.2)
  └─ Task 1.4: Implement Dataset (depends 1.3)

Wave 2 (depends Wave 1 complete)
  ├─ PARALLEL Track A (Model)
  │  ├─ Task 2.1: UNet backbone
  │  ├─ Task 2.2: Transformer (depends 2.1)
  │  ├─ Task 2.5: TTA (depends 2.1)
  │  └─ Task 2.6: Greedy assignment (depends 2.2)
  └─ PARALLEL Track B (Targets)
     ├─ Task 2.3: Heatmap benchmark (DECISION POINT)
     └─ Task 2.4: Edge targets + division loss (depends 2.3)

Wave 3 (depends Wave 2 complete)
  ├─ Task 3.1: Training loop
  ├─ Task 3.2: Kaggle script (depends 3.1)
  ├─ Task 3.3: Kaggle sanity-check (USER TRIGGER, depends 3.2)
  ├─ Task 3.4: Evaluate sanity checkpoint (depends 3.3)
  └─ Task 3.5: Review & decide full training (DECISION GATE)

Wave 4 (depends Wave 3 complete + user sign-off)
  ├─ Task 4.1: Full training submit (USER TRIGGER)
  ├─ Task 4.2: Local evaluation (depends 4.1)
  ├─ Task 4.3: Submission decision gate (DECISION GATE)
  ├─ Task 4.4: Test inference + CSV (depends 4.3)
  ├─ Task 4.5: Kaggle submission (depends 4.4)
  └─ Task 4.6: Final summary (depends 4.5 or 4.3 if no-submit)
```

**Parallel Execution Opportunities:**
- Wave 2: Track A (model) and Track B (targets) can run in parallel after Wave 1 complete
- Waves 3–4: Async Kaggle jobs (user triggers, system awaits completion)

### ✓ Waves Assigned for Parallel Execution

**Wave 1 (Sequential, ~2-4 hours):** Data decisions and infrastructure setup (cannot parallelize; each task feeds the next)

**Wave 2 (Parallel Tracks, ~4-6 hours):**
- **Track A:** UNet backbone → Transformer → TTA + Greedy assignment (model architecture pipeline)
- **Track B:** Heatmap benchmark (DECISION) → Edge targets + division loss (target generation pipeline)
- Both tracks can run in parallel after Wave 1, merge at end of Wave 2

**Wave 3 (Async + Sequential, ~2-4 hours local + 30-60 min Kaggle):**
- Local: Training loop implementation (Task 3.1–3.2, sequential)
- Kaggle: Submit sanity-check job (async, user awaits ~1 hour)
- Local: Evaluate results + decision (Task 3.4–3.5, sequential after Kaggle completes)

**Wave 4 (Async + Sequential, ~2-4 hours local + 2–4 hours Kaggle):**
- Kaggle: Submit full training job (async, user awaits ~2–4 hours)
- Local: Evaluate + decision → submission (Task 4.2–4.6, sequential after Kaggle completes)

### ✓ Must-Haves Derived from Phase Goal

Phase 2 Goal: **"Local score materially exceeds Phase 1 (target: >=0.80+), achieved by training the model on real targets, replacing naive peak extraction and hardcoded motion vectors with real predictions, and successfully submitting to Kaggle leaderboard."**

Each task directly contributes to one or more must-haves:

| Must-Have | Tasks | Verification |
|-----------|-------|--------------|
| Real heatmap targets from .geff | 2.3, 2.4 | 02-SUMMARY.md: heatmap benchmark results |
| Real edge targets from .geff | 2.4 | 02-SUMMARY.md: edge label generation verified |
| Naive peak extraction replaced | 2.1, 2.5 | 02-PLAN.md: UNet + TTA verified on real data |
| Motion vectors → edge probabilities | 2.2, 2.6 | 02-PLAN.md: Transformer + greedy assignment verified |
| Local score materially exceeds Phase 1 | 3.4, 4.2 | 03-SUMMARY.md, 04-SUMMARY.md: score comparison documented |
| Score >=0.80+ or >0.763+margin | 4.2, 4.3 | 04-SUMMARY.md: local score vs baselines, decision gate |
| Kaggle submission successful | 4.4, 4.5 | 04-SUMMARY.md: leaderboard score + submission ID |

### ✓ Normalization Decision Point Explicit

**Task 1.1: Benchmark & Decide Data Normalization Approach**

Decision Point from RESEARCH.md S3: Keep [0,1]-clipped zarr-quantile OR switch to host's [0,4.0]-clipped self-computed?

**Explicit Benchmark Approach:**
1. Load 2–3 competition samples via both normalization methods
2. Measure histogram statistics: mean, std, percentiles, foreground/background separation quality
3. Document findings with visual comparison (histograms or statistics table)
4. Make explicit choice: Option A or Option B
5. If Option B: add threshold recalibration task to Wave 2 (reuse sweep_threshold.py logic)

**Verification:**
- [ ] 01-SUMMARY.md contains: Option chosen, empirical measurements, reasoning
- [ ] No implicit assumptions — choice is explicit and justified

**Impact Awareness:** Phase 1's thresholds (CNN_THRESHOLD=0.85, UNET_THRESHOLD=0.9) are calibrated for [0,1]. Switching to [0,4.0] requires recalibration before model training begins.

### ✓ Target-Generation Empirical Benchmark Explicit

**Task 2.3: Benchmark & Implement Heatmap Target Generation**

Decision Point from RESEARCH.md S4: Point targets OR dilated Gaussian targets?

**Explicit Benchmark Approach:**
1. Implement both options:
   - **Option A:** Single voxel per centroid (sparse, ~0.1% positive)
   - **Option B:** Anisotropic Gaussian (sigma_z=1.0, sigma_yx=2.0 voxels, ~1–2% positive)
2. Benchmark on 10 held-out validation samples:
   - Train lightweight detection head (2–3 epochs) on each heatmap type
   - Evaluate via src/evaluation.py: compute edge_jaccard + adjusted_edge_jaccard
   - Log mean scores across 10 samples for each option
3. Pick winner (higher adjusted_edge_jaccard)
4. Lock heatmap target type for rest of training

**Verification:**
- [ ] 02-SUMMARY.md contains: Option A score (mean ± std), Option B score (mean ± std)
- [ ] Winner chosen with explicit reasoning ("Option X wins by +Y points")
- [ ] target_type parameter locked for all downstream tasks

**Impact Awareness:** Target type affects model convergence speed and final performance. Empirical testing required per CONTEXT.md's explicit locked decision.

### ✓ Plans Reflect Locked CONTEXT.md Decisions

All plans respect and enforce the locked decisions from 02-CONTEXT.md (verified in plans):

| Decision | Plan Reference | Enforcement |
|----------|-----------------|-------------|
| **Full host-style architecture** (UNet+Transformer, not incremental) | Task 2.1–2.2, 02-PLAN.md S1 | Architecture specified exactly (channels [32,64,128], strides (1,4,4), etc.) |
| **ILP kept as fallback, not deleted** | Task 2.6, 02-PLAN.md | Greedy assignment implemented; STHypergraphTracker left in repo as comparison |
| **Full 199-sample training from start** | Task 1.2, Task 3.3, 03-PLAN.md | Extract full dataset (Wave 1), run Kaggle training on all 199 (Wave 3), no smoke-test on 4 staged samples first |
| **Sanity-check run first (few epochs)** | Task 3.3, 03-PLAN.md | Kaggle sanity-check configured for 3–5 epochs before committing to 50+ |
| **Division loss weighting included** | Task 2.4, 02-PLAN.md | weight_division (~2.0–3.0x) applied to division edges |
| **TTA from the start** | Task 2.5, 02-PLAN.md | 4-view averaging (original, flip Y, flip X, flip YX) implemented in Wave 2, not deferred |
| **Submission only after local score beats baseline** | Task 4.3, 04-PLAN.md | Decision gate: only submit if adjusted_edge_jaccard >=0.80+ OR >0.763+margin |

All locked decisions explicitly wired into task descriptions and verification criteria.

### ✗ P0-2 CORRECTION (2026-07-16): the split below was NOT actually embryo-disjoint

**RESEARCH.md S2.3 Correction was itself wrong** -- it inferred "embryo" = individual
sample ID by comparing against the 4 staged `test/` placeholder copies (byte-identical
duplicates of 4 train samples, not the real hidden test set), not against authoritative
competition documentation. Kaggle's own Data description page states plainly that the
movie prefix (44b6/6bba) IS the embryo ID and "multiple samples may share the same
embryo." The stratified split this section describes therefore had real embryo-level
leakage (both `44b6` and `6bba` present in both `data_split.json`'s train and validation
lists, in its historical pre-P0-2 content). Superseded by
`scripts/build_train_val_split.py`'s leave-one-embryo-out fold generator -- see the P0-2 audit
for the full evidence trail. The root `data_split.json` has since been replaced with an exact
copy of `data_splits/embryo_44b6_validation.json` (a compatibility alias, genuinely
embryo-disjoint); prefer resolving the active fold via `src/split_utils.py`'s
`resolve_split_file_path()` rather than hardcoding either filename directly.

**Original (incorrect) claims, preserved for history:**

**Plans Enforce Correct Split:**
- Task 1.3: Partition 199 individual sample IDs into 149 train / 50 validation
- Stratify by movie prefix (44b6: 71 total → ~53 train / ~18 validation; 6bba: 128 total → ~96 train / ~32 validation)
- Never mix samples from same ID across train/validation
- Verified no leakage (split JSON with explicit sample lists)

**Verification:**
- [x] data_split.json: 199 total = 149 train + 50 validation (per-sample granularity, not prefix-based)
- [x] Both splits contain samples from both 44b6 and 6bba prefixes (no prefix-level split) -- **this checkbox itself describes the leakage; it was treated as a pass criterion under the incorrect embryo definition**
- [x] Dataset class respects split_file filtering (train set loads only 149 samples)

---

## Planning Artifacts

### Core Planning Documents
- **02-CONTEXT.md** (locked strategy from user/manager) — architecture choice, training scale, validation strategy
- **02-RESEARCH.md** (fact-checked findings) — hyperparameters, data infrastructure, empirical benchmarks, risks
- **01-PLAN.md** (Wave 1 execution) — infrastructure & decisions (4 tasks)
- **02-PLAN.md** (Wave 2 execution) — model & targets (6 tasks, 2 parallel tracks)
- **03-PLAN.md** (Wave 3 execution) — training & sanity-check (5 tasks, 1 async gate)
- **04-PLAN.md** (Wave 4 execution) — full training & submission (6 tasks, 1 async + 1 decision gate)
- **PLAN-INDEX.md** (this session's summary) — overview of all waves, dependencies, key decisions
- **PLANNING-COMPLETE.md** (this document) — quality gate verification

### Expected Outputs (Phase 2 Execution)

**Configuration & Decisions:**
- 01-SUMMARY.md — Normalization choice + data extraction stats
- 02-SUMMARY.md — Heatmap benchmark results + division loss weight
- 03-SUMMARY.md — Sanity-check training report + user decision
- 04-SUMMARY.md — Full training results + leaderboard score + Phase 2 exit verification
- data_split.json — originally a stratified train/val split (149/50); **the historical pre-P0-2 version of this file was NOT embryo-disjoint, see P0-2 correction above.** The current root `data_split.json` has since been replaced with an exact copy of `data_splits/embryo_44b6_validation.json` (a compatibility alias for the primary leave-one-embryo-out fold), so any caller still hardcoding this filename now gets a genuinely embryo-disjoint split.

**Code:**
- src/dataset.py — CompetitionDataset (Zarr v3 + .geff loader)
- src/model.py — UNet3D, SimpleNodeTransformer classes
- src/targets.py — Heatmap + edge target generation
- src/train.py — TrainingLoop with validation, checkpointing, early stopping
- src/inference.py — TTA inference, greedy edge assignment
- kaggle_kernel/train_kernel.py — Self-contained Kaggle training script

**Submissions & Logs:**
- submissions/submission_phase2_model_v1.csv — Test set predictions (if local score clears gate)
- Model checkpoint (best validation score)
- training_full_log.csv — Per-epoch metrics from full 50+ epoch run

---

## Phase 2 Success Criteria (Exit Condition)

Phase 2 is **complete** when ALL of the following are achieved:

1. **✓ Local validation score clearly exceeds Phase 1 baseline (0.0259)**
   - Target: adjusted_edge_jaccard >=0.80+ (primary goal)
   - Minimum: adjusted_edge_jaccard > 0.763 (classical baseline) + margin (~0.05)
   - Metric: Mean ± std across 50 held-out validation samples

2. **✓ Real targets used end-to-end**
   - Heatmap targets from .geff centroids (point or dilated, empirically chosen)
   - Edge targets from .geff edges (binary 0/1 labels)
   - Division loss weighting applied (weight_division on division edges)
   - Inverse-frequency weighting on both losses (class imbalance handling)

3. **✓ Naive peak extraction replaced with learned model**
   - UNet produces learned per-voxel detection logits (not raw intensity)
   - NMS peak extraction applied to learned logits (sigmoid >threshold, spatial NMS)
   - extract_peaks_from_volume() retired from main pipeline or gated behind a flag

4. **✓ Motion vectors replaced with learned edge probabilities**
   - Transformer predicts pairwise edge probabilities (learned, not hardcoded displacement vectors)
   - Greedy edge assignment builds directed track graphs from predicted edges
   - STHypergraphTracker (ILP) kept as fallback/comparison only, not active in production inference

5. **✓ Kaggle submission successful**
   - Submission CSV generated and validated (via validate_submission())
   - Uploaded to Kaggle via API
   - Public leaderboard score recorded and documented in 04-SUMMARY.md

6. **✓ All decision points resolved explicitly**
   - Normalization approach locked in 01-SUMMARY.md (Option A or B with reasoning)
   - Heatmap target type locked in 02-SUMMARY.md (point or dilated Gaussian with scores)

---

## Next Phase (Phase 3)

Phase 3 (Advanced Tracking) readiness depends on Phase 2 outcome:

| Outcome | Phase 3 Status | Recommended Action |
|---------|----------------|-------------------|
| Local score >=0.80+ | Ready to proceed | Phase 3 plans can execute as designed (ILP mitosis smoothing, edge refinement) |
| Local score ∈(0.763, 0.80) | Conditional proceed | Phase 3 can proceed with focus on refinement (hyperparameter tuning) before major architecture changes |
| Local score ≤0.763 | Blocked | Debug Phase 2 before advancing (likely data/target quality issues, not architecture problems) |

**NOTE from CONTEXT.md:** Phase 3 should explicitly revisit whether ILP requirements (TRACK-02/TRACK-03 from original roadmap) still apply, given Phase 2's pivot away from ILP as the active production path.

---

## Execution Readiness Checklist

Before executing via `/gsd:execute-phase`:

- [ ] Read all PLAN-*.md files (01, 02, 03, 04, INDEX)
- [ ] Understand decision gates (Tasks 1.1, 2.3, 3.5, 4.3 require user action/approval)
- [ ] Confirm Kaggle GPU setup (Tesla T4 x2 confirmed, verified working in Phase 0/1)
- [ ] Confirm access to competition dataset (/kaggle/input path verified in Phase 0/1)
- [ ] Confirm local evaluation harness (src/evaluation.py already verified working in Phase 0)
- [ ] Stage sample data locally for smoke testing (4 samples already available)
- [ ] Review CONTEXT.md & RESEARCH.md for any last-minute clarifications

---

## Summary

✓ **Planning is COMPLETE and ready for execution.**

Phase 2's 4-wave plan is designed to:
1. **Resolve critical decisions empirically** (normalization, heatmap type) before committing to training
2. **Run sanity-check first** (few epochs) before expensive full training (50+ epochs on Kaggle)
3. **Gate submissions strictly** (only submit once local score clearly beats baseline)
4. **Track all metrics thoroughly** (edge_jaccard, adjusted, division_jaccard, division_recall)
5. **Enable parallel execution** (model + targets tasks can run in parallel in Wave 2)

All quality gates satisfied. Plans are specific, actionable, and ready for the executor agent.

**Estimated Total Duration (All 4 Waves):**
- Wave 1: ~2–4 hours (local, sequential)
- Wave 2: ~4–6 hours (local, parallel tracks)
- Wave 3: ~2–4 hours (local) + ~30–60 min (Kaggle sanity-check, async)
- Wave 4: ~2–4 hours (local) + ~2–4 hours (Kaggle full training, async)
- **Total: ~12–22 hours active + ~3–5 hours Kaggle async**

Phase execution can begin with Wave 1 immediately.

