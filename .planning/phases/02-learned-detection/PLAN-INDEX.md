# Phase 2: Learned Detection — Complete Planning Index

**Phase:** 02 (Learned Detection)  
**Goal:** Train a real detection model on .geff ground-truth targets and wire it end-to-end, beating the classical baseline (0.763) and generating the first real Kaggle submission with local score >=0.80+.  
**Planning Status:** COMPLETE  
**Date:** 2026-07-08  

---

## Plan Structure

Phase 2 is organized into **4 waves** with explicit dependencies and parallel execution opportunities:

### Wave 1: Infrastructure & Decisions
**File:** `01-PLAN.md`  
**Depends:** None (startup)  
**Duration:** ~2-4 hours  
**Autonomous:** true

Establishes data pipeline and resolves two critical decision points:
1. **Normalization choice** (RESEARCH.md S3): empirically benchmark [0,1]-clipped zarr-quantile vs host's [0,4.0]-clipped self-computed
2. **Data infrastructure:** extract 199-sample competition set, build embryo-disjoint train/val split (per-sample granularity), implement PyTorch Dataset

**Key Tasks:**
- Task 1.1: Benchmark & decide normalization approach
- Task 1.2: Extract & organize full 199-sample dataset
- Task 1.3: Build embryo-disjoint train/val split (149/50, stratified by movie prefix)
- Task 1.4: Implement PyTorch CompetitionDataset class

**Exit Criteria:**
- Normalization approach locked in with empirical justification
- Data split validated (no leakage, stratified)
- Dataset class tested on all 199 real competition samples

**Outputs:** 01-SUMMARY.md, data_split.json, src/dataset.py

---

### Wave 2: Model Architecture & Target Generation
**File:** `02-PLAN.md`  
**Depends:** Wave 1 complete  
**Duration:** ~4-6 hours  
**Autonomous:** true

Implements full host-style learned detection architecture (UNet + Transformer) and target generation pipeline. Two parallel tracks:

**Track A (Model):**
- Task 2.1: 3D UNet backbone (32→64→128 channels, (1,4,4) strides, per-voxel logits)
- Task 2.2: SimpleNodeTransformer (edge probability prediction via cross-attention)
- Task 2.5: Test-time augmentation (4-view averaging: original, flip Y, flip X, flip YX)
- Task 2.6: Greedy edge assignment (not ILP at inference, per locked decision)

**Track B (Targets):**
- Task 2.3: **DECISION POINT:** Benchmark heatmap targets (point vs dilated Gaussian)
  - Empirically test on 10 held-out samples, pick winner based on local evaluation score
- Task 2.4: Edge probability targets & division loss weighting (edge_loss_weights *= weight_division for division edges)

**Exit Criteria:**
- Model architecture matches host paradigm (UNet → Transformer → greedy)
- Heatmap target generation empirically benchmarked (point vs dilated, winner documented)
- Edge targets & division loss weighting implemented
- All components verified on 10 real competition samples

**Outputs:** 02-SUMMARY.md (with heatmap benchmark results), src/model.py, src/targets.py, src/inference.py

---

### Wave 3: Training & Validation (Sanity Check)
**File:** `03-PLAN.md`  
**Depends:** Wave 2 complete  
**Duration:** ~2-4 hours (smoke test locally) + 30-60 min (Kaggle sanity-check run)  
**Autonomous:** false (user async trigger on Kaggle)

Builds training loop and runs sanity-check training on full 199-sample set with limited epochs. **Critical gate:** no commitment to full training until sanity-check validates setup.

**Key Tasks:**
- Task 3.1: Implement training loop (TrainingLoop class with loss computation, validation, early stopping, checkpointing)
- Task 3.2: Prepare Kaggle training script (self-contained, uses /kaggle/input paths)
- Task 3.3: **USER TRIGGER:** Run Kaggle sanity-check (3-5 epochs, full 199 samples, ~30-60 min runtime)
- Task 3.4: Retrieve & evaluate sanity-check checkpoint locally
- Task 3.5: **DECISION GATE:** User reviews sanity-check results and approves full training Y/N

**Exit Criteria:**
- Training loop implemented with proper loss computation, validation, early stopping
- Sanity-check runs to completion without errors (Kaggle job status "Completed")
- Loss curves reasonable (decreasing, not flat/increasing)
- Validation metrics non-NaN and meaningfully better than Phase 1 baseline (0.0259)
- User explicitly approves proceeding to full training

**Outputs:** src/train.py, kaggle_kernel/train_kernel.py, 03-SUMMARY.md (sanity-check report with plots)

---

### Wave 4: Full Training & Kaggle Submission
**File:** `04-PLAN.md`  
**Depends:** Wave 3 complete + user sign-off  
**Duration:** ~2-4 hours (Kaggle full training run on T4 x2, 50+ epochs)  
**Autonomous:** false (user async trigger + submission decision gate)

Executes full training run (50+ epochs, early-stop patience ~10) and conditionally submits to Kaggle leaderboard based on validation score.

**Key Tasks:**
- Task 4.1: **USER TRIGGER:** Submit full training to Kaggle (50+ epochs, ~2-4 hours runtime)
- Task 4.2: Retrieve best checkpoint and evaluate on 50 held-out validation samples via src/evaluation.py
- Task 4.3: **DECISION GATE:** Submission decision (only submit if adjusted_edge_jaccard >=0.80+ or >0.763+margin)
- Task 4.4: Run inference on 4 test samples and generate submission CSV (if gate passes)
- Task 4.5: Submit to Kaggle via API and track leaderboard score (if gate passes)
- Task 4.6: Finalize 04-SUMMARY.md and verify Phase 2 exit criteria

**Exit Criteria:**
- Full training completes (50+ epochs or early-stopped with best checkpoint)
- Local held-out evaluation: adjusted_edge_jaccard computed on 50 samples
- Submission decision documented with explicit reasoning
- **If local score >=0.80+:** Kaggle submission successful with public leaderboard score recorded
- Phase 2 exit criterion verified: real targets used, naive extraction replaced, score materially exceeds Phase 1

**Outputs:** 04-SUMMARY.md (full results + leaderboard score), submissions/submission_phase2_model_v1.csv (if submitted)

---

## Decision Points

Phase 2 planning explicitly resolves two critical decisions from RESEARCH.md as **empirical tasks**, not guesses:

### Decision 1: Data Normalization (Task 1.1)
**Question:** Keep existing [0,1]-clipped zarr-quantile normalization or switch to host's [0,4.0]-clipped self-computed quantile?

**Approach:**
- Load 2-3 real competition samples via both approaches
- Measure histogram statistics (mean, std, percentiles, foreground/background separation)
- Document findings with explicit recommendation: Option A (keep [0,1]) or Option B (switch to [0,4.0])
- **Output:** Documented decision in 01-SUMMARY.md

**Impact:** Phase 1's peak-finding thresholds (0.85, 0.9) are calibrated for [0,1] normalization. Switching to [0,4.0] requires recalibration before proceeding. Decision must be explicit, not implicit.

### Decision 2: Heatmap Target Type (Task 2.3)
**Question:** Point targets (single voxel per centroid) or dilated Gaussian targets (soft, anisotropic kernel)?

**Approach:**
- Implement both options as functions
- Benchmark on 10 held-out samples:
  - Train lightweight detection head (2-3 epochs) on each heatmap type
  - Evaluate via src/evaluation.py: edge_jaccard + adjusted_edge_jaccard
  - Calculate mean scores across 10 samples for each option
- Pick winner (higher adjusted_edge_jaccard)
- **Output:** Benchmark results + recommendation in 02-SUMMARY.md

**Impact:** Target type affects model convergence speed and final performance. Empirical benchmarking is required per CONTEXT.md's explicit locked decision.

---

## Locked Decisions (Non-Negotiable)

These decisions from CONTEXT.md (S1–5) are already locked and must be respected in all plans:

1. **Model Architecture:** Full host-style replacement (3D UNet → Transformer → greedy edge assignment), NOT incremental deepening of existing STHypergraphTracker. ILP tracker kept as fallback/comparison only, not deleted.

2. **Training Scale:** Train on FULL 199-sample set from the start, NOT smoke-testing on 4 staged samples first. Run sanity-check (few epochs) first before full training commitment.

3. **Test-Time Augmentation:** Include 4-view averaging from the start (not deferred to later phase).

4. **Division Loss Weighting:** Upweight division edges in loss (weight_division ~2.0–3.0x).

5. **Submission Timing:** Only submit once local held-out score clearly and materially exceeds 0.763 baseline — do not waste submissions validating upload/scoring loop early.

6. **Embryo-Disjoint Split:** Split at per-sample granularity (individual sample IDs like 44b6_0113de3b), NOT movie-prefix level (which is meaningless since both train and test draw from both prefixes).

---

## Must-Haves (Quality Gate Checklist)

All of these must be satisfied before Phase 2 is considered complete:

- [ ] **PLAN.md files created:** 01-PLAN.md, 02-PLAN.md, 03-PLAN.md, 04-PLAN.md in .planning/phases/02-learned-detection/
- [ ] **Each plan has valid frontmatter:** wave, depends_on, files_modified, autonomous
- [ ] **Tasks are specific and actionable:** each task has clear execution steps, verification criteria, and measurable outcomes
- [ ] **Dependencies correctly identified:** inter-wave dependencies (Wave 1→2→3→4), decision gates
- [ ] **Waves assigned for parallel execution:** Track A (model) and Track B (targets) in Wave 2 can run in parallel after Wave 1
- [ ] **Must-haves derived from phase goal:** Every task contributes to local score >=0.80+ goal
- [ ] **Normalization decision point explicit:** Task 1.1 empirically benchmarks [0,1] vs [0,4.0], locks choice in 01-SUMMARY.md
- [ ] **Target-generation empirical benchmark explicit:** Task 2.3 benchmarks point vs dilated Gaussian, locks choice in 02-SUMMARY.md
- [ ] **Plans reflect locked CONTEXT.md decisions:**
  - [ ] Full host-style architecture (UNet+Transformer)
  - [ ] ILP kept as fallback, not deleted
  - [ ] Full 199-sample training with sanity-check first
  - [ ] Division loss weighting included
  - [ ] TTA from the start
  - [ ] Submission only after local score beats baseline
- [ ] **Plans account for corrected embryo-disjoint split:** per-sample granularity (not movie-prefix level), no data leakage

---

## Downstream Execution

These plans are designed to be executed via `/gsd:execute-phase` orchestrator:

1. **Execute Wave 1** (autonomous): User triggers, system runs Task 1.1–1.4 sequentially (dependencies)
2. **Execute Wave 2** (autonomous): Tracks A & B can run in parallel after Wave 1 complete
3. **Execute Wave 3** (non-autonomous): User triggers Kaggle sanity-check (Task 3.3), awaits completion, reviews 03-SUMMARY.md, approves Wave 4
4. **Execute Wave 4** (non-autonomous): User triggers full training (Task 4.1), awaits completion, reviews 04-SUMMARY.md, makes submission decision (Task 4.3)

Executor tracks progress in checkpoints after each wave. If a task fails, executor stops and reports to user for diagnosis/rework.

---

## Key Files Generated by Phase 2

**Strategy/Config:**
- `.planning/phases/02-learned-detection/01-SUMMARY.md` — normalization choice, data setup stats
- `.planning/phases/02-learned-detection/02-SUMMARY.md` — heatmap benchmark results, division loss weight
- `.planning/phases/02-learned-detection/03-SUMMARY.md` — sanity-check results, loss curves, user decision
- `.planning/phases/02-learned-detection/04-SUMMARY.md` — full training results, leaderboard score, Phase 2 exit verification
- `data_split.json` — embryo-disjoint train/val split (149/50 samples)

**Code:**
- `src/dataset.py` — CompetitionDataset (PyTorch, loads Zarr v3 + .geff)
- `src/model.py` — UNet3D, SimpleNodeTransformer (architecture classes)
- `src/targets.py` — generate_heatmap_targets(), generate_edge_targets() (target generation)
- `src/train.py` — TrainingLoop (training loop with validation, checkpointing)
- `src/inference.py` — tta_inference(), greedy_edge_assignment() (inference utilities)
- `kaggle_kernel/train_kernel.py` — self-contained Kaggle training script

**Outputs:**
- `submissions/submission_phase2_model_v1.csv` — test set predictions (if submitted)
- Model checkpoint (best validation score)
- Training logs (per-epoch metrics)

---

## Success Metrics (Phase 2 Exit)

Phase 2 is complete when ALL of the following are satisfied:

1. **Local validation score >=0.80+** (target) OR **>0.763 + margin** (minimum)
   - Metric: adjusted_edge_jaccard on 50 held-out samples
   - Comparison: Phase 1 (0.0259), Classical baseline (0.763)

2. **Real targets used** ✓
   - Heatmap targets from .geff centroids (point or dilated Gaussian, empirically chosen)
   - Edge targets from .geff edges (binary labels, inverse-frequency weighting)
   - Division loss weighting implemented

3. **Naive peak extraction replaced** ✓
   - UNet produces learned detection logits (not raw intensity peaks)
   - NMS peak extraction applied to learned logits (not naive grid scan)

4. **Motion vectors replaced with edge probabilities** ✓
   - Transformer predicts pairwise edge probabilities (learned, not hardcoded vectors)
   - Greedy edge assignment builds tracks from predicted edges

5. **Kaggle submission successful** (if local score clears gate)
   - Submission CSV validated and uploaded
   - Public leaderboard score recorded

---

## Integration with Roadmap

This Phase 2 plan connects to the broader project roadmap (ROADMAP.md):

- **Phase 1 (Baseline Parity):** Attempted/closed at score 0.0259 (naive peak extraction + ILP). Phase 2 builds on Phase 1's infrastructure (evaluation harness, anisotropy fixes, data loader) but replaces detection approach (naive → learned).

- **Phase 2 (Learned Detection):** THIS PHASE. Exit when local score >=0.80+ and first Kaggle submission successful.

- **Phase 3 (Advanced Tracking):** Deferred. Will focus on (if needed) ILP mitosis smoothing, edge probability refinement, multi-frame gap-closing. NOTE: CONTEXT.md notes that Phase 3 should revisit whether ILP requirements still apply after Phase 2's pivot to greedy assignment.

---

## Next Steps (If Phase 2 Incomplete)

If Phase 2 does not achieve local score >=0.80+:

1. **Review loss curves + validation metrics** from 03-SUMMARY.md and 04-SUMMARY.md
2. **Diagnose issues** (overfitting? data preprocessing issue? hyperparams?):
   - Check heatmap target quality (visualization overlay on raw volumes)
   - Check edge label class imbalance (are positives too rare?)
   - Check normalization calibration (is threshold still valid?)
   - Check training stability (validation metrics moving or flat?)
3. **Iterate:** Modify hyperparams, target type, or loss weighting based on findings
4. **Re-run Kaggle training** after small adjustments (Task 4.1 can be repeated)
5. **Escalate to Phase 3** only when diagnosis requires architectural changes beyond Wave 2

---

## Document Versions

| Document | Version | Date | Status |
|----------|---------|------|--------|
| 02-CONTEXT.md | - | 2026-07-08 | Locked (already provided) |
| 02-RESEARCH.md | - | 2026-07-08 | Locked (already provided, fact-checked) |
| 01-PLAN.md | 1.0 | 2026-07-08 | Ready for execution |
| 02-PLAN.md | 1.0 | 2026-07-08 | Ready for execution |
| 03-PLAN.md | 1.0 | 2026-07-08 | Ready for execution |
| 04-PLAN.md | 1.0 | 2026-07-08 | Ready for execution |
| PLAN-INDEX.md | 1.0 | 2026-07-08 | This document (planning complete) |

---

## Planning Complete ✓

All quality-gate requirements satisfied:
- ✓ PLAN.md files (01, 02, 03, 04) created with valid frontmatter
- ✓ Tasks specific, actionable, measurable
- ✓ Dependencies correctly identified (Wave 1→2→3→4, decision gates)
- ✓ Parallel execution opportunities identified (Track A/B in Wave 2)
- ✓ Must-haves derived from phase goal (local score >=0.80+)
- ✓ Normalization decision point explicit (Task 1.1, empirical benchmark)
- ✓ Target-generation decision point explicit (Task 2.3, empirical benchmark)
- ✓ Locked CONTEXT.md decisions reflected throughout
- ✓ Corrected embryo-disjoint split (per-sample, not movie-prefix)

**Ready for execution via `/gsd:execute-phase`.**

