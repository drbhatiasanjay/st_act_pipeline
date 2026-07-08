---
wave: 4
depends_on:
  - 03-PLAN.md (Wave 3 complete + user sign-off)
files_modified:
  - run_pipeline.py (updated for inference)
  - .planning/phases/02-learned-detection/04-SUMMARY.md (output)
  - submissions/ (new submission CSV)
autonomous: false
---

# Phase 2 Wave 4: Full Training & Kaggle Submission

## Summary
Wave 4 executes the full training run (50+ epochs, early-stop patience ~10) on the full 199-sample train set, validates on held-out set, and (conditionally) submits the model to Kaggle leaderboard once local validation score materially exceeds the 0.763 classical baseline.

Per CONTEXT.md's locked decision: **Only submit once local held-out score clearly and materially exceeds 0.763** — do not waste a scarce Kaggle submission validating upload/scoring loop early.

**Workflow:**
1. **User sign-off on sanity-check results** (from Wave 3)
2. **Full training run on Kaggle** (50+ epochs, early-stop on val score, expected 2-4 hours on T4 x2)
3. **Retrieve best checkpoint** after training completes
4. **Local evaluation** on held-out set — compute final edge_jaccard, adjusted_edge_jaccard, division_jaccard
5. **Decision gate:** If adjusted_edge_jaccard >=0.80+ (or >0.763 + margin), proceed to submission
6. **Kaggle submission** (if gate passes): Run inference pipeline on test set, generate submission CSV, validate, submit via Kaggle API
7. **Leaderboard monitoring** (observational): Track public/private scores

Exit criterion: Local validation score >=0.80+ with successful Kaggle submission, OR documentation of local score + decision not to submit if below baseline.

## Must-Haves (Goal-Backward Verification)
- [ ] Full training completes with ≥50 epochs (or early-stop triggered after ≥10 epochs with best validation score)
- [ ] Best validation checkpoint retrieved and verified
- [ ] Local held-out evaluation: edge_jaccard + adjusted_edge_jaccard + division_jaccard computed
- [ ] **Submission decision documented:** local score vs 0.763 baseline, explicit go/no-go decision with reasoning
- [ ] If submitted: submission CSV generated, validated, and uploaded to Kaggle successfully
- [ ] Kaggle submission status tracked (in progress, completed)
- [ ] Leaderboard score recorded (public + private, if available)

---

## Tasks

### Task 4.1: Configure & Submit Full Training Run
**Depends:** Wave 3 complete + user sign-off  
**Owner:** User  
**Autonomous:** false

Configure kaggle_kernel/train_kernel.py for full training (50+ epochs) and submit to Kaggle. This is a long-running job (~2-4 hours on T4 x2); user must explicitly trigger.

**Execution Steps:**
1. User reviews Wave 3 sanity-check report (03-SUMMARY.md) and confirms "proceed to full training"

2. Executor updates kaggle_kernel/train_kernel.py config:
   - Set epochs_for_sanity_check → False (or epochs=50)
   - Set early_stop_patience=10
   - Confirm all hyperparams from Wave 2 (learning_rate=1e-4, batch_size=16, etc.)
   - Ensure checkpoint dir will save to /kaggle/working/ (for retrieval)

3. Verify GPU setup on Kaggle:
   - Confirm Tesla T4 x2 selected via website UI (not CLI)
   - Check kernel status page before submitting

4. Push to Kaggle:
   ```bash
   cd C:\Users\hemas\Downloads\st_act_pipeline
   py -m kaggle kernels push -p kaggle_kernel
   ```

5. Monitor job status:
   - Check Kaggle website: Kernels → [kernel name] → "Running" status
   - Expected runtime: 2-4 hours on T4 x2 for 50 epochs
   - If job stalls (> 4 hours with no progress): check Kaggle logs for OOM/errors, diagnose and re-run

6. Upon completion:
   - Kaggle will show "Completed" status
   - Download outputs: checkpoint, training_log.csv, model_summary.txt

**Verification:**
- [ ] User explicitly approved proceeding to full training
- [ ] Kaggle script updated for full 50+ epochs
- [ ] GPU T4 x2 confirmed on Kaggle website (not just guessed)
- [ ] Kernel pushed successfully
- [ ] Kernel reaches "Completed" status on Kaggle (not "Error")
- [ ] Training log saved with ≥50 epochs (or early-stopped after ≥10 with improving val score)

---

### Task 4.2: Retrieve Best Checkpoint & Local Evaluation
**Depends:** Task 4.1  
**Owner:** Executor  
**Autonomous:** true

Download trained model checkpoint from Kaggle and evaluate on held-out validation set locally via src/evaluation.py.

**Execution Steps:**
1. Download Kaggle kernel outputs (kernel ref is positional, `-p` is the output
   folder -- verified against real usage this session):
   ```bash
   py -m kaggle kernels output <owner>/<kernel-slug> -p ./kaggle_full_outputs/
   ```

2. Retrieve and load checkpoint:
   - File: training_full_checkpoint_best.pt (or latest epoch if not labeled "best")
   - Verify checkpoint contains model weights + hyperparams + training metadata

3. Load model and run inference on entire held-out validation set (~50 samples):
   - For each sample: load frames, run UNet (with TTA) + Transformer + greedy assignment
   - Generate predicted track graph via src/inference.py

4. Evaluate predicted graphs against GT via src/evaluation.py:
   - Function: evaluate_submission(pred_graphs, gt_graphs, scale=(1.625, 0.40625, 0.40625), max_distance=7.0, gt_metadata=...)
   - Compute: edge_jaccard, adjusted_edge_jaccard, division_jaccard, combined_score per sample
   - Compute: division_recall (divisions correctly predicted / GT divisions)
   - Log mean + std for each metric across 50 validation samples

5. Detailed results logging:
   - Per-sample scores in CSV (for inspection of outliers)
   - Summary statistics to 04-SUMMARY.md:
     - Mean edge_jaccard: X (std: S_x)
     - Mean adjusted_edge_jaccard: Y (std: S_y)
     - Mean division_jaccard: Z (std: S_z)
     - Mean division_recall: D (diagnostic)
     - Combined score: (Y + 0.1*Z) (competition formula)

6. Comparison to Phase 1:
   - Phase 1 score: 0.0259 (naive peak extraction)
   - Current score: Y (learned model)
   - Improvement: (Y - 0.0259) / 0.0259 × 100%

7. Comparison to baseline:
   - Classical baseline: 0.763
   - Current adjusted_edge_jaccard: Y
   - Delta to baseline: Y - 0.763

**Verification:**
- [ ] Checkpoint downloaded and loaded successfully
- [ ] Inference runs on all 50 validation samples without error
- [ ] All evaluation metrics (edge_jaccard, adjusted, division_jaccard, division_recall) computed
- [ ] No NaN values in final scores
- [ ] Per-sample results logged to CSV for inspection
- [ ] Summary statistics in 04-SUMMARY.md with comparisons to Phase 1 and baseline

---

### Task 4.3: Submission Decision Gate
**Depends:** Task 4.2  
**Owner:** Executor (analysis) + User (decision)  
**Autonomous:** false

Evaluate whether model performance warrants a Kaggle submission. Decision gate: only proceed if adjusted_edge_jaccard >=0.80+ (or materially exceeds 0.763 baseline with at least 0.05+ margin).

**Execution Steps:**
1. Executor compiles decision brief:
   - **Local held-out score (adjusted_edge_jaccard):** Y
   - **vs. Phase 1 (0.0259):** improvement of X%
   - **vs. baseline (0.763):** delta of (Y - 0.763)
   - **Target (Phase 2 exit criterion):** >=0.80+
   - **Recommendation:** "SUBMIT" if Y ≥0.80 or Y > 0.763 + 0.05 margin
                        "DO NOT SUBMIT" if Y ≤0.763 (doesn't clear baseline)

2. Explicit decision criteria:
   - ✓ SUBMIT if: adjusted_edge_jaccard ≥0.80+
   - ✓ SUBMIT if: adjusted_edge_jaccard ∈ (0.763, 0.80) but >0.763 + 0.05 and loss curves/validation stable
   - ✗ DO NOT SUBMIT if: adjusted_edge_jaccard ≤0.763 (below baseline)
   - ✗ DO NOT SUBMIT if: validation metrics unstable or high variance (suggests overfitting or data issue)

3. Present decision brief to user:
   - Include 04-SUMMARY.md with metrics
   - Include loss curve plots and per-sample score distribution
   - Explicit question: "Proceed with Kaggle submission? Y/N"

4. User decision:
   - If Y: proceed to Task 4.4 (Inference + Submission)
   - If N: document reasoning in 04-SUMMARY.md and skip Task 4.4
   - Approval recorded in CONTEXT.md or task comment

**Verification:**
- [ ] Decision brief created with clear metrics and comparison to baseline/target
- [ ] Recommendation (SUBMIT vs DO NOT SUBMIT) explicit
- [ ] User consulted and decision recorded

---

### Task 4.4: Run Inference on Test Set & Generate Submission CSV
**Depends:** Task 4.3 (user approved submission)  
**Owner:** Executor  
**Autonomous:** true

Run end-to-end inference on the official test set (4 samples from Kaggle) and generate submission CSV following competition format.

**Execution Steps:**
1. Load test samples:
   - Path: /kaggle/input/competitions/biohub-cell-tracking-during-development/test/ (if running locally, load from competition zip)
   - 4 samples: 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1

2. Run inference on each test sample:
   - Load all frames (T frames, Z=64, Y=256, X=256) via AnisotropicZarrLoader
   - For each (t, t+1) pair:
     - Concatenate frames → (1, 2, 64, 256, 256)
     - Forward through UNet with TTA → detection logits
     - Extract nodes (>threshold, NMS)
     - Forward through Transformer + greedy assignment → edges
   - Build directed graph G_sample with all predicted tracks

3. Generate output format:
   - Per sample: list of (parent_node_id, child_node_id, frame_t, frame_t1) tuples
   - Match competition format (see sample_submission.csv)
   - Validate output shape: should have ~100-500 edges per sample (sanity check)

4. Create submission CSV:
   - **Do not hand-build this schema.** Call `export_submission()` from
     src/submission_exporter.py (already tested in Phase 0, do not reimplement) --
     it produces the real, competition-required schema:
     `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` (node rows and edge
     rows interleaved per dataset, `-1` sentinels for unused fields, per-dataset
     `node_id` reset). Pass it the predicted graph structure from Task 4.4 step 2.
   - Validate via src/submission_exporter.py's validate_submission() (check format, no duplicates, etc.)

5. Save submission CSV:
   - File: submissions/submission_phase2_model_v1.csv (or similar timestamp)
   - Also save a backup: submissions/submission_phase2_model_v1_BACKUP.csv

6. Log submission details to 04-SUMMARY.md:
   - Num edges predicted per sample
   - File size and format validation result
   - Timestamp of generation

**Verification:**
- [ ] Inference completes on all 4 test samples without error
- [ ] Generated edges reasonable (100-500 per sample, order of magnitude correct)
- [ ] Submission CSV follows competition format (validated by validate_submission())
- [ ] No NaN/NULL values in CSV
- [ ] CSV saved to submissions/ directory with clear naming

---

### Task 4.5: Submit to Kaggle & Track Leaderboard Score
**Depends:** Task 4.4  
**Owner:** Executor  
**Autonomous:** true

Upload submission CSV to Kaggle competition and monitor leaderboard score.

**Execution Steps:**
1. Validate submission file before uploading:
   - Confirm file exists: submissions/submission_phase2_model_v1.csv
   - Check file size (should be ~1-5 KB for 4 samples)
   - Spot-check first 10 rows (format, values)

2. Submit via Kaggle API:
   ```bash
   py -m kaggle competitions submit -c biohub-cell-tracking-during-development -f submissions/submission_phase2_model_v1.csv -m "Phase 2: Learned Detection Model (UNet+Transformer, edge probs, greedy assignment)"
   ```

3. Monitor submission status:
   - Kaggle competition page → Submissions → [latest submission]
   - Expected: status "Running" → (1-5 min) → "Complete" with score displayed
   - If error: diagnose from Kaggle's error message (usually format/node ID issues)

4. Record leaderboard scores:
   - Public leaderboard score: X (displayed immediately)
   - Private leaderboard score: Y (revealed at competition end 2026-09-29)
   - Rank on public leaderboard

5. Log results to 04-SUMMARY.md:
   - Submission timestamp
   - Public leaderboard score + rank
   - Comparison to baseline (0.763)
   - Comparison to Phase 1 submission (0.0092, if submitted)

6. Archive submission:
   - Save submission CSV to .planning/phases/02-learned-detection/submissions/ (backup copy)
   - Save submission receipt/score screenshot (if possible)

**Verification:**
- [ ] Submission uploaded successfully (Kaggle status "Complete")
- [ ] Leaderboard score received and recorded
- [ ] Score is reasonable (not 0.000, not catastrophic regression)
- [ ] Submission metadata logged in 04-SUMMARY.md

---

### Task 4.6: Comprehensive Phase 2 Summary & Exit Criteria Verification
**Depends:** Task 4.5 (or Task 4.3 if no submission)  
**Owner:** Executor  
**Autonomous:** true

Finalize 04-SUMMARY.md and verify Phase 2 exit criteria achieved.

**Execution Steps:**
1. Create comprehensive 04-SUMMARY.md with sections:
   - **Full Training Results:**
     - Epochs run: N (actual)
     - Early-stop triggered? Y/N (after how many epochs)
     - Final validation loss/metrics
   - **Held-Out Evaluation:**
     - adjusted_edge_jaccard: Y (mean ± std across 50 validation samples)
     - edge_jaccard: X (mean ± std)
     - division_jaccard: Z (mean ± std)
     - division_recall: D (diagnostic)
     - Comparison to Phase 1 (0.0259) + baseline (0.763)
   - **Test Set Inference:**
     - Num edges predicted per sample
     - Inference time per sample
   - **Kaggle Submission (if applicable):**
     - Submission ID/timestamp
     - Public leaderboard score + rank
     - Private leaderboard score (TBD)
   - **Phase 2 Exit Criteria:** checklist with results

2. Verify Phase 2 exit criterion achievement:
   - **Exit Criterion (from ROADMAP.md):** "Local score materially exceeds Phase 1 (target: >=0.80+), achieved by training the model on real targets, replacing naive peak extraction and hardcoded motion vectors with real predictions, and successfully submitting to Kaggle leaderboard."
   - **Verification checklist:**
     - [ ] Model trained on real .geff targets? Yes (heatmap + edge targets from Tasks 2.3–2.4)
     - [ ] Naive peak extraction replaced? Yes (learned model inference in src/inference.py)
     - [ ] Motion vectors replaced with edge predictions? Yes (Transformer → greedy assignment, not hardcoded vectors)
     - [ ] Local score materially exceeds Phase 1 (0.0259)? Check: Y > 0.0259 + margin
     - [ ] Local score >=0.80+ (target) or >0.763 + margin (minimum)? Check: Y vs thresholds
     - [ ] Kaggle submission successful? Yes/No/Skipped

3. Log lessons learned:
   - What worked well in Wave 3/4
   - What could be improved for Phase 3
   - Hyperparameter tuning insights (if any)
   - Data quality observations

4. Recommend next phase (Phase 3):
   - If local score >=0.80+: Phase 3 can proceed as planned (ILP mitosis smoothing, edge probability refinement, etc.)
   - If local score <0.80 but >0.763: Phase 3 should focus on refinement before major architecture changes
   - If local score ≤0.763: Phase 3 should debug/revisit Phase 2 before advancing

**Verification:**
- [ ] 04-SUMMARY.md complete with all sections
- [ ] Phase 2 exit criteria checklist filled in with results
- [ ] Kaggle leaderboard score (public) recorded
- [ ] Recommendation for Phase 3 documented
- [ ] All artifacts (model, checkpoint, submission, logs) archived

---

## Verification Criteria (Phase 2 Exit)

All of the following must be satisfied:

- [ ] **Model training complete:** 50+ epochs or early-stopped after ≥10 with best checkpoint saved
- [ ] **Local validation score computed:** adjusted_edge_jaccard on 50 held-out samples, mean ± std
- [ ] **Score comparison documented:** vs. Phase 1 (0.0259), vs. baseline (0.763), vs. target (0.80+)
- [ ] **Submission decision made:** explicit go/no-go recorded with reasoning
- [ ] **If score >=0.80+ or >0.763+margin:** Kaggle submission successful with public leaderboard score recorded
- [ ] **Phase 2 exit criterion verified:** materiality of improvement established, real targets used, naive extraction replaced
- [ ] **All artifacts archived:** model checkpoint, training logs, submission CSV, 04-SUMMARY.md

---

## Output Artifacts
- `04-SUMMARY.md` — Full training results, evaluation scores, Kaggle submission details, Phase 2 exit verification
- `submissions/submission_phase2_model_v1.csv` — Test set predictions in competition format
- Best model checkpoint (retrieved from Kaggle, local copy)
- `training_full_log.csv` — Per-epoch metrics from full 50+ epoch run

