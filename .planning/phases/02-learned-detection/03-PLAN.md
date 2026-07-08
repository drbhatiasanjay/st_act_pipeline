---
wave: 3
depends_on:
  - 02-PLAN.md (Wave 2 complete)
files_modified:
  - src/train.py (new)
  - kaggle_kernel/train_kernel.py (new/updated)
  - .planning/phases/02-learned-detection/03-SUMMARY.md (output)
autonomous: false
---

# Phase 2 Wave 3: Training & Validation

## Summary
Wave 3 builds the training loop, validates on held-out data, and runs a critical sanity-check training job on the full 199-sample set (few epochs, no full commitment). Per CONTEXT.md's locked decision, run a short sanity-check first before committing to a long full training run.

**Workflow:**
1. **Local smoke test** (on staged samples): Verify training loop syntax/loss computation works end-to-end (quick, ~5 min)
2. **Kaggle sanity-check run** (full 199 samples, few epochs): Verify data loading, training, and validation on full competition scale (realistic setup, but limited epochs to catch data/config errors early, ~30-60 min)
3. **Checkpoint after sanity-check:** Retrieve model checkpoint, evaluate locally via src/evaluation.py
4. **User check-in:** Report sanity-check results (loss curves, validation score) before proceeding to full training

Exit criterion: Sanity-check training runs to completion without errors, loss curves reasonable, validation metrics logged. Full training deferred to Wave 4 after user sign-off.

## Must-Haves (Goal-Backward Verification)
- [ ] Training loop implemented with proper loss computation (edge + detection)
- [ ] Validation performed every epoch on held-out set, metrics logged (edge_jaccard, adjusted_edge_jaccard, division_jaccard, division_recall)
- [ ] Early stopping configured (patience ~10 epochs, monitor on adjusted_edge_jaccard)
- [ ] Data loader correctly produces (frame_t, frame_t+1, heatmap_targets, edge_targets, metadata) from src/dataset.py
- [ ] Kaggle sanity-check run completes on full 199-sample set with ≥3 epochs
- [ ] Loss curves and validation metrics retrieved and visualized
- [ ] Model checkpoint saved after sanity-check and retrievable for local evaluation

---

## Tasks

### Task 3.1: Implement Training Loop
**Depends:** Wave 2 complete (model, targets, dataset)  
**Owner:** Executor  
**Autonomous:** true

Build src/train.py with end-to-end training loop: data loading, loss computation, backprop, validation, logging, early stopping, checkpointing.

**Execution Steps:**
1. Create src/train.py with class TrainingLoop:
   - Constructor: model, optimizer, scheduler, device, hyperparams
   - Hyperparams: learning_rate=1e-4, grad_clip=1.0, batch_size=16, weight_decay (standard for AdamW), heatmap_loss_weight=1.0, division_loss_weight=2.5
   - Allow all hyperparams to be configurable (for tuning in Wave 4)

2. Implement forward pass:
   - Load (frame_t, frame_t+1) batch from src/dataset.py
   - Normalize frames (apply selected normalization from Wave 1 Task 1.1)
   - Concatenate frames into (B, 1, 2, 64, 256, 256) input
   - Forward through UNet3D → detection logits (B, 1, 64, 256, 256)
   - Extract nodes from detection logits (>threshold, NMS)
   - Forward through SimpleNodeTransformer → edge probabilities
   - Compute losses:
     - Detection loss: BCE with inverse-frequency weighting (per-sample computed from GT heatmap targets)
     - Edge loss: focal-weighted BCE with division weighting (per-sample computed from GT edge targets)
     - Total loss = edge_loss + heatmap_loss_weight * heatmap_loss

3. Implement backward pass:
   - zero_grad()
   - backward()
   - torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip=1.0)
   - optimizer.step()

4. Implement validation loop:
   - Per-epoch validation on held-out set
   - Compute edge predictions via full inference (UNet + Transformer + greedy assignment)
   - Evaluate against GT via src/evaluation.py: edge_jaccard, adjusted_edge_jaccard, division_jaccard
   - Also compute division_recall (pulled forward from Phase 4, observational)
   - Log all metrics to file (CSV format for plotting)

5. Implement early stopping:
   - Monitor adjusted_edge_jaccard on validation set
   - Early stop if no improvement for patience=10 epochs
   - Save best checkpoint when validation score improves

6. Implement checkpointing:
   - Save after every epoch: model weights + optimizer state + epoch number + validation scores
   - Filename: checkpoints/epoch_{N}_val_score_{SCORE:.4f}.pt
   - Delete old checkpoints to save space (keep last 3)

7. Implement logging:
   - Per-epoch: epoch, train_loss, val_edge_jaccard, val_adjusted, val_division_jaccard, division_recall
   - Save to CSV (training_log.csv) and console output
   - Include timestamp and hyperparams at start of log

8. Test loop on staged samples:
   - Load 4 staged samples, run 1 epoch of training + 1 epoch validation
   - Verify shapes through pipeline, loss computation, backward pass
   - Check that validation metrics are non-NaN
   - Ensure checkpoint is created and loadable

**Verification:**
- [ ] src/train.py exists with TrainingLoop class
- [ ] Forward pass produces detection logits + edge probabilities without shape mismatches
- [ ] Loss computation (edge + detection) produces scalar loss
- [ ] Backward pass and gradient clipping work
- [ ] Validation loop computes edge_jaccard + adjusted + division_jaccard + division_recall
- [ ] Early stopping configured with patience=10
- [ ] Checkpointing saves model weights + optimizer + scores
- [ ] Smoke test on staged samples completes without error
- [ ] Validation metrics are non-NaN on staged data

---

### Task 3.2: Prepare Kaggle Training Script
**Depends:** Task 3.1  
**Owner:** Executor  
**Autonomous:** true

Create kaggle_kernel/train_kernel.py: a self-contained training script for Kaggle that:
- Loads full 199-sample competition data from /kaggle/input
- Runs training loop from src/train.py
- Saves checkpoint to /kaggle/working for retrieval
- Logs loss curves and validation metrics

**Execution Steps:**
1. Create kaggle_kernel/train_kernel.py with structure:
   ```python
   # 1. Environment setup (imports, GPU check)
   # 2. Hyperparams config (batch_size, epochs, device, paths)
   # 3. Data loading (use /kaggle/input path)
   # 4. Model init (UNet3D + Transformer, load pretrained if available)
   # 5. Training loop call
   # 6. Checkpoint + log retrieval
   ```

2. Configure for Kaggle environment:
   - Data path: /kaggle/input/competitions/biohub-cell-tracking-during-development/
   - Working path: /kaggle/working/ (for checkpoints, logs)
   - Output path: /kaggle/output/ (for submission, if running full pipeline)
   - GPU detection: torch.cuda.is_available(), print device info
   - Seed for reproducibility: random seed + torch.manual_seed + torch.cuda.manual_seed

3. Sanity-check mode (for Wave 3):
   - Add config parameter epochs_for_sanity_check=3 (or 5)
   - Log should show "SANITY CHECK MODE" at start
   - Save checkpoint to /kaggle/working/sanity_checkpoint.pt
   - Save log to /kaggle/working/sanity_training_log.csv

4. Output final status:
   - Print final validation score + metrics
   - Save model summary (param counts) to /kaggle/working/model_summary.txt

5. Error handling:
   - Catch OOM errors and log gracefully (don't leave Kaggle job hanging)
   - Save partial checkpoint if training interrupted

6. Test script locally:
   - Run on staged samples (small subset) to verify no import/syntax errors
   - Simulate Kaggle paths locally for debugging

**Verification:**
- [ ] kaggle_kernel/train_kernel.py exists and is syntax-error free
- [ ] Script correctly detects GPU on Kaggle (should print device info)
- [ ] Data paths point to /kaggle/input correctly
- [ ] Sanity-check mode flag included + documented
- [ ] Checkpoints save to /kaggle/working/
- [ ] Logs save to /kaggle/working/sanity_training_log.csv
- [ ] Local test run on staged samples completes without error

---

### Task 3.3: Run Kaggle Sanity-Check Training (Few Epochs, Full Data)
**Depends:** Task 3.2  
**Owner:** User (async Kaggle job)  
**Autonomous:** false

Submit and run sanity-check training on Kaggle with full 199-sample train set but limited to 3-5 epochs. This is the critical "early error detection" checkpoint before committing to 50+ epoch full training.

**Execution Steps:**
1. Update kaggle_kernel/kernel-metadata.json:
   - Set kernel as private (for iteration)
   - Ensure GPU T4 x2 accelerator selected (via website UI, not CLI)

2. Push script to Kaggle:
   ```bash
   cd C:\Users\hemas\Downloads\st_act_pipeline
   py -m kaggle kernels push -p kaggle_kernel
   ```
   (Use `py -m kaggle` not bare `kaggle` command — confirmed working in prior sessions)

3. Monitor via Kaggle website:
   - Check kernel status (running/completed/error)
   - Expected runtime: 30-60 min for 3 epochs on 199 samples + GPU

4. Upon completion:
   - Download outputs via (kernel ref is positional, `-p` is the output folder --
     verified against real usage this session, e.g.
     `py -m kaggle kernels output drbhatiasanjay/st-act-gpu-smoke-test -p kaggle_kernel_output`):
     ```bash
     py -m kaggle kernels output <owner>/<kernel-slug> -p ./kaggle_sanity_outputs/
     ```
   - Retrieve sanity_checkpoint.pt and sanity_training_log.csv

5. If training fails:
   - Diagnose error from Kaggle logs (usually OOM, data path issues, or import errors)
   - Fix in src/train.py or kaggle_kernel/train_kernel.py
   - Resubmit (re-iterate on this task until passing)

**Verification:**
- [ ] Kaggle kernel pushed successfully (status "Running" or "Completed")
- [ ] Training runs for ≥3 epochs without errors
- [ ] Checkpoint saved to /kaggle/working/sanity_checkpoint.pt
- [ ] Loss and validation logs saved to /kaggle/working/sanity_training_log.csv
- [ ] Outputs downloaded locally for evaluation

---

### Task 3.4: Retrieve & Evaluate Sanity-Check Checkpoint Locally
**Depends:** Task 3.3  
**Owner:** Executor  
**Autonomous:** true

Download sanity-check outputs from Kaggle and evaluate checkpoint locally via src/evaluation.py.

**Execution Steps:**
1. Ensure kaggle CLI configured (from prior Phase work):
   ```bash
   py -m kaggle kernels output <owner>/<kernel-slug> -p ./kaggle_sanity_outputs/
   ```

2. Load sanity_training_log.csv:
   - Parse loss curves: plot train_loss, val_edge_jaccard, val_adjusted_jaccard per epoch
   - Check that loss is decreasing (not flat or increasing, which indicates a config bug)
   - Check that validation metrics improve (or at least non-NaN)
   - Log observations to 03-SUMMARY.md

3. Load sanity_checkpoint.pt:
   - Verify checkpoint structure (model weights, optimizer state, epoch)
   - Load model and run inference on 5 held-out competition samples
   - Use src/evaluation.py to compute local score on these 5 samples
   - Log results (mean edge_jaccard, adjusted, division_jaccard) to 03-SUMMARY.md

4. Comparison to Phase 1 baseline:
   - Phase 1's peak-finding score: 0.0259 (naive grid scan + NMS)
   - Sanity-check expected score: should be higher than 0.0259 if learning is happening (even just a few epochs of training should improve over naive peak extraction)
   - If score is ≤Phase 1's score, investigate: is model actually learning? Are targets correct? Is normalization mismatched?

5. Visualizations (optional but useful):
   - Plot loss curves (train_loss and val metrics vs epoch)
   - Save plots to .planning/phases/02-learned-detection/sanity_check_plots/

**Verification:**
- [ ] sanity_training_log.csv downloaded and parsed
- [ ] Loss curves show decreasing trend (not flat/increasing)
- [ ] Validation metrics are non-NaN across all epochs
- [ ] Sanity_checkpoint.pt loaded successfully
- [ ] Local evaluation on 5 held-out samples completes
- [ ] Sanity-check score >Phase 1 baseline (0.0259) OR investigation documented if not
- [ ] All results logged in 03-SUMMARY.md

---

### Task 3.5: Report Sanity-Check Results & Decide on Full Training
**Depends:** Task 3.4  
**Owner:** Executor (reporting), User (decision)  
**Autonomous:** false

Prepare comprehensive sanity-check report for user review. Decision gate: proceed to full training (Wave 4) only if sanity-check passes.

**Execution Steps:**
1. Create 03-SUMMARY.md with sections:
   - **Setup Verification:** Data loading, model architecture, hyperparams used
   - **Training Curve:** Loss plots, validation metrics per epoch
   - **Sanity-Check Score:** Local evaluation on 5 held-out samples, comparison to Phase 1 baseline (0.0259)
   - **Issues Found (if any):** Data anomalies, training instability, etc.
   - **Recommendation:** Proceed to full training Y/N, and any config adjustments needed

2. Include explicit metrics:
   - Epoch 1 val_adjusted_jaccard: X
   - Epoch N val_adjusted_jaccard: Y
   - Improvement per epoch: (Y-X)/(N-1)
   - Sanity-check local score: Z (vs baseline 0.0259)

3. Highlight decision criteria:
   - Loss decreasing? Y/N
   - Validation metrics non-NaN? Y/N
   - Score materially better than Phase 1? Y/N (allow smaller improvement if data/model setup is clearly correct)
   - No blocking errors (OOM, data corruption, etc.)? Y/N

4. User check-in prompt:
   - Present 03-SUMMARY.md
   - Explicit question: "Proceed to full training run with 50 epochs?" (decision required before Wave 4 proceeds)

**Verification:**
- [ ] 03-SUMMARY.md created with all required sections
- [ ] Loss curves and validation metrics clearly presented
- [ ] Sanity-check score calculated and compared to baseline
- [ ] Clear recommendation (proceed Y/N)
- [ ] User consulted and decision recorded (in CONTEXT.md or comments)

---

## Verification Criteria

All tasks must complete before Wave 4:

- [ ] **Training loop implemented:** Forward/backward, loss computation, validation, early stopping, checkpointing all working
- [ ] **Kaggle script prepared:** Script syntax-checked, data paths verified, GPU detection working
- [ ] **Sanity-check run submitted & completed:** Kaggle job finished with checkpoint + logs
- [ ] **Sanity-check evaluated locally:** Loss curves reasonable, validation metrics computed, comparison to baseline documented
- [ ] **User decision obtained:** Explicit go/no-go for full training recorded

---

## Output Artifacts
- `src/train.py` — TrainingLoop class with loss computation, validation, checkpointing
- `kaggle_kernel/train_kernel.py` — Kaggle submission script
- `03-SUMMARY.md` — Sanity-check report with loss curves, metrics, recommendation
- `sanity_checkpoint.pt` — Model checkpoint after sanity-check training
- `sanity_training_log.csv` — Per-epoch loss and validation metrics

