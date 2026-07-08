---
wave: 1
depends_on: []
files_modified:
  - src/data_loader.py (conditional: if normalization switch needed)
  - src/dataset.py (new)
  - .planning/phases/02-learned-detection/01-SUMMARY.md (output)
autonomous: true
---

# Phase 2 Wave 1: Infrastructure & Decisions

## Summary
Wave 1 establishes the data pipeline and resolves two critical decision points from RESEARCH.md:
1. **Normalization choice** (RESEARCH.md S3): keep existing [0,1]-clipped zarr-quantile vs. switch to host's [0,4.0]-clipped self-computed quantile — empirically benchmark and document the choice before Wave 2 proceeds.
2. **Data split and infrastructure**: enumerate the full 199-sample competition dataset directly from the zip (no local extraction, per CONTEXT.md's locked decision), build embryo-disjoint train/val split (per-sample granularity, not movie-prefix level), implement PyTorch Dataset class.

Exit criterion: normalization approach locked in with verification, embryo-disjoint split validated against the real enumerated 199-sample set, Dataset class tested on the locally-available real samples.

## Must-Haves (Goal-Backward Verification)
- [ ] Normalization decision documented in 01-SUMMARY.md with explicit reasoning and empirical benchmark results
- [ ] Data split correctly partitions 199 samples (enumerated from the zip, not extracted) into ~149 train / ~50 held-out, stratified by movie prefix (44b6/6bba), never mixing within a sample
- [ ] PyTorch Dataset loads real Zarr v3 volumes + .geff ground truth correctly
- [ ] Dataset produces (frame_t, frame_t+1) pairs with anisotropic physical metadata
- [ ] Locally-available real samples (~9: 4 staged + up to 5 spot-checked) successfully loaded and inspected; full 199-sample I/O validation explicitly deferred to Wave 3's Kaggle sanity-check run, not claimed here

---

## Tasks

### Task 1.1: Benchmark & Decide Data Normalization Approach
**Depends:** (none — startup task)  
**Owner:** Planner/Executor  
**Autonomous:** true

Resolve RESEARCH.md S3's explicit decision point:

**CRITICAL: This must be empirically benchmarked, not guessed.** Phase 1's peak-finding thresholds are calibrated for [0,1]-clipped zarr-quantile normalization. Switching to host's [0,4.0]-clipped self-computed approach would invalidate Phase 1's verified thresholds and require recalibration.

**Option A (RECOMMENDED for Phase 2):** Keep existing [0,1]-clipped zarr-quantile normalization (already in src/data_loader.py:95-119, 229-252). Rationale: Phase 1 peak-finding is already verified working at score 0.0259 with this normalization. Model heatmap targets and detection thresholds tune against [0,1] logits.

**Option B (if benchmarking shows clear benefit):** Switch to host's [0,4.0]-clipped self-computed quantile (q_min=0.001, q_max=0.999 computed per-sample). Requires: (1) modify AnisotropicZarrLoader quantile computation logic, (2) rerun peak-finding threshold calibration via sweep_threshold.py locally on a few samples, (3) document new thresholds in run_pipeline.py.

**Execution Steps:**
1. Read src/data_loader.py lines 95-119 and 229-252 to confirm current [0,1] implementation
2. Load 2-3 real competition samples (one from 44b6, one from 6bba) via both normalization approaches
3. Measure histogram statistics (mean, std, percentiles) for each approach on a few sample volumes
4. Document findings: which approach produces clearer separation of foreground (cells) vs background (noise)?
5. Make explicit decision: Option A or Option B
6. If Option B chosen: schedule a post-Wave1 threshold recalibration task (add to 02-PLAN.md's Wave 2)
7. Document decision + reasoning in 01-SUMMARY.md

**Verification:**
- [ ] 01-SUMMARY.md clearly states normalization approach chosen + empirical justification
- [ ] If Option B: threshold recalibration task added to 02-PLAN.md as a dependency for training

---

### Task 1.2: Enumerate & Verify Full Competition Dataset (no local extraction)
**Depends:** Task 1.1  
**Owner:** Executor  
**Autonomous:** true

**CORRECTED per plan verification:** the original draft of this task planned a full local
extraction of all 199 samples (~80GB) to local disk. This contradicts `02-CONTEXT.md`'s
locked "Training scale" decision: *"local staging stays as-is for harness/dev work...
Kaggle training sessions read from Kaggle's own mounted copy"* -- no local extraction was
ever asked for, and it would waste significant local disk (~80GB) and time (likely 1-2+
hours) for no actual downstream need (Wave 3's Kaggle sanity-check run is what genuinely
exercises the full 199-sample set, against Kaggle's already-working mount). This task is
corrected to **enumerate and spot-check the full train set directly from the zip, without
extracting it** -- matching this project's own established pattern (already used
successfully earlier this session via Python's `zipfile.ZipFile`, and originally by
`scripts/stage_subset.py` from Phase 0).

**Execution Steps:**
1. Verify zip file exists and is readable (`C:\Users\hemas\Downloads\biohub-cell-tracking-during-development.zip`, 81.6GB)
2. Using `zipfile.ZipFile` (no extraction), enumerate all `train/*.geff/` entries to confirm
   the real sample count and prefix distribution (expected: 199 total, `44b6`: 71, `6bba`: 128 --
   already verified once this session, confirm it still holds)
3. Spot-check 5 samples across both movie prefixes (44b6 and 6bba) by extracting *only
   those 5 samples'* `.zarr`/`.geff` entries to a temp location (not the full set): confirm
   Zarr v3 format, metadata shape (100,64,256,256), `.geff` parseable
4. Log enumeration stats to 01-SUMMARY.md: total samples, prefix distribution, confirmation
   that no full local extraction was performed (by design)

**Verification:**
- [ ] 199 train sample IDs enumerated directly from the zip (no full extraction)
- [ ] Prefix distribution confirmed: 44b6=71, 6bba=128
- [ ] 5 spot-checked samples (extracted individually, not the full set) confirm Zarr v3 format, shape (100,64,256,256) uint16, parseable .geff (via `tracksdata.graph.IndexedRXGraph.from_geff()`)
- [ ] Enumeration stats logged in 01-SUMMARY.md; local disk usage stays at the existing 4-staged-sample footprint, not +80GB

---

### Task 1.3: Build Embryo-Disjoint Train/Val Split
**Depends:** Task 1.2  
**Owner:** Executor  
**Autonomous:** true

Partition 199 individual samples into ~149 train / ~50 held-out, respecting embryo-disjoint constraint (per-sample granularity, not movie-prefix level — see RESEARCH.md S2.3 for corrected definition).

**CRITICAL:** Per RESEARCH.md S2.3 and CONTEXT.md, "embryo" means each individual sample ID (e.g., 44b6_0113de3b), NOT the 2-value movie prefix (44b6 or 6bba). Train and test sets both draw from both prefixes, so prefix-level split is meaningless. Split must be at sample level.

**Execution Steps:**
1. Enumerate all 199 train sample IDs
2. Partition randomly into 2 sets:
   - Train: ~149 samples (75%)
   - Held-out validation: ~50 samples (25%)
3. Stratify split to preserve movie prefix distribution in both sets:
   - 44b6: 71 total → ~53 train, ~18 held-out
   - 6bba: 128 total → ~96 train, ~32 held-out
4. Write split to a JSON file (e.g., data_split.json) with clear structure:
   ```json
   {
     "train": ["44b6_0113de3b", "6bba_05b6850b", ...],
     "validation": ["44b6_0b24845f", "6bba_05db0fb1", ...],
     "metadata": {
       "total_samples": 199,
       "train_count": 149,
       "validation_count": 50,
       "44b6_train": 53,
       "44b6_validation": 18,
       "6bba_train": 96,
       "6bba_validation": 32,
       "seed": 42
     }
   }
   ```
5. Verify no overlap between train/validation sets
6. Log split summary to 01-SUMMARY.md

**Verification:**
- [ ] data_split.json exists with 149 train + 50 validation = 199 total
- [ ] No sample appears in both train and validation
- [ ] Both train and validation contain samples from both 44b6 and 6bba prefixes
- [ ] Stratification roughly preserves prefix proportions (18/50 ≈ 36% from 44b6 in validation, matching 71/199 ≈ 36% global)

---

### Task 1.4: Implement PyTorch Dataset Class
**Depends:** Task 1.3  
**Owner:** Executor  
**Autonomous:** true

Create src/dataset.py with a PyTorch Dataset class that:
- Loads Zarr v3 volumes + .geff ground truth (works against either a local directory of
  staged samples or the Kaggle-mounted competition path -- `data_dir` is a parameter, not
  hardcoded, so the same class runs in both contexts)
- Produces (frame_t, frame_t+1) pairs with anisotropic metadata
- Respects the embryo-disjoint train/val split from Task 1.3
- Prepares infrastructure for target generation (Tasks 2.1–2.3 will add target computation)

**Scope correction (per Task 1.2's fix above):** since no local extraction of all 199
samples happens in this wave, this task tests the Dataset class against what's actually
available locally (the 4 originally-staged samples + the up-to-5 spot-checked samples from
Task 1.2 -- so up to ~9 real samples), NOT all 199. **Genuine validation against the full
199-sample set happens in Wave 3's Kaggle sanity-check run**, where the full competition
data is actually mounted -- that is this project's existing, correct place for that check,
not a duplicate one here.

**Execution Steps:**
1. Create src/dataset.py with class CompetitionDataset(torch.utils.data.Dataset):
   - `__init__(data_dir, split_file, split_type='train', normalize=True, anisotropy=(4.0, 1.0, 1.0))`
   - `__len__()` → number of valid (t, t+1) pairs in the split
   - `__getitem__(idx)` → (frame_t, frame_t+1, sample_id, timepoint_idx, metadata)
   - Metadata dict: `{sample_id, t_idx, volume_shape, physical_voxel_size, anisotropy_ratio}`
   
2. Integrate with existing AnisotropicZarrLoader:
   - Reuse src/data_loader.py's zarr loading + normalization logic
   - Confirm anisotropic metadata is correctly propagated (physical voxel size (1.625, 0.40625, 0.40625) um)

3. Add data augmentation hooks (not yet active, but scaffolded for Wave 3):
   - Optional elastic deformation (respect anisotropy)
   - Optional Y/X rotation (not Z)
   - Optional intensity jitter
   - Optional patch dropout

4. Test on real competition data (locally available samples only, ~9, not all 199 -- see
   scope correction above):
   - Load all locally-available samples (4 originally staged + up to 5 from Task 1.2's
     spot-check) to verify no I/O errors
   - Inspect shapes and dtypes (frame_t, frame_t+1 should be uint16, shape (1,64,256,256) or (64,256,256))
   - Verify split filtering works correctly against data_split.json (even though most of
     the 199 sample IDs it references aren't locally present yet -- the filtering logic
     itself is what's under test here, not full-set I/O)

5. Log dataset stats to 01-SUMMARY.md:
   - Total (t, t+1) pairs available in the locally-tested subset
   - Confirmation that split-filtering logic correctly includes/excludes samples per
     data_split.json's train/validation lists
   - Explicit note that full 199-sample I/O validation is deferred to Wave 3's Kaggle
     sanity-check run, not claimed as verified here

**Verification:**
- [ ] src/dataset.py exists with CompetitionDataset class
- [ ] __len__() and __getitem__() work without errors
- [ ] All locally-available real samples (~9) load successfully (no Zarr/geff parse errors)
- [ ] Split-filtering logic verified correct against data_split.json (not full-199 I/O, which is Wave 3's job)
- [ ] Metadata includes physical voxel size and anisotropy_ratio
- [ ] Shape verification: frames are (1, 64, 256, 256) or (64, 256, 256) uint16
- [ ] Dataset stats logged in 01-SUMMARY.md

---

## Verification Criteria

All tasks must pass before proceeding to Wave 2:

- [ ] **Normalization decision locked:** 01-SUMMARY.md contains explicit choice (Option A or B) + empirical justification
- [ ] **Data enumerated & spot-checked (not extracted):** 199 samples enumerated directly from the zip, 5 spot-checked individually for format/parsability -- no full local extraction (~80GB avoided per CONTEXT.md's locked decision)
- [ ] **Split created & validated:** data_split.json exists, 149 train + 50 validation with no overlap, stratified by prefix
- [ ] **Dataset class implemented & tested:** CompetitionDataset loads all locally-available real samples (~9) without error, respects split filtering, produces correct shapes and metadata; full 199-sample I/O validation explicitly deferred to Wave 3
- [ ] **No blocking issues:** All I/O, zarr, .geff parsing working end-to-end on the locally-available real competition data tested in this wave

---

## Output Artifacts
- `01-SUMMARY.md` — Normalization decision + empirical justification, enumeration stats, split summary, dataset test results
- `data_split.json` — Train/validation partition (149/50), stratified by movie prefix
- `src/dataset.py` — PyTorch Dataset class for the full competition set (works against local samples now, Kaggle-mounted full set from Wave 3 onward)

