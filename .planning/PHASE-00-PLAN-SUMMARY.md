# Phase 0 Planning Summary

**Date:** 2026-07-03
**Status:** PLANNING COMPLETE ✓

---

## What Was Planned

Phase 0 (Unblock) aims to generate a **schema-valid, locally-scoreable submission** from real Kaggle competition data in a single end-to-end run. This is the foundation for all future phases.

### Phase 0 Exit Criterion
A submission CSV with:
- Exact schema: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
- Separate node/edge rows
- Per-dataset node_id resets (node_id=1 for each new dataset)
- Generated from real Zarr v3 stores (`data/staging/`)
- Scored locally using exact competition metric: `adjusted_edge_jaccard + 0.1 × division_jaccard`

---

## Execution Plan Structure

### 4 Executable Plans (in `.planning/phases/00-unblock/`)

| Plan | Title | Wave | Effort | Dependencies |
|------|-------|------|--------|--------------|
| **00** | Vendor Scoring Code | 1 | 1h | None |
| **01** | Fix Data Loader & Config | 1 | 2h | None |
| **02** | Build Eval Harness | 2 | 2h | Plan 00 |
| **03** | Implement Submission Exporter | 2 | 2h | None |
| **04** | Wire Pipeline & Test | 3 | 2-3h | Plans 00, 01, 02, 03 |

**Total estimated time:** 9-10 hours (can be parallelized)

---

## Key Decisions Embodied in the Plan

1. **Use the host's reference implementation** (Plan 00)
   - Vendor `metrics.py`, `division_metrics.py`, `io.py` from `github.com/royerlab/kaggle-cell-tracking-competition`
   - This guarantees local scoring matches Kaggle's official scorer (no reimplementation risk)
   - `tracking_cellmot` is NOT on PyPI — must be cloned/vendored locally

2. **Fix data loading for real Zarr v3 OME-NGFF stores** (Plan 01)
   - Update `AnisotropicZarrLoader` to read at array path `0/`
   - Apply quantile-based normalization (raw uint16 → [0, 1])
   - Fix anisotropy to `(4.0, 1.0, 1.0)` everywhere (was hardcoded wrong)
   - Remove simulated fallback (enforce real data)

3. **Separate concerns into reusable modules** (Plans 02, 03)
   - `evaluation.py`: scoring logic (depends on vendored metrics)
   - `submission_exporter.py`: CSV generation (independent)
   - Both can be tested in isolation before integration

4. **Orchestrate via `run_pipeline.py`** (Plan 04)
   - Iterate over ALL datasets in `data/staging/` (not hardcoded single path)
   - Load → Track → Export → Score for each dataset
   - Generate single submission CSV with global id sequencing + per-dataset node_id reset
   - Validate schema before declaring "submit-ready"

---

## Critical Gotchas Addressed

### CRITICAL DEPENDENCY (Plan 00)
- `tracking_cellmot` (the host's scoring code) is **not on PyPI**
- Must be cloned from `github.com/royerlab/kaggle-cell-tracking-competition`
- Must NOT attempt to `pip install tracking_cellmot` (will fail)
- Solution: Clone once, vendor 3 files into `src/tracking_cellmot/`

### Zarr v3 OME-NGFF Format (Plan 01)
- Real competition data is Zarr v3 (not v2 legacy)
- Array is stored at path `0/` within the store
- Requires `zarr.open_array()` (not `zarr.open_group()` v2 API)
- Must extract `image_statistics.quantiles` from zarr metadata for normalization

### Anisotropy Was Wrong (Plan 01)
- Code hardcoded `(5.0, 1.0, 1.0)` — should be `(4.0, 1.0, 1.0)`
- Plan 01 fixes all references (config, loader, tracker, etc.)

### Schema Subtlety (Plan 03)
- `id`: globally sequential (0, 1, 2, ...)
- `node_id`: **per-dataset-LOCAL** and **resets to 1 for each new dataset**
- This is easy to get wrong; Plan 03 enforces it via validator

### Sparse GT Annotations (Plans 02, 04)
- Competition data is sparse-annotated (not all regions have ground truth)
- Evaluation formula is designed to handle this (unmatched pred nodes don't count as FP)
- This is built into the vendored `metrics.py` — Plan 02 just uses it as-is

---

## Must-Haves for Phase 0 Completion

- [ ] All 4 plans executed (Tasks 00-01 through 04-06)
- [ ] All unit tests pass (11 test files across Plans 00-04)
- [ ] Submission CSV generated: `submissions/phase_0_baseline_submission.csv`
- [ ] CSV is schema-valid (passes `validate_submission()`)
- [ ] All 4 staged datasets processed: 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1
- [ ] Evaluation harness computes score and reports:
  - `adjusted_edge_jaccard` (edge matching via 7µm gating)
  - `division_jaccard` (division matching via real bipartite algorithm)
  - Combined `score = adjusted_edge_jaccard + 0.1 × division_jaccard`
- [ ] Score is numeric (not NaN, Inf, or error)

---

## Stretch Goals (Not Blocking Phase 0)

- Combined score > 0.763 (classical baseline)
  - Phase 0 uses placeholder detector (untrained), so score may be lower
  - Phases 1-2 will train the model and improve
- Pipeline runtime < 30 minutes (for later scaling)
- All tests pass with zero warnings

---

## Files to Be Created/Modified

### New Files (Created by Plans)
```
src/tracking_cellmot/__init__.py
src/tracking_cellmot/metrics.py (vendored)
src/tracking_cellmot/division_metrics.py (vendored)
src/tracking_cellmot/io.py (vendored)
src/evaluation.py (~200 lines)
src/submission_exporter.py (~200 lines)
tests/test_scoring_baseline.py
tests/test_data_loader_real.py
tests/test_evaluation_harness.py
tests/test_submission_exporter.py
tests/test_e2e_pipeline.py
submissions/phase_0_baseline_submission.csv (output)
```

### Modified Files
```
src/data_loader.py (refactor for Zarr v3, add quantile norm, update anisotropy)
config/hyperparams.yaml (fix anisotropy constant)
run_pipeline.py (refactor for multi-dataset, integrate exporter & eval harness)
```

---

## How to Execute Phase 0

### Prerequisites
1. Staged data exists at `data/staging/` (already confirmed, ~3.55 GB)
2. Dependencies installed: `pip install -r requirements.txt` (already done this session)
   - Includes: `geff`, `tracksdata`, `napari`, etc.
3. Python 3.9+ environment ready

### Execution (via `/gsd:execute-phase`)
```bash
/gsd:execute-phase 0
```

This will:
1. Read `.planning/phases/00-unblock/PLAN-*.md` files (all 4 plans)
2. Execute each plan's tasks in wave order (1 → 2 → 3)
3. Run unit tests for each task
4. Report results: pass/fail, time elapsed, blockers if any

### Manual Execution (if needed)
```bash
# Wave 1 (parallel)
python -m pytest tests/test_scoring_baseline.py  # Plan 00
python -m pytest tests/test_data_loader_real.py  # Plan 01

# Wave 2 (after Wave 1)
python -m pytest tests/test_evaluation_harness.py  # Plan 02
python -m pytest tests/test_submission_exporter.py  # Plan 03

# Wave 3 (after Wave 2)
python run_pipeline.py --data-dir data/staging/ --output-path submissions/phase_0_baseline_submission.csv
python -m pytest tests/test_e2e_pipeline.py  # Plan 04
```

---

## Success Looks Like

After Phase 0 execution:
1. No exceptions, all tests pass
2. File `submissions/phase_0_baseline_submission.csv` exists
3. CSV has ~150-200 rows (4 datasets × ~40-50 nodes+edges each)
4. Header: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
5. Sample rows:
   ```
   0,44b6_0113de3b,node,1,0,5,10,15,-1,-1
   1,44b6_0113de3b,node,2,0,6,20,30,-1,-1
   2,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2
   ...
   ```
6. Evaluation output:
   ```
   Edge Jaccard: 0.45
   Division Jaccard: 0.10
   Adjusted Edge Jaccard: 0.40
   Combined Score: 0.41
   Above baseline (0.763): No (expected — placeholder detector is weak)
   ```

---

## Handoff to Executor

**To the gsd-executor agent:**
- All 4 plans are ready in `.planning/phases/00-unblock/`
- Each plan has valid frontmatter (wave, depends_on, files_modified, autonomous)
- All tasks are specific with file paths and exact requirements
- Unit tests are defined inline (no separate test designs needed)
- Must-haves are derived from Phase 0 exit criterion, not generic restatements
- Proceed with Wave 1 (Plans 00 & 01 in parallel)

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| `tracksdata` API differs from docstring | Task 02-01 hangs/fails | Read live tracksdata code; confirmed working this session |
| Zarr v3 metadata layout unexpected | Task 01-01 fails | Inspect real `data/staging/44b6_0113de3b/.zattrs` before writing code |
| Quantile normalization formula wrong | Task 01-01 produces bad data | Reference `io.py` from host repo (included in REFERENCE_IMPLEMENTATION.md) |
| Submission schema differs from sample | Task 04-05 fails validation | Manual spot-check against sample_submission.csv is Task 04-05 |
| ILP solver hangs on large graph | Task 04-04 timeouts | Staged data is small (~52 nodes); if it hangs, check solver config. Phase 3 will address scaling |

---

## Phase 0 → Phase 1 Handoff

Once Phase 0 is complete, Phase 1 will:
1. Validate baseline: Does the pipeline work with placeholder detector?
2. Establish baseline score: What does an untrained model score?
3. Prove embryo-disjoint split: Can we score on held-out embryos?
4. Document any schema issues found: Edge cases to handle in Phase 2+

Phase 1 does NOT require changes to Plans 00-04; it only validates them.

---

**Created by:** GSD Planner (gsd-planner agent)
**Date:** 2026-07-03
**Status:** READY FOR EXECUTION

