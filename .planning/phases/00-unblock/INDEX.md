# Phase 0: Unblock — Execution Plan Index

**Phase Goal:** Generate a schema-valid submission end-to-end from real competition data, scored locally.

**Phase 0 Exit Criterion:** A submission CSV with correct schema (`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`), separate node/edge rows, generated from real Zarr stores and ground-truth `.geff` annotations, scored locally using the exact competition metric (edge Jaccard + 0.1 × division Jaccard).

**Completion Target:** All 4 plans executed successfully, submission CSV generated and scored.

---

## Plan Overview

| Plan ID | Title | Wave | Dependencies | Key Deliverable |
|---------|-------|------|--------------|-----------------|
| 00 | Vendor Scoring Code | 1 | None | `src/tracking_cellmot/` with metrics.py, division_metrics.py, io.py |
| 01 | Fix Data Loader & Config | 1 | None | Updated `AnisotropicZarrLoader` for Zarr v3, anisotropy (4.0, 1.0, 1.0) |
| 02 | Build Eval Harness | 2 | 00 | `src/evaluation.py` with `evaluate_submission()` |
| 03 | Implement Submission Exporter | 2 | None | `src/submission_exporter.py` with `export_submission()` and `validate_submission()` |
| 04 | Wire Pipeline & Test | 3 | 00, 01, 02, 03 | `submissions/phase_0_baseline_submission.csv` with computed scores |

---

## Execution Strategy

### Wave 1 (Parallel)
- **PLAN-00**: Vendor the reference implementation (copy 3 Python files)
- **PLAN-01**: Update data loader and config (refactor loader, fix anisotropy constant)

Both are independent; execute in parallel for speed.

### Wave 2 (Parallel)
- **PLAN-02**: Build evaluation harness (requires Plan 00 complete)
- **PLAN-03**: Implement submission exporter (standalone, no dependencies)

Both can run in parallel once Wave 1 is done.

### Wave 3 (Sequential)
- **PLAN-04**: Wire everything together and run end-to-end test (requires all prior plans complete)

Orchestrates Plans 01-03; validates the complete pipeline.

---

## Key Requirements Mapped to Plans

| Requirement | Plan | Task | Status |
|-------------|------|------|--------|
| DATA-01: Read real Zarr v3 stores | 01 | 01-01 | Pending |
| DATA-02: No silent simulated-data fallback | 01 | 01-01 | Pending |
| DATA-03: Anisotropy (4.0, 1.0, 1.0) | 01 | 01-02 | Pending |
| DATA-04: Load .geff via tracksdata | 02 | 02-02 | Pending |
| DATA-05: Iterate all datasets at inference time | 04 | 04-01 | Pending |
| DATA-06: Quantile-based normalization | 01 | 01-01 | Pending |
| SUB-01: Correct schema with per-dataset node_id reset | 03 | 03-01 | Pending |
| SUB-02: One dataset block per folder | 04 | 04-01 | Pending |
| SUB-03: Validation before submit-ready | 03 | 03-02 | Pending |
| EVAL-01: Edge Jaccard via tracksdata (7µm matching) | 02 | 02-01 | Pending |
| EVAL-02: Division Jaccard via vendored code (real algorithm) | 02 | 02-01 | Pending |
| EVAL-03: Adjusted score formula with adjustment alpha | 02 | 02-01 | Pending |
| EVAL-04: traccuracy as secondary check only | 02 | 02-03 | Pending |

---

## Resource Requirements

- **Time:** ~2-3 hours per plan (4-6 hours Wave 1, 4-6 hours Wave 2, 2-3 hours Wave 3)
- **Compute:** Local Python environment (GPU not required for Phase 0 — using placeholder detector)
- **Disk:** ~10 GB (data/staging/ + generated submissions)
- **Network:** Internet access for `pip install` and git clone (first run only)

---

## Testing and Validation

### Per-Plan Testing
- Each plan includes 3-4 unit tests (files in `tests/` directory)
- Tests use real staged data where applicable
- All tests must pass before the plan is considered complete

### Integration Testing (Plan 04)
- Full pipeline execution on all 4 staged datasets
- Schema validation of generated CSV
- Evaluation harness scores the submission
- Spot-check against competition reference format

---

## Success Criteria (Go/No-Go)

### Hard Requirements (Must Have)
- [ ] All 4 plans complete with all tasks passing
- [ ] Submission CSV is generated
- [ ] CSV schema matches competition exactly
- [ ] All 4 datasets are processed (44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1)
- [ ] Evaluation harness scores the submission (combined score is numeric)
- [ ] No exceptions or critical errors

### Stretch Goals (Nice-to-Have)
- [ ] Combined score > 0.763 (classical baseline)
- [ ] Pipeline runtime < 30 minutes (for later scaling)
- [ ] All tests pass with no warnings

---

## Decision Points & Known Issues

### Known Dependency Gotchas
1. **`tracking_cellmot` not on PyPI** (Plan 00): Must be cloned/vendored from github.com/royerlab/kaggle-cell-tracking-competition — cannot `pip install`
2. **`tracksdata` is on PyPI** (Plan 02): Generic graph library, works out-of-the-box
3. **Zarr v3 API** (Plan 01): Must use zarr.open_array() (not v2 legacy), array at path "0/"

### Phase 0 Constraints
- Placeholder detector is untrained; score may be low (Phase 1/2 will improve this)
- Real competition data is sparse-annotated (ground truth not complete); evaluation handles this
- Test set is held-out by embryo; no data leakage

### Deferred (Not in Phase 0)
- ILP solver tuning (Phase 3)
- Model training (Phase 2)
- Advanced calibration (Phase 4-5)

---

## Files Created/Modified Summary

### New Files
- `src/tracking_cellmot/__init__.py`
- `src/tracking_cellmot/metrics.py` (vendored)
- `src/tracking_cellmot/division_metrics.py` (vendored)
- `src/tracking_cellmot/io.py` (vendored)
- `src/evaluation.py` (new, ~200 lines)
- `src/submission_exporter.py` (new, ~200 lines)
- `tests/test_scoring_baseline.py` (new, ~30 lines)
- `tests/test_data_loader_real.py` (new, ~40 lines)
- `tests/test_evaluation_harness.py` (new, ~100 lines)
- `tests/test_submission_exporter.py` (new, ~150 lines)
- `tests/test_e2e_pipeline.py` (new, ~50 lines)

### Modified Files
- `src/data_loader.py` (refactor AnisotropicZarrLoader, add Zarr v3 support, quantile norm)
- `config/hyperparams.yaml` (fix anisotropy constant)
- `run_pipeline.py` (refactor for multi-dataset, integrate exporter & eval harness)

### Output Files (Generated)
- `submissions/phase_0_baseline_submission.csv` (main deliverable)
- Optional: score report, timing log, validation report

---

## Next Steps After Phase 0

Once Phase 0 is complete:
1. **Phase 1** (Baseline Parity): Validate that the pipeline works with the placeholder detector; establish baseline score
2. **Phase 2** (Model Training): Build and train the detection model (U-Net or MONAI-based)
3. **Phase 3** (Tracker Scaling): Optimize ILP solver; handle real-scale data (1000s of cells)
4. **Phase 4-5** (Competitive Iteration): Grid-search tracker costs, ensembling, etc.

---

## Contact & Handoff

**Phase 0 Executor:** Claude Code (GSD gsd-executor agent)
**Phase 0 Planner:** Claude Code (GSD gsd-planner agent — this document)
**Handoff to:** gsd-executor for task implementation

---

**Created:** 2026-07-03
**Last Updated:** 2026-07-03
**Status:** PLANNING COMPLETE — Ready for execution
