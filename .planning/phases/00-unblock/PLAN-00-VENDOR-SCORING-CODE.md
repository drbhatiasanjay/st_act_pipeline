---
wave: 1
depends_on: []
files_modified:
  - src/tracking_cellmot/__init__.py
  - src/tracking_cellmot/metrics.py
  - src/tracking_cellmot/division_metrics.py
  - src/tracking_cellmot/io.py
autonomous: true
---

# Phase 0, Plan 00: Vendor Scoring Code

**Goal:** Obtain and integrate the host's own reference implementation of `metrics.py`, `division_metrics.py`, and `io.py` from `github.com/royerlab/kaggle-cell-tracking-competition` so that all local evaluation uses the exact same scoring code as Kaggle's official scorer.

**Rationale:** CRITICAL DEPENDENCY (flagged in REQUIREMENTS.md): `tracking_cellmot` (the host package containing the actual scoring logic) is **not** on PyPI. It must be cloned/vendored locally. This is a prerequisite for Plans 02, 04, and 05.

**Must-haves:**
- [ ] `src/tracking_cellmot/metrics.py` contains `EvaluationResult`, `DatasetsResult`, `evaluate()`, `evaluate_datasets()`, `per_sample_metrics()`, `summarise()` with constants `ADJUSTMENT_ALPHA = 0.1` and `SCORE_DIVISION_WEIGHT = 0.1`
- [ ] `src/tracking_cellmot/division_metrics.py` contains `evaluate_divisions()` (the real division-matching algorithm, not a simple out-degree check)
- [ ] `src/tracking_cellmot/io.py` contains `open_dataset()` pattern that loads Zarr stores and normalizes intensities via `image_statistics.quantiles`
- [ ] `src/tracking_cellmot/__init__.py` properly exports the key functions
- [ ] No local edits to scoring logic — code is verbatim from the reference repo (except for import path adjustments if needed)
- [ ] Unit test: `from src.tracking_cellmot.metrics import evaluate_datasets` succeeds, and calling `evaluate_datasets([])` returns a `DatasetsResult`. **Per the actual vendored source (REFERENCE_IMPLEMENTATION.md §2): `_jaccard(tp,fp,fn)` returns `float("nan")` when `tp+fp+fn==0`, not `0.0`.** An empty list means `edge_tp=edge_fp=edge_fn=0`, so `edge_jaccard` is `NaN`; `has_divisions` is `False` (0+0+0 not > 0), so `division_jaccard` is `NaN` and `score = edge_jaccard` (also `NaN`, not `+0`). The correct assertion is `math.isnan(result.score)`, not `score == 0.0` — assert NaN, don't assert zero.

## Tasks

### Task 00-01: Clone Reference Repo and Extract Files

```xml
<task id="00-01" title="Clone and extract metrics/division_metrics/io.py from host repo">
  <description>
    Clone github.com/royerlab/kaggle-cell-tracking-competition, locate the three target files 
    in src/tracking_cellmot/, and copy them to the local src/tracking_cellmot/ directory.
    Do not copy the entire repo or unnecessary files.
  </description>
  <files>
    <read>
      - (temp clone) royerlab/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py
      - (temp clone) royerlab/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py
      - (temp clone) royerlab/kaggle-cell-tracking-competition/src/tracking_cellmot/io.py
    </read>
    <write>
      - src/tracking_cellmot/metrics.py
      - src/tracking_cellmot/division_metrics.py
      - src/tracking_cellmot/io.py
    </write>
    <create_dirs>
      - src/tracking_cellmot/
    </create_dirs>
  </files>
  <verification>
    - File sizes match originals (within 5% — minor formatting changes only, no logic edits)
    - Key functions present: evaluate(), evaluate_datasets(), evaluate_divisions()
    - Constants ADJUSTMENT_ALPHA and SCORE_DIVISION_WEIGHT visible in metrics.py
  </verification>
</task>
```

### Task 00-02: Create __init__.py with Proper Exports

```xml
<task id="00-02" title="Create __init__.py with proper module exports">
  <description>
    Create src/tracking_cellmot/__init__.py that exports the key evaluation functions and result types.
    This allows Plan 02 and downstream code to import via `from src.tracking_cellmot.metrics import ...`
  </description>
  <files>
    <write>
      - src/tracking_cellmot/__init__.py
    </write>
  </files>
  <verification>
    - File exists and is non-empty
    - Contains at least: `from .metrics import evaluate, evaluate_datasets, EvaluationResult, DatasetsResult`
    - Contains at least: `from .division_metrics import evaluate_divisions`
  </verification>
</task>
```

### Task 00-03: Verify Empty-Dataset Edge Case

```xml
<task id="00-03" title="Unit test: evaluate_datasets on empty list returns correct DatasetsResult">
  <description>
    Write a minimal test (inline in run_pipeline.py or a dedicated test file) that:
    1. Imports evaluate_datasets from src.tracking_cellmot.metrics
    2. Calls evaluate_datasets([])  (or the expected empty-list signature)
    3. Confirms the result's edge_jaccard, division_jaccard, and score are all NaN
       (`math.isnan(...)`) -- per the actual vendored `_jaccard()` logic, a zero-denominator
       Jaccard is NaN, not 0.0. Do not assert equality to 0.0, it will fail.

    This ensures the vendored code is importable and handles the baseline case correctly.
  </description>
  <files>
    <write>
      - tests/test_scoring_baseline.py (or append to an existing test)
    </write>
  </files>
  <verification>
    - Test file exists
    - Test passes (run `python -m pytest tests/test_scoring_baseline.py` or equivalent)
    - Test asserts NaN (via math.isnan), not 0.0, for the empty-list case
    - Import statement works without ModuleNotFoundError
  </verification>
</task>
```

## Verification Criteria

- [ ] All three files (metrics.py, division_metrics.py, io.py) are copied to `src/tracking_cellmot/`
- [ ] No syntax errors when importing: `python -c "from src.tracking_cellmot import evaluate_datasets"`
- [ ] Functions have correct signatures and return types (spot-check via `help(evaluate_datasets)`)
- [ ] No unresolved dependencies (all imports in the vendored code are available in the environment — `polars`, `tracksdata`, standard library)

## Exit Criteria (Phase 0)

This plan is complete when:
1. Code is vendored (Tasks 00-01, 00-02)
2. Import and baseline test pass (Task 00-03)
3. A dependent plan (Plan 02: eval harness) can successfully import and call `evaluate_datasets()`
