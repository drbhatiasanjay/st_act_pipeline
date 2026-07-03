---
wave: 1
depends_on: []
files_modified:
  - src/data_loader.py
  - config/hyperparams.yaml
autonomous: true
---

# Phase 0, Plan 01: Fix Data Loader and Config

**Goal:** Update `AnisotropicZarrLoader` to read real Zarr v3 OME-NGFF stores (at array path `0/`) with correct anisotropy `(4.0, 1.0, 1.0)`, and fix config to use the correct value everywhere.

**Rationale:** DATA-01, DATA-03, DATA-06 in REQUIREMENTS.md. The current loader uses simulated fallback data and wrong anisotropy. Real competition data is Zarr v3 stored at `<dataset>.zarr/0/` with shape `(T,Z,Y,X)` uint16 and real-data quantile-based normalization.

**Must-haves:**
- [ ] `AnisotropicZarrLoader` reads real Zarr v3 OME-NGFF stores at path `<store>/0/` when given a real store path
- [ ] Simulated-data fallback is **never activated** against a real competition folder (removed or gated behind explicit flag)
- [ ] Anisotropy is `(4.0, 1.0, 1.0)` in all code paths and config
- [ ] Raw uint16 intensities are normalized via `image_statistics.quantiles` zarr attrs before any thresholding (DATA-06)
- [ ] Unit test on real staged data (e.g., `data/staging/train/44b6_0113de3b.zarr`) loads successfully with correct shape/dtype
- [ ] All downstream code (`extract_peaks_from_volume`, tracker, etc.) receives anisotropy as `(4.0, 1.0, 1.0)`

## Tasks

### Task 01-01: Update AnisotropicZarrLoader for Zarr v3 OME-NGFF

```xml
<task id="01-01" title="Refactor AnisotropicZarrLoader to read real Zarr v3 stores">
  <description>
    1. Remove simulated-data fallback or gate it behind `--simulate` flag (DATA-02)
    2. Update to use zarr.open_array() / zarr.open() for v3 stores (not v2 legacy)
    3. Read array at path "0/" (the OME-NGFF standard location)
    4. Extract anisotropy from zarr metadata or read from config (will be 4.0, 1.0, 1.0)
    5. Check for image_statistics.quantiles in zarr attrs; implement normalization:
       `(tensor - q_low) / (q_high - q_low)` clamped to [0, 1]
       (This is the host's pattern from io.py; the placeholder thresholds 0.4/0.45 assume [0,1])
    6. Return shape (T,Z,Y,X) uint16 raw or normalized float as appropriate
  </description>
  <files>
    <read>
      - src/data_loader.py (current AnisotropicZarrLoader)
      - data/staging/train/44b6_0113de3b.zarr/zarr.json (real store: train/{id}.zarr + train/{id}.geff,
        test/{id}.zarr for test-only; this is Zarr FORMAT 3, metadata file is `zarr.json`, not the
        v2-legacy `.zattrs`/`.zarray` this task description might otherwise assume; array data is at
        the nested `.../44b6_0113de3b.zarr/0/zarr.json`)
    </read>
    <write>
      - src/data_loader.py (updated AnisotropicZarrLoader)
    </write>
  </files>
  <verification>
    - Code uses zarr.open_array() or zarr.open() (not zarr.open_group() v2 legacy)
    - Path "0/" is hardcoded or parameterized correctly
    - Simulated fallback removed or guarded by flag check
    - Quantile normalization is implemented (check for presence of (q_low - q_high) calculation)
    - No import errors; zarr library is available
  </verification>
</task>
```

### Task 01-02: Fix Anisotropy in Config and All Code Paths

```xml
<task id="01-02" title="Update anisotropy to (4.0, 1.0, 1.0) everywhere">
  <description>
    1. Update config/hyperparams.yaml: set anisotropy to (4.0, 1.0, 1.0) [was (5.0, 1.0, 1.0)]
    2. Grep for hardcoded (5.0, 1.0, 1.0) in Python code; replace with (4.0, 1.0, 1.0)
    3. Check src/tracker.py, src/model.py, src/data_loader.py for any anisotropy hardcodes
    4. Ensure all references are consistent (either from config or from loader metadata)
  </description>
  <files>
    <read>
      - config/hyperparams.yaml
      - src/data_loader.py
      - src/tracker.py
      - src/model.py
      - run_pipeline.py (repo root, NOT under src/ -- if it has any hardcodes)
    </read>
    <write>
      - config/hyperparams.yaml (update value)
      - src/data_loader.py (if any hardcodes exist)
      - src/tracker.py (if any hardcodes exist)
      - src/model.py (if any hardcodes exist)
      - run_pipeline.py (repo root, NOT under src/ -- if any hardcodes exist)
    </write>
  </files>
  <verification>
    - No instances of (5.0, 1.0, 1.0) remain in Python code or config
    - All references to anisotropy use (4.0, 1.0, 1.0)
    - config/hyperparams.yaml has the correct value
    - All data-loading and tracker calls pass anisotropy correctly
  </verification>
</task>
```

### Task 01-03: Unit Test on Staged Real Data

```xml
<task id="01-03" title="Load real staged data and verify shape/dtype/quantiles">
  <description>
    Write a test that:
    1. Instantiates AnisotropicZarrLoader with a real staged path (e.g., data/staging/train/44b6_0113de3b.zarr)
    2. Loads a sample (calls loader.load() or __getitem__)
    3. Asserts:
       - Shape is (T, Z, Y, X) with T, Z, Y, X > 0
       - dtype is uint16 (raw) or float32 (normalized)
       - Anisotropy is (4.0, 1.0, 1.0)
       - Quantile normalization is applied (values in ~[0, 1] if q_low/q_high are present)
  </description>
  <files>
    <write>
      - tests/test_data_loader_real.py (new test file)
    </write>
  </files>
  <verification>
    - Test file exists and is executable
    - Test passes on staged data: `python -m pytest tests/test_data_loader_real.py`
    - No exceptions when loading (file not found, zarr format errors, etc.)
  </verification>
</task>
```

## Verification Criteria

- [ ] `AnisotropicZarrLoader` opens real Zarr v3 stores without fallback
- [ ] Anisotropy is consistently `(4.0, 1.0, 1.0)`
- [ ] Quantile normalization is implemented and applied
- [ ] Real staged data loads successfully with correct shape/dtype
- [ ] No hardcoded `(5.0, 1.0, 1.0)` remains in the codebase

## Exit Criteria (Phase 0)

This plan is complete when:
1. Data loader is refactored (Task 01-01)
2. Anisotropy is fixed everywhere (Task 01-02)
3. Real data loads successfully (Task 01-03 passes)
4. Plan 04 (wire pipeline) can use this loader without modification
