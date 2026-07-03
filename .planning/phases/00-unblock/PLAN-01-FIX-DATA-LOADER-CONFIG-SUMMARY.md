---
phase: 0
plan: 01
subsystem: Data Ingestion & Config
tags: [zarr-v3, anisotropy, data-loader, quantile-normalization, regression-tests]
completed: 2026-07-03
duration: ~30 minutes (interrupted mid-stream once, resumed and finished in the orchestrating session)
dependencies:
  requires: []
  provides:
    - AnisotropicZarrLoader reads real Zarr v3 OME-NGFF stores at array path 0/
    - Anisotropy corrected to (4.0, 1.0, 1.0) everywhere it's hardcoded
    - Quantile-based intensity normalization (DATA-06)
  affects:
    - Phase 0, Plan 04 (wire pipeline) consumes the updated loader directly
tech-stack:
  added: []
  patterns:
    - Zarr v3 metadata is zarr.json, not v2-legacy .zattrs/.zarray
    - Real per-sample quantile normalization via image_statistics.quantiles zarr attrs
---

# Phase 0, Plan 01: Fix Data Loader and Config — SUMMARY

**Objective:** Update `AnisotropicZarrLoader` to read real Zarr v3 OME-NGFF stores (array path `0/`) with correct anisotropy `(4.0, 1.0, 1.0)`, and fix the same constant everywhere else it's hardcoded.

**Status:** COMPLETE ✓

---

## Tasks Completed

| Task | Name | Commit | Files |
|---|---|---|---|
| 01-01 | Update AnisotropicZarrLoader for Zarr v3 OME-NGFF | 9a7efa0 | src/data_loader.py |
| 01-02 | Fix anisotropy to (4.0,1.0,1.0) everywhere | 8871c0d | config/hyperparams.yaml, run_pipeline.py |
| 01-03 | Unit tests on real staged data + tracker regression suite | 9f70240 | tests/test_data_loader_real.py, tests/test_tracker.py |

---

## What Was Built

### Task 01-01: Zarr v3 loader (9a7efa0)
`AnisotropicZarrLoader` now opens real Zarr v3 OME-NGFF stores at `<store>/0/` (not the v2-legacy
layout the simulated fallback produces), reads `image_statistics.quantiles` from zarr attrs, and
normalizes raw uint16 intensities via `(tensor - q_low) / (q_high - q_low)` before any thresholding
— the host's own `io.py` pattern (DATA-06). The simulated-data fallback no longer activates
silently against a real path (DATA-02).

### Task 01-02: Anisotropy correction (8871c0d)
`(5.0, 1.0, 1.0)` → `(4.0, 1.0, 1.0)` (Z:Y:X) in `config/hyperparams.yaml` and `run_pipeline.py`.
`src/tracker.py` and `src/model.py` were checked — no additional hardcodes found beyond what was
already caught; both already receive anisotropy as a parameter, not a hardcoded constant.

### Task 01-03: Test suites (9f70240)
- **`tests/test_data_loader_real.py`** (11 tests, all passing) — against the real staged sample
  `data/staging/train/44b6_0113de3b.zarr`: shape `(100,64,256,256)`, dtype `uint16`, anisotropy
  `(4.0,1.0,1.0)`, quantile normalization applied and values in expected range, multi-timepoint
  loading, and — importantly — that `simulate=False` against a nonexistent path fails loudly
  rather than silently fabricating data (DATA-02, verified not just implemented).
- **`tests/test_tracker.py`** (14 tests, all passing) — beyond this plan's formal scope, but
  written as regression coverage while testing the loader/tracker integration: covers positive
  linking, division, gap-closing, motion compensation, edge pruning, and explicitly regression-
  tests this session's earlier ILP flow-conservation fix (isolated/orphan nodes resolve instead of
  making the solver `Infeasible`).

Verified directly (not just trusted): `py -m pytest tests/test_data_loader_real.py
tests/test_tracker.py -v` → **25 passed**, 0 failed.

---

## Key Finding Worth Flagging to Phase 1/4 (logged in STATE.md)

`test_tracker.py` quantified something not previously measured: the ILP only prefers a link over
two isolated birth+death costs when `distance² < birth_cost + death_cost`. With production costs
(`birth_cost=15, death_cost=15`), that break-even is **~5.48 microns of movement per frame**. Also
confirmed: `prune_unphysical_edges` gates on *raw* (unwarped) coordinates, not motion-compensated
ones — a fast-moving cell whose raw per-frame jump exceeds the prune gate can never be rescued by
an accurate motion vector. If Phase 1's baseline-parity score comes in surprisingly low, check
tracker cost calibration before assuming the detector or data is broken.

---

## Deviations from Plan

None material. Task 01-03 additionally produced `test_tracker.py`, which wasn't explicitly
scoped in the plan file but exercises the same real-data-integration surface and caught a
quantifiable tuning risk worth carrying forward — kept rather than discarded.

---

## Verification Checklist

- ✓ `AnisotropicZarrLoader` reads real Zarr v3 stores at `<store>/0/`
- ✓ Simulated fallback never activates against a real path (tested directly, not assumed)
- ✓ Anisotropy is `(4.0,1.0,1.0)` in all checked code paths and config
- ✓ Quantile-based normalization implemented and tested
- ✓ 25/25 tests pass, run directly and confirmed (not just "should pass")
- ✓ All tasks committed individually
