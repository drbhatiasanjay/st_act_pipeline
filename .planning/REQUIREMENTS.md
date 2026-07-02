# Requirements: ST-ACT Cell Tracker

**Defined:** 2026-07-03
**Core Value:** A submission that is schema-valid, scores above the classical baseline (0.763),
and is engineered to compete for the top of the leaderboard

Derived directly from `PRD.md` §6 (Functional Requirements) — not re-interviewed, since that
document already fully scopes this from a real competition-spec audit.

## v1 Requirements

### Data Ingestion (PRD FR-1)

- [ ] **DATA-01**: Pipeline reads real Kaggle `train/{embryo}_{fov}/` and `test/{embryo}_{fov}/`
      Zarr v3 stores at array path `0/`, shape `(T,Z,Y,X)`, `uint16` — not the simulated fallback
- [ ] **DATA-02**: Simulated-data fallback in `_init_store` never silently activates against a
      real competition path (removed, or gated behind an explicit `--simulate` flag)
- [ ] **DATA-03**: Anisotropy ratio is `(4.0, 1.0, 1.0)` everywhere it's used (was hardcoded to
      the wrong `(5.0, 1.0, 1.0)`)
- [ ] **DATA-04**: `.geff` ground-truth reader loads `nodes/ids`, `nodes/props/{t,z,y,x}/values`,
      `edges/ids` into the same in-memory representation the tracker already consumes
- [ ] **DATA-05**: Pipeline iterates over all dataset folders present at inference time (not a
      single hardcoded path), emitting one `dataset` block per folder in the submission

### Detection Model (PRD FR-2)

- [ ] **MODEL-01**: Training script generates heatmap targets (anisotropy-aware Gaussian blobs at
      `.geff` centroids) and motion-vector targets (displacement to `t+1`, from GT edges)
- [ ] **MODEL-02**: Detection model trained with train/val split by **embryo** (never split within
      one embryo, matching the competition's embryo-disjoint train/test discipline)
- [ ] **MODEL-03**: `extract_peaks_from_volume`'s naive stride-scan replaced with real model
      inference (local maxima of predicted heatmap above a tuned threshold)
- [ ] **MODEL-04**: Tracker receives the model's predicted motion-vector field, not a hardcoded
      constant

### Tracker at Scale (PRD FR-3)

- [ ] **TRACK-01**: Tracker receives real predicted motion vectors (depends on MODEL-04)
- [ ] **TRACK-02**: Mitosis smoothing uses real local image-intensity evaluation, not the mock
      proxy
- [ ] **TRACK-03**: ILP is tractable at real scale (thousands of cells × 100 timepoints ×
      gap-closing) via windowed/rolling-horizon solving or a min-cost-flow reformulation —
      candidate: OR-Tools `SimpleMinCostFlow` for the bulk assignment (verified real fit for our
      flow-conservation formulation, not just "a mandatory framework")

### Submission Generation (PRD FR-4)

- [ ] **SUB-01**: Export emits `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` exactly,
      separate `node`/`edge` rows, integer voxel coords, `-1` sentinels
- [ ] **SUB-02**: One `dataset` block per test folder actually processed, matching real test
      basenames exactly
- [ ] **SUB-03**: Generated file validated against `sample_submission.csv`'s structure before a
      run is treated as "submit-ready"

### Local Evaluation Harness (PRD FR-5)

- [ ] **EVAL-01**: `edge_jaccard` implemented exactly per spec (7.0µm gating, bipartite matching,
      TP/(TP+FP+FN) with over-prediction penalty, sample-weighted averaging)
- [ ] **EVAL-02**: `division_jaccard` implemented exactly per spec (out-degree ≥2, connected-
      component check spanning both daughter lineages, micro-averaged)
- [ ] **EVAL-03**: Combined score `edge_jaccard + 0.1 × division_jaccard` computed against
      held-out **train** embryos with real `.geff` ground truth
- [ ] **EVAL-04**: Harness cross-checked against `traccuracy`'s CTC-standard metrics on the same
      graphs (won't match Kaggle's exact formula, but catches hand-rolled-harness bugs)

## v2 Requirements

Deferred — not required to clear the classical baseline or compete, but real leverage once v1
is solid.

### Competitive Iteration (PRD Phase 4-5)

- **ITER-01**: Grid/systematic calibration of tracker costs (birth/death/division/gap) against
  the local harness
- **ITER-02**: Model ensembling (multiple seeds/architectures)
- **ITER-03**: TF-GNN/JAX-MD node-embedding pre-processing — deferred as premature complexity;
  nothing in the metric currently justifies it (see conversation record, not adopted from the
  framework-spec prompt without justification)
- **ITER-04**: TensorStore-based ingestion for I/O throughput, if the 12-hour Kaggle runtime
  budget turns out to actually be tight (not assumed)

### Visualization

- **VIZ-01**: `napari` + `napari-geff` for local dev-time QC of predictions vs. ground truth
- **VIZ-02**: Neuroglancer-based web UI for visually showing tracking results (explicitly
  requested — "later we will build the UI, take some data and show it visually") — deferred
  until the core pipeline (v1 above) produces a real, scoreable submission

## Out of Scope

| Feature | Reason |
|---|---|
| General-purpose cell-tracking product/UI | This is a competition entry, not a product (PRD §5) |
| Non-light-sheet microscopy modality support | Out of scope per PRD §5 |
| Real-time/streaming inference | Batch offline inference against Kaggle test folders is sufficient (PRD §5) |
| Keras 3 + JAX model rewrite | Evaluated against the pasted framework-spec prompt; PyTorch + MONAI chosen instead — ecosystem maturity for 3D biomedical volumes (sliding-window inference, anisotropic augmentations) outweighs JAX's framework-purity, and GPU (not TPU) is the compute plan |

## Traceability

| Requirement | Phase | Status |
|---|---|---|
| DATA-01 | Phase 0 | In Progress |
| DATA-02 | Phase 0 | Pending |
| DATA-03 | Phase 0 | Pending |
| DATA-04 | Phase 0 | Pending |
| DATA-05 | Phase 0 | Pending |
| SUB-01 | Phase 0 | Pending |
| SUB-02 | Phase 0 | Pending |
| SUB-03 | Phase 0 | Pending |
| EVAL-01 | Phase 0 | Pending |
| EVAL-02 | Phase 0 | Pending |
| EVAL-03 | Phase 0 | Pending |
| EVAL-04 | Phase 0 | Pending |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| MODEL-04 | Phase 2 | Pending |
| TRACK-01 | Phase 2 | Pending |
| TRACK-02 | Phase 3 | Pending |
| TRACK-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

Phase 1 (Baseline parity, per PRD §8) has no dedicated new requirements of its own — it's the
*wiring and validation* of DATA-*/SUB-*/EVAL-* against the existing (untrained) placeholder
detector, proving the pipeline is sound before Phase 2's training investment.

---
*Requirements defined: 2026-07-03*
*Last updated: 2026-07-03 after initialization*
