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
- [ ] **DATA-04**: `.geff` ground truth loaded via `tracksdata`'s `IndexedRXGraph.from_geff()` —
      the host's own reference reader (confirmed real, pip-installable, same library the
      competition's official scorer is built on; see `REFERENCE_IMPLEMENTATION.md`) — not a
      hand-rolled parser, into the same in-memory representation the tracker already consumes
- [ ] **DATA-05**: Pipeline iterates over all dataset folders present at inference time (not a
      single hardcoded path), emitting one `dataset` block per folder in the submission
- [ ] **DATA-06**: Raw `uint16` intensities normalized via each sample's own
      `image_statistics.quantiles` (zarr attrs) before any thresholding — matches the host's own
      `open_dataset()` pattern (`(tensor - q_low) / (q_high - q_low)`, clamped). Confirmed necessary:
      current placeholder thresholds (`0.4`, `0.45` in `extract_peaks_from_volume`) assume a `[0,1]`
      range but real data ranges ~15–4319 raw — without this fix even Phase 1's placeholder detector
      produces meaningless output on real data, not just Phase 2's trained model

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
      separate `node`/`edge` rows, integer voxel coords, `-1` sentinels. **Confirmed from the real
      `sample_submission.csv` (2026-07-03) — non-obvious, easy to get wrong:** `id` is a single
      global sequential integer across the *entire file* (0,1,2,...), while `node_id` is
      **per-`dataset`-local** — starts at 1 and *resets to 1* for each new `dataset` block. `node_id`
      is only used to link `source_id`/`target_id` in edge rows; it is not, and must not be treated
      as, globally unique like `id`.
- [ ] **SUB-02**: One `dataset` block per test folder actually processed, matching real test
      basenames exactly
- [ ] **SUB-03**: Generated file validated against `sample_submission.csv`'s structure before a
      run is treated as "submit-ready"

### Local Evaluation Harness (PRD FR-5)

**Scope change (2026-07-03):** the host publishes and links a full reference implementation
(`royerlab/kaggle-cell-tracking-competition`, built on `tracksdata`) directly from the
competition's evaluation page — vendored/documented in `REFERENCE_IMPLEMENTATION.md`. EVAL-01..03
below are now "integrate the real `metrics.py`/`division_metrics.py`" tasks, not "reimplement a
bespoke spec from prose" — materially lower risk than originally scoped.

- [ ] **EVAL-01**: `edge_jaccard` computed via `tracksdata`'s `evaluate()`/`evaluate_datasets()`
      (7.0µm gated `DistanceMatching`, TP/(TP+FP+FN), micro-averaged across samples — counts summed
      before the ratio, not a mean of per-sample ratios)
- [ ] **EVAL-02**: `division_jaccard` computed via the vendored `division_metrics.evaluate_divisions()`
      — NOT a simple out-degree≥2 check; it does GT-division subgraph extraction, stage-coverage
      matching, and global bipartite max-matching between predicted/GT divisions (see
      `REFERENCE_IMPLEMENTATION.md` §3 — do not approximate this, the edge cases are the whole point)
- [ ] **EVAL-03**: Combined score `adjusted_edge_jaccard + 0.1 × division_jaccard` where
      `adjusted_edge_jaccard = max(0, jaccard · (1 − 0.1·(T_pred−T_true)/T_true))`, `T_true` = the
      `.geff`'s `estimated_number_of_nodes`; division term dropped entirely (not `+0`) when a split
      has zero GT divisions. Confirmed: unmatched predicted nodes are structurally excluded from FP
      (sparse-GT-safe by construction, not a fudge factor) — resolves the sparse-annotation concern
      in `data/staging/README.md`
- [ ] **EVAL-04**: `traccuracy` kept only as an optional secondary sanity check (generic CTC
      TRA/DET, a *different* formula from this competition's bespoke score) — `tracksdata` is
      primary per the Key Decision in `PROJECT.md`

**Verified 2026-07-03 (this session, not just documented):** `pip install tracksdata` works
cleanly (installed, confirmed against `requirements.txt`); `tracksdata.graph.IndexedRXGraph.
from_geff(path)` returns `(graph, GeffMetadata)` — note the tuple, not a bare graph — and correctly
parsed the real staged `44b6_0113de3b.geff` (52 nodes / 50 edges, exact match against a raw-zarr
read). **However:** `tracking_cellmot` (the host repo's own package containing the actual
`metrics.py`/`division_metrics.py` scoring code) is confirmed **not** on PyPI — it must be cloned
from `github.com/royerlab/kaggle-cell-tracking-competition` and installed locally (`pip install -e`)
or vendored by copying the specific files, not just `pip install`able like `tracksdata` itself.

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
| DATA-01 | Phase 0 | Complete |
| DATA-02 | Phase 0 | Complete |
| DATA-03 | Phase 0 | Complete |
| DATA-04 | Phase 0 | Complete |
| DATA-05 | Phase 0 | Complete |
| DATA-06 | Phase 0 | Complete |
| SUB-01 | Phase 0 | Complete |
| SUB-02 | Phase 0 | Complete |
| SUB-03 | Phase 0 | Complete |
| EVAL-01 | Phase 0 | Complete |
| EVAL-02 | Phase 0 | Complete |
| EVAL-03 | Phase 0 | Complete |
| EVAL-04 | Phase 0 | Complete |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| MODEL-04 | Phase 2 | Pending |
| TRACK-01 | Phase 2 | Pending |
| TRACK-02 | Phase 3 | Pending |
| TRACK-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 20 total (added DATA-06 2026-07-03 — real-data intensity normalization gap
  found via the host's reference `io.py`)
- Mapped to phases: 20
- Unmapped: 0 ✓

Phase 1 (Baseline parity, per PRD §8) has no dedicated new requirements of its own — it's the
*wiring and validation* of DATA-*/SUB-*/EVAL-* against the existing (untrained) placeholder
detector, proving the pipeline is sound before Phase 2's training investment.

---
*Requirements defined: 2026-07-03*
*Last updated: 2026-07-03 after initialization*
