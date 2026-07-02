# ST-ACT — Spatio-Temporal Anisotropic Cell Tracker

## What This Is

Our entry for the Kaggle competition **Biohub – Cell Tracking During Development**: for every
test embryo movie (3D+time light-sheet microscopy of a developing zebrafish embryo), detect every
cell nucleus, link detections into per-cell tracks over time, and correctly resolve mitosis
(division) events into a submittable tracking graph.

## Core Value

A submission that is schema-valid, scores above the classical non-learned baseline (0.763), and
is engineered to compete for the top of the leaderboard (#1 at PRD-authoring time: 0.875) — not
just "a pipeline that runs."

## Requirements

### Validated

<!-- Inferred from PRD.md §4.3 "What's actually good and worth keeping" -->

- ✓ `STHypergraphTracker`'s core ILP formulation (flow conservation with explicit birth/death/
  split variables, temporal gap-closing with exponential lookahead penalty, anisotropic edge
  pruning) — a real differentiator over nearest-neighbor and Hungarian-only baselines, directly
  maps onto the `division_jaccard` metric term
- ✓ `AnisotropicCoordinateTransformer` — correctly separates voxel-space from physical-space
  reasoning
- ✓ Flow-conservation ILP constraints fixed this session: isolated/orphan detections (no
  plausible neighbor in either direction — expected on real, sparse, noisy data) now resolve as
  legitimate one-frame singletons instead of making the solver `Infeasible` or silently tracking
  nothing. Verified via direct ILP objective-value comparison, not just "it runs."

### Active

<!-- From PRD.md §6 FR-1 through FR-5, and PRD.md §4.1 the 5 launch-blocking gaps -->

- [ ] Correct data ingestion: real Kaggle `train/`/`test/` Zarr v3 stores (not the simulated
  fallback), real physical anisotropy `(4.0,1.0,1.0)`, a `.geff` ground-truth reader
- [ ] Trained detection model wired into the pipeline (replacing the naive threshold placeholder),
  fed by real `.geff`-derived targets, real predicted motion vectors reaching the tracker
- [ ] Tracker made tractable at competition scale (thousands of cells × 100 timepoints × gap-
  closing) — windowed/rolling-horizon solving or min-cost-flow reformulation
- [ ] Correct submission generation: exact `id,dataset,row_type,node_id,t,z,y,x,source_id,
  target_id` schema, one `dataset` block per test folder
- [ ] Local evaluation harness implementing the exact competition metric
  (`edge_jaccard + 0.1 × division_jaccard`) against held-out train embryos with real `.geff`
  ground truth — required before spending any (rate-limited) Kaggle submission

### Out of Scope

<!-- From PRD.md §5 Non-goals -->

- General-purpose cell-tracking product/UI — this is a competition entry, not a product
- Microscopy modalities other than the competition's light-sheet zebrafish data
- Real-time/streaming inference — batch offline inference against Kaggle test folders is
  sufficient

## Context

- **Competition host:** Biohub (Royer Group). Builds on Ultrack (Royer's tracker, *Nature
  Methods* 2025). $60,000 prize pool, entry deadline 2026-09-22. ~49 teams on the board; public
  leaderboard reflects only ~29% of test data, so local validation against real `.geff` ground
  truth matters more than chasing the visible slice.
- **Full spec + audit:** `PRD.md` at repo root is the source of truth for competition spec,
  current-state audit, functional requirements, and phased roadmap (§8) — this PROJECT.md tracks
  GSD execution state alongside it, it does not replace it.
- **Cross-session verification already done:** `trackdiff.md` documents a comparison against a
  Google AI Studio export of this codebase — confirmed most of its UI is cosmetic/simulated
  (including a fabricated "pipeline run" score), and one of its two proposed Python fixes was
  actually a regression. The corrected version of that fix is already applied to `src/tracker.py`.
- **Real competition data located:** `C:\Users\hemas\Downloads\biohub-cell-tracking-during-
  development.zip` (81.6GB, unextracted) — 199 train samples (`.zarr`+`.geff` pairs across two
  source embryos `44b6`/`6bba`), 4 test samples, `sample_submission.csv`. Staging a small subset
  first (agreed) is the next concrete step (plan Step 1) once this GSD scaffolding is in place.
- **No local GPU** (torch is CPU-only). Kaggle Notebooks (free GPU, competition data pre-mounted)
  is the intended training environment for Phase 2, not Google AI Studio (no training compute).

## Constraints

- **Compute**: No local GPU — any model training happens on Kaggle Notebooks or Colab, not locally
- **Disk**: 169GB free vs. 81.6GB full dataset — fits, but stage incrementally rather than
  extracting everything up front
- **Kaggle submissions are rate-limited** — every change must clear the local evaluation harness
  (Active requirement above) before spending one
- **Train/test are embryo-disjoint** (PRD §3.2) — any train/val split must respect this, never
  split within an embryo

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Stage a small real-data subset (8 train + 4 test) before extracting all 199 train samples | Fast iteration loop; validate the pipeline against real data/format before paying the full 81GB extraction cost | — Pending |
| Classical (non-learned) detection baseline before CNN training | Establishes a real, scoreable submission fast without GPU dependency; floor to clear either way is 0.763 | — Pending |
| Use `geff` (reference reader), `traccuracy` (CTC-metric cross-check), `napari`+`napari-geff` (visual QC) | Verified real, MIT-licensed, same publisher as the GEFF format itself — reduces hand-rolled parsing/metric risk | — Pending |
| GSD for cross-session phase execution against PRD.md §8's roadmap | Multi-week effort; GSD already enabled with pre-approved permissions in this environment | — Pending |
| Ported a corrected version of the AI Studio tracker.py fix (drop `b_n+d_n<=1`, keep the flow equalities) rather than either the original bug or the AI Studio version verbatim | Empirically verified: AI Studio's fix reintroduces the "always zero edges" bug; the corrected version is feasible AND has a strictly better ILP objective value (995 vs. 1941) on the same test data | ✓ Good |

---
*Last updated: 2026-07-03 after initialization*
