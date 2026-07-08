# Phase 2: Learned Detection - Context

**Gathered:** 2026-07-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Train a real detection model on `.geff` ground-truth targets and wire it end-to-end,
beating the classical baseline (0.763) and generating the first real Kaggle submission.
Replaces Phase 1's placeholder detector (real peak-finding but no learned classification,
which topped out at 0.0259) with a genuinely learned model. Phase 3's scaling/correctness
work and Phase 4's metric-directed tuning are out of scope here.

</domain>

<decisions>
## Implementation Decisions

### Model architecture -- major pivot, ripple effects noted
- **Full host-style replacement**, not an incremental deepening of the existing
  `STACTCentroidPredictor` heatmap+motion CNN. Adopts the host's documented paradigm
  (`REFERENCE_IMPLEMENTATION.md` S5): a 3D UNet producing per-voxel detection logits, fed
  into a cross-attention Transformer (`SimpleNodeTransformer`-style) that predicts pairwise
  edge probabilities directly, with a greedy edge-assignment step (not our own ILP) turning
  those probabilities into tracks.
- **`STHypergraphTracker` (the existing, extensively-debugged, SCIP-optimized ILP tracker)
  is kept in the repo as a fallback/comparison, not deleted** -- but is not the active
  production path for Phase 2, and is not being actively maintained/extended going forward
  unless a later phase revisits it.
- **Scope-boundary note for the planner/researcher**: `ROADMAP.md`'s Phase 2 requirements
  MODEL-04 ("tracker receives the model's predicted motion-vector field") and TRACK-01
  ("tracker receives real predicted motion vectors") were written assuming the
  heatmap+motion+ILP paradigm. Under this host-style pivot, there is no per-cell "motion
  vector field" in the same sense -- motion/linking is implicit in the transformer's
  pairwise edge predictions over a frame window. These two requirements need
  reinterpretation during planning (e.g., "the model's predicted edge probabilities feed
  the assignment step" rather than literal motion vectors), not literal implementation.
  Similarly, Phase 3's TRACK-02/TRACK-03 (ILP mitosis smoothing, ILP scaling) may need
  re-scoping once this pivot is confirmed working in practice -- flagged as a Phase 3
  planning input, not resolved here.
- **Gap-closing**: unlike the host's strict 2-frame-window (t, t+1) design, Phase 2's
  edge-predictor should support multi-frame gap-closing (~2 frames), matching the
  capability our existing tracker already has, to handle missed detections across gaps.
- **Hyperparameters** (UNet channel depths, transformer hidden_dim/heads/blocks, exact
  window handling): Claude's discretion. Starting point is the host's documented values
  (UNet `[32,64,128]` channels; transformer hidden_dim 128, 4 heads, 4 blocks, dropout 0.3)
  -- treat as a known-working starting point, adjust only if something looks clearly wrong
  for this project's specific data.

### Training scale & iteration strategy
- **Train on the full 199-sample competition train set from the start** -- not smoke-testing
  on the 4 currently-staged local samples first. Kaggle GPU (confirmed: Tesla T4 x2) and the
  full competition data mount (`/kaggle/input/competitions/biohub-cell-tracking-during-development/`)
  are already verified working end-to-end.
- **The full competition dataset is already downloaded locally** (the original 81.6GB zip) --
  no re-download needed. Kaggle training sessions read from Kaggle's own mounted copy;
  local staging stays as-is for harness/dev work unless a specific need arises.
- **Even going straight to full-scale data, run a short few-epoch sanity-check training job
  first**, before committing to a long/full training run -- confirms the training loop,
  loss, and data pipeline work correctly on real data before spending significant GPU time,
  mirroring Phase 1's own successful "profile before full scale" pattern.
- **Check in with the user after each meaningful milestone** (sanity-check passes, first
  full training run completes, first real held-out score) rather than running long
  unsupervised stretches -- tighter feedback loop given how much Phase 1 diverged from
  initial time estimates.

### Heatmap/target generation
- **GT positive-voxel design (single exact point vs. small anisotropy-aware dilated region
  around each centroid): decide empirically, not by guessing.** Benchmark both against the
  local evaluation harness before locking one in for the full training run -- whichever
  measurably improves the real local score wins. Explicit user mandate: "benchmark against
  the exit criteria of the competition and design to achieve the target and be the winner
  in the leaderboard" -- this principle (validate score-impacting choices empirically, don't
  guess) applies to other similar decisions made during implementation too.
- **Add an auxiliary loss term weighting division edges more heavily** than standard
  edge-prediction loss alone -- division events are rare but flagged (PRD SS10,
  `STATE.md` blocker #3) as a likely differentiator among top scorers.
- **Include test-time augmentation from the start** (average detection logits across the
  original + 3 flips: Y, X, Y+X, before peak extraction) -- already documented
  (`REFERENCE_IMPLEMENTATION.md` S5) as a real accuracy lever, not an experimental add-on to
  defer.

### Validation & Kaggle submission cadence
- **Held-out validation set size**: Claude's discretion, informed by actually inspecting
  real embryo counts/diversity across the full 199-sample set (not a fixed a priori number
  like the earlier Phase 1 plan's "1 44b6 + 1 6bba sample").
- **Validation split stability** (fixed for the whole phase vs. reshuffled): Claude's
  discretion, decided based on what's practical once real training is underway.
- **Track `division_recall` as a separate diagnostic metric during Phase 2 validation**,
  pulled forward from its original Phase 4 scope -- cheap to add, gives direct feedback on
  whether the new auxiliary division-loss weighting is actually working. Not yet optimized
  for at this stage, just observed.
- **Only spend an actual Kaggle submission once the local held-out score clearly and
  materially exceeds the 0.763 baseline.** Do not spend a scarce submission early just to
  validate the upload/scoring loop -- local evaluation harness (already proven correct
  through Phase 0/1) is trusted for that validation instead.

### Claude's Discretion
- Exact model hyperparameters within the host-documented starting point.
- Held-out validation set size and split-stability policy.
- Backbone implementation details (hand-built vs. MONAI components) as long as the overall
  host-style paradigm (UNet detection + transformer edge-prediction + greedy assignment) is
  followed.

</decisions>

<specifics>
## Specific Ideas

- Primary architecture reference is `REFERENCE_IMPLEMENTATION.md` S5's documented host
  model (`UNetNodeTransformer`, `SimpleNodeTransformer`, `pool_kernel_from_um`-style NMS
  already implemented in Phase 1, greedy edge-assignment code snippet already vendored in
  that doc) -- follow this as the concrete reference, not a hypothetical redesign.
- User's explicit design principle for this phase: benchmark score-impacting
  implementation choices against the real competition exit criteria (local evaluation
  harness) and pick whichever wins, rather than guessing -- echoes Phase 1's own successful
  "sweep before committing" pattern (`scripts/sweep_threshold.py`).

</specifics>

<deferred>
## Deferred Ideas

None raised outside Phase 2's scope during this discussion. One related note (not deferred,
just flagged for the next phase's planning): Phase 3's TRACK-02 (ILP mitosis smoothing) and
TRACK-03 (ILP scaling) requirements assume the ILP tracker remains the active production
path -- given Phase 2's pivot away from it, Phase 3 planning should explicitly revisit
whether those requirements still apply as written, or need re-scoping around the
greedy-assignment paradigm instead.

</deferred>

---

*Phase: 02-learned-detection*
*Context gathered: 2026-07-08*
