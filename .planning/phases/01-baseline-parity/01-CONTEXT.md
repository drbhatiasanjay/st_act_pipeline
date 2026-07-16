# Phase 1: Baseline Parity - Context

**Gathered:** 2026-07-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the existing untrained placeholder detection model end-to-end and prove the pipeline reaches
the classical baseline (local score >= 0.763) with no new model training. Phase 0's real E2E run
confirmed this is reachable, not by tuning, but by giving the detector real peak-finding: the
current `extract_peaks_from_volume()` does a raw stride-8 grid scan, not genuine local-maxima
detection, so predicted coordinates land on a rigid grid and almost never fall within the 7.0um
match tolerance of a real labeled cell. Model training, learned motion vectors, and ILP scaling
are out of scope - those are Phase 2 and Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Peak-finding algorithm
- Replicate the host's own NMS approach exactly (documented in `REFERENCE_IMPLEMENTATION.md` S5):
  `max_pool3d`-based, kernel sized from real physical micrometers via `pool_kernel_from_um()`
  (host uses a 5.0um kernel), not a fixed voxel size.
- Applied directly to the (already quantile-normalized) raw intensity volume for Phase 1, since no
  model inference exists yet - this stands in for the host's logit-based version until Phase 2.
- Drop `offset_bias` from `extract_peaks_from_volume()` entirely - it was an artifact of the old
  grid-scan hack (used to fake two distinct "networks" from one grid) and is not meaningful once
  real peaks are found. `CNN_THRESHOLD`/`UNET_THRESHOLD` alone remain the two "views" that
  `ensemble_consensus_centroids()` combines via DBSCAN - that function needs no changes.

### Threshold calibration
- Small systematic sweep (5-10 values) across the staged train samples, not a single reasoned
  guess. Reuse the existing detection cache (`src/run_tracker.py`) so sweeping doesn't force full
  recomputation each time. Sanity-check candidate-count order of magnitude at each value (tens to
  low hundreds per timepoint, not thousands) before running the tracker on it - this project has a
  documented incident where a miscalibrated threshold caused an 18,000-candidate ILP blowup.
- The resulting threshold-vs-score data doubles as an early input to Phase 4's tuning work.

### Motion vectors
- Zero (`[0.0, 0.0, 0.0]`), not the current hardcoded `[0.05, 0.2, 0.3]` constant found at
  `run_pipeline.py:240` (confirmed via direct grep - the code is not actually zero today).
- Rationale: the evaluation metric only scores final predicted edges against GT within the 7.0um
  gate; motion vectors are purely an internal signal the tracker uses to warp a cell's position
  before measuring next-frame distance. Real measured inter-frame displacement across all 4 staged
  `.geff` files shows only 3.1% of edges exceed the tracker's 5.48um link/break-even threshold, so
  zero vectors should already be sufficient for the vast majority of true links. A hand-rolled
  heuristic would only help the narrow fast-moving tail and gets replaced by Phase 2's real learned
  model anyway - not worth building now.

### Held-out validation set
- Use all 4 currently staged train samples for the Phase 1 >=0.763 score check - no training
  happens this phase, so there is no overfitting mechanism to protect against yet.
- Separately, earmark one `44b6` sample and one `6bba` sample now as the embryo-disjoint
  validation set Phase 2's MODEL-02 requirement will need. Documentation decision, not a code
  change - avoids retroactively figuring out the split once training starts, and keeps the
  local-score-vs-leaderboard correlation risk (tracked in STATE.md) under control from the start.
  **P0-2 CORRECTION (2026-07-16): this was wrong** -- `44b6` and `6bba` are themselves the embryo
  IDs (Kaggle's own Data description page: "multiple samples may share the same embryo"), so
  holding out one sample per prefix is NOT embryo-disjoint if any other same-prefix sample stays
  in train. A real embryo-disjoint validation set must hold out an entire embryo's samples. See
  `scripts/build_train_val_split.py`'s leave-one-embryo-out fold generator.

### Claude's Discretion
- Exact sweep threshold values to try.
- Whether the sweep is a standalone script or a flag on `run_pipeline.py`.
- Which specific `44b6`/`6bba` sample pair to earmark for the future validation split.

</decisions>

<specifics>
## Specific Ideas

No specific product/UX references - this is an internal ML pipeline correctness phase. The
concrete reference is the host's own NMS code in `REFERENCE_IMPLEMENTATION.md` S5, which should be
followed closely rather than reinvented.

</specifics>

<deferred>
## Deferred Ideas

- Learned motion-vector estimation (heuristic or model-based) - Phase 2, once a real detector
  exists to replace hardcoded/zero vectors properly.
- ILP solver/scaling work - already handled separately (CBC to SCIP swap, Phase 3 windowing risk
  tracked in STATE.md), out of scope for Phase 1's detector-only focus.

</deferred>

---

*Phase: 01-baseline-parity*
*Context gathered: 2026-07-05*
