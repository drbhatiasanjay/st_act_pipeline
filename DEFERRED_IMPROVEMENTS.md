# Deferred Improvements — Revisit After Full-Scale Training

Cross-checked against an external AI-generated research report ("Cell Tracking Research Plan.md",
provided by user 2026-07-12) comparing SOTA cell-tracking approaches against this pipeline. The
items below were reviewed and found real/plausible but are explicitly **not** implemented —
either because they need a real trained checkpoint to validate against, are large enough in scope
to warrant their own dedicated planning pass, or (item 0) were verified numerically to not
actually change anything given how this codebase's NMS is constructed.

## 0. Sub-voxel intensity-weighted centroid refinement — INVESTIGATED, FOUND TO BE A NO-OP

**Idea from doc:** `extract_peaks_from_volume()` (both `run_pipeline.py` and `src/train.py`)
computes each detected peak's centroid via `ndimage.center_of_mass(is_peak, labeled, ...)`,
weighting by the binary `is_peak` mask rather than the real intensity/probability values in
`vol`. The doc's recommendation: weight by `vol` instead, for a true sub-voxel, intensity-weighted
centroid.

**Why this was NOT adopted, verified not just assumed:** `is_peak = (vol == pooled) & ...`
requires each peak voxel's own value to exactly equal the max of its own local window. For two
*adjacent* voxels to both satisfy this simultaneously, the overlapping-window max-filter math
forces them to share the *exact same value* (if p1's value exceeded p2's, p2's window would pick
up p1's higher value and fail `vol==pooled`). This means any single connected component of
`is_peak` can only ever contain voxels with identical `vol` values — so weighting by `vol` instead
of the binary mask is mathematically guaranteed to produce the *same* centroid every time.
Confirmed empirically before committing anything: constructed a synthetic plateau with a
deliberately-planted slightly-lower interior voxel (0.94 surrounded by 0.95 neighbors) — the NMS
construction excluded that voxel from the connected peak region entirely rather than including it
with different weight; max centroid difference across the test was exactly `0.0`.

**If sub-voxel refinement is still wanted later:** it would need a fundamentally different peak
construction than the current strict-equality NMS — e.g. a local intensity-weighted refinement
computed over a small window *around* each detected peak (not just the tied-equal region), as a
genuinely separate post-processing step. Not implemented — this is a real design change, not a
one-line fix, and not worth the scope for an unvalidated hypothesis before real training exists.

## 1. Isolated/singleton node pruning before export

**Idea:** `export_submission()`/`convert_nx_to_tracksdata()` currently export every node the
tracker produces, including unlinked singletons (a node with no incoming or outgoing edges). Since
the competition's `adjusted_edge_jaccard` penalizes over-prediction based on `T_pred` (total
predicted node count) vs `T_true`, pruning likely-spurious isolated nodes before export could
reduce that penalty.

**Why deferred:** this is a genuine precision/recall trade-off, not a free win. Some isolated
nodes are legitimate — a cell at the very first or last timepoint, or a real birth/death event,
will correctly have `b_n=1`/`d_n=1` with no edges (this is the deliberately-allowed one-frame
singleton case documented in `src/tracker.py`'s constraint comments — see `CLAUDE.md`). Pruning
indiscriminately would cut real true positives along with noise, potentially *hurting*
`edge_jaccard` recall. This needs empirical tuning against real ground truth to find a threshold
(e.g. by detection confidence, or by requiring a minimum size) that actually helps net score —
which isn't meaningfully testable until a real, non-undertrained checkpoint exists (the current
checkpoint predicts zero detections everywhere, so there's nothing to prune/tune against).

**Where it would go:** a new filtering step between `STHypergraphTracker.solve_lineage()`'s output
and `convert_nx_to_tracksdata()`/`export_submission()` — after the ILP has already solved, so it
does NOT touch the `b_n+d_n<=1`-absence design decision (that's about ILP feasibility during
solving; this is about what to keep in the final export).

**Revisit:** PRD.md Phase 4 (metric-directed tuning), once a real checkpoint's predictions can be
measured against real ground truth.

## 2. Multi-scale Difference-of-Gaussians (DoG) classical detector

**Idea from doc:** add a classical DoG-based blob detector as a baseline/fallback.

**Why not adopted:** we already have a trained detector architecture (`UNet3D`, 2-channel
detection head). The doc's emphasis on classical baselines applies to teams without a learned
model yet — our actual gap is that the *current checkpoint* is severely undertrained (200 of
~14,751 real batches), not that we lack a learned architecture. Adding a parallel classical
detection path would be redundant scope; the real fix is finishing a full-scale training run.
Revisit only if, after real training, the learned detector's recall is still worse than a quick
classical baseline would predict — not a default assumption.

## 3. Ultrack-style joint segmentation+tracking ILP / Penumbria / SurfDist volumetric segmentation

**Idea from doc:** replace point-detection + separate tracking with a joint segmentation
hypothesis graph (Ultrametric Contour Maps) solved via ILP simultaneously with tracking
(Ultrack), or swap in a dedicated volumetric instance-segmentation network (Penumbria/SurfDist)
upstream of tracking.

**Why not adopted:** this is a full architecture replacement, not an incremental change — it would
mean redesigning the entire detection stage (currently point/centroid detection via UNet3D +
NMS) around segmentation masks instead. High implementation risk and effort, uncertain payoff
before our current, much simpler architecture has even been validated with a real trained
checkpoint. Also worth noting: several of the doc's specific architecture citations (Penumbria,
SurfDist, ITEC) are unfamiliar and unverified from here — treat as informational leads to
investigate later, not established fact.

**Revisit:** only if a real trained checkpoint's score plateaus well below the competition's
target/leaderboard baselines, as a Phase 3/4-scale rearchitecture effort with its own dedicated
plan.

## Net outcome this session

None of this doc's specific recommendations were adopted as code changes. The one
initially-promising candidate (item 0, sub-voxel refinement) was verified numerically to be a
no-op before being committed, and reverted rather than shipped. The real, primary value from this
cross-check was independent external confirmation that bug 1.3 (anisotropy ratio vs. physical
microns, fixed earlier this session in `f5fd65c`) was correctly diagnosed and fixed — see the
"Independent Technical Code Review" section of the source document, which describes the identical
bug and an equivalent fix.
