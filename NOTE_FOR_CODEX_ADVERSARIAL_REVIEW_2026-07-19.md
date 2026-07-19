# NOTE FOR CODEX — Adversarial Review Handoff
**Date:** 2026-07-19  
**Project:** `st_act_pipeline`  
**Competition:** Kaggle `biohub-cell-tracking-during-development` (3D+time zebrafish cell tracking)  
**Score metric:** `adjusted_edge_jaccard + 0.1 × division_jaccard`  
**Floor to beat:** 0.763 (public classical baseline). Winner: 0.875.

---

## Context

This PR contains:
1. `dry_run_bugfix_impact.py` — original dry-run script (do not modify)
2. `dry_run_investigation.py` — fixed working version that runs end-to-end
3. `CODE_REVIEW.md` — full code review (claude code-review + Linus Torvalds taste principles)

The goal is for Codex to adversarially review the bug findings and the empirical results, and write back feedback as PR comments.

---

## Competition Pipeline (Brief)

- Raw data: `data/staging/train/{id}.zarr` — Zarr v3, shape T×Z×Y×X, uint16
- Voxel scale: z=1.625µm, y=x=0.40625µm (anisotropy ratio 4:1:1)
- GT: `data/staging/train/{id}.geff` — read via `tracksdata.graph.IndexedRXGraph.from_geff()`
- Pipeline: normalize → UNet3D → detection heatmap → NMS → ILP tracker → prediction graph → score
- Over-prediction penalty: `adj = max(0, J × (1 - 0.1 × (T_pred - T_true) / T_true))`
- T_true for sample `44b6_0b24845f`: **32,795** (from `.geff` metadata)

---

## Bugs Identified

### Bug 1 — Wrong normalization quantiles (`src/data_loader.py`)
- **Current:** `q_lo = q0.10`, `q_hi = q0.90`, clipped to `[0, 1]`
- **Reference (host implementation):** `q_lo = q0.001`, `q_hi = q0.999`, clipped to `[0, 4.0]`
- **Effect:** UNet3D receives wrong input distribution vs what the reference architecture was designed for. Any threshold tuned against reference normalization is meaningless against current normalization.

### Bug 2 — Detection threshold not recalibrated across normalization codomains
- **Current:** threshold=0.4 on `[0,1]` data = top-60th-percentile filter
- **After norm fix:** threshold=0.4 on `[0,4]` data = ~10th-percentile filter (almost all voxels pass)
- **Measured effect:** At thr=0.4 with reference norm → 11,759 peaks across 15 frames (capped to 800/frame). At thr=1.5+ → 0 peaks (exceeds 99.9th percentile). No working threshold exists for raw-intensity NMS.

### Bug 3 — `add_node_attr_key` called with 2 args, needs 3 (in dry_run_bugfix_impact.py)
- **Current:** `g.add_node_attr_key("z", 0.0)` — raises TypeError
- **Fix:** `g.add_node_attr_key("z", pl.Float64, 0.0)` — correct 3-arg API
- Confirmed against `src/prediction_graph.py:56`

### Bug 4 — `g.add_edge()` called with 2 args, needs 3
- **Current:** `g.add_edge(nid_a, nid_b)` — raises TypeError
- **Fix:** `g.add_edge(nid_a, nid_b, {})`

### Bug 5 — `run_scenario()` signature mismatch in dry_run_bugfix_impact.py
- `vol` positional slot conflicts with `vol=vol` keyword arg when `*args` is unpacked
- **Fix:** reorder signature so `vol` comes after all scenario-tuple positional args

---

## Empirical Dry Run Results (Verified on Real Data)

**Sample:** `44b6_0b24845f`, t=20..34, 15 frames, 1 GT node/frame  
**Script:** `dry_run_investigation.py`

| Scenario | Peaks | TP | edge_J | adj_J |
|---|---|---|---|---|
| BEFORE-A curr[0,1] thr=0.30 | 12,000 | 0 | 0.0000 | N/A |
| BEFORE-B curr[0,1] thr=0.40 (P1 baseline) | 12,000 | 0 | 0.0000 | N/A |
| BEFORE-C curr[0,1] thr=0.50 | 12,000 | 0 | 0.0000 | N/A |
| AFTER-1a ref[0,4] thr=0.40 | 12,000 | 0 | 0.0000 | N/A |
| AFTER-1b ref[0,4] thr=1.00 | 11,759 | 0 | 0.0000 | N/A |
| AFTER-1c ref[0,4] thr=1.50 | 0 | 0 | 0.0000 | N/A |
| AFTER-1d ref[0,4] thr=2.00 | 0 | 0 | 0.0000 | N/A |
| **Partial-window GT-centroid + greedy** *(not an upper bound)* | **15** | **14** | **0.2857** | **N/A** |

**BEFORE best edge_J: 0.0000 | AFTER best edge_J: 0.0000 | Delta: +0.0000**

> **adj_J = N/A for all rows.** This script evaluates only t=20..34 (15 frames, a
> partial window). The competition adjustment formula requires `estimated_number_of_nodes`
> for the **full sample** (`T_true` = 32,795). Feeding that value into a 15-frame
> prediction makes `T_pred ≪ T_true`, turning the over-prediction *penalty* into an
> under-prediction *reward* (~1.10× multiplier). The previously reported `adj_J = 0.3143`
> was produced by this invalid combination and is not a valid competition-comparable score.
> No window-specific `estimated_number_of_nodes` estimate exists in this repository;
> `gt_n_win` (20 sparse labels) is not a valid substitute.
>
> **node_recall = N/A for all rows.** The previously reported value divided
> `edge_tp` by `gt_n_win` — matched edges ÷ GT node count — which is dimensionally
> invalid. No correct node-recall computation is available for this partial-window
> diagnostic.

---

## Root Cause of Zero Score (Empirically Confirmed)

At t=20:
- GT cell: voxel `(z=56, y=167, x=214)`, intensity = **1,362**
- Brightest voxel in GT cell's NMS kernel: `(z=59, y=180, x=227)`, intensity = **1,684**
- Distance: **8.9µm** — outside the 7µm match gate

GT-labeled cell centroids are NOT local intensity maxima. Light-sheet microscopy background scatter and neighbouring structures are routinely brighter than the annotated centroid. Raw NMS on pixel intensity cannot detect GT cells at any threshold under any normalization.

**Implication:** Bug fixes 1 and 2 are real and must land in `src/data_loader.py` before the next training run. Their effect is exclusively on what the UNet3D sees during training — not on any heuristic detector.

---

## Questions for Codex — Adversarial Review

1. **Is the normalization bug already fixed in `src/data_loader.py`?**  
   Check `AnisotropicZarrLoader`'s normalize path for actual quantile values. The dry-run script computes its own quantiles independently — it is NOT evidence the training pipeline is fixed.

2. **Is the correct normalized input reaching `src/train.py` and `src/targets.py`?**  
   The heatmap GT generation and UNet3D forward pass must see `[0, 4]`-clipped data, not `[0, 1]`.

3. **Does the detection threshold in `run_pipeline.py` / inference kernel match the reference normalization codomain?**  
   Codex confirmed: in production validation/inference (`src/train.py:1108`,
   `src/submission_pipeline.py:135-139`), sigmoid is applied to detection logits before
   NMS thresholding. Production `detection_threshold` must remain in `[0,1]`; a value
   of 1.0–1.5 would yield zero detections or be rejected by the manifest validator.
   The original 1.0–1.5 recommendation conflated raw-input intensity (heuristic path
   only) with post-sigmoid model output probability — that recommendation was wrong and
   is retracted. Threshold calibration for the trained model must happen on post-sigmoid
   heatmap outputs after retraining under the chosen normalization.

4. **Was `adj_J = 0.3143` a valid "upper bound"? (Corrected: No)**
   Codex confirmed this was arithmetically reproducible but fundamentally wrong. Feeding
   a partial-window `T_pred` (≈15 nodes) against full-sample `T_true` (32,795) makes
   `(T_pred − T_true) / T_true ≈ −1`, which turns the over-prediction penalty into a
   ~1.10× reward. The label "upper bound" was also false — it was a greedy-linker
   diagnostic on one window, not a mathematical ceiling on full-sample or competition
   performance. **Fixed in `dry_run_investigation.py`**: `per_sample_metrics()` is now
   called with `n_total=float("nan")` (the documented API contract for "unavailable"),
   which returns `adj_edge_jaccard=NaN` for all scenarios. The row is relabelled
   "Partial-window GT-centroid + greedy-link diagnostic (NOT an upper bound)".
   `gt_n_win` was deliberately NOT used as `n_total` — the sparse annotated label count
   is not the `estimated_number_of_nodes` the formula requires.

5. **Has any checkpoint been trained with the wrong (current) normalization?**  
   If yes, applying the fix at inference without retraining would degrade, not improve, performance.

6. **Are there other call sites in the codebase that apply normalization independently of `AnisotropicZarrLoader`?**  
   e.g., `kaggle_kernel_inference/inference_kernel.py`, `run_pipeline.py`, any standalone eval scripts. Each must use the same quantile window.

---

## Key Files for Context

| File | Contents |
|---|---|
| `PRD.md` | Competition spec, phased roadmap, entry/exit criteria |
| `REFERENCE_IMPLEMENTATION.md` | Host's exact architecture and normalization spec |
| `src/data_loader.py` | `AnisotropicZarrLoader` — normalization path is primary suspect |
| `src/train.py` | `TrainingLoop` — consumes loader output |
| `src/targets.py` | Heatmap GT generation — input-distribution sensitive |
| `CODE_REVIEW.md` | Full prior review output (do not overwrite) |
| `dry_run_investigation.py` | Runnable verification script |

---

## Instructions to Codex

- Do not modify `dry_run_bugfix_impact.py` (it is the preserved original)
- Write adversarial findings as PR comments on this PR
- For each finding: state the file + line, the claim being challenged, and the evidence
- If a bug is confirmed fixed already, say so explicitly with the line reference
- If a bug is genuinely present and unfixed, propose the minimal surgical fix
