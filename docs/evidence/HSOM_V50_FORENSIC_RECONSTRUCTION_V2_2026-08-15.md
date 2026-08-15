Status: PENDING_INDEPENDENT_REVIEW

# HSOM BIOKaggle training-run-v50 Forensic Reconstruction (v2)
## Corrected, Evidence-Reconciled Reconstruction — Supersedes v1

**Investigation Date:** 2026-08-15
**Run Identity:** training-run-v50 (Kaggle kernel `drbhatiasanjay/st-act-gpu-smoke-test`, v50)
**Deployed SHA:** bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c
**Result:** val_score = 0.0019855762417342647 (0.001986), predicted_nodes_total = 554,366, is_structural_zero = False
**Status:** COMPLETE (20483.1s / 5.69h wall-clock)

**Supersedes:** `docs/evidence/HSOM_V50_FORENSIC_RECONSTRUCTION_2026-08-15.md` (v1, commit `6b31589`), which received an independent Codex review verdict of **REJECT** (5 blocker + 8 major corrections; `docs/evidence/HSOM_V50_CODEX_INDEPENDENT_REVIEW_2026-08-15.md`). All disputed claims between v1 and Codex were independently re-derived from primary evidence in this session — see `docs/evidence/HSOM_V50_RECONCILIATION_2026-08-15.md` for the full ledger and exact reproduction commands. Every numeric correction below was reproduced directly against git objects, the raw log, checkpoint hashes, the vendored evaluator source, and GEFF metadata — not copied from either v1's or Codex's stated conclusions.

---

## Changes from rejected v1

1. **Fallback-rate denominator corrected: 14,058, not 1,626.** `validate_epoch()` makes two `_peaks_for_channel()` calls per validation batch (channel 0, channel 1), and there are 7,029 validation batches (log: `Val loader batches: 7029`), giving 14,058 total detection calls. The 1,626 "severe under-confidence" warnings are **11.566368%** of that denominator, not 100%.
2. **Fixed detection_threshold=0.5 fired for the majority of calls, not zero times.** 12,432 of 14,058 calls (88.43%, by arithmetic complement) took the silent fixed-threshold path — the source code logs nothing when `0 < positive_fraction ≤ max_positive_fraction`. Validation-phase log lines directly corroborate this: 1,365 of 1,405 sampled validation-batch progress lines show `sigmoid_max > 0.5`.
3. **Effective adaptive-threshold values are fully recoverable, not "truncated/not recoverable."** All 1,626 values parse from the log: min 0.320821, max 0.420600, mean 0.383704879458793.
4. **The `~54–100K` ground-truth node estimate and the resulting "5–10× overprediction" conclusion are retracted.** No exact 71-sample `T_true` sum exists in any saved artifact. The two locally staged samples' exact `estimated_number_of_nodes` values (25,755 and 32,795, summing to 58,550) already exceed v1's own lower bound from just 2 of 71 samples, and on both of those samples v50's own predicted node counts (1,645 and 25,354) are *below*, not above, the sample's `T_true`. The full 71-sample direction is unresolved, not "5–10x over."
5. **`76bf901`'s full SHA, file scope, and attributed effect are corrected.** Full SHA is `76bf901126df7a70521be3b4923602a77188d565` (v1's stated `...abb836c5874d6107db431077ade57f8bf` suffix does not exist as a git object — it is spliced from `eb31af9`'s suffix). It changes `src/targets.py` + `tests/test_targets.py`, not `src/train.py`. It normalizes an existing weighted loss from `mean()`-style to sum-of-weights normalization; adaptive per-batch class weighting itself was introduced earlier, by `eef5700c24a9549579b248f6e07b178e371c2856`.
6. **Quantile-normalization fix's deployed ancestor and file scope corrected.** The deployed ancestor is `ba1bdb4a434d925c9e54cc608c039d38e93cd4ef`; `2a263c2` is a parallel cherry-pick **not** in `bc989ed`'s ancestry (v1 treated them as interchangeable). Files touched are `src/data_loader.py` + `tests/test_data_loader_real.py`, not `src/submission_pipeline.py`.
7. **Pre-v50 intervention chain expanded.** The LR change (1e-2→3e-3, `872743646b646f9ec117d7d2b95c75fd98153917`) and warmup enable (warmup_steps=300, `1c5c50f16efbb8c3c0ba4a482f50120925f37d44`) are both confirmed ancestors of `bc989ed` and are both referenced in v1's own configuration table (§D.2) without being named as fixes. The full confirmed chain is now 7 non-sync commits deep (see §C).
8. **Causal "these fixes solved collapse" language removed.** v50 is one combined run with ≥7 concurrent ancestor changes and no ablation. It observationally demonstrates that collapse is reversible; it does not causally isolate which change(s) mattered.
9. **Base edge Jaccard's semantic corrected: it is Jaccard (TP/(TP+FP+FN)), not recall.** v1's "0.18% of true edges are correctly predicted" framing is a recall-style reading of a Jaccard value.
10. **Adjusted-vs-base score relationship corrected — the direction is reversed from v1's account, and the mechanism is corrected again here.** v1 called the adjusted score's increase over base "counterintuitive" and left it unresolved. The correct mechanism, verified against `bc989ed:src/tracking_cellmot/metrics.py`: base micro-Jaccard (`sum(TP_i) / sum(w_i)`, `w_i = TP_i+FP_i+FN_i`) is algebraically identical to the `w_i`-weighted mean of the per-sample Jaccards `J_i` — base and adjusted use the *same* weighting, not different aggregation schemes. Adjusted exceeds base because the per-sample adjustment multiplier `(1 − 0.1·r_i)` is applied to each `J_i` *before* that shared weighted aggregation, and this multiplier exceeds 1 whenever that sample's predicted node count undershoots `T_true`.
11. **Division-Jaccard semantics corrected, and further marked UNRESOLVED.** `0.0` does not mean "possibly perfect division prediction" (perfect would be `1.0`); it means either zero true-positive divisions with a positive denominator, or a dropped (not zero-padded) NaN/no-division case. Confirmed directly from `bc989ed:src/train.py` (`validate_epoch()`, ~L1535–1541): the code sanitizes *any* NaN metric — including `division_jaccard` — to `0.0` before logging, and neither `division_tp`/`division_fp`/`division_fn` counts are persisted anywhere in v50's saved artifacts. This means the two cases are indistinguishable from the persisted `0.0` alone; which one actually occurred for v50 is UNRESOLVED, not resolved to either reading.
12. **Oracle-evaluation status corrected from "in progress / not yet completed" to "completed, 2-of-71-samples, checkpoint-identity-confirmed."** `oracle_check_training_run.log` was on disk with the exact v50 checkpoint hash **5.5 hours before v1's own commit timestamp**. The result exists, but only covers 2 of 71 validation samples and cannot by itself establish a full-fold bottleneck.
13. **44/216 "no matching nodes/edges" warning counts re-scoped.** These are `tracksdata` log records emitted once per whole match-operation that returns zero IDs, not once per individual unmatched GT node/edge. v1's derived "~0.6%"/"~1–2% unmatched" percentages are dropped as built on the wrong unit.
14. **Total-warning-count discrepancy resolved as two correct, differently-defined numbers.** 1,886 (v1's figure) is the exact count of project-logger `WARNING`-level records; 1,890 is the exact count of any log line containing "warning" case-insensitively, which additionally includes 4 unrelated third-party `FutureWarning`/`SyntaxWarning` lines from `dask`/`skimage`/`mistune`/`nbconvert`. Both are stated with their precise definitions below.
15. **H1–H5 rankings reconstructed from the corrected evidence** (§K) rather than carried forward from v1's now-invalidated 100%-fallback / 5–10×-overprediction framing.
16. **EV-V50 records reconstructed from scratch** (§V), reflecting corrected observations and, where evidence doesn't establish a causal defect/bottleneck, explicit `UNRESOLVED` status rather than `LIKELY`/`PROVEN`.

What v1 got right and is preserved unchanged: the raw score/count/checkpoint identity numbers (§F.1, F.2, F.5), the deployed-SHA verification (§E.1), the structural-health verdict (§G — zero errors/tracebacks/crashes, all enumerated fallback counters zero), the raw 1,626/44/216 log-line counts themselves (only their semantic denominators/units were wrong), and the eb31af9 instrumentation-commit code review (§Q).

---

## A. Executive Verdict

**training-run-v50 is the first non-structural-zero validation result recorded in the project's available run registry** (all prior registry rows scored 0.0). The score of 0.001986 is far below the 0.763 baseline, but represents real, non-degenerate output: 554,366 predicted nodes, 388,859 predicted edges, zero crashes, and a non-collapsed model (train_loss=2.226, sigmoid outputs spanning a real dynamic range up to 0.68 during validation, not stuck near 0).

Beyond that observational finding, most of v1's causal and quantitative interpretation does not survive independent re-verification against primary evidence:

- The fixed `detection_threshold=0.5` was **not** universally miscalibrated — it fired successfully (silently, no warning) for an estimated 88% of detection calls. Only 11.57% of calls hit the zero-positive-fraction fallback path.
- The "5–10× node overprediction" claim used a self-estimated ground-truth denominator that is contradicted by the two exact values actually available.
- An Oracle evaluation against the exact v50 checkpoint had already completed (2 of 71 samples) before v1 was written, and was misreported as pending.
- The specific commit identified as the "detection-head loss normalization fix" has a fabricated full SHA and wrong file attribution.
- The claim that "these four fixes solved collapse" is causally unsupported — collapse-relevant code changed across at least 7 confirmed ancestor commits in one combined, unablated run.

What remains solidly established: v50 ran clean, deployed the SHA it claims, evaluated the full 71-sample fold, and produced a real (if extremely low) nonzero score. Why the score is so low — undertraining, calibration, extraction policy, representation limits, or some combination — remains genuinely open, and is *more* open than v1 stated, because the strongest single piece of "smoking gun" evidence v1 offered (100% fallback activation) does not hold up.

---

## B. Current Repository and Execution State

*(Unchanged from v1 — independently re-confirmed, not disputed by Codex.)*

### B.1 Git State (as of 2026-08-15, this reconciliation)

| Property | Value |
|----------|-------|
| **Local Branch** | gpu-sanity-gate-wave2-v2 |
| **origin/master HEAD** | bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c (v50's deployed SHA) |
| **v1 report commit** | 6b315891a80fc2530d79ce29f8bff861b293e919 |
| **Codex review commit** | 6ebd38ce87f69f6b2c0bc32e69f790849da86c20 (branch `origin/codex/v50-independent-review`) |

### B.2–B.4

Unchanged from v1 — project-state documentation, run-registry path, and training-output artifact inventory were not disputed by Codex and are re-confirmed present in this session (`kaggle_runs.db`, `C:\Users\hemas\Downloads\kaggle_train_run_v50_output\{full_log.json, training_log.csv, training_progress.json, checkpoints\}`).

**One correction to B.3's registry description:** the registry's free-text `notes` field for `training-run-v50` itself repeats the incorrect "1626/1626 = 100%, raw sigmoid map never exceeded 0.5 anywhere" claim (added in a post-hoc note, timestamped after the raw run but apparently before this reconciliation). This note is **not** independent corroboration of the 100%-fallback claim — it is secondary narrative derived from the same flawed reconstruction being corrected here, and should not be treated as a second, independent source. It is superseded by this document; a correction to the registry note itself is a separate, explicitly-authorized action, not performed here (this reconciliation makes no database writes).

---

## C. Recovered Pre-v50 Fixes and Related Changes (corrected)

### C.1 Confirmed pre-v50 intervention chain

All entries independently confirmed as ancestors of `bc989ed` via `git merge-base --is-ancestor`, in the order they were authored. **Row order and file scopes below are corrected from an earlier draft of this table**, which misordered `76bf901` after the LR/warmup pair and understated the file scope of `eef5700`, `4a26f02`, and `ab5fcc3`; every commit/date/file below was independently re-verified via `git log --format='%H|%ad|%s' --date=iso-strict -1 <sha>` and `git diff-tree --no-commit-id --name-only -r <sha>` in this session:

| # | Commit (full SHA) | Authored | Message | Files | Effect |
|---|---|---|---|---|---|
| 1 | `eef5700c24a9549579b248f6e07b178e371c2856` | 2026-07-13T18:29:59+05:30 | fix(03-08): adaptive per-batch class-imbalance weighting in DetectionLoss | src/targets.py, src/train.py, tests/test_targets.py, DEFERRED_IMPROVEMENTS.md (+ kaggle_src_dataset/ mirror sync) | Introduces adaptive per-batch weighting (root cause analysis of v30's structural-zero run) |
| 2 | `76bf901126df7a70521be3b4923602a77188d565` | 2026-07-13T19:58:54+05:30 | fix(03-11): CRITICAL -- DetectionLoss normalized by numel, silencing cell-batch gradients | src/targets.py, tests/test_targets.py | Normalizes DetectionLoss from `mean()`-style to sum-of-weights normalization. **Precedes the LR/warmup pair below by ~19h/~30h respectively — corrected from an earlier draft that placed it after them.** |
| 3 | `872743646b646f9ec117d7d2b95c75fd98153917` | 2026-07-14T14:20:54+05:30 | fix(03-23): learning_rate 1e-2 -> 3e-3, geometric mean of two verified failures | kaggle_kernel/train_kernel.py | LR set to the value v50 actually trained with |
| 4 | `1c5c50f16efbb8c3c0ba4a482f50120925f37d44` | 2026-07-15T00:13:48+05:30 | feat(03-26): enable warmup_steps=300, targets v48's fast early-collapse pattern | kaggle_kernel/train_kernel.py | Warmup set to the value v50 actually trained with |
| 5 | `4a26f021e1d9e549f3d2e5393d036d6a59726ddb` | 2026-07-15T01:12:05+05:30 | fix(03-27): symmetric adaptive detection threshold, zero-detection bug | src/train.py, evaluate_checkpoint.py, kaggle_kernel_inference/inference_kernel.py — **does NOT touch `run_pipeline.py`, corrected from an earlier draft** | Adds the symmetric (both-directions) adaptive-threshold fallback used by v50 |
| 6 | `ab5fcc305cc2aa360486a6b6bcf2c0cd346443a1` | 2026-07-15T08:34:55+05:30 | fix(03-28): match DetectionLoss adaptive weighting to reference implementation | src/targets.py, src/train.py, tests/test_targets.py | Re-aligns adaptive weighting formula to reference implementation |
| 7 | `ba1bdb4a434d925c9e54cc608c039d38e93cd4ef` | 2026-08-14T20:27:13+05:30 | fix(data): correct quantile normalization to q0.001/q0.999 -> [0,4] | src/data_loader.py, tests/test_data_loader_real.py | Corrects data-loader quantile clipping range |
| — | `bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c` | — | (mirror sync of PR #11) | kaggle_src_dataset mirror | Declared v50 deployment SHA |

**Not in this ancestry chain:** `2a263c273bd62f2b60c709f7d8557c15c215378a` — a parallel cherry-pick of the same data-loader change that exists on the review branch but is not an ancestor of `bc989ed`. It should not be cited as evidence about what v50 ran.

**Status of causal attribution:** All 7 numbered commits above are confirmed present in v50's deployed tree. Whether any one of them (or which subset) is responsible for the transition from prior structural-zero runs to v50's nonzero score is **not established** — no run in the registry isolates a subset of these changes, and v50 is a single, unablated, combined run.

---

## D. training-run-v50 Execution Identity

*(Unchanged from v1's §D — deployment provenance, hyperparameters, data/validation scope, runtime, and checkpoint identity were not disputed and are re-confirmed here.)*

Confirmed directly in this session:
- Checkpoint `epoch_1_val_score_0.0020.pt` SHA-256 `8a788a192725d80a39c6ea4a5a4f74ade67cf4c259fa67cc943d9ede15c25092` (independently re-hashed).
- `checkpoint_manifest.json`: `training_code_sha=bc989ed...`, `validation_is_full_fold=true`, `validation_samples_evaluated=71`, `validation_samples_total=71`.
- `kaggle_runs.db` row for `training-run-v50`: matches all v1-reported fields; several telemetry fields (`gpu_name`, `cuda_available`, `max_sigmoid_*`, `peak_gpu_memory_*`) are `NULL`, confirmed pre-`eb31af9` deployment gap.

---

## E. Compliance Audit: Checklist Steps 1–4

Unchanged from v1 — all four steps re-verified independently in this session and confirmed: deployed SHA matches across log/progress/manifest/registry; heartbeat written once, complete; CSV structural-zero columns all show non-zero signal and zero fallback failures; no `RuntimeError`/circuit-breaker firing, full validation completed. **Not disputed by Codex; no changes.**

---

## F. Verified v50 Observations (corrected)

### F.1 Score and Structural Health — unchanged, re-confirmed

| Observation | Value | Classification |
|---|---|---|
| val_score | 0.0019855762417342647 (0.001986) | CONFIRMED |
| is_structural_zero | False | CONFIRMED |
| predicted_nodes_total | 554,366 | CONFIRMED |
| predicted_edges_total | 388,859 | CONFIRMED |

### F.2 Fallback and Failure Counters — unchanged, re-confirmed

heatmap_failures=0, edge_target_failures=0, edge_loss_failures=0, eval_failures=0 (all training/eval-side structural counters; distinct from the detection-threshold fallback discussed in F.4).

### F.3 Log Integrity — unchanged, re-confirmed

38,308 total log records; 0 ERROR; 0 Traceback; 0 CRITICAL.

### F.4 Detection and Adaptive Fallback Behavior — CORRECTED

| Observation | v1 claim | Corrected value | Evidence |
|---|---|---|---|
| Total detection calls (semantic denominator) | 1,626 (treated as exhaustive) | **14,058** (7,029 batches × 2 channels) | `bc989ed:src/train.py` L1306–1336; log `Val loader batches: 7029` |
| Zero-positive-fraction ("severe under-confidence") calls | 1,626 (=100% of denominator) | 1,626 (**=11.566368%** of 14,058) | Exact log-substring count, independently re-run |
| Channel split of the 1,626 | not reported | ch0=182, ch1=1,444 | Regex `ch=(\d+)` on warning lines |
| High-positive-fraction ("undertrained-model miscalibration") calls | not checked | **0** | Exact log-substring count |
| Fixed-threshold (silent, no warning) calls | 0 (claimed) | **12,432** (=88.433632%, arithmetic complement; not separately logged) | 14,058 − 1,626; corroborated by 1,365/1,405 sampled validation batches showing `sigmoid_max > 0.5` |
| Effective adaptive-threshold values | "not fully recoverable" | count=1,626, min=0.320821, max=0.420600, mean=0.3837048794587946 | Regex parse of all 1,626 log lines |

### F.5 Training Loss Trajectory — unchanged, re-confirmed

train_loss (final) = 2.226297828412056. Batch-level loss spikes (~batch 2480/2510) remain not independently re-verified in this session either (would require full batch-log parse; not attempted here as it is orthogonal to the corrected claims).

### F.6 Warnings (Non-Fatal, Caught) — CORRECTED (unit, not count)

| Warning Type | Count | Correct interpretation |
|---|---|---|
| "No matching nodes found" | 44 | One log record per whole match-operation returning zero IDs (`tracksdata/graph/_base_graph.py:1256`), **not** one per individual unmatched GT node. No valid "% unmatched" figure can be derived from this alone. |
| "No matching edges found" | 216 | Same semantics, `_base_graph.py:1281`. |
| Total records matching `"WARNING"` (exact case, project-logger records) | 1,886 | Confirmed exact. |
| Total records matching `"warning"` (case-insensitive, any source) | 1,890 | Confirmed exact; the +4 are unrelated `FutureWarning`/`SyntaxWarning` lines from `dask`/`skimage`/`mistune`/`nbconvert`, not pipeline-logger records. |

---

## G. Structural-Health Verdict

**Unchanged: HEALTHY.** Not disputed by Codex. Deployed SHA verified, zero crashes/tracebacks/criticals, full 71-sample validation completed, all training/eval-side fallback counters zero. The scientific question of why the score is low is orthogonal to this verdict, as v1 also stated.

---

## H. Telemetry Completeness Matrix

Unchanged from v1 — not disputed by Codex. The `eb31af9` instrumentation gap analysis (GPU name, CUDA availability, max_sigmoid stats, peak GPU memory all null pre-instrumentation) stands as originally reported.

---

## I. Threshold and Extraction Behavior (corrected)

### I.1 Detection Threshold Logic — corrected against primary source

Confirmed directly from `bc989ed:src/train.py` (`extract_inference_peaks()`, lines 165–204):

```python
vol_np = detection_probs[0, channel].cpu().numpy()
threshold = hyperparams['detection_threshold']          # 0.5
positive_fraction = float((vol_np > threshold).mean())
max_positive_fraction = hyperparams.get('max_positive_voxel_fraction', 0.005)

if positive_fraction > max_positive_fraction:
    # "undertrained-model miscalibration" warning path -- 0 occurrences in v50
    adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
    threshold = max(adaptive_threshold, threshold)
elif positive_fraction == 0.0:
    # "severe under-confidence" warning path -- 1,626 occurrences in v50
    adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
    threshold = adaptive_threshold
# else: 0 < positive_fraction <= 0.005 -- SILENT fixed-threshold path, NO log line.
#       This is the majority case in v50 (12,432 of 14,058 calls, by complement).
return extract_peaks_from_volume(vol_np, threshold=threshold, ...)
```

**Key correction to v1's §I.1:** v1 presented only the two warning branches as if they were the only two possible outcomes ("Fixed-Threshold Path" vs. "Adaptive Fallback Path"), when the actual fixed-threshold path is the silent `else` branch that fires when a *small but nonzero* fraction of voxels exceeds 0.5 — this is the branch that fired 88% of the time, and it is entirely unlogged, which is why v1 (reading only the warning lines) missed it.

### I.2 Effective Adaptive Threshold Values — corrected from "not recoverable"

count=1,626, min=0.320821, max=0.420600, mean=0.3837048794587946. Fully recovered by parsing all `severe under-confidence` log lines' `threshold={:.6f} instead` suffix.

### I.3 Fixed-Threshold Activation Rate — corrected

**Answer:** YES, for 12,432 of 14,058 calls (88.43%). v1's "NO (with high confidence)" answer and its "mathematical certainty" derivation both rested on treating 1,626 as the full call count, which is contradicted directly by the source code's silent branch.

### I.4 Multi-Channel Processing Structure — corrected detail

v1's inferred loop structure was directionally right (per-batch, per-channel) but its accounting was wrong: it is not "813 pairs × 2 channels = 1,626" — it is "7,029 batches × 2 channels = 14,058 calls," of which 1,626 hit the zero-positive branch. Read directly from source (§C.1 above / R-01 in the reconciliation ledger), not inferred.

### I.5 Extraction Failure Modes — corrected interpretation

Given the corrected 88%/12% split, the fact that the pipeline produces 388,859 edges from 554,366 nodes with zero enumerated fallback failures is no longer surprising evidence of "the fallback path always succeeding" — it is largely the **ordinary fixed-threshold path** succeeding, with the adaptive fallback contributing to a real but minority (11.57%) subset of detection calls.

---

## J. Scientific Interpretation (corrected)

### J.1 What v50 Proves

**Unchanged, still CONFIRMED (high confidence), restricted to the observational claim:** v50 achieved a nonzero, non-collapsed validation score (0.001986) after historical runs scored exactly 0.0. This is real signal, not an artifact of data corruption, early termination, or masked fallback failures (§G).

**Corrected/removed:** the causal claim that specific named fixes "solved" the collapse. See §C — 7 concurrent ancestor changes, one unablated run; causal attribution is unsupported (was UNSUPPORTED even at v1's more modest hedge level, and remains so here).

### J.2 What v50 Does NOT Prove — expanded

Everything v1 listed here remains not proven, plus:

- **That predicted nodes exceed the true full-fold scoring target.** v1 asserted this as established fact ("5-10x over-prediction... False-positive rate is likely very high"); it is now RETRACTED (§ Changes from rejected v1, item 4). The two exact locally-available T_true values (25,755 / 32,795) both *exceed* v50's own predicted counts on those same two samples (1,645 / 25,354).
- **That the detection-threshold fallback is a/the primary bottleneck.** With the correct 11.57% fallback rate, the "universal miscalibration" framing that drove this hypothesis in v1 no longer holds; whether the fallback path (minority) or fixed-threshold path (majority) contributes more to the low score is unresolved.

### J.3 Score Decomposition — corrected

**Reported Components (unchanged, re-confirmed exact):**
- val_edge_jaccard (base, micro-averaged) = 0.001845 (rounded; exact unrounded value not serialized)
- val_adjusted_edge_jaccard (final score) = 0.0019855762417342647
- val_division_jaccard = 0.0 (as logged; UNRESOLVED whether this is a genuine zero addend or a NaN case sanitized to 0.0 and dropped from the score sum — see below)

**Corrected formula and semantics**, read directly from `bc989ed:src/tracking_cellmot/metrics.py`:

```text
Per sample i:
  J_i   = TP_i / (TP_i + FP_i + FN_i)                       # Jaccard, not recall
  r_i   = (N_pred_i - T_true_i) / T_true_i                   # total_node_ratio
  adj_i = max(0, J_i * (1 - 0.1 * r_i))                       # ADJUSTMENT_ALPHA = 0.1

Aggregation:
  w_i               = TP_i + FP_i + FN_i
  base edge Jaccard = sum(TP_i) / sum(w_i)                     # == sum(w_i * J_i) / sum(w_i) algebraically
                                                                #    i.e. the w_i-weighted mean of per-sample J_i
  adjusted Jaccard  = sum(w_i * adj_i) / sum(w_i)              # the SAME w_i-weighted mean, applied to adj_i instead of J_i

  If total division denominator == 0 across all samples:
      division term is DROPPED (not added as 0): score = adjusted Jaccard
  else:
      score = adjusted Jaccard + 0.1 * division_jaccard         # SCORE_DIVISION_WEIGHT = 0.1
```

**Why adjusted (0.001986) > base (0.001845), corrected again:** base and adjusted use the *same* `w_i`-weighted aggregation, not "base minus a penalty" and not two structurally different aggregation schemes — base is algebraically the `w_i`-weighted mean of the raw per-sample `J_i`, exactly as adjusted is the `w_i`-weighted mean of the per-sample `adj_i`. The two scores differ only because the per-sample adjustment `adj_i = max(0, J_i · (1 − 0.1·r_i))` is applied *before* that shared aggregation, not because of a difference in how the aggregation itself weights samples. Whenever a sample underpredicts (`N_pred_i < T_true_i`, i.e. `r_i < 0`), its per-sample multiplier `(1 − 0.1·r_i)` exceeds 1, pulling that sample's `adj_i` above its own `J_i`; enough weighted mass on such samples pulls the aggregate `adjusted` value above the aggregate `base` value even though both aggregates use identical weights. v1's own two local-sample Oracle evidence is consistent with underprediction being present on at least one of the two locally checkable samples.

**Division term corrected, and marked UNRESOLVED:** `0.0` does not mean, and cannot mean, "possibly perfect division prediction" — perfect division prediction would score `division_jaccard = 1.0`. Beyond that, the persisted `0.0` is genuinely ambiguous between two structurally different cases that saved evidence cannot distinguish: (a) a positive division denominator with zero true-positive divisions — a real `division_jaccard = 0.0`, included in the score sum as `+0.1 × 0.0`; or (b) zero divisions present anywhere across the 71 samples (`division_total == 0`), where `summarise()` returns `division_jaccard = float('nan')` and drops the term from the score sum entirely (`score = adjusted Jaccard`). Confirmed directly from `bc989ed:src/train.py` (`validate_epoch()`, ~L1535–1541): the code explicitly replaces *any* NaN metric value — including `division_jaccard` — with `0.0` before logging ("Replace NaN with 0.0 for logging"), and neither `training_log.csv` nor `training_progress.json` persists `division_tp`/`division_fp`/`division_fn` for v50 (confirmed: zero matches for these keys in the saved artifacts). Cases (a) and (b) are therefore indistinguishable from the persisted `0.0` alone, and `val_score` would be numerically identical either way. This is marked **UNRESOLVED**, not resolved to either reading.

---

## K. Hypothesis H1–H5 (reconstructed from corrected evidence)

The corrected fallback rate (11.57%, not 100%) and the retraction of the 5–10× overprediction claim substantially change the evidentiary weight behind each hypothesis. Rankings are reconstructed from scratch, not carried forward from v1.

### K.1 H1: Undertraining

**Unchanged evidence base from v1** (5,000/12,392 batches, 1 epoch, no learning-curve data) — this hypothesis's evidence was not directly contradicted by Codex, but its relative priority changes because H4's evidence (below) is now much weaker.

**Status:** Possible, not proven, not falsified. Same cheapest discriminator as v1 (multi-epoch run with telemetry), same falsifier (plateau before epoch 5).

### K.2 H2: Absolute Calibration Mismatch

**Evidence changes materially.** v1 argued the 100%-fallback framing supported "model miscalibrated but not broken." With the corrected 11.57% fallback rate, the picture is different: the model *does* cross the fixed 0.5 threshold for the large majority of calls, meaning it is not uniformly suppressed below 0.5 — sigmoid outputs range up to at least 0.68 during validation. This weakens the specific "fixed threshold 0.5 is universally wrong" framing but does not resolve whether 0.5 (vs. some other fixed value) is well-calibrated for the small subset of true-positive detections vs. the presumably much larger population of false-positive voxels crossing it.

**Status:** Possible; corrected evidence is more ambiguous than v1's account, not more supportive or less.

### K.3 H3: Representation / Target / Loss Limitation

**Unchanged from v1.** Not directly addressed by Codex's review. Same evidence for/against, same discriminator (add cross-Z conv, brief retrain), same "expensive, not currently recommended" status.

### K.4 H4: Extraction-Policy Artifact (Over-Prediction)

**Substantially weakened.** v1's strongest evidence for H4 was "554K predicted vs. ~54–100K GT = 5–10x over-prediction" — this specific claim is retracted (§ item 4). The two exact locally-available comparisons go the *other* direction (v50 underpredicts relative to `T_true` on both checkable samples). The corrected 11.57% (not 100%) fallback rate also removes the "universal, static, untuned percentile fallback dominates every call" framing that supported H4.

**Status:** Downgraded from v1's "Strong / PRIORITY" to **unresolved, weakly supported at best** pending an exact full-71-sample `T_true` extraction. The 2-sample Oracle evidence (§L.3 below) is real but too small to establish a full-fold bottleneck either way.

### K.5 H5: Interaction (Undertraining + Extraction)

**Unchanged mechanically, weakened by H4's downgrade.** Since H4's evidentiary support is now much weaker, an H1×H4 interaction hypothesis inherits that weakness. Still not falsified, still not proven, still expensive to discriminate.

### K.6 Summary Table: Hypotheses and Evidence (reconstructed)

| Hypothesis | Supported by corrected evidence? | Falsified? | Change from v1 | Recommended priority |
|---|---|---|---|---|
| H1 (Undertraining) | Possible, unchanged evidence | NO | No change | Candidate for next discriminator |
| H2 (Calibration Mismatch) | Ambiguous — model does cross 0.5 often; whether *well*-calibrated is unknown | NO | Weakened from v1's dismissal; now genuinely open | Cheap to test (threshold sweep), worth doing |
| H3 (Architecture Limitation) | Unchanged | NO | No change | Low priority, expensive |
| H4 (Extraction Over-Prediction) | **Downgraded** — key supporting evidence retracted | NO (but no longer "strongly supported" either) | Major downgrade from v1's "Strong/PRIORITY" | Requires exact full-fold `T_true` extraction before further action |
| H5 (Interaction) | Possible, weakened alongside H4 | NO | Weakened | Defer |

**Reconstructed priority order:** (1) Extract exact full-71-sample `T_true` sum from `.geff` metadata directly — this is a one-off, low-cost query that would resolve H4's evidentiary status either way and was conspicuously never done in v1 despite being cited as the central number. (2) Run a threshold sweep (H2 discriminator) — cheap, CPU-only. (3) Only then decide between extended training (H1) and architecture changes (H3), informed by (1) and (2).

---

## L. Reconciliation with Previous Experiments (corrected §L.3)

### L.1, L.2, L.4, L.5

Unchanged from v1 — not disputed by Codex.

### L.3 Oracle Decomposition — CORRECTED from "In Progress" to "Completed (2/71 samples)"

**v1 claim:** "Currently running oracle-ceiling check against the real (non-collapsed) probe checkpoint" / "Not yet available."

**Corrected status:** A run had already completed. `oracle_check_training_run.log` records:

```text
Loading checkpoint: ..\kaggle_probe_output_v3\gpu_learning_probe\training_run_checkpoint.pt
  epoch=1 training_code_sha=bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c
Validation batches: 198 across samples ['44b6_0113de3b', '44b6_0b24845f']
```

Independent hashing confirms `training_run_checkpoint.pt` is byte-identical (SHA-256 `8a788a19...c25092`) to v50's own saved `epoch_1_val_score_0.0020.pt`. This file was written to disk at `2026-08-15 11:00:23` local time — **5.5 hours before** v1's own report commit (`16:30:35+05:30` = `11:00:35` UTC... i.e. essentially the same clock time as the Oracle log's write, and preceding v1's commit by the gap between an 11:00 file write and a 16:30 commit).

**Results (both modes, both samples):**

| Sample | Mode | Score | Edge Jaccard | Pred nodes | GT nodes (sparse) |
|---|---|---:|---:|---:|---:|
| 44b6_0113de3b | GT nodes + GT edges (sanity) | 1.0998 | 1.0000 | 52 | 52 |
| 44b6_0113de3b | GT nodes + model edges | 0.0000 | 0.0000 | 52 | 52 |
| 44b6_0113de3b | Model nodes + Oracle edges | 0.0000 | 0.0000 | 1,645 | 52 |
| 44b6_0b24845f | GT nodes + GT edges (sanity) | 1.0998 | 1.0000 | 51 | 51 |
| 44b6_0b24845f | GT nodes + model edges | 0.0000 | 0.0000 | 51 | 51 |
| 44b6_0b24845f | Model nodes + Oracle edges | 0.0040 | 0.0039 | 25,354 | 51 |

Note: the Oracle log's own "GT nodes" column here is the **sparse labeled node count** (52/51), a different and much smaller quantity than the `estimated_number_of_nodes` metadata field (25,755/32,795) that the real competition scoring formula actually uses as `T_true`. These two GT quantities must not be conflated (v1 conflated them implicitly by using sparse-count-scale numbers to estimate `~54-100K` in the first place).

**Classification:** COMPLETED, checkpoint-identity-confirmed, but **2 of 71 samples only**. Cannot establish a full-fold Oracle ceiling or definitively rank H1 vs. H4. Directionally, "GT nodes + model edges" scoring 0.0000 on both samples suggests the model's edge-linking is very weak even when given perfect node positions — a data point relevant to (but not conclusive for) H3/H4, that v1 never had access to because it believed this evaluation hadn't run yet.

---

## M. Hypothesis Updates (Post-v50, corrected)

### M.1 Updated Understanding of Collapse — unchanged (observational claim preserved)

Collapse is reversible: v50 is non-zero, prior registry runs were 0.0. High confidence, unchanged.

### M.2 Updated Understanding of Detection Calibration — CORRECTED

**v1's claim:** "Adaptive fallback activates 100% of the time... systematic, universal miscalibration."

**Corrected:** Adaptive (zero-positive-fraction) fallback activates 11.57% of the time. The fixed threshold succeeds (silently) for the remaining 88.43%. This is not "systematic, universal miscalibration" — it is a real but minority failure mode. Whether the *majority* fixed-threshold path is itself well-calibrated (i.e., whether its detections are true cells or noise) is a separate, still-open question that requires precision/recall analysis against exact GT, not addressed by either v1 or this correction.

### M.3 Updated Understanding of False-Positive Explosion — RETRACTED

**v1's claim:** "Predicted nodes (554K) are 5–10x ground truth... systematic property."

**Corrected:** This claim is retracted. No exact full-fold `T_true` exists in saved artifacts; the two exact values available (58,550 combined) already exceed v1's own lower estimate from 2 of 71 samples, and v50 *underpredicts* relative to `T_true` on both of those samples individually. The direction of the aggregate 71-sample comparison is unresolved, not established as overprediction.

---

## N. Score Loss and Oracle Interpretation (corrected)

### N.1 Where is Score Being Lost?

**Metric:** final_score = 0.0019855762417342647

**Corrected reading:**
1. **Base edge Jaccard (0.001845):** This is `TP/(TP+FP+FN)`, not recall. A value this low means the *overlap* between predicted and true edge sets is tiny relative to their union — consistent with either very poor node detection, very poor edge linking, or both, but does **not** on its own quantify "% of true edges recovered" (that would be `TP/(TP+FN)`, a different, unreported number).
2. **Adjusted exceeds base because of aggregation mechanics, not because an "over-prediction penalty" failed to apply** — see §J.3. There is no basis in the corrected evidence for saying an over-prediction penalty is "the dominant loss term"; if anything, the available per-sample multiplier evidence points toward underprediction contributing positively (multiplier >1) on at least the checkable samples.
3. **Division signal is zero** because the score-sum term is dropped, not because of a "perfect or absent" ambiguity — see §J.3.

**Verdict, corrected:** The dominant driver of the very low score is the low base Jaccard itself (poor overlap between predicted and true edge sets), which is consistent with node-detection and/or edge-linking failure — but the specific causal story v1 told ("over-prediction from a miscalibrated universal fallback overwhelming the linker") is not supported by the corrected evidence. The two-sample Oracle result (§L.3) — "GT nodes + model edges" scoring 0.0000 on both samples — is a real, if narrow, data point suggesting edge-linking weakness independent of node-detection quality, and deserves more weight in future investigation than v1 gave it (v1 didn't know this result existed).

---

## O. tracksdata Warning Interpretation (corrected)

### O.1 "No Matching Nodes" Warnings (44 total) — corrected unit

**Corrected meaning:** `tracksdata/graph/_base_graph.py` logs this once per whole match-operation returning zero matched IDs (`if len(node_ids) == 0: LOG.warning("No matching nodes found."); return`), confirmed by reading the installed package source directly. It is **not** a per-individual-unmatched-node counter. v1's derived "~0.6% unmatched" figure used the wrong denominator semantics (treating 44 as a numerator over an estimated GT-node count) and is dropped.

**Verdict:** 44 is a real, exact count of zero-match *operations*; no valid percentage-unmatched figure can be derived from it without additional instrumentation of the match operation itself (e.g., logging `len(node_ids)` on every call, not just when it's zero).

### O.2 "No Matching Edges" Warnings (216 total) — corrected unit

Same correction as O.1, applied to `_base_graph.py:1281`. The "~1–2% unmatched" derivation is dropped for the same reason.

---

## P. Run Registry Audit

Unchanged from v1's structural findings (§P.1, P.2, P.4) — not disputed by Codex. **Correction to P.3:** the registry's `notes` field is not independent corroboration of the fallback-rate/sigmoid claims — see §B.2 above; it repeats the same flawed reconstruction this document corrects.

---

## Q. Instrumentation Commit eb31af9 Assessment

Unchanged from v1 — not disputed by Codex. Code review of `eb31af9` (GPU name, CUDA availability, max_sigmoid min/max/final, peak GPU memory allocated/reserved; CPU/null-device backward compatibility; registry ingestion) stands as originally assessed: correct, complete for its stated scope, backward-compatible. Test-pass claims from the commit message remain not independently re-run in this session (would require executing pytest, out of this reconciliation's read-only-except-two-new-docs scope).

---

## R. Deferred Telemetry Prioritization (updated priority)

v1's §R stands largely as written, with one priority correction: **structured fallback telemetry is still the top recommendation, but for a different reason than v1 gave.** v1 argued it was needed because "v50 revealed 100% fallback rate, a critical finding" — that finding is now corrected to 11.57%. The corrected reason it remains the top priority: without structured per-call counters (fixed-vs-adaptive call counts, not just warning-line counts), *any* future run will require the same error-prone manual log-reconstruction this reconciliation had to perform to get the real denominator — a structured `fixed_threshold_calls` / `adaptive_fallback_calls` pair of counters, logged unconditionally per call (not only when a warning fires), would have prevented v1's core error outright.

**Revised proposed schema addition** (supersedes v1's, which omitted a counter for the silent fixed-threshold path):

```python
"fallback_stats_json": {
  "fixed_threshold_calls": int,       # NEW: was never counted in v1's proposal
  "zero_positive_fallback_calls": int,
  "high_positive_fallback_calls": int, # NEW: the "undertrained-model miscalibration" branch, 0 in v50 but not logged as a counter
  "total_detection_calls": int,        # NEW: batches * channels, so the denominator is never ambiguous again
  "adaptive_threshold_min": float,
  "adaptive_threshold_max": float,
  "adaptive_threshold_mean": float
}
```

---

## S. Remaining Unknowns (updated)

| Question | Why Unknown | Status change from v1 |
|---|---|---|
| Exact full-71-sample `T_true` sum | Not serialized; only 2/71 exact values available locally | Same gap, now explicitly identified as the cause of the retracted 5-10x claim |
| Full-fold Oracle ceiling | Only 2/71 samples evaluated | Corrected from "not started" to "started, narrow" |
| Which pre-v50 change caused the nonzero transition | No ablation; ≥7 concurrent changes | Same gap, chain now more completely enumerated |
| Whether the majority (88%) fixed-threshold detections are true cells or noise | No precision/recall breakdown by threshold-path exists | **New question, did not exist in v1's framing** because v1 believed there was no fixed-threshold path activity to analyze |
| Does 5 more epochs substantially improve score? | Single-epoch dataset | Unchanged |
| Are there actual divisions in GT? | Not queried | Unchanged |
| Whether v50's logged `val_division_jaccard=0.0` is a genuine zero-TP division Jaccard or a NaN "no divisions present" case dropped from the score | `validate_epoch()` sanitizes any NaN metric to 0.0 before logging (`bc989ed:src/train.py` ~L1535–1541); `division_tp`/`division_fp`/`division_fn` are not persisted in v50's saved artifacts | **New question**, identified in this correction round; UNRESOLVED |

---

## T. Ranked Next Experiments/Actions (reconstructed)

| Rank | Action | Cost | Rationale for rank |
|---|---|---|---|
| 1 | **Extract exact full-71-sample `estimated_number_of_nodes` sum** from all 71 `.geff` files' metadata (one-off script, no GPU) | VERY LOW | Directly resolves H4's evidentiary status either way; this was the single most consequential unverified number in v1 and is trivial to get exactly |
| 2 | **Threshold sweep** on v50 checkpoint (fixed thresholds 0.1–0.5) | LOW (CPU-only) | Cheap H2 discriminator; corrected evidence makes H2 more open than v1 thought, not less |
| 3 | **Full-71-sample Oracle evaluation** against v50's checkpoint (extending the existing 2-sample run) | LOW–MEDIUM (GPU inference only) | The 2-sample scaffold already exists and the checkpoint identity is confirmed; extending it to all 71 samples directly answers the H1-vs-H4 split at the scale that actually matters |
| 4 | **Structured fallback-call telemetry** (see §R) added to `src/train.py` before any further long run | LOW | Prevents recurrence of this exact denominator error |
| 5 | Extended training run (5 epochs) | VERY HIGH (~28.5h GPU) | Same as v1 — expensive, defer until 1–4 completed |
| 6 | Architecture modification (cross-Z conv) | VERY HIGH | Same as v1 — lowest priority |

---

## U. Recommended Next Action (corrected)

**v1's recommendation** ("complete the Oracle decomposition, already in progress") is moot — that evaluation already completed, at 2-of-71-sample scope, before v1 was even written.

**Corrected single highest-information next action: extract the exact full-71-sample `T_true` sum from local/staged `.geff` metadata.** This is lower cost than v1's recommended action (no GPU inference required, pure metadata read), and directly resolves the single most consequential retracted claim in this reconciliation (§ items 4, 11, K.4). Only after that number exists does extending the Oracle evaluation to the full fold (§T rank 3) become well-motivated as the next GPU-cost action.

---

## V. Proposed EV-V50 Evidence Records (reconstructed from scratch)

Per task instructions, no EV record is promoted here — these are proposals only.

### EV-V50-001: Non-Structural-Zero Result at Full-Fold Scale (observational only)

| Field | Value |
|---|---|
| **Claim** | v50 is the first recorded run in the available registry to produce a nonzero, non-degenerate validation score on the full 71-sample validation fold |
| **Observation** | val_score=0.001986, predicted_nodes_total=554,366, is_structural_zero=False, checkpoint SHA-256 confirmed, full 71/71 samples evaluated |
| **Source** | training_progress.json, training_log.csv, checkpoint_manifest.json, kaggle_runs.db |
| **Execution SHA** | bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c |
| **Evidence Grade** | CONFIRMED (observational) |
| **Defect Status** | N/A |
| **Effect Status** | PROVEN (nonzero output at full-fold scale) |
| **Bottleneck Status** | N/A — this record does not address current bottleneck |
| **Causal attribution to specific fixes** | EXPLICITLY NOT CLAIMED — see §C; 7 concurrent ancestor changes, no ablation |
| **Priority** | MEDIUM (retrospective validation only; not actionable on its own) |

### EV-V50-002: Detection-Threshold Path Split (corrected from v1's "universal fallback")

| Field | Value |
|---|---|
| **Claim** | Of 14,058 validation detection calls, 1,626 (11.566368%) hit the zero-positive-fraction adaptive-threshold fallback; the remaining 12,432 (88.433632%) used the fixed threshold=0.5 silently |
| **Observation** | Exact log counts + source-code branch structure, independently re-derived; adaptive-threshold values recovered: min=0.320821, max=0.420600, mean=0.383704879458793; channel split ch0=182/ch1=1,444 |
| **Source** | full_log.json, bc989ed:src/train.py |
| **Execution SHA** | bc989ed |
| **Evidence Grade** | CONFIRMED |
| **Defect Status** | UNRESOLVED — a minority zero-positive-fraction rate is not on its own evidence of a defect; could be expected behavior for a subset of low-signal timepoints |
| **Effect Status** | PROVEN (the split itself is real and exact) |
| **Bottleneck Status** | UNRESOLVED — no intervention/ablation test isolates this path's contribution to the low score |
| **Priority** | MEDIUM — worth structured telemetry (§R) and a threshold sweep (§T rank 2), not an emergency fix |

### EV-V50-003: Predicted Node Count (corrected — overprediction claim removed)

| Field | Value |
|---|---|
| **Claim** | predicted_nodes_total=554,366 is a real, exact count; its relationship to the true full-fold scoring denominator is **currently unknown** |
| **Observation** | 554,366 (exact, multi-source-confirmed); two exact local `T_true` comparisons (25,755 and 32,795) both exceed v50's own predicted counts on those specific samples (1,645 and 25,354) |
| **Source** | training_log.csv, training_progress.json, checkpoint_manifest.json, local `.geff` metadata for 2/71 samples |
| **Execution SHA** | bc989ed |
| **Evidence Grade** | CONFIRMED (count only); UNRESOLVED (relationship to T_true) |
| **Defect Status** | UNRESOLVED — no exact full-fold overprediction (or underprediction) has been demonstrated |
| **Effect Status** | NOT DEMONSTRATED — v1's "PROVEN" status for a 5-10x effect is withdrawn |
| **Bottleneck Status** | UNRESOLVED |
| **Priority** | HIGH — but the action is "extract the exact denominator" (§T rank 1), not "add node filtering," which was v1's premature prescription |

### EV-V50-004: Clean Execution, Healthy Status (unchanged from v1, re-confirmed)

| Field | Value |
|---|---|
| **Claim** | v50's execution was clean: no crashes, no enumerated fallback failures, full validation completed |
| **Observation** | health_status="healthy", heatmap/edge_target/edge_loss/eval_failures all 0, zero ERROR/Traceback/CRITICAL in 38,308 log records |
| **Source** | training_progress.json, training_log.csv, full_log.json |
| **Execution SHA** | bc989ed |
| **Evidence Grade** | CONFIRMED |
| **Defect Status** | N/A (infrastructure healthy) |
| **Effect Status** | PROVEN (system stable) |
| **Bottleneck Status** | N/A |
| **Priority** | MEDIUM (validates further experimentation is safe on this codebase) |

---

## W. ML-002 Assessment: Experiment Observability (corrected emphasis)

**Unchanged core lesson from v1:** "An experiment is incomplete if it fails to capture the measurements necessary to discriminate the hypotheses it was intended to test." v50 supports this lesson, but the corrected evidence sharpens *which* measurement gap actually mattered: it was not primarily the absence of GPU/max_sigmoid telemetry (which `eb31af9` already addresses) — it was the absence of an unconditional **fixed-vs-adaptive call counter**, whose absence directly caused v1's central 100%-fallback error. §R's revised schema addition targets this specific gap.

---

## X. Codex Verification Packet (updated)

### X.1–X.2 Repository Identifiers and Artifacts

Unchanged from v1, plus:

| Item | Value |
|---|---|
| **v1 report commit** | 6b315891a80fc2530d79ce29f8bff861b293e919 |
| **Codex review commit** | 6ebd38ce87f69f6b2c0bc32e69f790849da86c20 (`origin/codex/v50-independent-review`) |
| **Codex review SHA-256** | d52011efe7caf9bf0c4ebab9668c622b2b64429227f10c85acd2e65c468421a6 (independently re-verified in this session) |
| **Oracle log** | C:\Users\hemas\Downloads\oracle_check_training_run.log |
| **Oracle-run checkpoint** | C:\Users\hemas\Downloads\kaggle_probe_output_v3\gpu_learning_probe\training_run_checkpoint.pt (SHA-256 identical to v50's own checkpoint) |
| **GEFF metadata (2 samples)** | data/staging/train/44b6_0113de3b.geff, data/staging/train/44b6_0b24845f.geff |

### X.3 Key Verification Commands (corrected/extended)

| Verification | Command | Expected Result |
|---|---|---|
| Validation batch count | `grep "Val loader batches" full_log.json` | `7029` |
| Detection call denominator | 7029 × 2 | `14058` |
| Zero-positive fallback rate | count("severe under-confidence") / 14058 | `0.11566368` |
| Adaptive threshold stats | regex-parse all 1,626 `threshold=... instead` lines | min=0.320821, max=0.420600, mean=0.383704879458793 |
| Real `76bf901` SHA | `git rev-parse 76bf901` | `76bf901126df7a70521be3b4923602a77188d565` (NOT `...abb836c5874d6107db431077ade57f8bf`) |
| `76bf901` file scope | `git diff-tree --no-commit-id --name-only -r 76bf901` | `src/targets.py`, `tests/test_targets.py` |
| Deployed quantile-fix ancestor | `git merge-base --is-ancestor ba1bdb4 bc989ed` | exit 0 (true) |
| Off-ancestry cherry-pick | `git merge-base --is-ancestor 2a263c2 bc989ed` | exit 1 (false) |
| Checkpoint identity | `sha256sum` on both checkpoint files | identical: `8a788a192725d80a39c6ea4a5a4f74ade67cf4c259fa67cc943d9ede15c25092` |
| Exact T_true, 2 samples | `IndexedRXGraph.from_geff(path)` → `meta.extra['estimated_number_of_nodes']` | 25,755 and 32,795 |
| tracksdata warning unit | read `_base_graph.py` lines 1255–1256, 1280–1281 | one warning per zero-ID match operation, not per item |

### X.4–X.7

Superseded by the corrected content of §K, §T, §U above and by the full reconciliation ledger at `docs/evidence/HSOM_V50_RECONCILIATION_2026-08-15.md`.

---

## Final Decisions (reconstructed)

**1. What were the confirmed pre-v50 fixes?** Seven confirmed ancestor commits, in authored order (§C.1): adaptive weighting introduction (`eef5700`), loss normalization (`76bf901`), LR change (`872743646b`), warmup enable (`1c5c50f1`), symmetric adaptive threshold (`4a26f02`), reference-aligned weighting (`ab5fcc3`), quantile normalization (`ba1bdb4`). All confirmed ancestors of `bc989ed` via `git merge-base --is-ancestor`.

**2. What did v50 prove?** Observationally: collapse is reversible (nonzero, non-degenerate score after prior 0.0 runs). Not causally: which specific fix(es) mattered — unablated, combined run.

**3. What did v50 NOT prove?** Everything v1 listed, plus (newly explicit): that predicted nodes exceed the true scoring target (retracted), that the detection-threshold fallback is universal (corrected to 11.57%), that an Oracle evaluation hadn't started (it had, and completed on 2/71 samples).

**4. What is currently the strongest candidate failure mechanism?** UNRESOLVED at the "primary bottleneck" level. H4 (extraction over-prediction), v1's confident answer, is downgraded — its key evidence is retracted. The corrected evidence does not clearly favor any single hypothesis in K.1–K.5; the two-sample Oracle's "GT nodes + model edges → 0.0000" result is a real but narrow hint toward edge-linking weakness, not a resolved answer.

**5. Has any leaderboard bottleneck actually been established?** NO — same answer as v1, for stronger reasons: v1's own proposed discriminator (Oracle eval) had already run at narrow scope and does not resolve it; the overprediction evidence that would have supported H4 is retracted.

**6. Is the telemetry gap materially improved by eb31af9?** Unchanged assessment from v1 (partial: GPU/max_sigmoid/memory fields added, structured fallback-call counters still absent) — but see §R for the corrected, more specific counter gap that actually caused this report's central error.

**7. Is structured fallback telemetry required before another long run?** YES — same conclusion as v1, revised rationale (§R): not because v50 showed "100% fallback" (it didn't), but because the *absence* of an unconditional call counter is what allowed a 1,626-vs-14,058 denominator error to occur and go unnoticed through an entire report cycle.

**8. Is longer training currently scientifically justified?** NOT YET — same as v1's conclusion, but for a different reason: not "wait for Oracle eval" (already done, narrow), but "wait for the exact full-fold `T_true` extraction and threshold sweep" (§T ranks 1–2), which are both cheaper and more directly load-bearing than v1's recommended next step.

**9. What is the single highest-information next action?** Extract the exact full-71-sample `estimated_number_of_nodes` sum from `.geff` metadata (§U) — corrected from v1's "complete the Oracle eval," which was based on a false premise that it hadn't started.

**10. What evidence must an independent reviewer verify before authorizing the next long run?** All items in the reconciliation ledger's "Items independently re-verified" section (17 items) should be spot-checked by any subsequent reviewer using the exact commands in §X.3, before authorizing further GPU spend on H1 (extended training) or H4-driven (node-filtering) experiments — since both of v1's leading hypotheses for prioritizing between those two paths have now been either weakened (H4) or left open (H2) by this reconciliation.

---

## Final Report Status

**File:** `docs/evidence/HSOM_V50_FORENSIC_RECONSTRUCTION_V2_2026-08-15.md`

**Supersedes:** v1 (`docs/evidence/HSOM_V50_FORENSIC_RECONSTRUCTION_2026-08-15.md`, commit `6b31589`), which received Codex REJECT verdict.

**Reconciliation ledger:** `docs/evidence/HSOM_V50_RECONCILIATION_2026-08-15.md`

**Verification method:** Every numeric and structural correction in this document was independently reproduced from primary evidence (git objects, raw JSON log, checkpoint SHA-256 hashes, vendored evaluator source, GEFF metadata, SQLite registry) in this session — not copied from Codex's stated conclusions. Where Codex's own framing needed a footnote rather than a straight adoption (the 1,886-vs-1,890 warning count), both numbers are reported with exact definitions rather than picking one authority.

**No EV record in §V is promoted to canonical status by this document** — all four are proposals pending separate authorization, per task instructions.

Status: PENDING_INDEPENDENT_REVIEW
