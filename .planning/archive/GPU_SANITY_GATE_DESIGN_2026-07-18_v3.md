# GPU Sanity Gate — Design v3 (2026-07-18)

**Status: PLANNING ONLY. No repository code was modified to produce this revision. No
training, no Kaggle kernel run, no LOEO run has been executed. No commit, no push.**

Supersedes `GPU_SANITY_GATE_DESIGN_2026-07-18_v2.md`. Responds to the v2 review
verdict: **APPROVE WITH 4 FINAL CHANGES**.

## Prerequisite status update (verified against real git state, not assumed)

- `git rev-parse HEAD` on this branch = `8eeace8ab090aecb1cfad336f82865f752c825c4` —
  this **exactly matches** the SHA quoted in the v2 review (40/40 hex chars, no
  truncation this time, unlike the v1 round). `git log --oneline` confirms this commit
  is `8eeace8 fix(metrics): sync scoring with current competition metric`, consistent
  with "P0-7A is now integrated and pushed."
- **P0-7A prerequisite: satisfied**, confirmed by direct SHA check, not by trusting the
  status claim alone.
- **P0-7 (training-integrity freeze): not integrated.** No commit on this branch's
  visible history matches that description. **The formal gate remains blocked per §2 —
  this document is still planning only, and none of what follows authorizes running
  anything.**

---

## Changelog vs. v2 (map to the 4 required changes)

| # | Required change | Where addressed |
|---|---|---|
| 1 | Exact source-provenance equality (not ancestry); mounted `GIT_SHA.txt` equality; module-origin verification | §9 (was "ancestry" in v2 §11), §17 item 1 |
| 2 | Gradient probes must come from a genuinely edge-supervised batch, not an arbitrary final batch | §8 (fully rewritten) |
| 3 | Checkpoint save reordered to after verdict computation; exact required report fields | §10 (was §9 in v2) |
| 4 | Adaptive/fixed threshold rule evaluated per-timepoint, not averaged | §6, §11 |

All v2 improvements are retained unchanged except where a change above required a
rewrite: constructor-level `sample_id_allowlist` before pair-index construction (§3.0);
exact pair-index sample-ID assertion (§3.0); `filter_unannotated_pairs=True` (§3.0);
deterministic K=4..12 pre-flight expansion (§3.1); positive/negative/hard-negative
supervision predicate (§3.1); hard/easy negative distinction (§7); quantitative
edge-ranking criteria (§7); fixed/adaptive metrics reported separately (§6); dual
wall-clock limits (§10 wall-clock subsection, unchanged from v2 §10); zero-technical-
fallback policy (§9); biological-zero vs. technical-failure distinction (§9); SANITY
vs. DEPLOYMENT checkpoint separation (§10, §12); no P1 architecture changes (unchanged,
out of scope).

---

## New verification done for v3

- **`git rev-parse HEAD` checked directly** — matches the claimed P0-7A SHA exactly
  (see above). No P0-7-related commit found in `git log`.
- **`src/checkpoint_manifest.py:643-772` (`load_verified_checkpoint`) read in full.**
  This function *already implements exact-equality* SHA verification for the
  deployment-loading path: `if manifest["training_code_sha"] != expected_source_sha:
  raise ValueError(...)` (line 685) — a real, existing, exact (`!=`), not ancestry-based,
  check. v2's PASS-rule language ("post-dates the P0-7A merge commit... checked by SHA
  ancestry") was a v2-introduced weakening relative to infrastructure this codebase
  already has — v3 corrects this by requiring exact equality and reusing the existing
  pattern rather than inventing ancestry logic (§9).
- **`kaggle_kernel/train_kernel.py:108-117` read.** Confirms the exact mechanism
  already in place for Kaggle execution: `DEPLOYED_SHA` is read from
  `Path(KAGGLE_SRC_DATASET_DIR) / "GIT_SHA.txt"` at kernel start and logged
  ("Deployed code SHA: ..."). This is the real "mounted kaggle source GIT_SHA.txt"
  referenced in change 1 — not a new mechanism to build, just one to check against
  exactly, and to actually verify (not just log) before the gate proceeds.
- **`src/train.py:1429,1458`** confirm `training_code_sha` in the saved checkpoint
  comes from `self.deployed_sha` (fed by the same `DEPLOYED_SHA`/`GIT_SHA.txt` chain).
- **`src/train.py:785-822` (edge-target/edge-loss block inside `train_epoch`) read in
  full.** This is the exact code path change 2 concerns:
  - `generate_edge_targets()` is called per batch (line 787); on exception,
    `edge_targets = None` and `edge_target_generation_failure` increments (already
    existing fallback counter) — **the transformer forward call at line 808 is inside
    `if edge_targets is not None:`, so it never runs at all for a failed-generation
    batch.**
  - Even when `edge_targets is not None`, if `len(edge_logits) == 0`, `edge_loss` falls
    back to a fresh `torch.tensor(0.0, requires_grad=True)` **detached from the
    transformer's computation graph** — a batch like this contributes zero transformer
    gradient from `total_loss_item.backward()` even though the transformer's forward
    method was called.
  - **Conclusion, confirmed by reading the real code, not assumed:** a batch only
    produces a real, graph-connected transformer gradient contribution when
    `edge_targets is not None AND len(edge_logits) > 0`, and only proves supervision
    *from a positive link* when additionally `edge_metadata['num_positive_edges'] > 0`
    (all three of `num_positive_edges`, `num_negative_edges`, etc. are already
    returned per-batch by `generate_edge_targets`, `src/targets.py:321-334` — no new
    per-batch instrumentation is needed to know this, only a place to remember it
    across the loop, see §8).
  - The single `total_loss_item.backward()` call (line 834) is shared by both UNet
    and transformer parameters (`edge_loss + heatmap_loss_weight * detection_loss`,
    lines 825-828) — confirming detection and edge gradients are produced by the same
    backward pass, but only the edge branch's contribution depends on that specific
    batch having real, positive edge supervision.

---

## 2. Prerequisites (updated)

1. ~~P0-7A (metric parity) integrated~~ — **SATISFIED**, verified: `git rev-parse HEAD
   == 8eeace8ab090aecb1cfad336f82865f752c825c4`.
2. **P0-7 (training-integrity freeze) integrated and merged — NOT YET SATISFIED.**
3. Working tree clean at the SHA under test.
4. GPU confirmed available and stable for at least the projected runtime (§10).
5. `kaggle_src_dataset`/deployed-dataset mirror in sync with the SHA under test —
   verified by the exact-equality check in §9, not assumed.

**If any of 1–5 is false: the gate does not run. Item 2 is currently false. The gate
remains blocked.** Nothing in this document authorizes execution.

---

## 3. Data selection (unchanged from v2 — kept per review verdict)

### 3.0 Training-subset injection point

Constructor-level `sample_id_allowlist` param on `CompetitionDataset`, applied
immediately after `self.sample_ids = split_data[split_type]` (`src/dataset.py:102`),
**before** the pair-index-build loop (`src/dataset.py:109` onward):

```
self.sample_ids = split_data[split_type]
if sample_id_allowlist is not None:
    allowlist_set = set(sample_id_allowlist)
    self.sample_ids = [s for s in self.sample_ids if s in allowlist_set]
    missing = allowlist_set - set(self.sample_ids)
    if missing:
        raise ValueError(f"sample_id_allowlist contains IDs not in split '{split_type}': {sorted(missing)}")
```

Mandatory pre-training assertion, after `train_dataset` construction, before any
optimizer step:

```
configured_ids = set(TRAINING_SAMPLE_IDS)
actual_ids = {sample_id for sample_id, _ in train_dataset.pairs}
assert actual_ids == configured_ids, (
    f"Training pair index does not match configured sample IDs. "
    f"configured={sorted(configured_ids)} actual={sorted(actual_ids)}"
)
```

`filter_unannotated_pairs=True` for the training dataset (existing constructor param,
`src/dataset.py:56,76-84`).

### 3.1 Training subset content and edge-supervision pre-flight (unchanged)

Deterministic candidate selection: start `K=4` — `6bba_05b6850b`, `6bba_05db0fb1`,
`6bba_062c8d37`, `6bba_07477033` (first 4 of `embryo_44b6_validation.json`'s `train`
list, file order). Pre-flight predicate before any training step:
`total_positive_edges > 0 AND total_negative_edges > 0 AND total_hard_negative_edges >
0`, accumulated via `generate_edge_targets()` (`src/targets.py:197`) over every
frame-pair in the candidate set. If it fails at K=4, expand `K→K+1` deterministically
up to `K=12`; if still failing, **FAIL: "TRAINING SUBSET SELECTION EXHAUSTED"** and stop
for human review before any GPU spend.

Real GEFF evidence already gathered (unchanged from v2): `6bba_05b6850b` has 845 real
lineage edges, `6bba_05db0fb1` has 1,183 — strong evidence for 2 of the 4 candidates.
`6bba_062c8d37` / `6bba_07477033` remain unverified locally (not staged); the pre-flight
check is the real safety net.

### 3.2 Validation subset (unchanged)

`max_validation_samples=2` (`src/train.py:1001-1067`), first 2 of the 71 `44b6`
validation sample_ids, each evaluated completely. `validation_is_full_fold=False`.

### 3.3 Seed and ordering (unchanged)

`SEED=42`. `shuffle=False` for both loaders.

---

## 4. Training budget (unchanged from v2)

`max_batches_per_epoch=40`, `num_epochs=3`, training hard cap 1200s
(`max_wall_clock_seconds`), checkpoint once at end of final epoch (see §10 for the
*reordered* lifecycle), validation once after final epoch.

## 5. Validation budget (unchanged)

2 complete 44b6 samples, full `validate_epoch()` path, P0-7A-corrected metric. No
competitive-score expectation; structural zero still disallowed (§11).

---

## 6. Detection metrics — per-timepoint pathology rule (change 4)

**The adaptive/fixed threshold rule is evaluated per timepoint, never averaged or
aggregated across timepoints before the pass/fail decision.**

For **every** validated timepoint (both timepoints of both selected 44b6 validation
samples — i.e. every `t` at which peaks are extracted during validation, not just one
representative `t` per sample):

| Metric | Definition |
|---|---|
| `fixed_threshold_peak_count[t]` | peak count at `detection_threshold=0.5`, this exact timepoint |
| `adaptive_threshold_triggered[t]` | boolean, this exact timepoint (`src/train.py:180-198`'s existing branch) |
| `adaptive_threshold_peak_count[t]` | reported separately, this exact timepoint, never substituted for the fixed count |
| `gt_node_count_at_t[t]` | real sparse GT node count at this exact `t`, from the sample's `.geff` |
| `peak_gt_ratio[t]` | `fixed_threshold_peak_count[t] / max(gt_node_count_at_t[t], 1)` |

**Per-timepoint pathology predicate (must hold at every single evaluated `t`, not on
average):**

```
adaptive_threshold_triggered[t] == False
AND 1 <= fixed_threshold_peak_count[t] <= 500
AND peak_gt_ratio[t] <= 50
```

**If this predicate fails at even one evaluated timepoint, the gate's detection
criterion fails for the sample containing that timepoint — full stop.** A validation
sample with 9 good timepoints and 1 timepoint that required adaptive rescue is a
**FAIL for that sample**, not a 90%-pass. Precision/recall are additionally aggregated
at the sample level (mean/micro-average across a sample's timepoints) **only after**
the per-timepoint pathology gate has been confirmed to hold everywhere in that sample
— aggregation is for reporting detection quality, never for deciding whether the
fixed-threshold measurement itself was trustworthy.

The report must include the full per-timepoint table (all 5 columns above, every
evaluated `t`), not just a sample-level summary — so a reviewer can see exactly which
timepoint (if any) required adaptive rescue, rather than inferring it from an aggregate
number.

Numeric grounding (unchanged from v2): `peak_gt_ratio <= 50` and absolute cap `500`
are set against real per-timepoint sparse GT density measured for `6bba_05b6850b`
(6–11 nodes/frame, mean 8.61) and the documented ~18,000-candidate pathological
incident on record in `.claude/CLAUDE.md`.

---

## 7. Edge/transformer metrics (unchanged from v2)

`num_positive_edges`, `num_negative_edges` (existing, `src/targets.py:321-322`);
new `num_hard_negative_edges`/`num_easy_negative_edges` split (existing negative count
split by whether both endpoints independently GT-matched); `true_edge_logits` and
`hard_negative_edge_logits` collected from the transformer's per-batch output.

Predicates unchanged:
1. `mean(true_edge_logits) > mean(hard_negative_edge_logits)` (always evaluated).
2. When `count(true_edge_logits) >= 5 AND count(hard_negative_edge_logits) >= 5`:
   `ROC-AUC > 0.5 OR AUPRC > positive-class prevalence`.
3. Below that count: report `"NOT COMPUTED — insufficient samples (n_pos=X, n_neg=Y)"`,
   fall back to predicate 1 alone.

---

## 8. Gradient evidence — rewritten per change 2

**v2's rule (capture gradients on "the last training batch of the final epoch,
arbitrary") is retracted.** As confirmed by reading `src/train.py:785-822` directly
(see "New verification done for v3"), the final batch of a run may legitimately be a
batch with `edge_targets is None` (a caught generation failure that isn't itself an
integrity violation if isolated) or a legitimate zero-positive-edge batch — capturing
gradients there would either see `None`/zero transformer gradients for reasons that
have nothing to do with whether the transformer *can* learn, and would produce a false
FAIL.

### 8.1 Two independently-tracked capture targets

Both are updated **conditionally, live, during the existing training loop** — no extra
forward/backward pass, no re-running anything. Each is simply "the most recent
qualifying batch's gradient snapshot," so by construction the final stored value is
from the *last* qualifying batch, without needing to know in advance which batch index
that will be:

- **UNet gradient evidence**: captured (overwriting any prior capture) immediately
  after `total_loss_item.backward()` (`src/train.py:834`) on **every** batch where
  detection-loss computation succeeded (i.e., no `heatmap_generation_failure` for that
  batch) — this is "the final valid detection-supervised training batch" by
  construction, since detection loss is computed on nearly every batch and this is
  simply the last one that didn't hit the existing fallback path.
- **Transformer gradient evidence**: captured (overwriting any prior capture) at the
  same point, but **only** on batches satisfying all of:
  - `edge_targets is not None` (no generation/computation exception this batch), **and**
  - `len(edge_logits) > 0`, **and**
  - `edge_metadata['num_positive_edges'] > 0`.

  This is exactly "the last batch satisfying `num_positive_edges > 0` and containing
  valid edge supervision," per the review verdict's wording.

### 8.2 Counters (new, required)

- `edge_supervised_batches_total`: count of batches, across the whole run, satisfying
  the three conditions above (real edge target, real logits, real positive edges).
- `edge_supervised_batches_with_nonzero_transformer_grad`: of those, the count where
  the transformer gradient probes captured at that batch were finite and nonzero.

**Both must be `> 0`.**

### 8.3 Fail conditions

- **If `edge_supervised_batches_total == 0` across the entire run:** automatic FAIL
  with the explicit, named reason `"NO EDGE-SUPERVISED BATCH"` — distinct from a
  gradient-value failure; this means the selected training subset never produced a
  qualifying batch at runtime, a data/config problem to root-cause (and a case the
  §3.1 pre-flight check exists specifically to catch in advance — this is the runtime
  belt-and-suspenders check, not the primary line of defense).
- **Given at least one edge-supervised batch exists:** the transformer gradient probes
  (§8.4) are evaluated against the *last* edge-supervised batch's captured snapshot
  only. A biological batch with zero positive edges elsewhere in the run **never**
  counts against the transformer gradient check — it simply isn't a capture candidate,
  by construction of §8.1. This directly satisfies "do not let biological zero-edge
  batches generate false failures."

### 8.4 Probe points and fail conditions (unchanged attribute paths from v1/v2)

| Probe point | Exact attribute | Captured at |
|---|---|---|
| Early UNet layer | `training_loop.unet3d.enc0[0].weight.grad` | last valid detection-supervised batch |
| Detection head | `training_loop.unet3d.det_head[-1].weight.grad` | last valid detection-supervised batch |
| Transformer node embedding | `training_loop.transformer.node_embed.weight.grad` | last edge-supervised batch (§8.1/§8.3) |
| Transformer attention/encoder block | `training_loop.transformer.encoder_t.layers[0].self_attn.in_proj_weight.grad` | last edge-supervised batch |
| Transformer edge scorer | `training_loop.transformer.edge_scorer[0].weight.grad` | last edge-supervised batch |

**Fail condition, each probe:** `grad is None`, `torch.isnan(grad).any()`,
`torch.isinf(grad).any()`, or `grad.abs().max() == 0`. Any single failure among the
five (given their respective qualifying batch existed) is an automatic gate FAIL.

---

## 9. Provenance — exact equality, not ancestry (change 1, rewritten)

**v2's PASS-rule language ("`training_code_sha` matching a commit that post-dates the
P0-7A merge commit, checked by SHA ancestry") is retracted.** Ancestry is insufficient
— a checkpoint trained on a *later* commit than P0-7A's merge could still not be the
exact commit under test (e.g. an unrelated intervening commit, or a dirty/uncommitted
local change never pushed). Required instead, all three checked independently and all
must hold **exactly**:

1. **`checkpoint["training_code_sha"] == exact_source_sha_under_test`** — direct
   string equality, reusing the exact pattern already implemented in
   `src/checkpoint_manifest.py:685` (`load_verified_checkpoint`'s existing
   `manifest["training_code_sha"] != expected_source_sha` check) rather than inventing
   new logic. `exact_source_sha_under_test` is supplied by whoever configures the gate
   run (e.g. `8eeace8ab090aecb1cfad336f82865f752c825c4` once P0-7 also lands and a new
   SHA is designated).
2. **For Kaggle execution specifically:** `Path(KAGGLE_SRC_DATASET_DIR /
   "GIT_SHA.txt").read_text().strip() == exact_source_sha_under_test` — the exact
   mechanism already implemented at `kaggle_kernel/train_kernel.py:112-117`
   (`DEPLOYED_SHA`), checked programmatically before the gate proceeds, not just
   logged and eyeballed.
3. **Imported production module origin verification:** for every production module the
   gate run actually imports (`src.train`, `src.model`, `src.dataset`, `src.targets`,
   `src.evaluation`, `src.checkpoint_manifest`, `src.split_utils`), assert its resolved
   file path is physically located under the mounted `KAGGLE_SRC_DATASET_DIR`:
   ```
   import src.train, src.model, src.dataset, src.targets, src.evaluation, src.checkpoint_manifest, src.split_utils
   mounted_root = Path(KAGGLE_SRC_DATASET_DIR).resolve()
   for mod in (src.train, src.model, src.dataset, src.targets, src.evaluation,
               src.checkpoint_manifest, src.split_utils):
       resolved = Path(mod.__file__).resolve()
       assert resolved.is_relative_to(mounted_root), (
           f"{mod.__name__} resolves to {resolved}, not under mounted source "
           f"{mounted_root} -- a stale/duplicate copy of src/ is shadowing the "
           f"intended deployed code."
       )
   ```
   This guards against a real, distinct failure mode from checks 1/2: even if
   `GIT_SHA.txt` and the checkpoint's `training_code_sha` both say the right SHA, a
   Python import-path ordering bug (e.g. a stray `src/` copy earlier on `sys.path`, or
   a cached `.pyc`/package install shadowing the mounted dataset) could still execute
   different code than the SHA claims. All three checks are needed together; none is
   individually sufficient.

**If any of the three mismatches: automatic FAIL, named reason
`"SOURCE PROVENANCE MISMATCH"`, with the specific check (1/2/3) and the actual vs.
expected values reported.**

---

## 10. Checkpoint lifecycle — reordered (change 3, rewritten)

**v2's lifecycle** (train → save checkpoint, with validation/verdict computed
somewhere around/after that) **is replaced with this explicit order:**

```
1. train  (§4)
2. validate  (§5, §6)
3. compute gate verdict + full report  (§11/§12/§13, incl. §8's gradient checks and
   §9's provenance checks, all of which must be evaluated before step 4 — the
   checkpoint is not saved until the verdict is known)
4. save sanity checkpoint + gate report together
```

This ordering matters: under v2's ambiguous ordering, a checkpoint could in principle
be written to disk before the run's own verdict was known, inviting a race where a
partially-written or premature checkpoint gets picked up by something else before the
gate has actually finished evaluating it. v3 makes "compute the full verdict first,
then save" an explicit, sequential requirement.

**The saved sanity artifact + report must include, verbatim:**

```json
{
  "training_code_sha": "<exact 40-char sha>",
  "split_membership_sha256": "<exact sha256>",
  "selected_training_sample_ids": ["6bba_05b6850b", "6bba_05db0fb1", "..."],
  "selected_validation_sample_ids": ["44b6_...", "44b6_..."],
  "validation_is_full_fold": false,
  "validation_samples_evaluated": 2,
  "validation_samples_total": 71,
  "gate_verdict": "PASS | CONDITIONAL PASS | FAIL",
  "deployment_manifest": "NOT GENERATED"
}
```

(plus every metric/predicate result from §6, §7, §8, §9, §11-§13 — the JSON sketch
above is the minimum required subset called out explicitly by the review verdict, not
the full report schema.)

**The gate runner must never import or call `write_checkpoint_manifest()`** — this is
structural, not merely relying on `deployment_eligibility_errors()` rejecting an
ineligible checkpoint after the fact (unchanged from v2, restated here because it now
lives in this renumbered section). The runner calls `save_checkpoint_file()`
(`src/checkpoint_manifest.py:79`) directly for the `.pt` file and writes its own
gate-report JSON (the schema above) alongside it — a different file, under a different
name, than `checkpoint_manifest.json`, so nothing downstream could mistake it for a
real deployment manifest even by filename collision.

**SANITY vs. DEPLOYMENT distinction, unchanged:** a checkpoint from this gate is a
SANITY CHECKPOINT (partial-fold validation, no manifest) by construction, never a
DEPLOYMENT-ELIGIBLE CHECKPOINT (§12).

---

## 11. Explicit PASS rules (updated for changes 1, 2, 4)

All of the following true simultaneously:

- No NaN/Inf in any logged loss or metric value.
- **UNet gradient probe** (last valid detection-supervised batch, §8.1): finite,
  nonzero.
- **Transformer gradient probes** (last edge-supervised batch, §8.1): finite, nonzero,
  AND `edge_supervised_batches_total > 0` AND
  `edge_supervised_batches_with_nonzero_transformer_grad > 0`.
- All technical fallback/integrity counters (§9-of-v2/now folded into this section) `==
  0`.
- Detection loss trend: `mean(detection_loss, epoch 3) <= mean(detection_loss, epoch
  1)` OR GT/background sigmoid separation increases epoch 1 → epoch 3.
- **Per-timepoint detection pathology predicate (§6) holds at every single evaluated
  timepoint** in both validation samples — not on average, not "mostly."
- `recall_at_7um > 0` on at least one of the 2 validation samples.
- Edge-ranking criterion (§7): mean comparison holds, plus ROC-AUC/AUPRC when sample
  counts allow.
- `validation_samples_evaluated == 2` exactly.
- `predicted_nodes_total > 0 AND predicted_edges_total > 0 AND is_structural_zero ==
  False`.
- `evaluate_submission()` executed under source verified by **all three** §9 exact-
  equality checks (checkpoint SHA, mounted `GIT_SHA.txt`, module-origin) — not
  ancestry.
- `measured_training_wall_clock <= 1200s AND measured_total_gate_wall_clock <= 2400s`.

## 12. Explicit CONDITIONAL PASS rules (unchanged in kind, restated)

- Edge-ranking satisfies only the minimum mean-comparison clause (insufficient N or
  ROC-AUC ≤ 0.5 / AUPRC ≤ prevalence).
- `recall_at_7um > 0` but numerically low.
- Wall-clock within budget but the computed short-LOEO projection is close to (not
  clearly over) whatever budget is agreed at that time.

## 13. Explicit FAIL rules (updated)

Any single one, automatic FAIL:

- Any technical fallback/integrity counter nonzero.
- UNet or transformer gradient probe None/zero/NaN/Inf **on their respective qualifying
  batch** (§8).
- **`edge_supervised_batches_total == 0`** → named reason `"NO EDGE-SUPERVISED BATCH"`
  (new, §8.3).
- `is_structural_zero == True`, `predicted_nodes_total == 0`, or
  `predicted_edges_total == 0`.
- `validation_samples_evaluated != 2`.
- Detection loss NaN/Inf at any point.
- **Any of the three §9 provenance checks mismatched** → named reason
  `"SOURCE PROVENANCE MISMATCH"` (rewritten from ancestry-based to exact-equality).
- Adaptive-threshold peak count substituted for, or used to rescue, a failing
  fixed-threshold criterion **at any single evaluated timepoint** (§6) — a sample
  cannot pass by averaging away one bad timepoint.
- Gate executed while P0-7A or P0-7 is not merged into the SHA under test (P0-7
  currently unmet — see §2).
- Training-subset selection exhausted at K=12 (§3.1) without satisfying the
  edge-supervision predicate.
- `train_dataset.pairs`'s actual sample IDs don't exactly equal the configured training
  sample ID list (§3.0 assertion).
- `measured_training_wall_clock > 1200s` or `measured_total_gate_wall_clock > 2400s`
  without a conclusive verdict already reached.

## 14. Stop/go decision tree (unchanged in structure; conditions point at updated predicates)

```
P0-7A + P0-7 not both integrated at test SHA?
  -> STOP. Do not run the gate. (P0-7 currently unmet.)

Any §9 provenance check fails before training even starts?
  -> STOP. Fix the deployment/mount/import-path issue before spending any GPU time.

Training-subset pre-flight predicate (§3.1) fails up to K=12?
  -> STOP. Root-cause edge-target generation / split content first.

edge_supervised_batches_total == 0 after training?
  -> FAIL: NO EDGE-SUPERVISED BATCH. Root-cause before re-running.

Gate FAILs any other §13 condition?
  -> Diagnose root cause locally/offline first. Do not re-run unchanged.

Gate PASSes fully (§11)?
  -> GO: short primary LOEO (6bba->44b6, larger budget, same orientation).

Gate CONDITIONAL PASS (§12)?
  -> Re-run with a modestly larger budget / expanded K before committing to LOEO.

Detector predicates PASS, transformer/edge predicates FAIL (real gradient/ranking
break, not just insufficient-N)?
  -> Do NOT proceed to LOEO. Root-cause the edge/transformer path in isolation first.

Transformer PASSes but validation score is poor?
  -> Proceed cautiously to short primary LOEO; not evidence for/against P1 changes.

Runtime/memory projection incompatible with a short LOEO's realistic budget?
  -> Do NOT proceed to LOEO at current config. Re-measure with one deliberate knob
     changed.

Everything above resolved and short primary LOEO passes its own gate?
  -> Reciprocal LOEO (44b6->6bba, data_splits/embryo_6bba_validation.json).

Both LOEO orientations pass?
  -> P1 architecture experiments become in-scope. Not before.
```

## 15. Estimated GPU time (unchanged)

Training ≤20 min hard cap. Overall gate ≤40 min hard cap. Actual measured timing
reported separately from any projection.

## 16. Exact command/config changes likely needed (design only — not applied, updated)

- Hyperparams: `max_batches_per_epoch=40`, `num_epochs=3`,
  `max_wall_clock_seconds=1200`, `overall_gate_wall_clock_seconds=2400`,
  `max_validation_samples=2`, `batch_size=1`, `SEED=42`,
  `sample_id_allowlist=[...]` (§3.1), `filter_unannotated_pairs=True`.
- `exact_source_sha_under_test` passed explicitly into the gate config (not inferred),
  checked against checkpoint, `GIT_SHA.txt`, and module `__file__` origins (§9) before
  training starts.
- Two live-updated gradient-snapshot variables inside the training loop (§8.1),
  conditionally overwritten each qualifying batch — no extra backward passes.
- Checkpoint/report save moved to *after* verdict computation (§10) — ordering change
  to the gate-runner script's control flow, not to `TrainingLoop` itself.

## 17. Repository changes required later to actually run the gate (updated)

1. `sample_id_allowlist` constructor param + pre-pair-index filtering + post-
   construction assertion (`src/dataset.py:102`).
2. Pre-flight edge-supervision predicate check with deterministic K-expansion (§3.1).
3. `generate_edge_targets` hard/easy negative split (`src/targets.py:324-334`).
4. GT-vs-background sigmoid mean/median instrumentation (`src/train.py:950-951`).
5. Per-timepoint peak/GT ratio + `adaptive_threshold_triggered` reporting, threaded
   through validation so every timepoint's row is retained (not aggregated away) —
   **broadened in v3** from v2's single-flag-per-sample framing to a full per-`t` table
   (§6).
6. True-edge/hard-negative logit collection + ROC-AUC/AUPRC (`sklearn`, already
   pinned).
7. **Two independently-tracked gradient-snapshot capture points inside `train_epoch`**
   (§8.1) — rewritten from v2's single "final batch" capture; requires tracking
   per-batch `edge_metadata['num_positive_edges']` and `len(edge_logits)` state across
   the loop, not just reading gradients once at the very end.
8. **Exact-equality provenance check** (§9): checkpoint-SHA equality (reuses existing
   `checkpoint_manifest.py:685` pattern), `GIT_SHA.txt` equality
   (`kaggle_kernel/train_kernel.py:112-117`'s existing read, now asserted not just
   logged), and new module-`__file__`-origin verification (genuinely new code, no
   existing equivalent found).
9. Gate-runner script: sample-ID assertion, dual wall-clock enforcement, **reordered
   lifecycle** (train → validate → verdict → save, §10), non-manifest checkpoint save
   path, gate-report JSON with the exact required fields (§10).

## 18. Classification of each item in §17

| # | Item | Classification |
|---|---|---|
| 1 | `sample_id_allowlist` + assertion | **REQUIRED INFRASTRUCTURE** — unchanged from v2 |
| 2 | Pre-flight edge-supervision check | **REQUIRED INFRASTRUCTURE** — unchanged from v2 |
| 3 | Hard/easy negative split | **REQUIRED INFRASTRUCTURE** — unchanged from v2 |
| 4 | GT-vs-background sigmoid mean/median | **REQUIRED INFRASTRUCTURE** — unchanged |
| 5 | Per-timepoint peak/GT ratio + adaptive-trigger table | **REQUIRED INFRASTRUCTURE** — broadened scope in v3 (full per-`t` table, not one flag per sample) |
| 6 | ROC-AUC/AUPRC | **REQUIRED INFRASTRUCTURE**, low-risk (`sklearn` pinned) — unchanged |
| 7 | Dual live gradient-snapshot capture (edge-supervised vs. detection-supervised) | **REQUIRED INFRASTRUCTURE** — replaces v2's single "final batch" capture (item 7 in v2's §18), materially more involved: needs conditional state tracked across the whole loop, not a one-shot read at the end |
| 8 | Exact-equality provenance check (checkpoint SHA + `GIT_SHA.txt` + module origin) | **REQUIRED INFRASTRUCTURE** — 2 of 3 sub-checks reuse existing exact-equality patterns already in the codebase (`checkpoint_manifest.py:685`, `train_kernel.py:112-117`); module-origin verification is genuinely new |
| 9 | Gate-runner script (assertion, dual wall-clock, reordered lifecycle, non-manifest save, exact report schema) | **REQUIRED INFRASTRUCTURE** — unchanged in kind from v2, reordered internally |
| — | Split file, `max_batches_per_epoch`/`max_validation_samples`, `filter_unannotated_pairs`, embryo-disjointness validation, checkpoint schema/manifest fields, global seeding, `num_positive_edges`/`num_negative_edges` | **NOT REQUIRED** — unchanged from v2, all already exist |

---

## Requested from Codex

All 4 required changes addressed above, each grounded in code read specifically for
this revision (`checkpoint_manifest.py:643-772`, `train_kernel.py:100-140`,
`train.py:785-822`), not just reworded. P0-7A's integration was independently verified
against real `git rev-parse HEAD` (exact match) rather than accepted on the status
claim alone. **P0-7 remains unintegrated — this design still does not authorize
running anything.**

Please respond with one of:

- **APPROVE** — ready to move to implementing §17/§18's REQUIRED INFRASTRUCTURE items
  once P0-7 lands (§2).
- **APPROVE WITH CHANGES** — list the specific section(s) and exact change needed.
- **REQUEST CHANGES** — list what's wrong or unsafe as designed.

One open item worth your explicit judgment: §9's module-origin verification (change
1's "verify imported production module origins resolve under the mounted source
dataset") is new — no existing equivalent was found anywhere in this codebase. The
`Path(mod.__file__).resolve().is_relative_to(mounted_root)` approach is standard but
untested in this repo's actual Kaggle execution environment (e.g. behavior under
however Kaggle's dataset-mount symlinks or path-normalization actually works has not
been verified against a real running kernel). Flag if you want this checked against an
actual Kaggle environment before being accepted as sufficient, rather than accepted on
the logic alone.
