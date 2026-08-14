# GPU Sanity Gate — Design v2 (2026-07-18)

**Status: PLANNING ONLY. No repository code was modified to produce this revision. No
training, no Kaggle kernel run, no LOEO run has been executed.**

Supersedes `GPU_SANITY_GATE_DESIGN_2026-07-18.md` (v1). This revision responds point by
point to the v1 review verdict: **APPROVE WITH CHANGES** (10 required changes). Each
change below is addressed with concrete repo evidence gathered for this revision, not
just rephrased — see the "New verification done for v2" callouts.

---

## Changelog vs. v1 (map to the 10 required changes)

| # | Required change | Where addressed |
|---|---|---|
| 1 | Subset restriction must happen before/during pair-index construction, with an assertion | §3, §17 item 1 |
| 2 | Prove the training subset has usable edge supervision before running | §3.1, §17 item 2 |
| 3 | Replace subjective PASS language with executable predicates | §11 |
| 4 | Detection gate: peak/GT ratio + explicit non-pathological criterion | §6, §11 |
| 5 | Quantitative edge-ranking success (mean comparison, ROC-AUC/AUPRC, insufficient-N behavior) | §7, §11 |
| 6 | Split wall-clock: training cap vs. overall gate cap; report actual timing | §10 |
| 7 | Sanity checkpoint must never call the manifest writer (not just be rejected by eligibility checks) | §9 |
| 8 | Updated REQUIRED INFRASTRUCTURE list | §18 |
| 9 | No code implemented | (this document is prose + pseudocode only) |
| 10 | Return v2 for review | this document |

Everything the review verdict asked to keep is retained: P0-7A + P0-7 hard
prerequisites (§2), primary 6bba→44b6 orientation (§3), whole-sample validation cap
(§3.2), separate fixed/adaptive threshold reporting (§6), five gradient probes (§7 —
now §8), zero-technical-fallback policy (§8 — now §9), biological-zero-edge vs.
technical-failure distinction (§9 — now §9), corrected-metric requirement (§2, §11),
SANITY vs. DEPLOYMENT checkpoint distinction (§12), and no P1 architecture changes
during the gate (unchanged, out of scope).

---

## New verification done for v2 (real repo evidence, not assumed)

- **Real GEFF ground truth loaded for all 4 locally-staged samples**
  (`tracksdata.graph.IndexedRXGraph.from_geff`, per `.claude/CLAUDE.md`'s documented
  loading method):

  | sample_id | nodes | edges | estimated_number_of_nodes (T_true, full-embryo) |
  |---|---|---|---|
  | `6bba_05b6850b` | 861 | 845 | 6,362 |
  | `6bba_05db0fb1` | 1,229 | 1,183 | 69,800 |
  | `44b6_0113de3b` | 52 | 50 | 25,755 |
  | `44b6_0b24845f` | 51 | 49 | 32,795 |

  Both real 6bba training recordings have hundreds to low-thousands of real GEFF
  lineage edges — strong direct evidence they contain genuine positive edge
  supervision. (`estimated_number_of_nodes` is the whole-embryo scoring-formula
  estimate, not something a per-frame peak count should ever be compared against
  directly — see §6.)
- **Per-timepoint sparse GT node density**, `6bba_05b6850b`: 100 distinct timepoints
  (t=0..99), per-timepoint node count ranges 6–11, mean 8.61. This is the real,
  measured order of magnitude of "how many labeled cells exist in one 3D frame" for
  this recording — used to ground the peak/GT ratio band in §6, instead of an
  arbitrary number.
- **`src/targets.py:197-336` (`generate_edge_targets`) read in full.** It already
  returns `num_positive_edges` and `num_negative_edges` per frame-pair — but
  `num_negative_edges` is **not** currently split into "hard" (both endpoints matched
  a real GT node, but no true edge between them) vs. "easy" (one/both endpoints
  matched no real GT node at all) negatives. This split does not exist yet and is
  needed for change 2/5's "hard-negative" requirement — see §17 item 3.
- **`src/dataset.py:48-110` (`CompetitionDataset.__init__`) read in full.**
  `self.sample_ids = split_data[split_type]` (line 102) runs immediately before
  `self.pairs = []` and the pair-index-build loop (line 109 onward). This is the
  exact, correct injection point for a training-subset allowlist — confirms the
  change-1 concern was valid: mutating `dataset.sample_ids` *after* `__init__`
  returns would have zero effect on `dataset.pairs`, which is already built.
  `filter_unannotated_pairs` (constructor param, default `False`) already exists and
  is documented as required for the real training dataset (`src/dataset.py:76-84`) —
  v1 did not set this explicitly; v2 does (§3).
- **`src/train.py` hyperparameter defaults confirmed**: `detection_threshold=0.5`,
  `edge_threshold=0.5` (`kaggle_kernel/train_kernel.py:210-211`, `src/train.py:306-307`),
  `max_positive_voxel_fraction` defaults to `0.005` (`src/train.py:182`) — the existing
  guard that silently switches to an adaptive threshold if the fixed threshold flags
  more than 0.5% of voxels as positive (`src/train.py:180-198`). This is real,
  already-shipped logic — v2 requires the gate to explicitly report whether this
  guard fired (§6, §11), rather than let a fixed-threshold number pass through without
  disclosing that it was actually adaptive underneath.
- **`scikit-learn>=1.2.0` is already a pinned dependency** (`requirements.txt:21`), not
  currently imported anywhere in `src/*.py`. Computing ROC-AUC/AUPRC for §7 is
  therefore a new *import*, not a new *dependency*.

---

## 2. Prerequisites (unchanged from v1)

1. P0-7A (metric parity) integrated and merged to the branch under test.
2. P0-7 (training-integrity freeze) integrated and merged to the branch under test.
3. Working tree clean at the SHA under test.
4. GPU confirmed available and stable for at least the projected runtime (§10).
5. `kaggle_src_dataset`/deployed-dataset mirror in sync with the SHA under test.

**If any of 1–5 is false: the gate does not run.**

---

## 3. Data selection (revised — change 1 and change 2)

**Primary orientation, unchanged: train=6bba, validate=44b6**, using the existing
`data_splits/embryo_44b6_validation.json` (already embryo-disjoint, no new split-file
logic).

### 3.0 Training-subset injection point (change 1)

The training subset **must** be applied as a constructor-level allowlist to
`CompetitionDataset`, applied immediately after `self.sample_ids = split_data[split_type]`
(`src/dataset.py:102`) and **before** the pair-index-build loop begins
(`src/dataset.py:109` onward) — never by mutating `dataset.sample_ids` on an
already-constructed instance. Concretely (design intent, not yet implemented):

```
# inside CompetitionDataset.__init__, immediately after line 102:
self.sample_ids = split_data[split_type]
if sample_id_allowlist is not None:
    allowlist_set = set(sample_id_allowlist)
    self.sample_ids = [s for s in self.sample_ids if s in allowlist_set]
    missing = allowlist_set - set(self.sample_ids)
    if missing:
        raise ValueError(f"sample_id_allowlist contains IDs not in split '{split_type}': {sorted(missing)}")
# ... existing pair-index build loop follows, now only over the filtered self.sample_ids
```

**Mandatory pre-training assertion** (fails loud, not a warning):

```
configured_ids = set(TRAINING_SAMPLE_IDS)          # the exact list from §3.1
actual_ids = {sample_id for sample_id, _ in train_dataset.pairs}
assert actual_ids == configured_ids, (
    f"Training pair index does not match configured sample IDs. "
    f"configured={sorted(configured_ids)} actual={sorted(actual_ids)}"
)
```

This assertion runs **after** `train_dataset` is constructed and **before** the first
optimizer step. It is a real, executable check — not a documentation note — and its
failure is an automatic gate abort (before any GPU time is spent training).

Also set `filter_unannotated_pairs=True` for this training dataset (existing
constructor param, `src/dataset.py:56,76-84` — documented as required for exactly this
use case: "the actual training dataset used for optimizer/backpropagation"). v1 did not
set this explicitly; leaving it at the default `False` would let zero-GT-node frame
pairs into the training subset, which the codebase's own docstring says is "actively
harmful during training." **Not doing this would have been a real correctness gap in
v1.**

### 3.1 Training subset content, and proof of usable edge supervision (change 2)

**Candidate selection procedure (deterministic, ordered, expandable):**

1. Start with `K=4`: the first 4 sample_ids from `embryo_44b6_validation.json`'s
   `train` list in file order — `6bba_05b6850b`, `6bba_05db0fb1`, `6bba_062c8d37`,
   `6bba_07477033`.
2. **Pre-flight check, required before any training step runs:** for every
   `(t, t+1)` frame-pair produced by `CompetitionDataset` over this candidate set
   (with `filter_unannotated_pairs=True`), call `generate_edge_targets()` (already
   exists, `src/targets.py:197`) and accumulate, across the whole candidate set:
   - `total_positive_edges = sum(num_positive_edges)`
   - `total_negative_edges = sum(num_negative_edges)`
   - `total_hard_negative_edges` (see §17 item 3 — requires a small metadata
     extension to `generate_edge_targets` that does not yet exist: split
     `num_negative_edges` into hard vs. easy by whether **both** endpoints matched
     a real GT node)
3. **Predicate that must hold before training starts:**
   `total_positive_edges > 0 AND total_negative_edges > 0 AND total_hard_negative_edges > 0`.
4. **If the predicate fails at K=4:** increment `K` by 1 (still first-`K`-in-file-order,
   still fully deterministic — no re-sampling, no shuffling) and re-check. Hard cap
   `K=12` (chosen to keep the training budget in §4 small even in the worst case —
   12 recordings is still a small multiple of the original 4, not a silent drift
   toward "just use everything").
5. **If the predicate still fails at `K=12`:** the gate does not proceed to training.
   Report a distinct blocked state — `"TRAINING SUBSET SELECTION EXHAUSTED — no
   deterministic prefix of the 6bba train list up to K=12 satisfies the edge-supervision
   predicate"` — and stop for human review. This is deliberately not a silent fallback
   to a larger K or a different embryo; it is a signal that something is wrong with
   edge-target generation or the split itself, worth root-causing before spending more
   GPU time.

**What's actually verified vs. what remains to be checked live:** real GEFF ground
truth was loaded for `6bba_05b6850b` (845 real lineage edges) and `6bba_05db0fb1`
(1,183 real lineage edges) — both strongly positive evidence the K=4 predicate is
likely to hold for at least those two. `6bba_062c8d37` and `6bba_07477033` are not
staged locally (only present on the Kaggle-mounted competition data) and **could not be
checked in this design pass** — this is stated honestly, not assumed. The pre-flight
check in step 2 is exactly the mechanism that makes this gap safe: it is a real,
executable check that runs before training, not a design-time assumption.

### 3.2 Validation subset (unchanged from v1)

`max_validation_samples=2` (existing hyperparam, `src/train.py:1001-1067`), selecting
the first 2 of the 71 `44b6` validation sample_ids in file order, each evaluated
**completely**. `validation_is_full_fold=False` (2 of 71) — deliberate, since this is
a sanity gate, not a deployment validation run (§12).

### 3.3 Seed and ordering (unchanged from v1)

`SEED=42` (matches `kaggle_kernel/train_kernel.py:150-151`). `shuffle=False` for both
train and validation loaders in the gate — determinism comes from the fixed,
asserted sample_id list (§3.0), not from trusting seeded-shuffle reproducibility
across environments.

---

## 4. Training budget (unchanged from v1)

- `max_batches_per_epoch=40` (existing hyperparam, `src/train.py:1000`).
- `num_epochs=3`.
- **Training hard cap: 20 minutes** (existing `max_wall_clock_seconds` param to
  `TrainingLoop.fit()`, `src/train.py:1323`).
- Checkpoint once, at the end of the final completed epoch.
- Validation once, after the final training epoch only.

---

## 5. Validation budget (unchanged from v1)

2 complete 44b6 samples (§3.2), evaluated once, full `validate_epoch()` path against the
P0-7A-corrected metric. No attempt to reach a competitive `val_score`; a structural
zero is still disallowed (§11).

---

## 6. Detection metrics and the non-pathological fixed-threshold criterion (change 4)

Reported, per validated timepoint:

| Metric | Definition |
|---|---|
| `fixed_threshold_peak_count` | peak count from `extract_peaks_from_volume()` at `detection_threshold=0.5`, **only counted if the adaptive-threshold guard did not fire** |
| `adaptive_threshold_triggered` | boolean — did `positive_fraction > max_positive_voxel_fraction (0.005)` force the adaptive path (`src/train.py:180-198`)? Reported for every timepoint, never silently absorbed |
| `adaptive_threshold_peak_count` | peak count under the adaptive path — reported **separately**, never substituted for `fixed_threshold_peak_count` |
| `gt_node_count_at_t` | real sparse GT node count at that exact timepoint (from the sample's `.geff`, via `graph.node_attrs(attr_keys=['t'])` filtered to `t`) |
| `peak_gt_ratio` | `fixed_threshold_peak_count / max(gt_node_count_at_t, 1)` |
| `precision_at_7um`, `recall_at_7um` | existing match logic vs. `DEFAULT_MAX_DISTANCE`/`DEFAULT_SCALE` |

**Explicit non-pathological fixed-threshold criterion (executable predicate):**

```
adaptive_threshold_triggered == False
AND 1 <= fixed_threshold_peak_count <= 500
AND peak_gt_ratio <= 50
```

Grounding for the numbers: real per-timepoint sparse GT node counts measured for
`6bba_05b6850b` range 6–11 (mean 8.61, §"New verification done for v2"). `peak_gt_ratio
<= 50` is loose enough to tolerate a severely undertrained checkpoint over-predicting
by more than an order of magnitude, while the absolute cap `<= 500` is well below the
documented pathological failure mode already on record in `.claude/CLAUDE.md`: the
stride-8 grid-scan degenerate case that produced **~18,000 false candidates per
timepoint** and caused a 2.5-hour ILP blowup. `500` is comfortably below `18,000` and
comfortably above the real ~6–11 GT baseline scaled by even a generous multiplier —
this is a deliberately wide, not-yet-tuned-for-quality band, whose only job is to catch
the known catastrophic failure signature, not to judge model quality.

---

## 7. Edge/transformer metrics and quantitative ranking success (change 5)

Collected once, on the final training epoch's batches plus the validation pass:

- `num_positive_edges`, `num_negative_edges` (already computed,
  `src/targets.py:321-322`).
- `num_hard_negative_edges`, `num_easy_negative_edges` — **new metadata field**,
  splitting `num_negative_edges` by whether both endpoints independently matched a
  real GT node (hard) or not (easy). Requires extending `generate_edge_targets`'s
  return dict (`src/targets.py:324-334`) — not yet implemented (§17 item 3).
- `true_edge_logits`: `edge_logits` (from `self.transformer(...)`, `src/train.py:808`)
  at positions where `edge_labels == 1`.
- `hard_negative_edge_logits`: same tensor, at positions where the new hard-negative
  mask is `True`.

**Executable predicates, in order of preference:**

1. **Minimum requirement (always evaluated):**
   `mean(true_edge_logits) > mean(hard_negative_edge_logits)`
2. **Preferred, when sample counts allow:**
   `ROC-AUC(true_edge_logits vs. hard_negative_edge_logits) > 0.5`
   `OR AUPRC > (num_true_edges / (num_true_edges + num_hard_negatives))` (i.e. AUPRC
   beats the positive-class prevalence baseline)
3. **Insufficient-sample behavior (explicit, not a silent skip):** if
   `count(true_edge_logits) < 5 OR count(hard_negative_edge_logits) < 5`, ROC-AUC/AUPRC
   are reported as `"NOT COMPUTED — insufficient samples (n_pos=X, n_neg=Y)"` and the
   gate's edge-ranking verdict falls back to predicate 1 alone. This is a defined,
   named state — not a silent pass, not a crash.

Computed via `sklearn.metrics.roc_auc_score` / `average_precision_score`
(`scikit-learn>=1.2.0` already pinned, `requirements.txt:21`) — a new import in the
gate-runner script, not a new dependency.

**Deliberate scope note:** ranking is computed against *hard* negatives specifically
(both endpoints real, no true link), not against the full negative pool including easy
negatives — including easy negatives would make the discrimination task trivially easy
and would not answer "does the transformer distinguish real-but-unlinked cells from
truly-linked ones," which is the actual question worth asking here.

---

## 8. Gradient requirements (unchanged from v1 — kept as-is per review verdict)

Measured once, at the last training batch of the final epoch, after `loss.backward()`
and before `optimizer.step()`/`zero_grad()`:

| Probe point | Exact attribute |
|---|---|
| Early UNet layer | `training_loop.unet3d.enc0[0].weight.grad` |
| Detection head | `training_loop.unet3d.det_head[-1].weight.grad` |
| Transformer node embedding | `training_loop.transformer.node_embed.weight.grad` |
| Transformer attention/encoder block | `training_loop.transformer.encoder_t.layers[0].self_attn.in_proj_weight.grad` |
| Transformer edge scorer | `training_loop.transformer.edge_scorer[0].weight.grad` |

**Fail condition, each probe:** `grad is None`, or `torch.isnan(grad).any()`, or
`torch.isinf(grad).any()`, or `grad.abs().max() == 0`. Any single failure is an
automatic gate **FAIL**.

---

## 9. Fallback/integrity gate and checkpoint provenance (change 7)

**Zero-technical-fallback policy, unchanged from v1:**

| Counter | Required value |
|---|---|
| `heatmap_generation_failure` | 0 |
| `edge_target_generation_failure` (technical) | 0 — distinct from legitimate zero-GT-edge batches |
| `edge_loss_computation_failure` | 0 |
| `evaluation_failure` | 0 |
| GT/GEFF load failure, missing sample, unreadable Zarr, malformed GEFF | 0 (surfaces as hard `RuntimeError`, `src/train.py:649,669,1054,1194`) |
| provenance mismatch (`training_code_sha`, `split_membership_sha256`) | 0 |

**Biological-vs-technical zero-edge distinction, unchanged:** a batch with a real,
correctly generated target that happens to contain zero GT edges is legitimate and
tallied separately; only technical failure counters gate the run.

**Checkpoint save path (change 7 — structural, not just eligibility rejection):**

v1 relied on `checkpoint_manifest.deployment_eligibility_errors()` to reject a
partial-fold checkpoint after the fact. v2 requires the gate runner to **structurally
never call `write_checkpoint_manifest()` at all** for a sanity checkpoint — not call it
and expect rejection. Concretely: the gate-runner script's checkpoint-save step must
call `save_checkpoint_file()` (`src/checkpoint_manifest.py:79`) directly and then write
its own gate-report JSON containing `"deployment_manifest": "NOT GENERATED — sanity
checkpoint only, validation_is_full_fold=False"`. There is no code path in the
gate-runner script that imports or invokes `write_checkpoint_manifest()`. This removes
the manifest-eligibility-rejection safety net as the *only* line of defense — it
becomes belt-and-suspenders (the function still exists and would still reject this
checkpoint if somehow called), but the primary guarantee is structural absence of the
call.

---

## 10. Runtime and memory (change 6)

- **Training hard cap: 20 minutes** (unchanged, §4).
- **Overall gate hard cap: ≤40 minutes**, including validation and report generation
  (not just training) — a new, explicit ceiling distinct from the training-only cap.
- Both caps are enforced independently: hitting the training cap aborts training and
  the gate proceeds to report a `FAIL` (incomplete training run) rather than silently
  extending; hitting the overall cap aborts whatever step is in progress and reports
  `FAIL` (gate did not complete within budget).
- **Actual timing is reported separately from any projection**: wall-clock per training
  step (measured mean), validation wall-clock per sample (measured, both of the 2
  samples), total measured wall-clock, and only then a *computed* (not assumed) short-LOEO
  projection = measured mean step time × planned short-LOEO step count.
- Peak `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`,
  measured, not estimated.
- No fixed ceiling is asserted in advance for the short-LOEO projection — consistent
  with "do not assume the old 12-hour estimate remains valid." The gate's job is to
  produce the number; the go/no-go judgment on that number happens at the next stop-go
  checkpoint (§13), informed by whatever the actual competition-environment budget is
  at that time.

---

## 11. Explicit PASS rules (change 3 — every clause below is an executable predicate; no subjective language remains)

All of the following true simultaneously:

- `no NaN/Inf in any logged loss or metric value` (explicit `torch.isnan`/`torch.isinf`
  check on every logged tensor).
- All 5 gradient probes (§8) satisfy: not `None`, no NaN, no Inf, `abs().max() > 0`.
- All technical fallback/integrity counters (§9) `== 0`.
- **Detection loss trend, executable form:** `mean(detection_loss over epoch 3 batches)
  <= mean(detection_loss over epoch 1 batches)` **OR**
  `mean(gt_center_sigmoid over epoch 3) - mean(background_sigmoid over epoch 3) >
  mean(gt_center_sigmoid over epoch 1) - mean(background_sigmoid over epoch 1)`.
  (Replaces v1's "positive learning trend.")
- **Detection non-pathological criterion, executable form (§6):**
  `adaptive_threshold_triggered == False AND 1 <= fixed_threshold_peak_count <= 500 AND
  peak_gt_ratio <= 50`, evaluated per validated timepoint, required to hold for both of
  the 2 validation samples. (Replaces v1's "plausible order of magnitude.")
- `recall_at_7um > 0` on at least one of the 2 validation samples (unchanged, already
  numeric).
- **Edge-ranking criterion, executable form (§7):**
  `mean(true_edge_logits) > mean(hard_negative_edge_logits)`, plus (when
  `count(true_edge_logits) >= 5 AND count(hard_negative_edge_logits) >= 5`) either
  `ROC-AUC > 0.5` or `AUPRC > positive-class prevalence`. (Replaces v1's "ranking
  signal.")
- `validation_samples_evaluated == 2` exactly (no silently-skipped sample).
- `predicted_nodes_total > 0 AND predicted_edges_total > 0 AND is_structural_zero ==
  False`.
- `evaluate_submission()` executed under `training_code_sha` matching a commit that
  post-dates the P0-7A merge commit (checked by SHA ancestry, not assumed).
- `measured_training_wall_clock <= 1200s AND measured_total_gate_wall_clock <= 2400s`.

## 12. Explicit CONDITIONAL PASS rules

Downgrades a PASS to CONDITIONAL only if one or more of:

- Detector-side predicates all pass, but the edge-ranking criterion satisfies only the
  minimum mean-comparison clause (predicate 1 of §7) — insufficient sample count or
  ROC-AUC ≤ 0.5 / AUPRC ≤ prevalence.
- `recall_at_7um > 0` but numerically low (fewer than, e.g., 2 matched GT nodes across
  both validation samples combined) — real signal, not yet strong.
- `measured_total_gate_wall_clock` is within 2400s but the computed short-LOEO
  projection is uncomfortably close to (without clearly exceeding) whatever budget is
  agreed at that time.

CONDITIONAL PASS → re-run with a modestly larger budget (same, or `K+1`-expanded per
§3.1's deterministic procedure, sample selection), not a direct jump to short LOEO.

## 13. Explicit FAIL rules

Any single one, automatic FAIL, no rescue:

- Any technical fallback/integrity counter nonzero.
- Any gradient probe `None`/zero/NaN/Inf.
- `is_structural_zero == True`, `predicted_nodes_total == 0`, or
  `predicted_edges_total == 0`.
- `validation_samples_evaluated != 2`.
- Detection loss NaN/Inf at any point.
- Provenance mismatch (`training_code_sha` or `split_membership_sha256` don't match the
  SHA under test).
- **Adaptive-threshold peak count reported in place of, or used to rescue, a failing
  fixed-threshold criterion** — an explicit, named trap, still forbidden in v2.
- Gate executed while P0-7A or P0-7 is not merged into the SHA under test.
- **Training-subset selection exhausted (§3.1 step 5)** without satisfying the
  edge-supervision predicate up to `K=12` — new in v2, a direct consequence of change 2.
- `train_dataset.pairs`'s actual sample IDs do not exactly equal the configured
  training sample ID list (§3.0's mandatory assertion) — new in v2, a direct
  consequence of change 1.
- `measured_training_wall_clock > 1200s` or `measured_total_gate_wall_clock > 2400s`
  without the run having reached a conclusive PASS/FAIL verdict on the metrics above —
  new in v2, a direct consequence of change 6.

## 14. Stop/go decision tree (unchanged in structure from v1; conditions now point at the executable predicates above)

```
P0-7A + P0-7 not both integrated at test SHA?
  -> STOP. Do not run the gate.

Training-subset pre-flight predicate (§3.1) fails up to K=12?
  -> STOP. Root-cause edge-target generation / split content before any GPU spend.

Gate FAILs any §13 condition?
  -> Diagnose root cause locally/offline first. Do not re-run unchanged.

Gate PASSes fully (§11)?
  -> GO: short primary LOEO (6bba->44b6, larger budget, same orientation).

Gate CONDITIONAL PASS (§12)?
  -> Re-run with a modestly larger budget / expanded K before committing to LOEO.

Detector predicates PASS, transformer/edge predicates FAIL (zero/NaN gradient, or
edge-ranking criterion not even computable due to a real structural break, not just
insufficient-N)?
  -> Do NOT proceed to LOEO. Root-cause the edge/transformer path in isolation first.

Transformer PASSes but validation score is poor?
  -> Proceed cautiously to short primary LOEO; do not treat this as evidence for or
     against any P1 architecture change.

Runtime/memory projection incompatible with a short LOEO's realistic budget?
  -> Do NOT proceed to LOEO at current config. Re-measure with one deliberate knob
     changed, per CLAUDE.md's existing "profile before raising" rule.

Everything above resolved and short primary LOEO passes its own, separately-defined
gate?
  -> Reciprocal LOEO (44b6->6bba, data_splits/embryo_6bba_validation.json).

Both LOEO orientations pass?
  -> P1 architecture experiments become in-scope. Not before.
```

## 15. Estimated GPU time

Training: ≤20 min hard cap (§4/§10). Overall gate: ≤40 min hard cap including
validation and reporting (§10, new in v2). Actual measured timing is reported
separately and is the number that matters for any subsequent projection — the estimate
here is only a planning-time budget, not a claim about real performance.

## 16. Exact command/config changes likely needed (design only — not applied)

- Hyperparams: `max_batches_per_epoch=40`, `num_epochs=3`,
  `max_wall_clock_seconds=1200` (training), `overall_gate_wall_clock_seconds=2400`
  (new, enforced by the gate-runner script itself, not by `TrainingLoop`),
  `max_validation_samples=2`, `batch_size=1`, `SEED=42`,
  `sample_id_allowlist=["6bba_05b6850b","6bba_05db0fb1","6bba_062c8d37","6bba_07477033"]`
  (or the K-expanded list per §3.1), `filter_unannotated_pairs=True` for the training
  dataset only.
- `train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)`,
  `val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)`.
- Pre-flight edge-supervision check (new script or function) run before
  `TrainingLoop.fit()` is called at all.
- Post-construction sample-ID assertion (§3.0) run immediately after `train_dataset`
  is built.
- Gradient-probe capture wired into the final training batch only.
- Gate-runner script writes its own JSON report; never imports
  `write_checkpoint_manifest`.

## 17. Repository changes required later to actually run the gate

1. **`CompetitionDataset` constructor**: add `sample_id_allowlist: list[str] | None =
   None`, applied at `src/dataset.py:102` (immediately after `self.sample_ids =
   split_data[split_type]`), before the pair-index build loop. Raise `ValueError` if
   any allowlist ID isn't in the split. (§3.0)
2. **Pre-flight edge-supervision check script**: iterates the candidate training
   window's frame-pairs, calls `generate_edge_targets()` per pair, accumulates
   `total_positive_edges`/`total_negative_edges`/`total_hard_negative_edges`, checks
   the §3.1 predicate, and implements the deterministic `K→K+1` expansion up to
   `K=12`. (§3.1)
3. **`generate_edge_targets` metadata extension** (`src/targets.py:324-334`): split
   `num_negative_edges` into `num_hard_negative_edges` (both endpoints matched a real
   GT node, no true edge) and `num_easy_negative_edges` (the rest). Small, additive,
   backward-compatible (existing `num_negative_edges` field stays, new fields added).
4. GT-vs-background sigmoid mean/median instrumentation (extends existing
   `sig_min`/`sig_max`, `src/train.py:950-951`).
5. Peak/GT ratio + `adaptive_threshold_triggered` boolean surfaced from
   `extract_peaks_from_volume()` (`src/train.py:90-198`) — the adaptive-vs-fixed
   branch already exists; it just needs to return which branch actually ran, plus a
   GT-node-count-at-t lookup from the sample's `.geff`.
6. True-edge vs. hard-negative logit collection + ROC-AUC/AUPRC computation
   (`sklearn.metrics.roc_auc_score`/`average_precision_score` — new import, existing
   pinned dependency) with the insufficient-N fallback behavior (§7).
7. Gradient-probe capture code (the 5 attribute reads, §8).
8. Gate-runner script itself, including the mandatory sample-ID assertion (§3.0), the
   dual wall-clock enforcement (training vs. overall, §10), and the sanity-checkpoint
   save path that never calls `write_checkpoint_manifest()` (§9).

## 18. Classification of each item in §17 (change 8 — updated)

| # | Item | Classification |
|---|---|---|
| 1 | `sample_id_allowlist` constructor param + pre-pair-index filtering + post-construction assertion | **REQUIRED INFRASTRUCTURE** — v1 under-specified this; v2 makes it a precise, minimal `src/dataset.py` change plus a runner-side assertion |
| 2 | Pre-flight edge-supervision predicate check (with deterministic K-expansion) | **REQUIRED INFRASTRUCTURE** — new in v2, direct response to change 2 |
| 3 | `generate_edge_targets` hard/easy negative split | **REQUIRED INFRASTRUCTURE** — new in v2; without it, "hard-negative edge-target count > 0" cannot be checked at all |
| 4 | GT-vs-background sigmoid mean/median | **REQUIRED INFRASTRUCTURE** — unchanged from v1 |
| 5 | Peak/GT ratio + adaptive-threshold-triggered flag | **REQUIRED INFRASTRUCTURE** — new in v2, direct response to change 4 |
| 6 | True-edge/hard-negative logits + ROC-AUC/AUPRC | **REQUIRED INFRASTRUCTURE**, but low-risk (`sklearn` already pinned, not yet imported in `src/`) — direct response to change 5 |
| 7 | Gradient-probe capture code | **OPTIONAL INSTRUMENTATION** in size (a few log lines), functionally mandatory for the gradient-evidence goal — unchanged from v1 |
| 8 | Gate-runner script (assertion, dual wall-clock, non-manifest checkpoint save) | **REQUIRED INFRASTRUCTURE** — unchanged from v1, sharpened in v2 |
| — | Split file (`embryo_44b6_validation.json`) | **NOT REQUIRED** — already exists, already validated |
| — | `max_batches_per_epoch`, `max_validation_samples` caps | **NOT REQUIRED** — already exist (P0-4) |
| — | `filter_unannotated_pairs` | **NOT REQUIRED** — already exists (`src/dataset.py:56,76-84`); just needs to be set `True` in the gate's config, which v1 omitted |
| — | Embryo-disjointness validation | **NOT REQUIRED** — already exists (`split_utils.py`) |
| — | Checkpoint schema / manifest contract fields | **NOT REQUIRED** — already exist (P0-6, `checkpoint_manifest.py`) |
| — | Global seeding | **NOT REQUIRED** — already exists (`SEED=42`, `kaggle_kernel/train_kernel.py:150-151`) |
| — | `num_positive_edges`/`num_negative_edges` (undifferentiated) | **NOT REQUIRED** — already exists (`src/targets.py:321-322`); only the hard/easy split (item 3) is new |

---

## Requested from Codex

Ten required changes from the v1 review are addressed above, each grounded in fresh
repo evidence (real GEFF node/edge counts, exact constructor/line references, real
hyperparameter defaults) gathered specifically for this revision — not just reworded.
Still planning only: nothing has been implemented, no GPU/Kaggle run has occurred.

Please respond with one of:

- **APPROVE** — ready to move to implementing §17/§18's REQUIRED INFRASTRUCTURE items
  (still gated on P0-7A + P0-7 landing first, per §2).
- **APPROVE WITH CHANGES** — list the specific section(s) and exact change needed.
- **REQUEST CHANGES** — list what's wrong or unsafe as designed, with the specific
  section/claim in question.

Two items worth your explicit attention, since they involve honest gaps rather than
settled facts:

1. §3.1's edge-supervision predicate is verified against real GEFF data for 2 of the 4
   proposed training sample_ids (`6bba_05b6850b`, `6bba_05db0fb1` — both have hundreds
   of real lineage edges); the other 2 (`6bba_062c8d37`, `6bba_07477033`) are not
   staged locally and could not be checked in this design pass. The pre-flight check
   (§17 item 2) is the real safety net here — confirm you're comfortable with that,
   or ask for a different verification path before implementation.
2. §6's peak/GT ratio band (`<=50`, absolute cap `500`) is grounded in one recording's
   real per-timepoint GT density (6–11 nodes/frame) and the historical ~18,000-candidate
   pathological incident already on record in `.claude/CLAUDE.md` — it is deliberately
   loose (a catastrophic-failure catch, not a quality bar). Flag if you want it tighter
   or want it derived per-sample instead of from a single reference recording.
