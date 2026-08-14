# GPU Sanity Gate — Design v4 (2026-07-18)

**Status: PLANNING ONLY. No repository code was modified to produce this revision. No
training, no Kaggle kernel run, no LOEO run has been executed. No commit, no push.**

Supersedes `GPU_SANITY_GATE_DESIGN_2026-07-18_v3.md`. Responds to the v3 review
verdict: **APPROVE WITH 2 FINAL MINOR CORRECTIONS**. Per the reviewer's framing, this
is expected to be the final frozen design unless P0-7's actual implementation
introduces a genuine incompatibility.

## Prerequisite status (re-verified, unchanged from v3)

- `git rev-parse HEAD` = `8eeace8ab090aecb1cfad336f82865f752c825c4` — re-checked again
  for this revision, still an exact match. P0-7A: **satisfied**.
- P0-7 (training-integrity freeze): **not integrated**. No corresponding commit exists
  in `git log`. **The formal gate remains blocked. Nothing in this document authorizes
  running anything.**

---

## Changelog vs. v3 (map to the 2 required corrections)

| # | Required correction | Where addressed |
|---|---|---|
| 1 | Remove/correct wording implying an isolated technical edge-target failure is tolerable; make the zero-tolerance policy explicit and non-contradictory | §8 |
| 2 | Reuse P0-6's actual canonical source-provenance helpers instead of inventing a second interpretation | §9 |

Everything else from v3 is retained unchanged: constructor-level `sample_id_allowlist`
before pair-index construction (§3.0); exact pair-index sample-ID assertion (§3.0);
`filter_unannotated_pairs=True` (§3.0); deterministic K=4..12 pre-flight expansion
(§3.1); positive/negative/hard-negative supervision predicate (§3.1); hard/easy
negative distinction (§7); quantitative edge-ranking criteria (§7); per-timepoint
(not averaged) fixed/adaptive threshold rule (§6); dual wall-clock limits (§10);
reordered checkpoint lifecycle — train → validate → verdict → save (§10);
zero-technical-fallback policy (§9); biological-zero vs. technical-failure distinction
(§8, §9); SANITY vs. DEPLOYMENT checkpoint separation (§10, §12); no P1 architecture
changes (out of scope, unchanged).

---

## New verification done for v4

**Correction 2 required checking whether v3's claim ("module-origin verification is
genuinely new, no existing equivalent found") was actually true. It was not — this was
a real gap in v3, found by re-reading code that v3 had not checked closely enough.**

Read `kaggle_kernel_inference/inference_kernel.py` in full for its P0-6 provenance
logic (the file's own docstring labels these "Part C1"–"Part C4" — real, already
integrated, already running in production):

- **`find_all_kaggle_input_dirs(marker_relpath)` / `find_exactly_one_kaggle_input_dir(marker_relpath)`**
  (lines 33-56, "Part C1"): walks `/kaggle/input` up to `MAX_SEARCH_DEPTH=5`, collects
  every directory containing a marker file (`src/dataset.py`), and **raises** if zero or
  more than one match — never silently picks the first, never uses directory order or
  mtime. This *is* the canonical "exact-one source dataset discovery" mechanism.
- **GIT_SHA.txt strict validation** (lines 89-104, "Part C3"): raises if the file is
  missing, empty/whitespace-only, or not exactly 40 lowercase hex characters — stricter
  than what v2/v3 described (which only checked existence + string equality). **Real
  divergence found**: `kaggle_kernel/train_kernel.py:112` (the training kernel, not the
  inference kernel) still uses the older, looser pre-P0-6 pattern —
  `DEPLOYED_SHA = "unknown (GIT_SHA.txt not found...)"` as a silent fallback string,
  with no format validation and no raise. **This is exactly the failure mode
  `.claude/CLAUDE.md` already warns about** ("a fix living in one kernel script does
  not automatically apply to a sibling script") — the strict version shipped in P0-6
  for `inference_kernel.py` but was never carried over to `train_kernel.py`. This is
  direct, current evidence supporting Correction 2's concern: two independently
  evolving interpretations of the same "verify the mounted source" idea already exist
  in this repository today, not just a hypothetical risk.
- **`verify_import_origins(expected_root)`** (lines 205-235, "Part C2"): resolves
  `__file__` for every production module the kernel imports
  (`src.dataset, src.model, src.train, src.submission_pipeline,
  src.checkpoint_manifest, src.submission_exporter, src.prediction_graph,
  src.inference`), and for each, calls `resolved.relative_to(expected_root_resolved)`
  inside a `try/except ValueError`, raising `RuntimeError` on any module resolving
  outside the expected root. **This is the exact module-origin verification v3 claimed
  didn't exist.** It does exist, is already running in the real inference kernel, and
  uses `Path.relative_to()` + `except ValueError` (not `Path.is_relative_to()`, which
  v3's pseudocode used) — a real, minor implementation-pattern detail that matters for
  "reuse the same semantics," not a rewrite of it.
- **`main()`** (lines 242-250) ties it together: `find_single_manifest("/kaggle/input")`
  (from `src/checkpoint_manifest.py:265`) + `load_verified_checkpoint(manifest_path,
  expected_source_sha=DEPLOYED_SHA, ...)` (`src/checkpoint_manifest.py:643`, the exact
  string-equality check already cited in v3 §9).

**Conclusion: v4 does not invent anything for provenance verification. It cites and
reuses the exact three pieces above, verbatim in logic, and additionally flags the
real `train_kernel.py`/`inference_kernel.py` divergence already found — which the gate
design should not paper over, but should require be resolved (or explicitly bypassed
with a stated reason) before the gate can claim to reuse "the canonical P0-6
semantics" rather than a third, different-again copy.**

---

## 2. Prerequisites (unchanged)

1. P0-7A — **satisfied** (`git rev-parse HEAD == 8eeace8ab090aecb1cfad336f82865f752c825c4`).
2. P0-7 — **not satisfied**.
3. Working tree clean at the SHA under test.
4. GPU confirmed available and stable for the projected runtime (§10).
5. Deployed-dataset mirror in sync with the SHA under test — verified by §9's exact
   equality checks, not assumed.

**If any of 1–5 is false: the gate does not run. Item 2 is currently false.**

---

## 3. Data selection (unchanged from v3)

### 3.0 Training-subset injection point

`sample_id_allowlist` constructor param on `CompetitionDataset`, applied immediately
after `self.sample_ids = split_data[split_type]` (`src/dataset.py:102`), before the
pair-index-build loop (`:109` onward):

```
self.sample_ids = split_data[split_type]
if sample_id_allowlist is not None:
    allowlist_set = set(sample_id_allowlist)
    self.sample_ids = [s for s in self.sample_ids if s in allowlist_set]
    missing = allowlist_set - set(self.sample_ids)
    if missing:
        raise ValueError(f"sample_id_allowlist contains IDs not in split '{split_type}': {sorted(missing)}")
```

Mandatory pre-training assertion after `train_dataset` construction, before any
optimizer step:

```
configured_ids = set(TRAINING_SAMPLE_IDS)
actual_ids = {sample_id for sample_id, _ in train_dataset.pairs}
assert actual_ids == configured_ids
```

`filter_unannotated_pairs=True` for the training dataset (`src/dataset.py:56,76-84`).

### 3.1 Training subset content and edge-supervision pre-flight

Deterministic candidates: `K=4` — `6bba_05b6850b`, `6bba_05db0fb1`, `6bba_062c8d37`,
`6bba_07477033` (first 4 of `embryo_44b6_validation.json`'s `train` list). Pre-flight
predicate before any training step: `total_positive_edges > 0 AND total_negative_edges
> 0 AND total_hard_negative_edges > 0`, via `generate_edge_targets()`
(`src/targets.py:197`). Expand `K→K+1` deterministically up to `K=12` if it fails;
`FAIL: "TRAINING SUBSET SELECTION EXHAUSTED"` beyond that.

Real GEFF evidence: `6bba_05b6850b` has 845 real lineage edges, `6bba_05db0fb1` has
1,183 (both verified via `IndexedRXGraph.from_geff` against the real locally-staged
files). `6bba_062c8d37`/`6bba_07477033` remain unverified locally; the pre-flight check
is the real safety net for those.

### 3.2 Validation subset

`max_validation_samples=2` (`src/train.py:1001-1067`), first 2 of the 71 `44b6`
validation sample_ids, each evaluated completely. `validation_is_full_fold=False`.

### 3.3 Seed and ordering

`SEED=42`, `shuffle=False` for both loaders.

---

## 4. Training budget (unchanged)

`max_batches_per_epoch=40`, `num_epochs=3`, training hard cap 1200s
(`max_wall_clock_seconds`), checkpoint save deferred to after verdict computation
(§10), validation once after the final epoch.

## 5. Validation budget (unchanged)

2 complete 44b6 samples, full `validate_epoch()` path, P0-7A-corrected metric. No
competitive-score expectation; structural zero still disallowed (§11).

---

## 6. Detection metrics — per-timepoint pathology rule (unchanged from v3)

For **every** evaluated validation timepoint (not averaged, not aggregated before the
pass/fail decision):

| Metric | Definition |
|---|---|
| `fixed_threshold_peak_count[t]` | peak count at `detection_threshold=0.5` |
| `adaptive_threshold_triggered[t]` | boolean (`src/train.py:180-198`) |
| `adaptive_threshold_peak_count[t]` | reported separately, never substituted |
| `gt_node_count_at_t[t]` | real sparse GT node count at this `t`, from `.geff` |
| `peak_gt_ratio[t]` | `fixed_threshold_peak_count[t] / max(gt_node_count_at_t[t], 1)` |

Per-timepoint predicate, must hold at **every** evaluated `t`:

```
adaptive_threshold_triggered[t] == False
AND 1 <= fixed_threshold_peak_count[t] <= 500
AND peak_gt_ratio[t] <= 50
```

A single failing timepoint fails the containing sample's detection criterion, full
stop — no averaging it away. Precision/recall are aggregated at sample level only
*after* the per-timepoint gate is confirmed to hold everywhere in that sample. Full
per-timepoint table required in the report, not just a sample-level summary.

Numeric grounding unchanged: real per-timepoint GT density for `6bba_05b6850b` (6–11
nodes/frame, mean 8.61) and the documented ~18,000-candidate pathological incident on
record in `.claude/CLAUDE.md`.

---

## 7. Edge/transformer metrics (unchanged from v3)

`num_positive_edges`, `num_negative_edges` (existing, `src/targets.py:321-322`);
new `num_hard_negative_edges`/`num_easy_negative_edges` split. `true_edge_logits`,
`hard_negative_edge_logits` from the transformer's per-batch output.

1. `mean(true_edge_logits) > mean(hard_negative_edge_logits)` — always evaluated.
2. When `count(true_edge_logits) >= 5 AND count(hard_negative_edge_logits) >= 5`:
   `ROC-AUC > 0.5 OR AUPRC > positive-class prevalence`.
3. Below that count: `"NOT COMPUTED — insufficient samples (n_pos=X, n_neg=Y)"`, fall
   back to predicate 1 alone.

---

## 8. Gradient evidence — corrected wording, zero-tolerance made explicit (Correction 1)

**v3's introductory wording for this section described a caught
`edge_target_generation_failure` as "not itself an integrity violation if isolated."
That sentence is retracted — it contradicts this design's own zero-technical-fallback
policy (§9) and is removed. There is no isolated-failure tolerance anywhere in this
gate.** The two rules below are independent and must never be conflated:

### 8.0 Two independent rules — stated with no ambiguity

**Rule A — gradient-capture eligibility** (a bookkeeping/measurement concern only):
when deciding *which single batch's gradients to snapshot* for the transformer probes,
a batch where `generate_edge_targets()` raised (caught, `edge_targets = None`) is
simply **not eligible to be the captured batch** — the transformer's forward pass never
ran for it, so there is nothing meaningful to snapshot. This is purely about not
picking a meaningless batch as the measurement point.

**Rule B — overall gate pass/fail** (a correctness/integrity concern, absolute, no
exceptions): **any `edge_target_generation_failure` count greater than zero, anywhere
in the run, fails the overall gate.** This was already true in v1/v2/v3's
zero-technical-fallback policy (§9) and remains true, unchanged, unweakened, in v4.
There is no count of technical failures — not one, not "isolated," not "explained
away" — that is tolerable. A batch being ineligible for gradient capture (Rule A) says
nothing about whether it's allowed to have occurred at all (Rule B says it isn't, if
it was a technical failure).

**The only thing allowed to reduce the pool of gradient-capture-eligible batches
without failing the gate is a *legitimate biological zero-positive-edge batch* — a
batch where `generate_edge_targets()` succeeded, returned a real target tensor, but
that target genuinely contains zero true links** (e.g. no GT divisions/continuations
in that exact frame pair). This is categorically different from a technical failure:

| Case | `generate_edge_targets()` outcome | Counted toward | Gate effect |
|---|---|---|---|
| **TECHNICAL EDGE-TARGET FAILURE** | raised an exception, caught | `edge_target_generation_failure` counter | **Count > 0 anywhere → gate FAILS** (§9, §13) |
| **LEGITIMATE ZERO-POSITIVE-EDGE BATCH** | succeeded, `num_positive_edges == 0` | a separate, allowed "biological zero" counter (not a failure counter) | Allowed; simply not eligible for transformer-gradient capture (Rule A) |

### 8.1 Two independently-tracked capture targets (unchanged mechanism from v3)

Both updated conditionally, live, during the existing training loop — no extra
forward/backward pass:

- **UNet gradient evidence**: captured (overwriting any prior capture) immediately
  after `total_loss_item.backward()` (`src/train.py:834`) on every batch where
  detection-loss computation succeeded (no `heatmap_generation_failure` for that
  batch).
- **Transformer gradient evidence**: captured (overwriting any prior capture) at the
  same point, only on batches satisfying all of: `edge_targets is not None` (i.e., not
  a technical failure — Rule A/B distinction above), `len(edge_logits) > 0`, and
  `edge_metadata['num_positive_edges'] > 0`.

### 8.2 Counters (unchanged from v3, now unambiguously scoped)

- `edge_target_generation_failure`: existing technical-failure counter
  (`src/train.py:410,802`). **Must be exactly 0 for the gate to pass — this is Rule B,
  restated, and it is not softened by anything in §8.0/§8.1.**
- `legitimate_zero_positive_edge_batches`: new counter, batches where
  `generate_edge_targets()` succeeded but `num_positive_edges == 0`. Allowed to be
  nonzero; never gates the run.
- `edge_supervised_batches_total`: count of batches satisfying all three conditions in
  §8.1's transformer-capture rule (real target, real logits, real positive edges).
  Must be `> 0`.
- `edge_supervised_batches_with_nonzero_transformer_grad`: of those, count where the
  captured gradient probes were finite and nonzero. Must be `> 0`.

### 8.3 Fail conditions

- **`edge_target_generation_failure > 0` anywhere in the run → gate FAILS**, named
  reason `"TECHNICAL EDGE-TARGET GENERATION FAILURE"` — unconditional, no isolated-
  occurrence exception (this is the corrected statement of Correction 1).
- **`edge_supervised_batches_total == 0`** (i.e., every batch was either a technical
  failure — already independently fatal per the rule above — or a legitimate
  zero-positive-edge batch) → FAIL, named reason `"NO EDGE-SUPERVISED BATCH"`.
- Given at least one edge-supervised batch exists and zero technical failures
  occurred: the transformer gradient probes are evaluated against the *last*
  edge-supervised batch's captured snapshot. A legitimate zero-positive-edge batch
  elsewhere in the run never counts against this check — it simply was never a capture
  candidate.

### 8.4 Probe points and fail conditions (unchanged)

| Probe point | Exact attribute | Captured at |
|---|---|---|
| Early UNet layer | `training_loop.unet3d.enc0[0].weight.grad` | last valid detection-supervised batch |
| Detection head | `training_loop.unet3d.det_head[-1].weight.grad` | last valid detection-supervised batch |
| Transformer node embedding | `training_loop.transformer.node_embed.weight.grad` | last edge-supervised batch |
| Transformer attention/encoder block | `training_loop.transformer.encoder_t.layers[0].self_attn.in_proj_weight.grad` | last edge-supervised batch |
| Transformer edge scorer | `training_loop.transformer.edge_scorer[0].weight.grad` | last edge-supervised batch |

Fail condition, each probe: `grad is None`, NaN, Inf, or `abs().max() == 0`.

---

## 9. Provenance — reuse P0-6's canonical helpers exactly (Correction 2, rewritten)

**v3 stated module-origin verification was "genuinely new, no existing equivalent
found." That was incorrect — it exists, is real, already runs in production, and this
section now cites it directly instead of re-deriving it.**

The gate reuses, verbatim in logic, three pieces already implemented in
`kaggle_kernel_inference/inference_kernel.py` (P0-6, "Part C1"/"Part C2"/"Part C3"):

1. **Exact-one source-root discovery** — `find_exactly_one_kaggle_input_dir(marker_relpath)`
   (`inference_kernel.py:47-56`, built on `find_all_kaggle_input_dirs`, lines 33-44):
   walks `/kaggle/input` for a directory containing a marker file (`src/dataset.py`),
   raises if zero or more than one match. The gate must call this same function (or an
   identical shared copy — see below), not a re-derived version.
2. **`GIT_SHA.txt` strict validation** — the exact block at `inference_kernel.py:89-104`:
   file must exist, must be non-empty after stripping whitespace, and must be exactly
   40 lowercase hex characters; any violation raises. **Not** `train_kernel.py`'s
   looser current behavior (`:112`, silent `"unknown (...)"` fallback string, no format
   check) — that older pattern must not be the one the gate reuses.
3. **`verify_import_origins(expected_root)`** — the exact function at
   `inference_kernel.py:205-235`: for every production module the gate run imports,
   resolve `__file__`, call `.relative_to(expected_root_resolved)` inside
   `try/except ValueError`, raise `RuntimeError` on any module resolving outside the
   verified root. The gate's own module list must be adapted to whatever it actually
   imports (at minimum `src.dataset`, `src.model`, `src.train`, `src.targets`,
   `src.evaluation`, `src.checkpoint_manifest`, `src.split_utils`), but the
   verification *mechanism* — `Path.relative_to()` + `except ValueError`, not
   `Path.is_relative_to()` — must match exactly, so a future review of both call sites
   sees one pattern, not two that happen to currently agree.

**Combined with the already-cited exact-equality checkpoint check**
(`src/checkpoint_manifest.py:685`, `load_verified_checkpoint`'s
`manifest["training_code_sha"] != expected_source_sha`), all four pieces together give
the required invariant:

> Every production module executed by the gate resolves physically inside the single
> verified source mount associated with the exact source SHA under test — checked by
> reusing the same four mechanisms P0-6 already established, not a fifth, independently
> maintained interpretation of what "under mounted source" means.

**A real, current gap this verification surfaced, which the gate design must not
paper over:** `train_kernel.py` and `inference_kernel.py` already disagree on how
strictly `GIT_SHA.txt` is validated (§"New verification done for v4"). Before the gate
reuses "the canonical P0-6 semantics," this design requires one of:

- (preferred) extracting `find_exactly_one_kaggle_input_dir`,
  `verify_import_origins`, and the strict `GIT_SHA.txt` validation block out of
  `inference_kernel.py` into a shared `src/` module (e.g.
  `src/deployment_provenance.py`), imported identically by `train_kernel.py`,
  `inference_kernel.py`, and the new gate-runner script — this is the only way to
  guarantee there is exactly one interpretation, not three independently-copied ones
  that could silently drift (the exact failure class `.claude/CLAUDE.md` already
  documents from the polars `--force-reinstall` incident); or
- at minimum, before implementation, `train_kernel.py`'s looser SHA handling is
  brought up to the same strict standard as `inference_kernel.py`'s, so the gate isn't
  reusing "P0-6 semantics" from one script while a sibling script silently still runs
  the pre-P0-6 pattern.

This is listed as new REQUIRED INFRASTRUCTURE in §17/§18 — not implemented now, per
"do not implement code."

**If any of the four checks fails: automatic FAIL, named reason
`"SOURCE PROVENANCE MISMATCH"`, reported with which of the four checks failed and the
actual vs. expected values.**

---

## 10. Checkpoint lifecycle (unchanged from v3)

```
1. train  (§4)
2. validate  (§5, §6)
3. compute gate verdict + full report  (§11/§12/§13, including §8's gradient checks
   and §9's provenance checks — all evaluated before step 4)
4. save sanity checkpoint + gate report together
```

Saved artifact must include, verbatim:

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

(plus every metric/predicate result from §6-§9, §11-§13 — this is the minimum
explicitly required subset, not the full schema.)

**The gate runner must never import or call `write_checkpoint_manifest()`** —
structural, not just relying on `deployment_eligibility_errors()` rejecting it after
the fact. Saves via `save_checkpoint_file()` directly (`src/checkpoint_manifest.py:79`)
plus its own differently-named gate-report JSON.

SANITY vs. DEPLOYMENT distinction unchanged (§12): this gate only ever produces a
SANITY CHECKPOINT.

---

## 11. Explicit PASS rules

All simultaneously:

- No NaN/Inf in any logged loss or metric.
- UNet gradient probe (last valid detection-supervised batch): finite, nonzero.
- Transformer gradient probes (last edge-supervised batch): finite, nonzero, AND
  `edge_supervised_batches_total > 0` AND
  `edge_supervised_batches_with_nonzero_transformer_grad > 0`.
- **`edge_target_generation_failure == 0`** and all other technical fallback/integrity
  counters `== 0` (§8.0 Rule B, §9) — unconditional, restated explicitly per
  Correction 1.
- Detection loss trend: epoch-3 mean ≤ epoch-1 mean, or GT/background sigmoid
  separation increases epoch 1 → epoch 3.
- Per-timepoint detection pathology predicate (§6) holds at every evaluated timepoint
  in both validation samples.
- `recall_at_7um > 0` on at least one of the 2 validation samples.
- Edge-ranking criterion (§7): mean comparison holds, plus ROC-AUC/AUPRC when sample
  counts allow.
- `validation_samples_evaluated == 2` exactly.
- `predicted_nodes_total > 0 AND predicted_edges_total > 0 AND is_structural_zero ==
  False`.
- All four §9 provenance checks pass exactly (source-root discovery succeeds
  exact-one; `GIT_SHA.txt` strictly valid and equal to `exact_source_sha_under_test`;
  checkpoint's `training_code_sha` equal to the same; every imported production module
  resolves under the verified root).
- `measured_training_wall_clock <= 1200s AND measured_total_gate_wall_clock <= 2400s`.

## 12. Explicit CONDITIONAL PASS rules (unchanged)

- Edge-ranking satisfies only the minimum mean-comparison clause.
- `recall_at_7um > 0` but numerically low.
- Wall-clock within budget but computed short-LOEO projection is close to (not clearly
  over) the budget agreed at that time.

## 13. Explicit FAIL rules

Any single one, automatic FAIL:

- **`edge_target_generation_failure > 0`** → named reason
  `"TECHNICAL EDGE-TARGET GENERATION FAILURE"` (Correction 1 — no isolated-occurrence
  exception, ever).
- Any other technical fallback/integrity counter nonzero.
- UNet or transformer gradient probe None/zero/NaN/Inf on their respective qualifying
  batch.
- `edge_supervised_batches_total == 0` → named reason `"NO EDGE-SUPERVISED BATCH"`.
- `is_structural_zero == True`, `predicted_nodes_total == 0`, or
  `predicted_edges_total == 0`.
- `validation_samples_evaluated != 2`.
- Detection loss NaN/Inf at any point.
- Any of the four §9 provenance checks fails → named reason
  `"SOURCE PROVENANCE MISMATCH"`.
- Adaptive-threshold peak count substituted for, or used to rescue, a failing
  fixed-threshold criterion at any single evaluated timepoint.
- Gate executed while P0-7A or P0-7 is not merged into the SHA under test (P0-7
  currently unmet).
- Training-subset selection exhausted at K=12 without satisfying the edge-supervision
  predicate.
- `train_dataset.pairs`'s actual sample IDs don't exactly equal the configured training
  sample ID list.
- `measured_training_wall_clock > 1200s` or `measured_total_gate_wall_clock > 2400s`
  without a conclusive verdict already reached.

## 14. Stop/go decision tree

```
P0-7A + P0-7 not both integrated at test SHA?
  -> STOP. Do not run the gate. (P0-7 currently unmet.)

Any §9 provenance check fails before training starts?
  -> STOP. Fix the deployment/mount/import-path issue first.

Training-subset pre-flight predicate (§3.1) fails up to K=12?
  -> STOP. Root-cause edge-target generation / split content first.

edge_target_generation_failure > 0 anywhere during the run?
  -> FAIL: TECHNICAL EDGE-TARGET GENERATION FAILURE. No exceptions. Root-cause
     before any re-run.

edge_supervised_batches_total == 0 after training (and zero technical failures)?
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

## 16. Exact command/config changes likely needed (design only — not applied)

- Hyperparams: `max_batches_per_epoch=40`, `num_epochs=3`,
  `max_wall_clock_seconds=1200`, `overall_gate_wall_clock_seconds=2400`,
  `max_validation_samples=2`, `batch_size=1`, `SEED=42`, `sample_id_allowlist=[...]`,
  `filter_unannotated_pairs=True`.
- `exact_source_sha_under_test` passed explicitly, checked against checkpoint,
  `GIT_SHA.txt` (strict format), and module `__file__` origins (§9) before training
  starts — reusing `find_exactly_one_kaggle_input_dir` and `verify_import_origins`
  verbatim (or their shared-module extraction, see §9).
- Two live-updated gradient-snapshot variables inside the training loop (§8.1).
- New counters: `edge_target_generation_failure` (existing, now explicitly
  zero-tolerance restated), `legitimate_zero_positive_edge_batches` (new, allowed
  nonzero), `edge_supervised_batches_total`,
  `edge_supervised_batches_with_nonzero_transformer_grad` (new).
- Checkpoint/report save moved to after verdict computation.

## 17. Repository changes required later to actually run the gate (updated)

1. `sample_id_allowlist` constructor param + pre-pair-index filtering + post-
   construction assertion (`src/dataset.py:102`).
2. Pre-flight edge-supervision predicate check with deterministic K-expansion (§3.1).
3. `generate_edge_targets` hard/easy negative split (`src/targets.py:324-334`).
4. GT-vs-background sigmoid mean/median instrumentation (`src/train.py:950-951`).
5. Per-timepoint peak/GT ratio + `adaptive_threshold_triggered` reporting table (§6).
6. True-edge/hard-negative logit collection + ROC-AUC/AUPRC (`sklearn`, pinned).
7. Two independently-tracked gradient-snapshot capture points inside `train_epoch`,
   plus the new `legitimate_zero_positive_edge_batches` /
   `edge_supervised_batches_total` /
   `edge_supervised_batches_with_nonzero_transformer_grad` counters (§8).
8. **Provenance verification — reused, not invented** (§9): call
   `find_exactly_one_kaggle_input_dir`, the strict `GIT_SHA.txt` validation block, and
   `verify_import_origins`, all currently living in
   `kaggle_kernel_inference/inference_kernel.py:33-104,205-235`. **New recommended
   sub-item**: extract these three into a shared `src/` module so `train_kernel.py`,
   `inference_kernel.py`, and the gate-runner script all import one identical
   implementation, closing the real divergence found during this revision (v3
   `train_kernel.py` still uses the pre-P0-6 loose SHA pattern).
9. Gate-runner script: sample-ID assertion, dual wall-clock enforcement, reordered
   lifecycle (train → validate → verdict → save), non-manifest checkpoint save path,
   gate-report JSON with the exact required fields (§10).

## 18. Classification of each item in §17

| # | Item | Classification |
|---|---|---|
| 1 | `sample_id_allowlist` + assertion | **REQUIRED INFRASTRUCTURE** |
| 2 | Pre-flight edge-supervision check | **REQUIRED INFRASTRUCTURE** |
| 3 | Hard/easy negative split | **REQUIRED INFRASTRUCTURE** |
| 4 | GT-vs-background sigmoid mean/median | **REQUIRED INFRASTRUCTURE** |
| 5 | Per-timepoint peak/GT ratio + adaptive-trigger table | **REQUIRED INFRASTRUCTURE** |
| 6 | ROC-AUC/AUPRC | **REQUIRED INFRASTRUCTURE**, low-risk (`sklearn` pinned) |
| 7 | Dual live gradient-snapshot capture + new counters | **REQUIRED INFRASTRUCTURE** |
| 8 | Provenance verification reuse (source-root discovery, strict `GIT_SHA.txt`, module-origin check) | **NOT REQUIRED to build — already exists** in `inference_kernel.py`; **REQUIRED INFRASTRUCTURE** only for the shared-module extraction (closing the `train_kernel.py` divergence) — this is a refactor/consolidation task, not new logic |
| 9 | Gate-runner script (assertion, dual wall-clock, reordered lifecycle, non-manifest save, exact report schema) | **REQUIRED INFRASTRUCTURE** |
| — | Split file, `max_batches_per_epoch`/`max_validation_samples`, `filter_unannotated_pairs`, embryo-disjointness validation, checkpoint schema/manifest fields, global seeding, `num_positive_edges`/`num_negative_edges`, checkpoint-SHA exact-equality check | **NOT REQUIRED** — all already exist |

---

## Requested from Codex

Both corrections addressed:

1. §8 no longer contains any language suggesting an isolated technical
   `edge_target_generation_failure` is tolerable. The zero-tolerance rule
   (`edge_target_generation_failure > 0` anywhere → gate FAIL, no exceptions) is now
   stated independently of, and without conflict with, the separate gradient-capture-
   eligibility rule (which only concerns which batch to snapshot, never whether the
   run is allowed to have had a technical failure at all).
2. §9 no longer invents a second provenance mechanism. It cites and reuses
   `find_exactly_one_kaggle_input_dir`, the strict `GIT_SHA.txt` validation block, and
   `verify_import_origins` — all three read directly from
   `kaggle_kernel_inference/inference_kernel.py` for this revision — and additionally
   surfaces a real, currently-existing divergence (`train_kernel.py` still uses the
   looser pre-P0-6 SHA-handling pattern) that should be resolved as part of
   implementing this gate, not left as a second interpretation running alongside the
   correct one.

This design is now presented as the frozen v4, per your framing, pending only P0-7's
actual landing. **Still planning only** — nothing implemented, no GPU run, no commit,
no push. If P0-7's real implementation introduces something incompatible with any
predicate above (most likely candidates: the exact shape of the new technical-failure
counters it introduces, or a change to `generate_edge_targets`'s metadata contract),
that would warrant a v5 addressing the specific incompatibility — not a speculative
revision now.
