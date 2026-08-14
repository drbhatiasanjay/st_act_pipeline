# GPU Sanity Gate — Design for Review (2026-07-18)

**Status: PLANNING ONLY. No repository code was modified to produce this design. No
training, no Kaggle kernel run, no LOEO run has been executed.**

This document is the deliverable of a read-only design task: define a deterministic,
fail-closed GPU sanity gate that runs *after* P0-7A (metric parity) and P0-7
(training-integrity freeze) are integrated, and that answers whether the current
architecture (UNet3D + SimpleNodeTransformer) shows real, non-collapsed, dual-branch
learning before any LOEO-scale or P1 spend.

Written for Codex to review and either **APPROVE** or **REQUEST CHANGES** (see the
request at the end). Everything below was checked against the real repository state at
this SHA before being written — not guessed or templated.

---

## 0. Grounding facts (verified against the real repo, not assumed)

- **HEAD note:** the SHA quoted in the originating task prompt
  (`4901d450cb12ed6577633b71d4f50bc1bc3c6`) is missing 3 hex characters compared to the
  real `git rev-parse HEAD` (`4901d450cb12ed6577634633b71d4f50bc1bc3c6`). Same commit
  (`git log` confirms `4901d45` = "fix(submission): enforce verified deployment and
  canonical graph assembly") — flagging the mismatch for the record, not blocking.
- Working tree has uncommitted changes (`.claude/CLAUDE.md`, `bash.exe.stackdump`,
  `training_log_smoke_test.csv`) and three new untracked planning docs. None touch
  `src/`, so none affect this design — but the gate must not be **run** against a dirty
  tree (see §9, prereq 3).
- **The primary LOEO orientation (train=6bba, validate=44b6) already exists as real,
  validated infrastructure**: `data_splits/embryo_44b6_validation.json` —
  `train_embryos=['6bba']` (128 samples), `validation_embryos=['44b6']` (71 samples),
  embryo-disjoint (`src/split_utils.py`), built by `scripts/build_train_val_split.py`
  (commits `14d0cdb`, `3cf2a45`). **No new split file is needed for the fold itself.**
- Sample IDs are `{embryo}_{field_of_view}` — one per real recording. The 128 6bba
  entries are 128 *distinct* recordings, not 128 frame-pairs from one recording. Only 2
  of the 128 (`6bba_05b6850b`, `6bba_05db0fb1`) and 2 of the 71 (`44b6_0113de3b`,
  `44b6_0b24845f`) are physically staged under `data/staging/`; the rest exist only on
  the Kaggle-mounted competition dataset — **this gate is a Kaggle-GPU gate, not a local
  one**, consistent with "local sustained PyTorch/IO execution is environmentally
  unstable."
- **`max_batches_per_epoch`** (`src/train.py:1000`, wired at `:701-705`) already caps
  training steps. **`max_validation_samples`** (`src/train.py:1001-1067`) already caps
  validation by *whole sample_id*, tracks `validation_is_full_fold`, and is exactly the
  "cap by whole sample count only" mechanism this design needs — **not something to
  build**.
- **No training-side equivalent exists** for restricting to specific sample_ids (only
  validation has `allowed_sample_ids`). `CompetitionDataset` (`src/dataset.py:245,102`)
  iterates `sample_ids` in split-file list order, deterministically — this ordering is
  what the training-subset selection below relies on.
- Model attribute paths (`src/train.py:283-284`, `src/model.py`): `self.unet3d` (early
  layer `unet3d.enc0`, head `unet3d.det_head`), `self.transformer`
  (`SimpleNodeTransformer`: `node_embed`, `encoder_t` — `nn.TransformerEncoder` —,
  `edge_scorer`). Checkpoint keys: `unet3d_state_dict`, `transformer_state_dict`
  (`src/checkpoint_manifest.py:56-66`).
- Manifest contract fields are exact and already enforced
  (`src/checkpoint_manifest.py:38-54,382-488`): `model_contract` must equal
  `"edge_logits_v1"`, `is_structural_zero` must be `False`, plus
  `validation_is_full_fold`/`validation_samples_evaluated`/`validation_samples_total`,
  `training_code_sha`, `split_membership_sha256`, node/edge totals,
  `adjusted_edge_jaccard`.
- `evaluate_checkpoint.py` is a **CPU-only, locally-capped** tool (`max_pairs`, capped
  specifically to dodge a reproducible local Windows segfault at batch ~33 — not a
  deliberate sanity-budget design). Useful as a cross-check only; the formal gate's
  validation must go through `TrainingLoop.validate_epoch()` on the real GPU run.
- `src/evaluation.py:36` `DEFAULT_SCALE=(1.625,0.40625,0.40625)` is present now, but
  P0-7A ("metric parity") is stated to be in progress separately — this design treats
  current `evaluate_submission()` as **not yet the corrected metric** until P0-7A lands.

---

## 1. Gate name

**GPU-SANITY-GATE-01** ("the v-first-light gate"): a fail-closed, deterministic,
GPU-executed check that a trained checkpoint exhibits real, non-collapsed, dual-branch
learning on the real embryo-disjoint 6bba→44b6 fold, before any LOEO-scale or P1 spend.

## 2. Prerequisites (hard blockers — gate must not run without all of these)

1. P0-7A (metric parity) integrated and merged to the branch under test.
2. P0-7 (training-integrity freeze) integrated and merged to the branch under test.
3. Working tree clean at the SHA under test (no uncommitted `src/` changes).
4. GPU confirmed available and stable for at least the projected runtime (§10) at gate
   time.
5. `kaggle_src_dataset`/deployed-dataset mirror in sync with the SHA under test if run on
   Kaggle (per `scripts/sync_kaggle_src.py`'s existing verify step).

**If any of 1–5 is false: the gate does not run.** Not a soft recommendation — this is
the single most important fail-closed condition in this design, because a "pass"
measured against pre-P0-7A metric code or pre-P0-7 fallback plumbing is evidence about
the wrong system.

## 3. Data selection

**Primary orientation: train=6bba, validate=44b6**, using the existing
`data_splits/embryo_44b6_validation.json` as the base split (already embryo-disjoint,
already validated — no new split-membership logic).

- **Training subset:** first **4** sample_ids from that file's `train` list, in file
  order: `6bba_05b6850b`, `6bba_05db0fb1`, `6bba_062c8d37`, `6bba_07477033`.
  Deterministic (fixed list order, no RNG dependency), genuinely 4 distinct recordings
  (not one pair repeated), includes both recordings already physically staged locally
  plus 2 more from the Kaggle mount.
- **Validation subset:** `max_validation_samples=2`, which — combined with existing
  P0-4 logic (`src/train.py:1032-1067`) — selects the **first 2** of the 71 44b6
  validation sample_ids in file order and evaluates each **completely** (all its
  frame-pairs), never truncating within a sample. This explicitly sets
  `validation_is_full_fold=False` (2 of 71 selected) — deliberately, since this is a
  sanity gate, not a deployment validation run (see §9).
- **Seed:** `SEED=42`, matching `kaggle_kernel/train_kernel.py:150-151`'s existing
  global seeding, set before model init and DataLoader construction, unchanged.
- **DataLoader shuffle:** `shuffle=False` for the sanity-gate's train_loader (deviating
  from the production kernel's `shuffle=True`). Determinism comes from the *fixed
  sample_id list*, not from trusting global-RNG-seeded shuffle order to reproduce
  identically across environments/PyTorch versions. `val_loader` stays `shuffle=False`
  (already required by `PredictionGraphAssembler`'s chronological-order invariant,
  `src/train.py:995-998`).
- **Selection is fixed, not sampled** — no `random.choice`/stratification anywhere in
  this plan.

## 4. Training budget

- **Max optimizer steps:** `max_batches_per_epoch=40` (existing hyperparam,
  `src/train.py:1000`).
- **num_epochs:** 3.
- **Max wall-clock:** hard cap 20 minutes total training (existing
  `max_wall_clock_seconds` param to `TrainingLoop.fit()`, `src/train.py:1323`).
- **Checkpoint frequency:** once, at the end of the final completed epoch.
- **Validation frequency:** once, after the final training epoch only.

## 5. Validation budget

2 complete 44b6 samples (§3), evaluated once, full `validate_epoch()` path (real peak
extraction, real graph assembly, real `evaluate_submission()` call against the
P0-7A-corrected metric). Explicit non-goal: no attempt to reach a competitive
`val_score` — low/near-zero is expected and acceptable here; a **structural zero** is
not (see §11-13).

## 6. Metrics table

| Category | Metric | Source |
|---|---|---|
| Detection | detection loss (per batch + epoch mean) | `TrainingLoop.train_epoch()` |
| Detection | GT-center sigmoid mean / median | extend existing `sig_min/sig_max` instrumentation (`src/train.py:950-951`) |
| Detection | background sigmoid statistic (mean/median away from GT) | same instrumentation point |
| Detection | GT/background separation | derived (GT mean − background mean) |
| Detection | max sigmoid | already computed (`sig_max`, `:950-951`) |
| Detection | precision @ 7µm, recall @ 7µm | `src/evaluation.py` match logic vs. `DEFAULT_MAX_DISTANCE`/`DEFAULT_SCALE` |
| Detection | fixed-threshold peak count | `extract_peaks_from_volume()`, fixed threshold |
| Detection | adaptive-threshold peak count | same function, adaptive path — **logged separately, never substituted for the fixed-threshold number** |
| Detection | NMS matched-GT count, false-positive peak count | derived from precision/recall computation |
| Edge/transformer | edge-supervised batch count, legitimate zero-edge-target batch count | `epoch_fallback_counts` bookkeeping (`:409-413`) — needs a third counter distinguishing real-batch-zero-edges from fallback |
| Edge/transformer | edge-target/edge-loss failure counts | `epoch_fallback_counts['edge_target_generation_failure']`, `['edge_loss_computation_failure']` (already exist) |
| Edge/transformer | true-edge vs. hard-negative logit distributions + ranking | new instrumentation around `edge_logits = self.transformer(...)` (`:808`) |
| Edge/transformer | edge AUPRC | derived if hard-negative sampling ships with P0-7; else explicitly marked "not feasible this gate" |
| Edge/transformer | candidate edge count, accepted edge count | graph assembly / tracker candidate generation |

## 7. Gradient requirements

Measured **once, at the last training batch of the final epoch**, after
`loss.backward()` and before `optimizer.step()`/`zero_grad()`:

| Probe point | Exact attribute |
|---|---|
| Early UNet layer | `training_loop.unet3d.enc0[0].weight.grad` |
| Detection head | `training_loop.unet3d.det_head[-1].weight.grad` |
| Transformer node embedding | `training_loop.transformer.node_embed.weight.grad` |
| Transformer attention/encoder block | `training_loop.transformer.encoder_t.layers[0].self_attn.in_proj_weight.grad` |
| Transformer edge scorer | `training_loop.transformer.edge_scorer[0].weight.grad` |

**Fail condition, each probe:** `grad is None`, or `torch.isnan(grad).any()`, or
`torch.isinf(grad).any()`, or `grad.abs().max() == 0`. Any single failure among the five
is an automatic gate **FAIL** — this directly answers "do both UNet and transformer
receive real gradients," and a `None`/zero transformer gradient while UNet gradients are
healthy is exactly the "detector learns, transformer doesn't" split scenario §14 plans
for.

## 8. Fallback/integrity gate

Expected counters after P0-7:

| Counter | Required value |
|---|---|
| `heatmap_generation_failure` | 0 |
| `edge_target_generation_failure` (technical) | 0 — **distinct from** legitimate zero-GT-edge batches, counted separately and allowed to be nonzero |
| `edge_loss_computation_failure` | 0 |
| `evaluation_failure` | 0 |
| GT/GEFF load failure, missing expected sample, unreadable Zarr, malformed GEFF | 0 (should surface as hard `RuntimeError`s already, `src/train.py:649,669,1054,1194`, not silent counter increments) |
| provenance mismatch (`training_code_sha`, `split_membership_sha256`) | 0 — checked via existing `checkpoint_manifest.py` validation |

**Biological-vs-technical zero-edge distinction:** a batch with a real, correctly
generated target that happens to contain zero GT edges (e.g. no true divisions/links in
that frame pair) is legitimate and must be tallied in a dedicated counter, never merged
into `edge_target_generation_failure`. This gate fails only on the *technical* failure
counters being nonzero.

## 9. Checkpoint/provenance requirements

**Two distinct checkpoint classes — do not conflate them:**

- **SANITY CHECKPOINT** (what this gate produces): satisfies checkpoint schema,
  `model_contract="edge_logits_v1"`, real `training_code_sha`, real
  `split_membership_sha256`, valid state dicts and hyperparams — but
  `validation_is_full_fold=False` (2 of 71 samples) and `validation_samples_evaluated <
  validation_samples_total`. **No `checkpoint_manifest.json` may be written for this
  checkpoint** — writing one would make it silently `load_verified_checkpoint()`-eligible
  for submission, which it is not. State explicitly in the gate's output artifact:
  `"deployment_manifest": "NOT GENERATED — sanity checkpoint only, validation_is_full_fold=False"`.
- **DEPLOYMENT-ELIGIBLE CHECKPOINT**: everything above, plus `validation_is_full_fold=True`
  (all 71, or a full reciprocal/complete fold), passing every check in
  `checkpoint_manifest.deployment_eligibility_errors()`. Out of scope for this gate.

## 10. Runtime/memory requirements

Record, no threshold guessed from the old 12h estimate:

- Wall-clock per training step (mean, over the ~120 executed steps).
- Validation wall-clock per sample (2 measured samples).
- Total run wall-clock (target: well under 30 minutes given §4/§5's budget).
- Peak `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
- **Projected short-LOEO time** = measured mean step time × planned short-LOEO step
  count — computed once the gate's numbers exist, not assumed in advance.

## 11. Explicit PASS rules

All of the following true simultaneously:

- No NaN/Inf anywhere in loss or logged metrics.
- All 5 gradient probes (§7) finite and nonzero.
- All technical fallback/integrity counters (§8) exactly 0.
- Detection loss trend across the 3 epochs is non-increasing on net (batch-level noise
  allowed), or GT/background sigmoid separation increases epoch-over-epoch.
- Fixed-threshold peak count is nonzero and in a plausible order of magnitude (not the
  stride-8 grid-scan degenerate case).
- Recall @ 7µm > 0 on at least one validation sample.
- True-edge logits rank above hard-negative logits on average.
- Validation coverage matches the configured cap exactly (`validation_samples_evaluated
  == 2`, no silently-skipped sample).
- `predicted_nodes_total > 0` and `predicted_edges_total > 0`; `is_structural_zero ==
  False`.
- `evaluate_submission()` ran under the P0-7A-corrected metric code (verified by
  checking the deployed `training_code_sha`/module against the P0-7A merge commit).
- Total wall-clock under the §10 budget with a computed short-LOEO projection.

## 12. Explicit CONDITIONAL PASS rules

Any of these, and only these, downgrade a PASS to CONDITIONAL:

- Detector-side metrics all PASS, but transformer/edge metrics show real nonzero
  gradients yet no clear true-edge/hard-negative ranking signal.
- Recall @ 7µm is positive but low (e.g. single digits of matched GT).
- Runtime projection for short-LOEO is uncomfortably close to (but not clearly over)
  whatever budget is agreed at that time.

CONDITIONAL PASS means: proceed to a slightly larger, still-short diagnostic (more
steps/epochs, same fold), not directly to full LOEO.

## 13. Explicit FAIL rules

Any single one of these is an automatic FAIL, with no rescue:

- Any technical fallback/integrity counter nonzero.
- Any gradient probe None/zero/NaN/Inf.
- `is_structural_zero == True`, or `predicted_nodes_total == 0`, or
  `predicted_edges_total == 0`.
- `validation_samples_evaluated != 2` (silently skipped or over-ran the cap).
- Detection loss NaN/Inf at any point.
- Provenance mismatch: `training_code_sha` or `split_membership_sha256` don't match the
  SHA under test.
- **Adaptive-threshold peak count used to claim fixed-threshold success** — if the
  fixed-threshold count is degenerate but adaptive threshold "looks fine," that is a
  FAIL, not a pass-by-substitution.
- Gate executed while P0-7A or P0-7 is not actually merged into the SHA under test.

## 14. Stop/go decision tree

```
P0-7A + P0-7 not both integrated at test SHA?
  -> STOP. Do not run the gate.

Gate FAILs any §13 condition?
  -> Diagnose root cause locally/offline first before spending more GPU time.
     Do not re-run the same gate unchanged hoping for a different result.

Gate PASSes fully (§11)?
  -> GO: proceed to short primary LOEO (6bba->44b6, larger budget, same orientation).

Gate CONDITIONAL PASS (§12)?
  -> Re-run gate with a modestly larger budget (same 4+2 sample selection)
     before committing to short LOEO.

Detector metrics PASS, transformer/edge metrics FAIL?
  -> Do NOT proceed to LOEO. Root-cause the edge/transformer path in isolation
     (e.g. a focused unit test forcing known nonzero edge targets through
     transformer.edge_scorer) before spending GPU time on a longer run.

Transformer PASSes but validation score is poor?
  -> Proceed cautiously to short primary LOEO, but do NOT yet interpret this
     as evidence for or against any P1 architecture change.

Runtime/memory projection incompatible with a short LOEO's realistic budget?
  -> Do NOT proceed to LOEO at the current batch_size/candidate-cap config.
     Re-measure with one deliberate knob changed (e.g. MAX_CANDIDATES_PER_TIMEPOINT)
     rather than guessing a fix.

Everything above resolved and short primary LOEO passes its own gate?
  -> Reciprocal LOEO (44b6->6bba, data_splits/embryo_6bba_validation.json,
     already exists, same treatment).

Both LOEO orientations pass?
  -> P1 architecture experiments become in-scope. Not before.
```

## 15. Estimated GPU time

Training: <=20 min (hard cap, §4). Validation: a few minutes for 2 complete samples
(order-of-magnitude estimate; GPU validation should be faster than
`evaluate_checkpoint.py`'s CPU-only per-sample timings, but this is an estimate to be
replaced by the gate's own measurement). **Total estimate: 25-35 minutes.**

## 16. Exact command/config changes likely needed (design only — not applied)

- Hyperparams dict: `max_batches_per_epoch=40`, `num_epochs=3`,
  `max_wall_clock_seconds=1200`, `max_validation_samples=2`, `batch_size=1`, `SEED=42`.
- `train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)` (deviates from
  production kernel's `shuffle=True` — sanity-gate-only change, documented as such).
- A small script (Kaggle-kernel-style, mirroring `kaggle_kernel/train_kernel.py`'s
  structure) that: loads the existing `data_splits/embryo_44b6_validation.json`,
  restricts `train_dataset.sample_ids` to the first 4 entries (needs a constructor
  param or post-init filter — see §17 item 1), runs `TrainingLoop.fit(num_epochs=3,
  max_wall_clock_seconds=1200)`, then one `validate_epoch()` call, then writes a
  gate-report JSON (metrics table + PASS/CONDITIONAL/FAIL verdict) instead of a
  `checkpoint_manifest.json`.

## 17. Repository changes required later to actually run the gate

1. A way to restrict `CompetitionDataset`'s training-side `sample_ids` to a fixed small
   list (no existing equivalent to validation's `allowed_sample_ids` on the training
   side).
2. Extended instrumentation for GT-center-vs-background sigmoid mean/median (currently
   only min/max are computed, `src/train.py:950-951`).
3. Instrumentation for true-edge-logit vs. hard-negative-logit distributions and the
   ranking check (scope depends on what P0-7 actually lands for edge supervision —
   cannot be finalized pre-P0-7).
4. A distinct fallback counter for "legitimate zero-GT-edge batch" separate from
   `edge_target_generation_failure`, if P0-7 doesn't already add this.
5. The gate-runner script itself (§16).
6. Gradient-probe capture code (the 5 attribute reads in §7), wrapped so it runs once
   post-backward on the final batch without disturbing the existing training loop.
7. A "sanity checkpoint" save path that explicitly skips `write_checkpoint_manifest()`
   and instead writes a plain marker (`deployment_manifest: NOT GENERATED`) — reusing
   `save_checkpoint_file()` but not the manifest-writing call.

## 18. Classification of each item in §17

| # | Item | Classification |
|---|---|---|
| 1 | Training-side sample_id restriction | **REQUIRED INFRASTRUCTURE** — the gate cannot select "4 of 128" recordings without it |
| 2 | GT-vs-background sigmoid mean/median instrumentation | **REQUIRED INFRASTRUCTURE** — directly named in the metrics list |
| 3 | True-edge vs. hard-negative logit distribution/ranking instrumentation | **REQUIRED INFRASTRUCTURE**, scope pending P0-7's actual edge-supervision design |
| 4 | Legitimate-zero vs. technical-failure edge counter split | **REQUIRED INFRASTRUCTURE** if P0-7 doesn't already provide it; **NOT REQUIRED** if it does — check P0-7's actual diff once it lands |
| 5 | Gate-runner script | **REQUIRED INFRASTRUCTURE** |
| 6 | Gradient-probe capture code | **OPTIONAL INSTRUMENTATION** in size (a few log lines), but functionally mandatory for the gradient-evidence goal |
| 7 | Sanity-checkpoint save path (no manifest) | **REQUIRED INFRASTRUCTURE** — without it, a partial-fold checkpoint risks acquiring full deployment-manifest status by accident |
| — | Split file (`embryo_44b6_validation.json`) | **NOT REQUIRED** — already exists, already validated |
| — | `max_batches_per_epoch`, `max_validation_samples` caps | **NOT REQUIRED** — already exist (P0-4) |
| — | Embryo-disjointness validation | **NOT REQUIRED** — already exists (`split_utils.py`) |
| — | Checkpoint schema / manifest contract fields | **NOT REQUIRED** — already exist (P0-6, `checkpoint_manifest.py`) |
| — | Global seeding | **NOT REQUIRED** — already exists (`SEED=42`, `kaggle_kernel/train_kernel.py:150-151`) |

---

## Requested from Codex

This document is planning only — nothing above has been implemented, and no GPU/Kaggle
run has occurred. Please review against the actual current repo state at your end and
respond with one of:

- **APPROVE** — design is sound as written; ready to move to implementing §16/§17's
  REQUIRED INFRASTRUCTURE items (still gated on P0-7A + P0-7 landing first, per §2).
- **APPROVE WITH CHANGES** — list the specific section(s) and the exact change needed.
- **REQUEST CHANGES** — list what's wrong or what would make the gate unsafe/unreliable
  as designed, with the specific section/claim in question.

If anything in §0's grounding facts doesn't match what you see in the repo (file paths,
line numbers, split-file contents, existing hyperparam names), flag it explicitly —
those claims were verified against the repo at the time of writing, not assumed, but a
second independent check is exactly the point of this review step.
