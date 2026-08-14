# P0-7 TRAINING INTEGRITY — FROZEN SPECIFICATION v1 — FINAL CANDIDATE
<!-- 2026-07-19. Incorporates Draft v1 code-evidence findings + second-prompt
     adversarial-review deltas + 12 reviewer decisions above.
     Reviewer decisions override all prior draft language on conflict.
     DO NOT IMPLEMENT until this document is marked FROZEN. -->

---

## REPOSITORY BASELINE

- **Exact HEAD**: `8eeace8ab090aecb1cfad336f82865f752c825c4`
- **Meaning**: P0-7A competition-metric parity integrated; last merged prior fix
- All `src/*.py` are byte-identical to `kaggle_src_dataset/src/` (verified MD5 at session start)
- P0-1 through P0-7A are CLOSED — do not reopen, do not touch their test suites beyond confirming they still pass

---

## CONFIRMED DEFECTS AT HEAD (evidence-based; all file:line verified)

### F1 — `_get_gt_nodes()` silently swallows ALL technical failures
**File**: `src/train.py:569–592`
**Evidence**: entire function body inside `try/except Exception: return None`
**Problem**: technical geff parse error and legitimately-absent geff are identical to the caller; no counter; no audit trail; a parse crash silently produces the same None as "this frame is unannotated"
**Assigned handler class**: `SILENTLY_CONTINUED` → must become `FATAL` on any technical error

### F2 — Edge supervision silently skipped for retained pairs with no counter
**File**: `src/train.py:770`
**Evidence**: `if nodes_t is not None and nodes_t1 is not None and nodes_t.shape[0] > 0 and nodes_t1.shape[0] > 0:` — entire edge block skipped when any frame is None or empty
**Problem**: for a pair retained by `filter_unannotated_pairs=True`, both frames are guaranteed ≥1 GT node by the filter; None or empty tensor reaching `train_epoch` for a retained pair is a technical failure, not a biological zero — but it produces no counter and no error
**Assigned handler class**: `SILENTLY_CONTINUED` → must become `FATAL` for retained pairs

### F3 — `generate_edge_targets()` failure counted then continued
**File**: `src/train.py:800–803`
**Evidence**: `except Exception: self.epoch_fallback_counts['edge_target_generation_failure'] += 1; edge_targets = None`
**Assigned handler class**: `COUNTED_AND_CONTINUED` → must become `COUNTED_THEN_FATAL` (increment then raise)

### F4 — Edge-loss computation failure counted then continued
**File**: `src/train.py:819–822`
**Evidence**: `except Exception: self.epoch_fallback_counts['edge_loss_computation_failure'] += 1; edge_loss = tensor(0.0)`
**Problem**: technical failure silently produces `edge_loss = 0.0`, indistinguishable from a legitimate no-candidates batch
**Assigned handler class**: `COUNTED_AND_CONTINUED` → must become `COUNTED_THEN_FATAL`

### F5 — Dataset silently skips missing/unreadable expected Zarrs
**File**: `src/dataset.py:248–261`
**Evidence**: missing Zarr → `logger.debug() + continue`; unreadable Zarr → `logger.warning() + continue`
**Assigned handler class**: `SILENTLY_CONTINUED` → must become `FATAL` when `strict_sample_coverage=True`

### F6 — `validation_samples_total` uses survivor count, not full expected fold size
**File**: `src/train.py:1036–1043, 1269–1272`
**Evidence**: `unique_sample_ids` is derived from `val_dataset.pairs` (only pairs from successfully-opened Zarrs)
**Problem**: if 5 of 71 expected fold Zarrs are absent, manifests `validation_samples_total=66` — a lie about the fold's true size

### F7 — Zero-retained-pairs expected sample disappears silently
**Files**: `src/dataset.py:248–261` + `src/train.py:1036–1043`
**Problem**: expected sample that opens its Zarr but produces 0 pairs has no entry in `val_dataset.pairs`, no entry in `unique_sample_ids`, and no error anywhere — complete silent drop

### F8 — Training-side expected coverage not asserted
**File**: `src/dataset.py:245–327`, `kaggle_kernel/train_kernel.py:409–434`
**Problem**: no post-construction check that all split-defined training sample IDs produced at least one pair; any drop is silent

### F9 — `validation_is_full_fold=True` set unconditionally when no cap
**File**: `src/train.py:1067`
**Evidence**: `else: validation_is_full_fold = True` — always True when `allowed_sample_ids is None`, regardless of whether any expected samples are missing

### F10 — `train_kernel.py` has weaker SHA provenance than `inference_kernel.py`
**File**: `kaggle_kernel/train_kernel.py:112–117`
**Evidence**:
- first-match `os.walk` (takes first hit, no error on multiple matches) vs exact-one in inference_kernel
- `if sha_file.exists(): DEPLOYED_SHA = sha_file.read_text().strip()` has silent "unknown" fallback
- no 40-char lowercase hex validation
- no `verify_import_origins()` — so `training_code_sha` can claim SHA X while Python has imported `src/` from a different location
**Inference kernel (correct reference)**: raises on absent GIT_SHA.txt, validates 40-char hex, exact-one discovery, verifies all production module origins

### F11 — Stale P0-5 test asserts [0,1] range on transformer output
**File**: `tests/test_model.py:160`
**Evidence**: `assert torch.all((out >= 0) & (out <= 1)), "edge_scorer ends in Sigmoid, output must be in [0,1]"`
**Problem**: this assertion is only satisfiable if Sigmoid is reintroduced (a regression); it FAILS with the correct current architecture (raw logits, P0-5 contract)

### F12 — No distinction between technical and biological zeros
**File**: `src/train.py:764–822`
**Problem**: no counter tracks (a) technical GT-load failure vs (b) legitimate empty-node frame; no counter tracks legitimate zero-positive-edge batches — cannot audit whether training is seeing real biology or silent data failures

---

## SEMANTIC FAILURE TAXONOMY

Every failure mode the production path can encounter must be classified. Implementation must
not invent a new class; if a real failure doesn't fit one of these, raise it as a gap before
implementing.

**FATAL**: raise immediately; no continuation; no checkpoint; no further counter increment

**COUNTED_THEN_FATAL**: increment the appropriate legacy counter for backward CSV compatibility, then raise immediately; the counter value must NEVER authorize continuation

**COUNTED_AND_CONTINUED**: count the event, then continue — ONLY for explicitly approved cases (currently: `evaluation_failure` in non-strict validation mode)

**BIOLOGICAL_ZERO_COUNTED**: increment a biological-zero counter, continue normally — ONLY for events proven to be real biology, not code failures

---

### Case A — TECHNICAL_GT_LOAD_FAILURE
**Trigger**: geff file absent for an expected-to-have-geff sample; `IndexedRXGraph.from_geff()` raises; required node-attribute keys missing from parsed graph
**Training batch** (`train_epoch`, retained pair, `filter_unannotated_pairs=True`): `COUNTED_THEN_FATAL`
- Increment `gt_node_load_failure` counter
- Re-raise immediately
**Validation batch** (strict_integrity_mode=True): `COUNTED_THEN_FATAL`
- Increment `evaluation_failure` counter
- Re-raise immediately (fails the entire epoch)
**Validation batch** (strict_integrity_mode=False): `COUNTED_AND_CONTINUED`
- Increment `evaluation_failure` counter
- Continue; sample excluded from `validation_samples_evaluated`
- Epoch fails if `evaluation_failure` > 50% of selected samples

### Case B — LEGITIMATE_NO_GT_NODES_AT_FRAME (biological zero)
**Trigger**: geff parsed successfully; the specific timepoint genuinely has zero GT nodes
**For training retained pairs** (`filter_unannotated_pairs=True`): IMPOSSIBLE — the filter guarantees ≥1 annotated GT node per retained frame. If `_get_gt_nodes(strict=True)` returns an empty tensor for a retained pair, treat as Case A (COUNTED_THEN_FATAL — the filter's guarantee was violated, indicating a coordinate or data-boundary code error, not biology)
**For non-filter callers** (e.g. exploratory or future inference-path callers that don't use the filter): `BIOLOGICAL_ZERO_COUNTED` — increment `legitimate_zero_gt_node_batches`, continue, no error
**Note**: distinguish the call sites clearly in implementation — the `strict` parameter on `_get_gt_nodes` controls this (see INV-3)

### Case C — TECHNICAL_EDGE_TARGET_GENERATION_FAILURE
**Trigger**: `generate_edge_targets()` raises for any technical reason
**Classification**: `COUNTED_THEN_FATAL`
- Increment `edge_target_generation_failure` counter
- Re-raise immediately — `edge_targets = None` from a technical exception must never occur

### Case D — LEGITIMATE_ZERO_POSITIVE_EDGE (biological zero)
**Trigger**: `generate_edge_targets()` succeeds and returns all-negative targets (detected nodes exist but none match a GT edge within threshold)
**Classification**: `BIOLOGICAL_ZERO_COUNTED` — increment `legitimate_zero_positive_edge_batches`, continue; BCE loss over all-negative targets is a valid non-zero training signal

### Case E — TECHNICAL_EDGE_LOSS_FAILURE
**Trigger**: `self.transformer()` raises; `self.division_loss_fn()` raises; combined loss is NaN or Inf after computation
**Classification**: `COUNTED_THEN_FATAL`
- NaN/Inf: increment `edge_loss_computation_failure`, raise
- Exception: increment `edge_loss_computation_failure`, re-raise
- `edge_loss = 0.0` must NEVER result from a caught technical exception

### Case F — MISSING_EXPECTED_SAMPLE
**Trigger**: sample ID is in the split's expected set; Zarr path does not exist on disk
**strict_sample_coverage=True**: `FATAL` at dataset construction time (recorded in `failed_sample_ids` before raising)
**strict_sample_coverage=False**: warning + continue; recorded in `failed_sample_ids`

### Case G — UNREADABLE_EXPECTED_SAMPLE
**Trigger**: Zarr path exists; `AnisotropicZarrLoader` raises on open or first read
**strict_sample_coverage=True**: `FATAL` at dataset construction time (recorded in `failed_sample_ids`)
**strict_sample_coverage=False**: warning + continue; recorded in `failed_sample_ids`

### Case H — MALFORMED_GEFF
**Trigger**: geff file exists but `IndexedRXGraph.from_geff()` raises or returns a graph missing required structural invariants
**All modes**: `FATAL` — this is a data-integrity error with no safe fallback

### Case I — ZERO_USABLE_PAIRS (expected sample that opened successfully but has no retained pairs)
**Trigger**: Zarr opened; `_build_pair_index()` ran; all pairs excluded (e.g. all frames unannotated, all pairs filtered)
**strict_sample_coverage=True**: `FATAL` — an expected training or validation sample with no usable pairs cannot contribute to training or honest validation accounting
**strict_sample_coverage=False**: record in `zero_pairs_sample_ids`, warn, continue
**Coverage metadata**: must ALWAYS distinguish Case F/G (`failed_sample_ids`) from Case I (`zero_pairs_sample_ids`); these are different failure modes with different diagnoses

### Case J — DUPLICATE_SAMPLE_ID
**Trigger**: same sample ID appears more than once in the split definition
**All modes**: `FATAL` at split validation time; check whether `load_and_validate_split()` already handles this — see OPEN REVIEWER DECISIONS ORD-4

### Case K — PARTIAL_VALIDATION_COVERAGE (summary case, drives `validation_is_full_fold`)
`validation_is_full_fold = True` ONLY when ALL of:
1. The set of samples actually fully evaluated (`validation_samples_evaluated_ids`) equals the COMPLETE expected fold sample-ID set (`val_dataset.expected_sample_ids`) — no cap exclusions, no missing samples, no evaluation failures
2. `validation_samples_evaluated == len(val_dataset.expected_sample_ids)`

`validation_is_full_fold = False` whenever ANY of:
- `max_validation_samples` cap is applied and cap < len(expected fold)
- Any expected sample was absent from `val_dataset.pairs` (Case F/G/I)
- Any expected sample's evaluation failed (Case A in non-strict mode)

**Do NOT infer full-fold from `allowed_sample_ids is None` alone** — the current F9 bug does exactly this

---

## VALIDATION ACCOUNTING CONTRACT (reviewer decision §1 — canonical)

This contract overrides all prior language on `validation_samples_total`.

### Definitions (exact, not approximate)

```
expected_fold_ids       = val_loader.dataset.expected_sample_ids
                          (the full split-defined fold, always before any cap)

selected_validation_ids = expected_fold_ids[:max_validation_samples]
                          (the capped evaluation target; equals expected_fold_ids
                           when no cap is applied or cap >= len(expected_fold_ids))

validation_samples_total    = len(expected_fold_ids)          # ALWAYS full fold size
validation_samples_evaluated = count of selected_validation_ids
                               for which evaluation fully completed without exception
validation_is_full_fold     = (set(evaluated_sample_ids) == set(expected_fold_ids))
```

### Required example (from reviewer)
```
full expected fold = 71 samples
max_validation_samples cap = 2
evaluations that completed successfully = 2

validation_samples_total    = 71   ← FULL fold size; NEVER the capped count
validation_samples_evaluated = 2
validation_is_full_fold      = False  ← selected set ≠ full fold set
```

### What to track in the validation loop

`validate_epoch()` must maintain:
- `selected_validation_ids` (the capped ordered list, built at loop start from `expected_fold_ids[:cap]`)
- `evaluated_sample_ids` (set of IDs for which evaluation actually completed)
- `failed_evaluation_ids` (set of IDs for which evaluation raised)

These are ephemeral (not saved to checkpoint or manifest), but must be logged and used to
compute `validation_is_full_fold` and `validation_samples_evaluated` correctly.

---

## DATASET COVERAGE CONTRACT

### `strict_sample_coverage` parameter (reviewer decision §3)

`CompetitionDataset.__init__` must accept an explicit parameter:

```python
strict_sample_coverage: bool = False
```

**Do NOT infer from environment variables, path patterns, or any implicit signal.** Callers
must pass the value explicitly.

**Kaggle training kernel**: must pass `strict_sample_coverage=True` for both training and
validation datasets.

**Local/CI**: `False` is the default; compatible warning/continue behavior preserved.

### Coverage metadata properties

After `_build_pair_index()` completes, the dataset must expose:

```
expected_sample_ids          : list[str]  # the original self.sample_ids from split file;
                                           # frozen before any Zarr is opened
successfully_opened_sample_ids : list[str] # opened Zarr AND produced ≥1 usable pair
zero_pairs_sample_ids          : list[str] # opened Zarr successfully; 0 usable pairs retained
failed_sample_ids              : list[str] # Zarr absent OR Zarr raised on open/read
```

These four sets must be DISJOINT and their union must equal `expected_sample_ids`.
`failed_sample_ids` and `zero_pairs_sample_ids` must NEVER be conflated.

### Behavior table

| Event | strict_sample_coverage=True | strict_sample_coverage=False |
|---|---|---|
| Zarr path absent | FATAL (record in failed_sample_ids, then raise) | record in failed_sample_ids, warn, continue |
| Zarr open/read raises | FATAL (record in failed_sample_ids, then raise) | record in failed_sample_ids, warn, continue |
| Zarr opens; 0 retained pairs | FATAL (record in zero_pairs_sample_ids, then raise) | record in zero_pairs_sample_ids, warn, continue |
| Zarr opens; ≥1 retained pair | record in successfully_opened_sample_ids, continue | same |

### "Unexpected" sample IDs

Do NOT scan for or discover arbitrary sample IDs not in the split definition. The only
invariant is: every expected split ID must be accounted for exactly once. Unexpected
IDs discovered via directory scanning are out of scope for P0-7.

---

## STRICT VALIDATION INTEGRITY MODE (reviewer decision §2)

`TrainingLoop` must accept an explicit parameter:

```python
strict_integrity_mode: bool = False
```

**Do NOT infer from environment.** Callers must pass it explicitly.

**Kaggle training kernel**: must pass `strict_integrity_mode=True`.

**Local/CI**: `False` preserves current count-and-continue behavior for validation.

### Behavior table for `validate_epoch()`

| Failure | strict_integrity_mode=True | strict_integrity_mode=False |
|---|---|---|
| GT/GEFF technical failure per sample | `COUNTED_THEN_FATAL` (increment evaluation_failure, raise) | `COUNTED_AND_CONTINUED` (increment, continue) |
| Sample evaluation raises for any reason | `COUNTED_THEN_FATAL` | `COUNTED_AND_CONTINUED` |
| Missing expected sample in val_dataset.pairs | `FATAL` before evaluation loop | warn, continue with reduced evaluated count |
| `failed_sample_ids` non-empty | `FATAL` before evaluation loop | warn, continue |
| `evaluation_failure` > 50% of selected | N/A (fatal on first failure) | epoch-level RuntimeError |

The ">50% tolerance" policy applies ONLY when `strict_integrity_mode=False`.

---

## TRAINING FAILURE POLICY (reviewer decisions §2, §5)

### Default: IMMEDIATE FATAL for all technical integrity failures in `train_epoch()`

All `except Exception` blocks in `train_epoch()` for technical failures must be replaced
with `COUNTED_THEN_FATAL` behavior:
1. Increment the appropriate legacy counter (preserves CSV backward compatibility)
2. Re-raise immediately — the raise must propagate out of `train_epoch()` uncaught

The counter must NEVER authorize continuation. The sequence is always: count → raise.

### Policy table

| Failure in train_epoch() | Policy | Counter incremented | Continuation? |
|---|---|---|---|
| `_get_gt_nodes(strict=True)` raises for retained pair | `COUNTED_THEN_FATAL` | `gt_node_load_failure` (NEW) | NO |
| `_get_gt_nodes(strict=True)` returns empty tensor for retained pair | `COUNTED_THEN_FATAL` | `gt_node_load_failure` | NO |
| `generate_edge_targets()` raises | `COUNTED_THEN_FATAL` | `edge_target_generation_failure` (existing, kept) | NO |
| Transformer call raises | `COUNTED_THEN_FATAL` | `edge_loss_computation_failure` (existing, kept) | NO |
| `DivisionLoss` raises | `COUNTED_THEN_FATAL` | `edge_loss_computation_failure` | NO |
| Loss is NaN or Inf | `COUNTED_THEN_FATAL` | `edge_loss_computation_failure` | NO |
| Legitimate zero-positive-edge batch (Case D) | `BIOLOGICAL_ZERO_COUNTED` | `legitimate_zero_positive_edge_batches` (NEW) | YES |
| Non-filter caller gets empty-node frame (Case B) | `BIOLOGICAL_ZERO_COUNTED` | `legitimate_zero_gt_node_batches` (NEW) | YES |
| `heatmap_generation_failure` (P0-1, retained pair) | `FATAL` as before (P0-1 already fail-loud) | existing counter | NO |

### On `heatmap_generation_failure`

P0-1 already makes heatmap failure fatal for retained pairs. Do not change this.
The existing counter name is preserved.

---

## FALLBACK COUNTER CONTRACT (reviewer decision §5 — Option B)

Existing counter names are KEPT for CSV backward compatibility.
New counters are ADDED for biological-zero tracking.
No existing counter authorizes continuation for technical failures after P0-7.

| Counter | Trigger | Class | Authorizes continuation? | In CSV log? | In checkpoint? | In manifest? |
|---|---|---|---|---|---|---|
| `heatmap_generation_failure` | P0-1 heatmap fails for retained pair | Technical | NO (re-raise) | YES (existing) | NO | NO |
| `gt_node_load_failure` | `_get_gt_nodes(strict=True)` raises or returns empty for retained pair | Technical | NO (re-raise) | YES (new column) | NO | NO |
| `edge_target_generation_failure` | `generate_edge_targets()` raises | Technical | NO (re-raise) | YES (existing) | NO | NO |
| `edge_loss_computation_failure` | transformer/loss raises or NaN/Inf | Technical | NO (re-raise) | YES (existing) | NO | NO |
| `evaluation_failure` | GT load or evaluation raises in `validate_epoch()` | Technical | In strict=False mode only: YES if ≤50% | YES (existing) | NO | NO |
| `legitimate_zero_gt_node_batches` | Non-filter caller; empty-node frame; Case B | Biological | YES | YES (new column) | NO | NO |
| `legitimate_zero_positive_edge_batches` | `generate_edge_targets()` succeeds; zero positive edges; Case D | Biological | YES | YES (new column) | NO | NO |

**Counter increment ordering**: increment counter, THEN raise. This is safe because
COUNTED_THEN_FATAL means the run terminates before any checkpoint is written.
The in-flight epoch counter dict is discarded along with the epoch.

---

## TRAINING PROVENANCE CONTRACT (reviewer decision §4)

**Decision**: Minimal strict parity in `kaggle_kernel/train_kernel.py` for P0-7 now.
Shared-helper extraction deferred to GPU-gate implementation phase.

**`verify_import_origins()` is NOT GPU-GATE ONLY** (reviewer override of draft language).
It is required for trustworthy training provenance: without it, `training_code_sha` in the
manifest can claim SHA X while Python has imported `src/` modules from a different location
(e.g. a stale system-path copy or a second attached dataset). This gap closes the full
provenance chain.

### Required changes to `kaggle_kernel/train_kernel.py`

The following three requirements must ALL be met in Kaggle mode (`KAGGLE_MODE=True`):

**A — Exact-one source mount discovery**
Inline the `find_exactly_one_kaggle_input_dir()` logic from `inference_kernel.py`
(lines 33–56 of that file). Raise `RuntimeError` on zero matches. Raise `RuntimeError`
listing all candidates on multiple matches. First-match `os.walk` is not acceptable.

**B — Strict 40-lowercase-hex SHA validation, no "unknown" fallback**
The sequence must be:
1. `if not sha_file.exists(): raise RuntimeError(...)`
2. `raw_sha = sha_file.read_text(encoding="utf-8").strip()`
3. `if not raw_sha: raise RuntimeError(...)`
4. `if len(raw_sha) != 40 or raw_sha != raw_sha.lower() or not all(c in "0123456789abcdef" for c in raw_sha): raise RuntimeError(...)`
5. `DEPLOYED_SHA = raw_sha`

The string `"unknown"` must NEVER appear as `DEPLOYED_SHA` in Kaggle mode.

**C — Production module origin containment**
Inline `verify_import_origins()` from `inference_kernel.py` (lines 205–235 of that file).
Call it after importing all production `src.*` modules, passing `KAGGLE_SRC_DATASET_DIR`.
Modules to verify: `src.dataset`, `src.model`, `src.train`, `src.checkpoint_manifest`,
`src.submission_pipeline`, `src.submission_exporter`, `src.prediction_graph`, `src.inference`
(the same 8 modules listed in `inference_kernel.py:216–223`).

**Do not modify `kaggle_kernel_inference/inference_kernel.py`** in P0-7.
It is already correct and is the reference implementation for the above three blocks.

---

## STALE P0-5 TEST CORRECTION

**File**: `tests/test_model.py`
**Method**: `TestSimpleNodeTransformer.test_output_shape_is_all_pairwise_edges_in_row_major_order`
**Stale line 160**: `assert torch.all((out >= 0) & (out <= 1)), "edge_scorer ends in Sigmoid, output must be in [0,1]"`

**Exact replacement** (minimal; do not change surrounding lines):
```python
assert out.shape == (n_t * n_t1,), f"Expected shape ({n_t * n_t1},), got {out.shape}"
assert isinstance(transformer.edge_scorer[-1], nn.Linear), (
    "edge_scorer must end in Linear (no Sigmoid) — transformer returns raw logits (P0-5 contract)"
)
```

This is the only change to `tests/test_model.py`. No further modifications.

---

## CHECKPOINT INTEGRITY INTERACTION (reviewer decision §6)

1. **Can a checkpoint be saved after any COUNTED_THEN_FATAL event?**: NO. The re-raise propagates out of `train_epoch()`/`validate_epoch()` before `_save_last_checkpoint()` is reached.
2. **Can a deployment-candidate checkpoint be promoted after a technical failure?**: NO. Same reason.
3. **Should P0-7 counters appear in checkpoint metadata?**: NO. COUNTED_THEN_FATAL guarantees all technical counters are zero in any checkpoint that was successfully written.
4. **Should P0-6 manifest eligibility be updated to require zero P0-7 counters?**: NO manifest schema expansion. P0-7 fail-fast prevents technically-failed runs from reaching checkpoint promotion. The existing eligibility checks are sufficient.

---

## AUTHORIZED FILE SCOPE (reviewer decision §11 — narrowest proven set)

**REQUIRED — must change:**

| File | Change | Mirror required? |
|---|---|---|
| `src/train.py` | F1/F2/F3/F4/F6/F9/F12: strict_integrity_mode param; _get_gt_nodes strict param; COUNTED_THEN_FATAL; validation accounting; biological counters | YES |
| `kaggle_src_dataset/src/train.py` | byte-identical mirror of above | — |
| `src/dataset.py` | F5/F7/F8: strict_sample_coverage param; 4 coverage metadata properties; Case F/G/I behavior | YES |
| `kaggle_src_dataset/src/dataset.py` | byte-identical mirror of above | — |
| `kaggle_kernel/train_kernel.py` | F10: exact-one discovery; 40-hex SHA; verify_import_origins(); strict_sample_coverage=True; strict_integrity_mode=True | NO mirror (kernel entrypoint) |
| `tests/test_model.py` | F11: replace stale line 160 only | NO |
| `tests/test_p07_training_integrity.py` | NEW: all P0-7 negative and positive regression tests | NO |

**Add a path only if directly proven necessary.** Do not modify:
- `tests/test_train.py` (unless a concrete production-path test provably cannot be expressed in the new dedicated file)
- `tests/test_dataset.py` (same caveat)
- `src/evaluation.py`, `src/model.py`, `src/targets.py`, `src/tracker.py`
- `src/inference.py`, `src/prediction_graph.py`, `src/split_utils.py`, `src/checkpoint_manifest.py`
- `kaggle_kernel_inference/inference_kernel.py`
- Any GPU-gate, P1, or submission-pipeline files

---

## INVARIANT → CODE → TEST MATRIX (updated for all reviewer decisions)

| INV-ID | Invariant | Enforcement in code | File/Lines | Negative test | Positive compat test |
|---|---|---|---|---|---|
| INV-1 | Missing expected Zarr fatal when strict_sample_coverage=True | `CompetitionDataset._build_pair_index()` | `src/dataset.py:248–261` | T-01: strict=True, Zarr absent → RuntimeError | T-01P: strict=False, Zarr absent → warning, continues |
| INV-2 | Unreadable expected Zarr fatal when strict=True | `CompetitionDataset._build_pair_index()` | `src/dataset.py:256–261` | T-02: strict=True, Zarr raises → RuntimeError | T-02P: strict=False → continues |
| INV-3 | `_get_gt_nodes(strict=True)` re-raises on ALL technical errors; never returns None for technical errors | `TrainingLoop._get_gt_nodes(strict=True)` | `src/train.py:569–592` | T-03: parse exception → re-raises; T-04: empty tensor for retained pair → COUNTED_THEN_FATAL | T-03P: strict=False, missing geff → returns None |
| INV-4 | `generate_edge_targets()` failure is COUNTED_THEN_FATAL | `TrainingLoop.train_epoch()` | `src/train.py:786–803` | T-05: raises → increments counter then propagates | T-05P: zero-positive-edge success → continues, increments biological counter |
| INV-5 | Edge-loss failure and NaN/Inf are COUNTED_THEN_FATAL | `TrainingLoop.train_epoch()` | `src/train.py:808–822` | T-06: transformer raises → COUNTED_THEN_FATAL; T-07: NaN loss → COUNTED_THEN_FATAL | T-06P: empty-logit no-candidates → continues normally |
| INV-6 | `validation_samples_total` = full expected fold size, never capped | `TrainingLoop.validate_epoch()` | `src/train.py:1036–1043, 1269–1272` | T-08: fold=71, cap=2, evaluated=2 → total=71, evaluated=2, full_fold=False | T-08P: no cap, all present, all evaluated → total=N, evaluated=N, full_fold=True |
| INV-7 | `validation_is_full_fold=True` requires evaluated set equals full expected fold set | `TrainingLoop.validate_epoch()` | `src/train.py:1067` | T-09: one expected sample absent from pairs → full_fold=False | T-09P: all expected samples present and evaluated → full_fold=True |
| INV-8 | `selected_validation_ids` tracked separately from expected fold | `TrainingLoop.validate_epoch()` | `src/train.py` (new tracking) | T-10: cap=2, fold=5 → selected has exactly 2 IDs; total=5 | T-10P: cap≥fold → selected==expected |
| INV-9 | Strict validation integrity mode: technical failure is COUNTED_THEN_FATAL | `TrainingLoop.validate_epoch(strict_integrity_mode=True)` | `src/train.py` | T-11: strict=True, GT load raises → COUNTED_THEN_FATAL; epoch fails | T-11P: strict=False, GT load raises, count<50% → epoch continues |
| INV-10 | Dataset exposes all 4 coverage metadata properties; failed_sample_ids ≠ zero_pairs_sample_ids | `CompetitionDataset` post-construction | `src/dataset.py` | T-12: absent Zarr → in failed_sample_ids; zero-pair → in zero_pairs_sample_ids; not conflated | T-12P: full coverage → all IDs in successfully_opened; all others empty |
| INV-11 | Zero-pair expected sample fatal when strict_sample_coverage=True | `CompetitionDataset._build_pair_index()` | `src/dataset.py` | T-13: strict=True, Zarr opens, 0 pairs → RuntimeError | T-13P: strict=False, 0 pairs → zero_pairs_sample_ids, continues |
| INV-12 | train_kernel.py: exact-one source mount discovery | Inline find_exactly_one logic | `kaggle_kernel/train_kernel.py:47–66` | T-14: two src datasets attached → RuntimeError (lists both); T-15: zero src datasets → RuntimeError | T-14P: exactly one → proceeds |
| INV-13 | train_kernel.py: SHA strict 40-lowercase-hex validation; no "unknown" in Kaggle mode | SHA validation block | `kaggle_kernel/train_kernel.py:112–117` | T-16: absent GIT_SHA.txt → RuntimeError; T-17: malformed SHA (wrong length) → RuntimeError; T-18: malformed SHA (non-hex) → RuntimeError | T-16P: valid SHA → DEPLOYED_SHA set correctly |
| INV-14 | train_kernel.py: verify_import_origins() called after all src.* imports | Inline verify_import_origins | `kaggle_kernel/train_kernel.py` (new block) | T-19: one module resolves outside src dataset dir → RuntimeError | T-19P: all modules under selected src dir → proceeds |
| INV-15 | Stale [0,1] assertion removed; edge_scorer ends in nn.Linear | tests/test_model.py:160 replacement | `tests/test_model.py` | T-20: logit output outside [0,1] → test passes (not fails) | T-20P: isinstance(edge_scorer[-1], nn.Linear) → True |
| INV-16 | Biological-zero counters are only incremented for confirmed biological events, never for technical failures | `train_epoch()` counter paths | `src/train.py` | T-21: Case A (technical GT failure) → biological counters NOT incremented | T-21P: Case D (zero positive edges, successful) → legitimate_zero_positive_edge_batches incremented |

---

## NEGATIVE TEST MATRIX (minimum 23, target ~28)

All tests go in `tests/test_p07_training_integrity.py` unless noted.

| Test ID | What to inject | Path exercised | Expected outcome | Anti-vacuous check |
|---|---|---|---|---|
| T-01 | Expected Zarr absent; strict_sample_coverage=True | `CompetitionDataset._build_pair_index()` | RuntimeError; sample in failed_sample_ids | Check exception type and failed_sample_ids |
| T-02 | Zarr open raises; strict_sample_coverage=True | `_build_pair_index()` | RuntimeError; sample in failed_sample_ids | Check NOT in zero_pairs_sample_ids |
| T-03 | `load_geff_cached` raises; `_get_gt_nodes(strict=True)` | `TrainingLoop._get_gt_nodes` | Re-raises the exception | strict=False same fixture → returns None |
| T-04 | `_get_gt_nodes(strict=True)` returns empty tensor for retained pair | `train_epoch()` retained-pair path | COUNTED_THEN_FATAL: gt_node_load_failure incremented, then raises | Counter is 1; exception propagates |
| T-05 | `generate_edge_targets()` raises | `train_epoch()` | COUNTED_THEN_FATAL: edge_target_generation_failure incremented, then raises | Counter=1; no edge_targets=None continuation |
| T-06 | Transformer call raises | `train_epoch()` | COUNTED_THEN_FATAL: edge_loss_computation_failure incremented, then raises | Counter=1 |
| T-07 | Combined loss is NaN | `train_epoch()` | COUNTED_THEN_FATAL: edge_loss_computation_failure incremented, then raises | Explicit NaN check in source must fire |
| T-08 | fold=71, cap=2, evaluations succeed=2 | `validate_epoch()` | validation_samples_total=71, evaluated=2, full_fold=False | Confirm total is NOT 2 |
| T-09 | One expected sample absent from val_dataset.pairs | `validate_epoch()` | full_fold=False; validation_samples_total unchanged | Must check both fields |
| T-10 | cap=2, fold has 5 samples, all 2 succeed | `validate_epoch()` | full_fold=False (selected≠full fold); total=5, evaluated=2 | full_fold must be False even though all selected completed |
| T-11 | strict_integrity_mode=True; GT load raises during validation | `validate_epoch()` strict path | COUNTED_THEN_FATAL: evaluation_failure incremented, epoch raises | strict=False same fixture → counted, continues |
| T-12 | Zarr absent; strict=False | `_build_pair_index()` | sample in failed_sample_ids; NOT in zero_pairs_sample_ids; warning logged | Check disjoint sets |
| T-13 | Zarr opens; 0 pairs; strict_sample_coverage=True | `_build_pair_index()` | RuntimeError; sample in zero_pairs_sample_ids | Not in failed_sample_ids |
| T-14 | Two src datasets attached (two matches for marker file) | `train_kernel.py` exact-one logic | RuntimeError listing both candidates | Check both paths in error message |
| T-15 | Zero src datasets attached | `train_kernel.py` exact-one logic | RuntimeError "No directory" | Not a silent continue |
| T-16 | GIT_SHA.txt absent in Kaggle mode | `train_kernel.py` SHA block | RuntimeError | Not "unknown" fallback |
| T-17 | GIT_SHA.txt contains 37-char string | `train_kernel.py` SHA block | RuntimeError (wrong length) | Check len check fires |
| T-18 | GIT_SHA.txt contains 40-char string with uppercase | `train_kernel.py` SHA block | RuntimeError (not lowercase hex) | Check charset check fires |
| T-19 | One src.* module resolves outside selected src dir | `train_kernel.py` verify_import_origins inline | RuntimeError naming the offending module | Check module name in message |
| T-20 | Transformer output outside [0,1] | `tests/test_model.py` (only test in that file) | Test passes (not fails) — logits are unbounded | `test_p05_double_sigmoid_fix.py::test_transformer_output_unbounded` still passes |
| T-21 | Case A (technical GT failure in train_epoch) | `train_epoch()` biological counters | `legitimate_zero_gt_node_batches` NOT incremented | Confirm biological counter is 0 |
| T-22 | Case D (zero positive edges; generate_edge_targets succeeds) | `train_epoch()` | `legitimate_zero_positive_edge_batches` incremented; training continues | Counter increments exactly once per injected batch |
| T-23 | Case B non-filter caller: valid geff; frame has zero GT nodes | `_get_gt_nodes(strict=False)` non-retained caller | Returns torch.zeros((0,3)); `legitimate_zero_gt_node_batches` incremented | NOT None; NOT raises |
| T-24 | strict_integrity_mode=False; evaluation_failure > 50% of selected | `validate_epoch()` non-strict | Epoch-level RuntimeError (existing circuit-breaker) | Confirm threshold is 50%, not 0 |
| T-25 | All 4 coverage metadata properties correct when one Zarr absent (strict=False) | `CompetitionDataset` | expected=N, successfully_opened=N-1, zero_pairs=0, failed=1; sum=N | Verify union equals expected_sample_ids |
| T-26 | strict_sample_coverage=True; all samples load successfully; all have ≥1 pair | `CompetitionDataset` | failed_sample_ids=[], zero_pairs_sample_ids=[], successfully_opened=expected | No false positives |
| T-27 | loss is Inf (not NaN) | `train_epoch()` | COUNTED_THEN_FATAL | Both NaN and Inf branches covered |
| T-28 | validation_is_full_fold=True only when fully evaluated set == full expected set | `validate_epoch()` | full_fold=True iff all samples present and all evaluations completed | Inject one failure → full_fold must flip to False |

### Test state safety requirements (all tests)

- Must pass identically: development worktree, fresh `git apply` worktree, staged/committed state
- No `git grep` or git-index-dependent logic anywhere in tests
- Use `monkeypatch`, `tmp_path`, `make_bare_training_loop()` from existing `tests/test_train.py`
- Do not use `subprocess`, environment variable sniffing, or path-existence checks for invariant verification

---

## IMPLEMENTATION ORDER (dependency-aware)

**Phase 1 — Dataset coverage primitives**
- `src/dataset.py`: add `strict_sample_coverage` param; add 4 coverage metadata properties; implement Case F/G/I behavior per table
- Mirror: `kaggle_src_dataset/src/dataset.py`

**Phase 2 — Training-loop enforcement** (reads dataset API from Phase 1)
- `src/train.py`:
  - `_get_gt_nodes()`: add `strict: bool = False`; re-raise on any technical error when strict=True; return empty tensor for biological zero when strict=False (non-retained callers)
  - `train_epoch()`: COUNTED_THEN_FATAL for F1/F2/F3/F4; NaN/Inf check; add 2 biological-zero counters
  - `validate_epoch()`: add `strict_integrity_mode` usage; fix `validation_samples_total` from expected_fold_ids; track `selected_validation_ids` separately; fix `validation_is_full_fold` logic
- Mirror: `kaggle_src_dataset/src/train.py`

**Phase 3 — Training kernel provenance** (independent of Phases 1-2)
- `kaggle_kernel/train_kernel.py`: exact-one discovery; 40-hex SHA; verify_import_origins(); pass `strict_sample_coverage=True`; pass `strict_integrity_mode=True`

**Phase 4 — Stale test fix** (independent)
- `tests/test_model.py`: line 160 only

**Phase 5 — Regression tests** (depends on Phases 1-3 being implemented)
- `tests/test_p07_training_integrity.py`: T-01 through T-28

**Phase 6 — Mirror verification and patch**
- Run mirror hash check
- Run full test suite
- Generate patch

---

## VERIFICATION COMMANDS

```bash
# Verify baseline
git rev-parse HEAD
# Must output exactly: 8eeace8ab090aecb1cfad336f82865f752c825c4

# P0-7 new tests
py -m pytest -q tests/test_p07_training_integrity.py -v

# Stale test fix
py -m pytest -q tests/test_model.py -v

# P0-5 regression (must not regress)
py -m pytest -q tests/test_p05_double_sigmoid_fix.py -v

# P0-6 regression (must not regress)
py -m pytest -q tests/test_p06_submission_deployment.py -v

# P0-7A regression (must not regress)
py -m pytest -q tests/test_p07a_metric_parity.py -v

# Ruff on all authorized changed paths
py -m ruff check src/train.py src/dataset.py kaggle_kernel/train_kernel.py tests/test_model.py tests/test_p07_training_integrity.py

# No trailing whitespace
git diff --check

# Mirror hash check
python -c "
import hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
for f in os.listdir('src'):
    if not f.endswith('.py'): continue
    r, m = f'src/{f}', f'kaggle_src_dataset/src/{f}'
    if not os.path.exists(m): print(f'MISSING MIRROR: {f}'); continue
    print('OK' if h(r)==h(m) else 'DIFFER', f)
"

# Full suite (with known pre-existing exclusion)
py -m pytest -q -rA --ignore=scripts/test_dataset.py
```

---

## PATCH ARTIFACT PROCEDURE

**Baseline SHA**: `8eeace8ab090aecb1cfad336f82865f752c825c4`
**Patch path**: `C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch`

```powershell
New-Item -ItemType Directory -Force "C:\Users\hemas\Downloads\st_act_p07_review"

git diff 8eeace8ab090aecb1cfad336f82865f752c825c4..HEAD -- `
  src/train.py `
  src/dataset.py `
  kaggle_src_dataset/src/train.py `
  kaggle_src_dataset/src/dataset.py `
  kaggle_kernel/train_kernel.py `
  tests/test_model.py `
  tests/test_p07_training_integrity.py `
  > "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"

# Verify file count (expect 7; 9 if optional files added)
(Select-String "^diff --git" "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch").Count

# Verify no prohibited files
Select-String "^diff --git" "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"

# SHA-256
python -c "
import hashlib
with open(r'C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch', 'rb') as f:
    d = f.read()
print('SHA-256:', hashlib.sha256(d).hexdigest())
print('Bytes:', len(d))
"

# Clean-apply check
git worktree add ..\p07-verify 8eeace8ab090aecb1cfad336f82865f752c825c4
cd ..\p07-verify
git apply --check "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"
git apply "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"
py -m pytest -q -rA --ignore=scripts/test_dataset.py
```

---

## IMPLEMENTATION AGENT FINAL REPORT TEMPLATE

```
## P0-7 IMPLEMENTATION FINAL REPORT

### Baseline SHA (must be exact)
8eeace8ab090aecb1cfad336f82865f752c825c4

### Final implementation SHA
[40-char hex]

### Patch location
C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch

### Patch SHA-256
[sha256 hex]

### Patch byte size
[integer]

### Files in patch (7 base; 9 if optional)
- src/train.py: [description]
- kaggle_src_dataset/src/train.py: mirror
- src/dataset.py: [description]
- kaggle_src_dataset/src/dataset.py: mirror
- kaggle_kernel/train_kernel.py: [description]
- tests/test_model.py: line 160 stale assertion replaced
- tests/test_p07_training_integrity.py: NEW — T-01 through T-28

### `git apply --check` result
[PASS / FAIL + full output]

### P0-7 test output (literal)
[py -m pytest -q tests/test_p07_training_integrity.py -v]

### test_model.py output (literal)
[py -m pytest -q tests/test_model.py -v]

### P0-5 regression output (literal)
[py -m pytest -q tests/test_p05_double_sigmoid_fix.py -v]

### P0-6 regression output (literal)
[py -m pytest -q tests/test_p06_submission_deployment.py -v]

### P0-7A regression output (literal)
[py -m pytest -q tests/test_p07a_metric_parity.py -v]

### Full suite output (literal)
[py -m pytest -q -rA --ignore=scripts/test_dataset.py]

### Pre-existing failures (with baseline evidence)
[list with evidence from baseline run before any changes]

### Mirror hash check output (literal)
[python hash script output]

### Ruff output (literal)
[ruff check on all changed paths]

### Reviewer decisions confirmed taken
- strict_sample_coverage: explicit param (not env-inferred); default False; Kaggle kernel passes True
- strict_integrity_mode: explicit param; default False; Kaggle kernel passes True
- validation_samples_total: full fold size; never capped count
- verify_import_origins: inlined in train_kernel.py (not deferred)
- counters: Option B (kept + COUNTED_THEN_FATAL)
- manifest: no schema expansion

### Confirmed: NO CODE OUTSIDE AUTHORIZED SCOPE
YES / NO (if NO: list each unauthorized file with justification)

### Confirmed: NO GIT COMMITS
YES

### Confirmed: NO PUSH
YES

### Confirmed: NO TRAINING RUN STARTED
YES
```

---

## ADVERSARIAL REVIEW CHECKLIST (final)

**Baseline**
- [ ] `git apply --check` against `8eeace8ab090aecb1cfad336f82865f752c825c4` succeeds clean
- [ ] `diff --git` count = 7 (or 9 with optional files); no unauthorized files

**Dataset coverage**
- [ ] `CompetitionDataset.__init__` accepts `strict_sample_coverage: bool = False`
- [ ] Parameter is explicit, not inferred from environment or paths
- [ ] 4 coverage metadata properties exist and are DISJOINT: `expected_sample_ids`, `successfully_opened_sample_ids`, `zero_pairs_sample_ids`, `failed_sample_ids`
- [ ] Their union equals `expected_sample_ids`
- [ ] strict=True: absent Zarr → RuntimeError (in failed_sample_ids) not silent skip
- [ ] strict=True: unreadable Zarr → RuntimeError (in failed_sample_ids)
- [ ] strict=True: zero usable pairs → RuntimeError (in zero_pairs_sample_ids, NOT failed_sample_ids)
- [ ] strict=False: all three cases → warn, continue, metadata populated
- [ ] No "unexpected sample" scanning machinery added

**Training-loop: GT failures**
- [ ] `_get_gt_nodes` has `strict: bool = False` parameter
- [ ] strict=True: any exception → `gt_node_load_failure` incremented THEN re-raised
- [ ] strict=True: empty tensor for retained pair → `gt_node_load_failure` incremented THEN raised
- [ ] strict=False: missing geff → returns None (backward compat)
- [ ] `train_epoch()` calls `_get_gt_nodes(..., strict=True)` on retained pairs
- [ ] `legitimate_zero_gt_node_batches` incremented ONLY for non-filter callers returning biological empty tensor

**Training-loop: edge failures**
- [ ] No `except Exception` around `generate_edge_targets()` that allows continuation
- [ ] `edge_target_generation_failure` incremented THEN raised on `generate_edge_targets()` exception
- [ ] No `except Exception` around transformer call that allows continuation
- [ ] `edge_loss_computation_failure` incremented THEN raised on transformer/loss exception
- [ ] Explicit NaN/Inf check after loss computation; `edge_loss_computation_failure` incremented THEN raised
- [ ] `edge_loss = 0.0` never results from a caught technical exception
- [ ] `legitimate_zero_positive_edge_batches` incremented ONLY for Case D (successful generate_edge_targets; zero positive targets)

**Validation accounting**
- [ ] `validation_samples_total` = `len(val_dataset.expected_sample_ids)` — NEVER capped count
- [ ] `selected_validation_ids` built as `expected_sample_ids[:cap]`
- [ ] `validation_samples_evaluated` counts only IDs for which evaluation fully completed
- [ ] `validation_is_full_fold=True` only when `evaluated_ids == set(expected_sample_ids)`
- [ ] Verified with fold=71, cap=2, evaluated=2 → total=71, evaluated=2, full_fold=False

**Strict validation integrity**
- [ ] `TrainingLoop.__init__` accepts `strict_integrity_mode: bool = False`
- [ ] strict=True: GT load failure in validate_epoch → `evaluation_failure` incremented THEN epoch fails
- [ ] strict=True: any sample evaluation failure → `COUNTED_THEN_FATAL`
- [ ] strict=False: count-and-continue; ">50%" circuit-breaker retained
- [ ] Kaggle kernel passes `strict_integrity_mode=True`

**Provenance (all three)**
- [ ] Exact-one source mount discovery in train_kernel.py (raises on 0 or 2+ matches)
- [ ] SHA: absent GIT_SHA.txt → raises; empty → raises; not 40-char lowercase hex → raises
- [ ] "unknown" string never appears as DEPLOYED_SHA in Kaggle mode
- [ ] `verify_import_origins()` inlined in train_kernel.py; called after all src.* imports
- [ ] All 8 production modules verified under KAGGLE_SRC_DATASET_DIR
- [ ] `inference_kernel.py` NOT modified

**Stale test**
- [ ] `tests/test_model.py` line 160 replaced with Linear-check and shape-check
- [ ] No other modifications to test_model.py
- [ ] `test_p05_double_sigmoid_fix.py` all tests pass

**P0 regression preservation**
- [ ] P0-1: heatmap fail-loud unchanged
- [ ] P0-2: split identity in checkpoints unchanged
- [ ] P0-3: post-pass zero-detection circuit-breaker unchanged
- [ ] P0-4: `max_validation_samples` cap mechanism unchanged
- [ ] P0-5: `test_p05_double_sigmoid_fix.py` all pass
- [ ] P0-6: `test_p06_submission_deployment.py` all pass
- [ ] P0-7A: `test_p07a_metric_parity.py` all pass

**Counters**
- [ ] All 4 technical counters (heatmap, gt_node_load, edge_target, edge_loss) follow COUNTED_THEN_FATAL
- [ ] `evaluation_failure` follows COUNTED_THEN_FATAL in strict=True; COUNTED_AND_CONTINUED in strict=False
- [ ] 2 new biological counters follow BIOLOGICAL_ZERO_COUNTED only
- [ ] No counter authorizes continuation for a technical failure

**Mirrors and scope**
- [ ] All changed src/*.py are byte-identical to kaggle_src_dataset/src/
- [ ] Hash check passes for all mirrored files
- [ ] No unauthorized file in patch

---

## OPEN REVIEWER DECISIONS (remaining; not settled from repository)

### ORD-1 — CLOSED: Provenance helper extraction
Decision: Option C (inline parity in train_kernel.py; shared helper deferred to GPU-gate).
Implementation: inline exact-one logic + SHA validation + verify_import_origins() from
`inference_kernel.py` into `train_kernel.py`. No new shared files in P0-7.

### ORD-2 — CLOSED: Counter cleanup
Decision: Option B (keep existing counter names; add COUNTED_THEN_FATAL behavior;
add 2 new biological-zero counters). No removal of existing counter names.

### ORD-3 — CLOSED: Manifest schema
Decision: No expansion. COUNTED_THEN_FATAL prevents failed runs from reaching checkpoint
promotion; existing eligibility checks are sufficient.

### ORD-4 — OPEN: Duplicate sample ID handling in `load_and_validate_split()`
**Question**: Does the existing `src/split_utils.py::load_and_validate_split()` already
raise on duplicate IDs in the split file?
**Why not resolved here**: requires reading `src/split_utils.py` directly (not read in this
session). If yes → close as already handled. If no → smallest required change: add a
`len(ids) != len(set(ids))` check with `raise ValueError` before returning; this is a
one-line fix and fits within the already-authorized `src/dataset.py` scope (since
`load_and_validate_split` is called from the dataset constructor) OR `src/split_utils.py`
(a new file addition; requires reviewer sign-off before adding to scope).
**Decision required by**: implementation agent before writing Case J tests.

### ORD-5 — OPEN: `selected_validation_ids` logging location
**Question**: Should `selected_validation_ids` (the capped ordered ID list) be logged to
`training_progress.json` for auditability, or only used ephemerally in `validate_epoch()`?
**Recommendation**: log it (as a list) to `training_progress.json` alongside
`validation_samples_total`/`validation_samples_evaluated`/`validation_is_full_fold` — this
makes the fold selection auditable mid-run via the already-working `kernels output` check.
**Decision required by**: implementation agent before finalizing `validate_epoch()` changes.
If logging to training_progress.json, confirm this doesn't expand the manifest/checkpoint
schema (it should not — training_progress.json is a separate file, not the checkpoint).

---

## GAPS IDENTIFIED (author's analysis — not yet reviewer decisions)

The following are potential issues in the specification itself that the reviewer should
confirm before the implementation agent starts.

### GAP-1: `strict_integrity_mode` vs `strict_sample_coverage` naming coherence
The spec now has two separate "strict" parameters on different classes.
Risk: an implementer may conflate them or wire them together incorrectly.
Recommendation: the spec should explicitly state these are independent:
- `strict_sample_coverage` is a dataset-construction-time parameter
- `strict_integrity_mode` is a training-loop-time parameter
- The Kaggle kernel must pass BOTH explicitly; neither is derived from the other
This is documented above but should be called out prominently in the implementation
brief to the agent.

### GAP-2: `_get_gt_nodes(strict=True)` parameter naming collision
The `strict` flag on `_get_gt_nodes` (which controls whether to re-raise on technical
errors) uses the same word as `strict_sample_coverage` and `strict_integrity_mode` for
semantically different things. If there is a future reader confusion risk, consider naming
the `_get_gt_nodes` parameter `fail_on_error: bool = False` to be unambiguous. This is a
naming decision — not a behavior decision — and should be resolved before implementation.
If `strict` is kept, the spec should explicitly note it's a private parameter of a private
method and not user-facing.

### GAP-3: `gt_node_load_failure` NEW counter not in existing CSV schema
The existing `training_log.csv` has a fixed column set. Adding `gt_node_load_failure` as a
new column will produce NaN/missing values in any log-parsing script that reads existing
CSVs. Similarly for `legitimate_zero_gt_node_batches` and `legitimate_zero_positive_edge_batches`.
**Action required**: confirm whether `scripts/kaggle_check_run.py` and any CI parsers need
updates for new columns, OR whether they are written-to CSV only in new runs (no backward
read required). If backward read is required, this is a schema-evolution concern.

### GAP-4: `verify_import_origins()` module list in `train_kernel.py`
The inference_kernel.py verifies 8 specific modules (lines 216–223). The training kernel
imports a partially different set (e.g. it imports `src.train`, which inference_kernel.py
also imports, but train_kernel.py may not import some submission-pipeline modules like
`src.submission_pipeline`). The spec says "the same 8 modules listed in inference_kernel.py"
but the training kernel may not import all 8 — calling `verify_import_origins()` on an
unimported module would fail. **Action**: the implementation agent must enumerate which
`src.*` modules `train_kernel.py` actually imports (at the time of P0-7 implementation) and
verify only those. The spec requirement is: every module that `train_kernel.py` imports from
`src` must be verified — not necessarily all 8.

### GAP-5: `zero_pairs_sample_ids` in strict training mode vs annotation filtering
An expected training sample can produce zero retained pairs because ALL its consecutive
frame pairs are unannotated (filtered by `filter_unannotated_pairs=True`). In strict
production training mode, this should be fatal (per the spec's Case I). However, this is a
DATA QUALITY event — the geff exists, the Zarr loads, but the sample contributes nothing to
supervised training. Is this the correct behavior for Kaggle production? Specifically: if
ONE of 71 training samples has all unannotated pairs and therefore zero retained pairs,
should the ENTIRE Kaggle training run fail? If yes, the spec is correct. If the correct
behavior is "warn loudly but still train on the other 70," then strict=True should only be
fatal for completely missing/unreadable samples (Cases F/G), not for zero-pair samples
(Case I). **Decision required by reviewer before implementation.**

### GAP-6: Validation-side `strict_sample_coverage` interaction
When `strict_integrity_mode=True` in `validate_epoch()`, the spec says "missing expected
sample → fatal." But the validation dataset is constructed at `TrainingLoop` init time, not
inside `validate_epoch()`. If `strict_sample_coverage=True` was used at dataset construction,
a missing Zarr would have already raised at construction time — making the validate_epoch
check redundant. Conversely, if `strict_sample_coverage=False` was used for construction but
`strict_integrity_mode=True` is used for validation, a sample that silently dropped from the
dataset would be caught at validation time via `failed_sample_ids`. The spec should clarify
the expected pairing: Kaggle kernel should use BOTH `strict_sample_coverage=True` (at
construction) AND `strict_integrity_mode=True` (at validation). These are defense-in-depth
layers, not alternatives. This pairing should be stated explicitly in the kernel spec.

### GAP-7: `validation_samples_total` in checkpoint vs training_progress.json
The spec fixes `validation_samples_total` to mean the full fold size. This value goes into
the checkpoint (and therefore the P0-6 manifest). The manifest's existing field
`validation_samples_total` must now always equal the full fold size — even for capped runs.
This changes the semantic of an existing manifest field. Confirm: is there any existing
downstream code (e.g. `deployment_eligibility_errors()`) that interprets
`validation_samples_total` as the capped count and would break if it now returns the full
fold size? This requires reading `src/checkpoint_manifest.py` — if eligibility logic uses
`validation_samples_total > 0` as a proxy for "some validation happened," it will still
work; if it compares against `validation_samples_evaluated` expecting them to be equal for
a full-fold run, the comparison logic must be updated to check
`validation_is_full_fold=True` instead. **Action required before implementation.**
