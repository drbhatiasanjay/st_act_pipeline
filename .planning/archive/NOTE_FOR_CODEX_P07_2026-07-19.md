# CODEX HANDOFF NOTES — P0-7 TRAINING INTEGRITY IMPLEMENTATION
<!-- Written 2026-07-19 — consolidates Draft v1 + second-prompt adversarial-review deltas.
     Second prompt overrides conflicts with Draft v1. -->

## REPOSITORY BASELINE
- **Exact HEAD**: `8eeace8ab090aecb1cfad336f82865f752c825c4`
- **Meaning**: P0-7A metric parity integrated (last merged prior fix)
- All `src/*.py` files are byte-identical to `kaggle_src_dataset/src/` counterparts (verified by MD5)
- P0-1 through P0-7A are CLOSED — do not reopen

---

## CONFIRMED DEFECTS (all verified from code at HEAD)

### F1 — `_get_gt_nodes()` silently swallows technical failures
**File**: `src/train.py:569–592`
**Current**: entire function body in `try/except Exception: return None`
**Problem**: technical geff parse error and legitimately-absent geff produce identical None returns; no counter; no audit trail
**Handler classification**: `SILENTLY_CONTINUED` — must become `FATAL`

### F2 — Edge supervision silently skipped with no counter
**File**: `src/train.py:770`
**Current**: `if nodes_t is not None and nodes_t1 is not None and nodes_t.shape[0] > 0 and nodes_t1.shape[0] > 0:` — skip entire edge block with no counter if any fails
**Problem**: For a retained pair (`filter_unannotated_pairs=True`), both frames are guaranteed ≥1 GT node; None or empty tensor here IS a technical failure but is uncounted
**Handler classification**: `SILENTLY_CONTINUED` — must become `FATAL`

### F3 — `generate_edge_targets()` failure counted and continued
**File**: `src/train.py:800–803`
**Current**: `except Exception: self.epoch_fallback_counts['edge_target_generation_failure'] += 1; edge_targets = None`
**Handler classification**: `COUNTED_AND_CONTINUED` — must become `FATAL`

### F4 — Edge-loss computation failure counted and continued
**File**: `src/train.py:819–822`
**Current**: `except Exception: self.epoch_fallback_counts['edge_loss_computation_failure'] += 1; edge_loss = tensor(0.0)`
**Problem**: technical failure becomes `edge_loss = 0.0` — identical to legitimate no-candidates batch
**Handler classification**: `COUNTED_AND_CONTINUED` — must become `FATAL`

### F5 — Dataset silently skips missing/unreadable Zarr
**File**: `src/dataset.py:248–261`
**Current**: Missing Zarr → `logger.debug() + continue`; unreadable Zarr → `logger.warning() + continue`
**Handler classification**: `SILENTLY_CONTINUED` — must become `FATAL` in Kaggle/production mode

### F6 — `validation_samples_total` uses survivor count, not expected
**File**: `src/train.py:1036–1043, 1269–1272`
**Current**: `unique_sample_ids` derived from `val_dataset.pairs` (only Zarr-loaded samples)
**Problem**: If 5 of 50 expected validation Zarrs are missing, reports `validation_samples_total=45`

### F7 — Zero-retained-pairs sample disappears silently
**Files**: `src/dataset.py:248–261` + `src/train.py:1036–1043`
**Problem**: No entry in `val_dataset.pairs` → absent from all accounting, no error

### F8 — Training-side expected coverage not asserted
**Files**: `src/dataset.py:245–327`, `kaggle_kernel/train_kernel.py:409–434`
**Problem**: No post-construction check that all split-defined training sample IDs produced pairs

### F9 — `validation_is_full_fold=True` unconditionally when no cap
**File**: `src/train.py:1067`
**Current**: `else: validation_is_full_fold = True` — always True when `allowed_sample_ids is None`
**Problem**: Even if expected samples are absent from `val_dataset.pairs`

### F10 — `train_kernel.py` has weaker SHA provenance than `inference_kernel.py`
**File**: `kaggle_kernel/train_kernel.py:112–117`
**Current**: `if sha_file.exists(): DEPLOYED_SHA = sha_file.read_text().strip()` — has "unknown" fallback string; no 40-hex validation; first-match source discovery (not exact-one); no `verify_import_origins()`
**Inference kernel (correct)**: raises on absent GIT_SHA.txt, validates 40-char lowercase hex, exact-one discovery, verifies import origins

### F11 — STALE P0-5 TEST
**File**: `tests/test_model.py:160`
**Stale line**: `assert torch.all((out >= 0) & (out <= 1)), "edge_scorer ends in Sigmoid, output must be in [0,1]"`
**Problem**: Asserts transformer output is in [0,1] — contradicts P0-5 raw logit contract; would PASS only if Sigmoid reintroduced (a regression), FAILS with correct current architecture

### F12 — Fallback counters don't distinguish technical from biological
**File**: `src/train.py:764–822`
**Problem**: `_get_gt_nodes()` returning None for missing geff (technical) vs unannotated frame (biological zero) are both uncounted; no counter for "GT supervision skipped entirely for batch"

---

## SEMANTIC FAILURE TAXONOMY (Cases A–K)

**A. TECHNICAL_GT_LOAD_FAILURE**
- Condition: geff absent for expected-to-have-geff sample; geff parse exception; required node attrs malformed
- Required: RAISE IMMEDIATELY; do not continue training batch; do not increment legacy counter
- Training continues: NO
- Checkpoint: NO
- Deployment eligible: NO

**B. LEGITIMATE_NO_GT_NODES_AT_FRAME_PAIR**
- Condition: geff parsed successfully; timepoint has zero GT nodes
- With `filter_unannotated_pairs=True`: IMPOSSIBLE for retained pairs — filter guarantees both frames have ≥1 GT node; if zero-node result reaches `train_epoch` for a retained pair, it IS a TECHNICAL failure (coordinate/bounds error), not biological
- Without filter (validation, submission inference): legitimate — no error, no counter
- Training continues: YES (non-filter callers); NO (treat as FATAL for retained pairs)
- Counter: new `legitimate_zero_gt_node_batches` for non-filter callers

**C. TECHNICAL_EDGE_TARGET_GENERATION_FAILURE**
- Condition: `generate_edge_targets()` raises for any reason
- Required: RAISE IMMEDIATELY; never catch and continue
- Training continues: NO

**D. LEGITIMATE_ZERO_POSITIVE_EDGE_CASE**
- Condition: `generate_edge_targets()` succeeds, returns all-negative targets
- Meaning: detected nodes too far from GT nodes, or no GT edges in this time window
- Required: training continues; loss computed on all-negative targets (non-zero BCE loss)
- Counter: new `legitimate_zero_positive_edge_batches` (informational only)
- Training continues: YES

**E. TECHNICAL_EDGE_LOSS_FAILURE**
- Condition: `self.transformer()` raises OR `self.division_loss_fn()` raises OR loss is NaN/Inf
- Required: RAISE IMMEDIATELY; `edge_loss = 0.0` must NEVER result from a caught technical exception
- Training continues: NO

**F. MISSING_EXPECTED_SAMPLE**
- Condition: sample ID in split's expected set; Zarr path does not exist
- Required: in Kaggle/production mode — RAISE at dataset construction time; in local/CI mode — warn + continue
- Production training: FATAL

**G. UNREADABLE_ZARR**
- Condition: Zarr path exists; `AnisotropicZarrLoader` raises on open
- Required: same as F — FATAL in Kaggle/production mode

**H. MALFORMED_GEFF**
- Condition: geff exists; `IndexedRXGraph.from_geff()` raises or returns structurally invalid graph
- Required: FATAL (raise immediately) in all contexts

**I. DUPLICATE_SAMPLE_ID**
- Condition: same sample ID appears twice in split definition OR twice in discovered samples
- Required: FATAL at split validation time

**J. UNEXPECTED_SAMPLE_ID**
- Strict production: log WARNING; do not add to expected set; do not include in validation metrics
- Local/exploratory: log WARNING; may include if explicitly opted in

**K. PARTIAL_VALIDATION_COVERAGE**
- `validation_is_full_fold = True` PERMITTED ONLY WHEN:
  - `selected_validation_ids == expected_fold_validation_ids` (no cap, or cap ≥ total)
  - AND every selected sample was successfully evaluated (`validation_samples_evaluated == validation_samples_total`)
  - AND no expected sample was absent from `val_dataset.pairs`
- `validation_is_full_fold = False` REQUIRED WHEN:
  - Any cap is applied that excludes any expected fold sample
  - OR any expected sample is absent from constructed dataset
  - OR any expected sample's evaluation failed/raised

---

## DATASET/SPLIT COVERAGE CONTRACT (10 Invariants)

1. Expected sample IDs come from `split_data[split_type]` (validated split definition), established BEFORE any Zarr is opened.

2. Expected membership is frozen before dataset construction. Dataset construction may not redefine it.

3. `CompetitionDataset` must expose explicit coverage metadata after `_build_pair_index()`:
   - `expected_sample_ids`: the original `self.sample_ids` list
   - `successfully_loaded_sample_ids`: samples that opened Zarr and produced ≥1 pair
   - `zero_pairs_sample_ids`: samples that opened Zarr but produced 0 retained pairs
   - `failed_sample_ids`: samples that failed Zarr open or Zarr read

4. Production training/validation (Kaggle mode): RAISE if `failed_sample_ids` is non-empty.

5. `validation_samples_total` = `len(expected_validation_ids_after_cap)` — from split file, not from `val_dataset.pairs`. Source: `val_loader.dataset.expected_sample_ids[:cap]`.

6. `validation_samples_evaluated` = count of expected samples whose complete evaluation succeeded.

7. `validation_is_full_fold = True` only if evaluated_set == expected_set AND every expected sample completed evaluation (see Case K above).

8. `max_validation_samples` cap applies to the ordered expected ID list, not to surviving pairs. A capped run must have `validation_is_full_fold = False` unless cap ≥ len(expected fold).

9. Pair count must never substitute for sample coverage. Zero pairs from an expected sample is an explicit state, not a silent disappearance.

10. A sample yielding zero retained pairs must appear in `zero_pairs_sample_ids` explicitly.

---

## TRAINING FAILURE POLICY

**Default policy: IMMEDIATE FATAL for all technical integrity failures.**

Exception handlers in `train_epoch()` for TECHNICAL failures must be REMOVED and the failures must propagate. The existing ">50% batch failure" threshold guard is retained ONLY for `evaluation_failure` in `validate_epoch()`.

| Failure type | Policy | Justification if not FATAL |
|---|---|---|
| `_get_gt_nodes()` raises in training | FATAL — re-raise | None |
| `generate_edge_targets()` raises in training | FATAL — re-raise | None |
| Transformer call raises in training | FATAL — re-raise | None |
| `DivisionLoss` raises in training | FATAL — re-raise | None |
| Loss is NaN/Inf | FATAL — re-raise | None |
| GT load failure in `validate_epoch()` per sample | COUNTED — fail if >50% | Validation has many samples; isolated failure is recoverable |
| Missing Zarr in Kaggle mode | FATAL at construction | None |
| Missing Zarr in local mode | WARN + CONTINUE | Partial staging is normal locally |

---

## FALLBACK COUNTER CONTRACT (Exact Matrix)

| Counter | Trigger | Biological or Technical | Required value for integrity-valid training run | Fatal immediately? | In checkpoint? | In CSV log? | In manifest eligibility? |
|---|---|---|---|---|---|---|---|
| `heatmap_generation_failure` | REMOVED — now FATAL | — | N/A | YES (re-raise) | NO | NO | NO |
| `gt_node_load_failure` | NEW: GT load raises in `train_epoch()` | Technical | 0 (FATAL means can't be non-zero) | YES (re-raise) | NO | N/A | NO |
| `edge_target_generation_failure` | REMOVED — now FATAL | — | N/A | YES (re-raise) | NO | NO | NO |
| `edge_loss_computation_failure` | REMOVED — now FATAL | — | N/A | YES (re-raise) | NO | NO | NO |
| `evaluation_failure` | GT load raises per sample in `validate_epoch()` | Technical | ≤50% of evaluated samples | NO (counted) | NO | YES | NO |
| `legitimate_zero_gt_node_batches` | NEW: `_get_gt_nodes()` returns empty tensor (0,3) for non-retained callers | Biological | Uncapped (informational) | NO | NO | YES | NO |
| `legitimate_zero_positive_edge_batches` | NEW: `generate_edge_targets()` succeeds, zero positive edges | Biological | Uncapped (informational) | NO | NO | YES | NO |

**Notes**:
- The two new "legitimate" counters exist solely to prove biological zeros ARE happening and are distinguishable from technical failures
- Do NOT add P0-7 counters to checkpoint metadata or P0-6 manifest schema — IMMEDIATE FATAL means these counters should be 0 always
- `evaluation_failure` already exists and is retained unchanged

---

## CHECKPOINT INTEGRITY INTERACTION

1. **Can last-checkpoint be saved after technical failure?**: NO — IMMEDIATE FATAL means the run raises before reaching `_save_last_checkpoint()`
2. **Can deployment-candidate checkpoint be saved after technical failure?**: NO — same reason
3. **Should P0-7 counters be in checkpoint metadata?**: NO — IMMEDIATE FATAL makes them always-zero; adding to manifest schema adds complexity with no benefit
4. **Should P0-6 manifest eligibility require zero P0-7 counters?**: NO manifest schema expansion — **REVIEWER DECISION REQUIRED** if reviewer disagrees

The existing P0-6 `deployment_eligibility_errors()` checks are sufficient if P0-7's fail-fast prevents any technically-failed run from reaching `save_checkpoint()` at all.

---

## TRAINING PROVENANCE CONTRACT

**Recommended option: C — make train_kernel.py fail-closed NOW without shared helper**

Rationale: shared helper (Option A) adds 2 new mirror files and import risk; minimal inline parity (Option C) closes the immediate gap. Shared extraction deferred to GPU-gate infrastructure.

**Required changes to `kaggle_kernel/train_kernel.py` in Kaggle mode ONLY:**

1. Replace first-match source discovery with exact-one discovery:
   - Inline the `find_exactly_one_kaggle_input_dir()` logic from `inference_kernel.py` (lines 33–56 there) directly into `train_kernel.py`
   - Raise `RuntimeError` if zero or multiple matches found

2. Replace lenient SHA read with fail-fast:
   - Raise `RuntimeError` if `GIT_SHA.txt` does not exist
   - Raise `RuntimeError` if content is empty
   - Raise `RuntimeError` if `len(raw_sha) != 40 or raw_sha != raw_sha.lower() or not all(c in '0123456789abcdef' for c in raw_sha)`
   - Remove "unknown" fallback string entirely in Kaggle mode

3. `verify_import_origins()` — **GPU-GATE ONLY** — DO NOT ADD TO P0-7

**REVIEWER DECISION REQUIRED**: Whether to extract shared helper now (Option A) or defer (Option C). Recommendation: Option C.

---

## STALE P0-5 TEST CORRECTION

**File**: `tests/test_model.py`
**Method**: `TestSimpleNodeTransformer.test_output_shape_is_all_pairwise_edges_in_row_major_order`
**Stale line 160**: `assert torch.all((out >= 0) & (out <= 1)), "edge_scorer ends in Sigmoid, output must be in [0,1]"`

**Fix**: Replace line 160 with:
```python
assert out.shape == (n_t * n_t1,), f"Expected shape ({n_t * n_t1},), got {out.shape}"
assert isinstance(transformer.edge_scorer[-1], nn.Linear), (
    "edge_scorer must end in Linear (no Sigmoid) — transformer returns raw logits (P0-5 contract)"
)
```

No other test files found with stale probability assertions on transformer output.

---

## AUTHORIZED FILE SCOPE

**REQUIRED (must change):**
- `src/train.py` — INV-1 through INV-6 (GT fail-fast, edge fail-fast, validate coverage, full-fold fix)
- `kaggle_src_dataset/src/train.py` — byte-identical mirror
- `src/dataset.py` — coverage audit, production-mode fail-fast on missing Zarr
- `kaggle_src_dataset/src/dataset.py` — byte-identical mirror
- `kaggle_kernel/train_kernel.py` — SHA validation, exact-one discovery
- `tests/test_model.py` — F11 stale test fix
- `tests/test_train.py` — new P0-7 tests
- `tests/test_dataset.py` — new P0-7 dataset coverage tests

**CONDITIONAL (reviewer decision):**
- `src/kaggle_kernel_utils.py` — only if shared provenance helper chosen (Option A)
- `kaggle_src_dataset/src/kaggle_kernel_utils.py` — mirror of above

**PROHIBITED (do not touch):**
- `src/evaluation.py`, `src/model.py`, `src/targets.py`, `src/tracker.py`
- `src/inference.py`, `src/prediction_graph.py`, `src/split_utils.py`, `src/checkpoint_manifest.py`
- `kaggle_kernel_inference/inference_kernel.py` (already correct — reference only)
- Any GPU-gate runner scripts
- Any P1 architecture files

---

## INVARIANT → CODE → TEST MATRIX

| INV-ID | Invariant | Enforcement point | File/Function | Negative test | Positive compat test | Mirror? |
|---|---|---|---|---|---|---|
| INV-1 | Missing expected Zarr fatal in Kaggle mode | `CompetitionDataset._build_pair_index()` + kernel | `src/dataset.py:248–261`, `train_kernel.py` | T-01: expected sample Zarr absent → kernel raises | T-01P: local mode missing Zarr → warns, continues | Yes |
| INV-2 | Unreadable Zarr fatal in Kaggle mode | `CompetitionDataset._build_pair_index()` | `src/dataset.py:256–261` | T-02: Zarr open raises → fatal | T-02P: local mode → continues | Yes |
| INV-3 | `_get_gt_nodes()` raises on parse failure in strict mode | `TrainingLoop._get_gt_nodes(strict=True)` | `src/train.py:569–592` | T-03: `load_geff_cached` raises → re-raises | T-03P: strict=False → returns None (compat) | Yes |
| INV-4 | `generate_edge_targets()` failure is fatal | `TrainingLoop.train_epoch()` | `src/train.py:786–803` | T-04: `generate_edge_targets` raises → propagates | T-04P: legitimate zero-positive-edge → continues | Yes |
| INV-5 | Edge-loss failure is fatal | `TrainingLoop.train_epoch()` | `src/train.py:808–822` | T-05: transformer raises → propagates; T-06: loss is NaN → raises | T-05P: no-candidates batch (empty logits) → continues | Yes |
| INV-6 | `validation_samples_total` uses expected IDs | `TrainingLoop.validate_epoch()` | `src/train.py:1036–1043` | T-07: val_dataset missing one Zarr → total unchanged | T-07P: all samples present → total equals len(expected) | Yes |
| INV-7 | `validation_is_full_fold=True` requires all expected evaluated | `TrainingLoop.validate_epoch()` | `src/train.py:1067` | T-08: one expected sample absent → full_fold=False | T-08P: all present and evaluated → full_fold=True | Yes |
| INV-8 | Dataset exposes coverage metadata | `CompetitionDataset` post-construction | `src/dataset.py` | T-09: check `failed_sample_ids` populated correctly | T-09P: full dataset → all fields correct | Yes |
| INV-9 | train_kernel.py SHA is validated 40-hex | `train_kernel.py` provenance block | `kaggle_kernel/train_kernel.py:112–117` | T-10: malformed SHA → raises; T-11: absent file → raises | T-10P: valid SHA → proceeds | No mirror |
| INV-10 | train_kernel.py exact-one source discovery | `train_kernel.py` discovery block | `kaggle_kernel/train_kernel.py:47–66` | T-12: multiple matches → raises; T-13: zero matches → raises | T-12P: exactly one match → proceeds | No mirror |
| INV-11 | Stale test removed from test_model.py | `tests/test_model.py:160` | `tests/test_model.py` | T-14: current logit output does NOT satisfy [0,1] bound | T-14P: Linear is final layer of edge_scorer | No |

---

## COMPREHENSIVE NEGATIVE TEST MATRIX (23 minimum)

| Test ID | What to inject | Production path exercised | Expected outcome | Cannot pass vacuously |
|---|---|---|---|---|
| T-01 | Expected training Zarr absent (Kaggle mode) | `CompetitionDataset._build_pair_index()` | RuntimeError at dataset construction | Checks exact sample is in `failed_sample_ids` |
| T-02 | Zarr open raises | `_get_loader()` / `loader.get_shape()` | Fatal in Kaggle mode | Checks error not swallowed |
| T-03 | Geff parse fails in `_get_gt_nodes(strict=True)` | `TrainingLoop._get_gt_nodes()` | Re-raises; no None return | strict=False test still returns None |
| T-04 | Geff file missing for retained pair | `TrainingLoop._get_gt_nodes(strict=True)` | RuntimeError (not None) | Must check exception type |
| T-05 | Geff malformed (returns bad graph object) | `_get_gt_nodes(strict=True)` | RuntimeError | Checks malformed structure doesn't silently pass |
| T-06 | Legitimate zero-GT-node frame (valid geff, t has no nodes) | `_get_gt_nodes(strict=False)` | Returns `torch.zeros((0,3))`, not None | Must check it is NOT None and NOT raises |
| T-07 | `generate_edge_targets()` raises | `TrainingLoop.train_epoch()` | RuntimeError propagates | No `edge_target_generation_failure` counter |
| T-08 | Edge-target output structurally invalid | `TrainingLoop.train_epoch()` | RuntimeError or assertion | Not silently continued |
| T-09 | Legitimate zero-positive-edge batch | `TrainingLoop.train_epoch()` | Training continues; `legitimate_zero_positive_edge_batches++` | Batch count in counter matches injection count |
| T-10 | Transformer raises | `TrainingLoop.train_epoch()` | RuntimeError propagates | No `edge_loss_computation_failure` counter |
| T-11 | `DivisionLoss` returns NaN | `TrainingLoop.train_epoch()` | RuntimeError detected and raised | NaN check must be explicit |
| T-12 | `DivisionLoss` raises | `TrainingLoop.train_epoch()` | RuntimeError propagates | — |
| T-13 | Expected Zarr missing from validation split | `TrainingLoop.validate_epoch()` | `validation_samples_total` unchanged; `validation_is_full_fold=False` | Must check TOTAL, not just full_fold |
| T-14 | One validation sample absent from `val_dataset.pairs` | `TrainingLoop.validate_epoch()` | `validation_samples_evaluated < validation_samples_total` | Both fields must be checked |
| T-15 | Duplicate sample ID in split | `load_and_validate_split()` | ValueError or RuntimeError | Not silently deduplicated |
| T-16 | Unexpected sample ID in strict mode | `CompetitionDataset` coverage audit | Warning logged; not added to expected set | Expected set size unchanged |
| T-17 | Expected sample yields zero retained pairs | `CompetitionDataset._build_pair_index()` | Sample in `zero_pairs_sample_ids`; not in `failed_sample_ids` | Explicit accounting state |
| T-18 | One validation sample evaluation raises | `TrainingLoop.validate_epoch()` | `evaluation_failure++`; sample excluded from `validation_samples_evaluated` | Must check counter |
| T-19 | Capped validation (`max_validation_samples < total`) | `TrainingLoop.validate_epoch()` | `validation_is_full_fold=False` | Even if all selected samples succeed |
| T-20 | Full-fold coverage: all expected evaluated | `TrainingLoop.validate_epoch()` | `validation_is_full_fold=True` | Requires ALL to succeed, not just "most" |
| T-21 | Malformed GIT_SHA.txt in Kaggle mode | `train_kernel.py` provenance block | RuntimeError | Must check both wrong-length AND wrong-charset cases |
| T-22 | Missing GIT_SHA.txt in Kaggle mode | `train_kernel.py` provenance block | RuntimeError (not silent fallback) | Must verify "unknown" string never used |
| T-23 | Multiple source mounts match | `train_kernel.py` exact-one discovery | RuntimeError | Must verify two-mount case raises |

Additional tests beyond 23 minimum:
- T-24: Edge-loss NaN/Inf detection (explicit NaN check after loss computation)
- T-25: Stale [0,1] test removed; raw logits produce values outside [0,1]
- T-26: `verify_import_origins()` NOT added in P0-7 scope (GPU-GATE only)

---

## TEST STATE SAFETY

All new tests must:
- Pass identically in development worktree, fresh `git apply` worktree, staged/committed state
- NOT use `git grep` or Git-index-dependent logic for any repository content invariant
- Use `monkeypatch`, `tmp_path`, and `make_bare_training_loop()` patterns already established in `tests/test_train.py`
- The `make_bare_training_loop()` `__new__` bypass pattern is already established — reuse it

---

## MIRROR CONTRACT

All `src/*.py` changes (train.py, dataset.py, optional kaggle_kernel_utils.py) must be mirrored to `kaggle_src_dataset/src/` as byte-identical copies. Verification command:

```python
import hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
for f in os.listdir('src'):
    if not f.endswith('.py'): continue
    r, m = f'src/{f}', f'kaggle_src_dataset/src/{f}'
    if not os.path.exists(m): print(f'MISSING MIRROR: {f}'); continue
    print('OK' if h(r)==h(m) else 'DIFFER', f)
```

`kaggle_kernel/train_kernel.py` is NOT mirrored (pushed separately as kernel entrypoint).

---

## IMPLEMENTATION ORDER (dependency-aware)

**Phase 1: Coverage and audit primitives**
- `src/dataset.py`: Add `expected_sample_ids`, `successfully_loaded_sample_ids`, `zero_pairs_sample_ids`, `failed_sample_ids` coverage metadata. Add Kaggle-mode production fail-fast.
- Mirror: `kaggle_src_dataset/src/dataset.py`

**Phase 2: Training-loop enforcement (depends on Phase 1 concepts)**
- `src/train.py`:
  - `_get_gt_nodes()`: add `strict: bool = False` parameter; re-raise on failures when `strict=True`
  - `train_epoch()`: remove all `except Exception` catches for edge-target, edge-loss; add NaN/Inf check; add two new biological-zero counters
  - `validate_epoch()`: fix `validation_samples_total` source; fix `validation_is_full_fold` logic
- Mirror: `kaggle_src_dataset/src/train.py`

**Phase 3: Provenance (independent of Phases 1-2)**
- `kaggle_kernel/train_kernel.py`: exact-one discovery, 40-hex SHA validation, no-unknown-fallback in Kaggle mode
- (Optional) `src/kaggle_kernel_utils.py` + mirror if shared helper chosen

**Phase 4: Stale test fix (independent)**
- `tests/test_model.py`: line 160 fix

**Phase 5: New tests (depends on Phases 1-3)**
- `tests/test_train.py`: T-01 through T-26
- `tests/test_dataset.py`: coverage audit tests

**Phase 6: Verification and mirrors**
- Run hash check
- Run full test suite
- Generate patch

---

## VERIFICATION COMMANDS

```bash
# Step 0: Verify baseline
git rev-parse HEAD
# Must output: 8eeace8ab090aecb1cfad336f82865f752c825c4

# Step 1: Focused P0-7 new tests
py -m pytest -q tests/test_model.py::TestSimpleNodeTransformer::test_output_shape_is_all_pairwise_edges_in_row_major_order -v
py -m pytest -q tests/test_model.py -v
py -m pytest -q tests/test_train.py -v
py -m pytest -q tests/test_dataset.py -v

# Step 2: P0-5 regression suite
py -m pytest -q tests/test_p05_double_sigmoid_fix.py -v

# Step 3: P0-6 regression suite
py -m pytest -q tests/test_p06_submission_deployment.py -v

# Step 4: P0-7A regression
py -m pytest -q tests/test_p07a_metric_parity.py -v

# Step 5: Ruff on all authorized changed paths
py -m ruff check src/train.py src/dataset.py kaggle_kernel/train_kernel.py tests/test_model.py tests/test_train.py tests/test_dataset.py

# Step 6: git diff --check (no trailing whitespace)
git diff --check

# Step 7: Mirror hash check
python -c "
import hashlib, os
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
for f in os.listdir('src'):
    if not f.endswith('.py'): continue
    r, m = f'src/{f}', f'kaggle_src_dataset/src/{f}'
    if not os.path.exists(m): print(f'MISSING MIRROR: {f}'); continue
    print('OK' if h(r)==h(m) else 'DIFFER', f)
"

# Step 8: Complete suite with known-issue workaround
py -m pytest -q -rA --ignore=scripts/test_dataset.py
```

---

## PATCH ARTIFACT PROCEDURE

**Patch path**: `C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch`

```powershell
# Create patch directory
mkdir "C:\Users\hemas\Downloads\st_act_p07_review"

# Generate patch against frozen baseline
git diff 8eeace8ab090aecb1cfad336f82865f752c825c4..HEAD -- `
  src/train.py src/dataset.py `
  kaggle_src_dataset/src/train.py kaggle_src_dataset/src/dataset.py `
  kaggle_kernel/train_kernel.py `
  tests/test_model.py tests/test_train.py tests/test_dataset.py `
  > "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"

# Verify file count (must match authorized file count)
(Select-String "^diff --git" "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch").Count

# Compute patch SHA-256
python -c "
import hashlib
with open(r'C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch', 'rb') as f:
    d = f.read()
print('SHA-256:', hashlib.sha256(d).hexdigest())
print('Bytes:', len(d))
"

# Fresh-baseline apply check
git worktree add ..\p07-verify 8eeace8ab090aecb1cfad336f82865f752c825c4
cd ..\p07-verify
git apply --check "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"
git apply "C:\Users\hemas\Downloads\st_act_p07_review\p07-training-integrity.patch"
py -m pytest -q -rA --ignore=scripts/test_dataset.py
```

The patch must NOT include any commit, push, Kaggle kernel deployment, GPU training run, or changes outside the authorized file list.

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

### Files changed in patch (list each with brief description)
- src/train.py: [description]
- kaggle_src_dataset/src/train.py: mirror
- src/dataset.py: [description]
- kaggle_src_dataset/src/dataset.py: mirror
- kaggle_kernel/train_kernel.py: [description]
- tests/test_model.py: [description]
- tests/test_train.py: [description]
- tests/test_dataset.py: [description]
[optional: src/kaggle_kernel_utils.py if shared helper chosen]

### `git apply --check` result
[PASS/FAIL + output]

### Focused P0-7 test output (literal)
[pytest output]

### P0-5 regression test output (literal)
[pytest output]

### P0-6 regression test output (literal)
[pytest output]

### P0-7A metric parity test output (literal)
[pytest output]

### Full suite output (literal)
[py -m pytest -q -rA --ignore=scripts/test_dataset.py output]

### Pre-existing failures (with baseline evidence)
[list with evidence from baseline run]

### Mirror hash check output
[python hash script output]

### Ruff output
[ruff output on changed paths]

### Reviewer decisions taken
- U/Provenance: chose Option [A/B/C] — [one sentence reason]
- Counters: kept/removed legacy counters — [one sentence reason]
- Manifest schema: did/did not expand — [one sentence reason]

### Confirmed: NO CODE OUTSIDE AUTHORIZED SCOPE
YES / NO (if NO, list unauthorized changes)

### NO GIT COMMITS MADE
YES

### NO PUSH MADE
YES

### NO TRAINING STARTED
YES
```

---

## OPEN REVIEWER DECISIONS

### ORD-1 (CRITICAL): Provenance helper extraction
- **Option A**: Create `src/kaggle_kernel_utils.py` + mirror; both kernels import from it. Pros: DRY, prevents future drift. Cons: adds 2 new mirror files, import dependency, larger scope.
- **Option C**: Inline exact-one discovery + SHA validation directly in `train_kernel.py` only. Pros: minimum scope, no new files. Cons: documented tech debt for GPU-gate phase.
- **Recommendation: Option C** — closes the immediate training integrity gap; shared extraction deferred to GPU-gate infrastructure phase where it will be needed anyway.

### ORD-2 (LOW RISK): Counter cleanup
- **Option A**: Remove `heatmap_generation_failure`, `edge_target_generation_failure`, `edge_loss_computation_failure` counters (IMMEDIATE FATAL makes them always 0). Breaks CSV backward compatibility; requires updating `scripts/kaggle_check_run.py`.
- **Option B**: Keep all existing counters but add assertion after epoch that they are exactly 0 (defense in depth). CSV format unchanged.
- **Recommendation: Option B** — keep counters as zero-assertion sentinels; smaller diff; backward compatible with log parsing.

### ORD-3 (REVIEWER DECISION REQUIRED): Manifest schema expansion
- **Question**: Should P0-6 `deployment_eligibility_errors()` check that `training_code_sha != "unknown ..."`?
- **Recommendation: No manifest change needed** — IMMEDIATE FATAL in train_kernel.py removes the source of the "unknown" value upstream; manifest eligibility is sufficient as-is.

---

## ADVERSARIAL REVIEW CHECKLIST (for patch reviewer)

**Baseline**
- [ ] `git apply --check` on exact baseline `8eeace8ab090aecb1cfad336f82865f752c825c4` succeeds clean
- [ ] `diff --git` header count matches authorized file count exactly
- [ ] No unauthorized files in patch

**F1/F2: GT-load failures**
- [ ] `_get_gt_nodes(strict=True)` re-raises on parse failure — no None return for technical errors
- [ ] `_get_gt_nodes(strict=False)` still returns None for missing geff (backward compat)
- [ ] `train_epoch()` calls `_get_gt_nodes(..., strict=True)` on both frames
- [ ] Legitimate zero-GT-node frame (non-filter callers) still returns empty tensor, not None

**F3: Edge-target failure**
- [ ] No `except Exception` around `generate_edge_targets()` in `train_epoch()`
- [ ] `edge_target_generation_failure` counter either removed or proven unreachable
- [ ] `edge_targets = None` path is GONE for the technical failure case

**F4: Edge-loss failure**
- [ ] No `except Exception` around transformer call or `division_loss_fn()` in `train_epoch()`
- [ ] NaN/Inf check added AFTER loss computation
- [ ] `edge_loss_computation_failure` counter either removed or proven unreachable

**F5: Dataset coverage**
- [ ] `CompetitionDataset` exposes `failed_sample_ids` after construction
- [ ] Kaggle-mode kernel raises if `failed_sample_ids` non-empty
- [ ] Local-mode: still warns and continues (no regression)
- [ ] `expected_sample_ids` property returns original `self.sample_ids` from split file

**F6/F7/F9: Validation coverage truthfulness**
- [ ] `validation_samples_total` = count of expected IDs from split, NOT from `val_dataset.pairs`
- [ ] `validation_is_full_fold=True` requires: all expected IDs evaluated AND no missing samples
- [ ] Missing expected sample in `val_dataset.pairs` → `full_fold=False`

**F10: Training provenance**
- [ ] `train_kernel.py` raises if GIT_SHA.txt absent in Kaggle mode
- [ ] `train_kernel.py` raises if SHA is not 40-char lowercase hex
- [ ] "unknown" string never appears as `DEPLOYED_SHA` in Kaggle mode
- [ ] Exact-one or equivalent discovery (raises on multiple matches)

**F11: Stale test**
- [ ] `tests/test_model.py` line 160 no longer asserts [0,1] bound
- [ ] Replacement asserts `isinstance(transformer.edge_scorer[-1], nn.Linear)`
- [ ] `test_p05_double_sigmoid_fix.py` AST tests still pass

**F12: Biological zeros properly tracked**
- [ ] `legitimate_zero_positive_edge_batches` counter incremented for zero-positive-edge batches
- [ ] `legitimate_zero_gt_node_batches` counter incremented for non-filter zero-GT-node results
- [ ] These counters are never incremented for technical failures

**P0 preservation**
- [ ] P0-1: `filter_unannotated_pairs` mechanism unchanged; retained-pair heatmap fail-loud preserved
- [ ] P0-2: split identity embedding in checkpoints unchanged
- [ ] P0-3: post-pass circuit breaker in `validate_epoch()` unchanged
- [ ] P0-4: `max_validation_samples` whole-sample cap logic unchanged
- [ ] P0-5: AST tests in `test_p05_double_sigmoid_fix.py` all pass
- [ ] P0-6: manifest eligibility tests in `test_p06_submission_deployment.py` all pass
- [ ] P0-7A: metric parity tests in `test_p07a_metric_parity.py` all pass

**Scope**
- [ ] No architecture changes (model.py, targets.py loss functions, tracker.py)
- [ ] No GPU-gate infrastructure (gradient instrumentation, ROC-AUC, gate runner)
- [ ] No P1 items (cross-attention, positional encoding, temporal convolution)
- [ ] `verify_import_origins()` NOT added to train_kernel.py (GPU-GATE only)

**Mirrors**
- [ ] All changed `src/*.py` are byte-identical to `kaggle_src_dataset/src/`
- [ ] Hash check passes for all files

**Tests**
- [ ] T-01 through T-23 minimum tests present and passing
- [ ] Tests use real production code paths (not test-only duplicates)
- [ ] No test uses git-index-dependent logic
- [ ] Tests pass in fresh `git apply` worktree equivalently
