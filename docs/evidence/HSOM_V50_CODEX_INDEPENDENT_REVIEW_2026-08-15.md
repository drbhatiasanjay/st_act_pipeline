# Independent Codex Review — HSOM v50 Forensic Reconstruction

**Review date:** 2026-08-15  
**Canonical report reviewed:** `docs/evidence/HSOM_V50_FORENSIC_RECONSTRUCTION_2026-08-15.md`  
**Canonical report commit:** `6b315891a80fc2530d79ce29f8bff861b293e919`  
**Canonical report SHA-256:** `c165bc3dee2601ee6bdfe6630dc57f1f3f332afd74d2db206b9be78743714ea7`  
**v50 declared execution SHA:** `bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c`  
**Review mode:** independent, repository- and artifact-first; no trust placed in the report's conclusions

## Final verdict

**REJECT**

The report contains genuine primary observations, including the v50 score, prediction counts, full-fold coverage, deployed-SHA marker, checkpoint identity, and the raw count of 1,626 severe-under-confidence warnings. Its central scientific interpretation is nevertheless invalidated by four independently reconstructed facts:

1. Validation contained **7,029 batches**, and `validate_epoch()` made **two detection calls per batch**, so the semantic denominator was **14,058 detection calls**, not 1,626. The 1,626 adaptive fallbacks were **11.566368%**, not 100%.
2. The fixed `0.5` threshold was retained for **12,432 calls (88.433632%)**, not zero. Logged validation maxima above 0.5 independently corroborate this.
3. The report never establishes the exact 71-sample `estimated_number_of_nodes` population. Its `~54–100K` denominator is only an estimate and is inconsistent with the two locally available exact metadata values, which already sum to 58,550. The claimed 5–10× overprediction conclusion must therefore be withdrawn.
4. The evaluator does not apply a global overprediction penalty to the reported base score. It micro-averages base edge Jaccard, computes adjusted Jaccard per sample, and then denominator-weight-averages those adjusted values. A sample with `N_pred < T_true` receives a multiplier greater than one. This is why adjusted `0.001986` can correctly exceed base `0.001845`; the report's “penalty reduced base to adjusted” interpretation is reversed.

In addition, an Oracle execution did use the exact v50 checkpoint: the Oracle input file and v50 epoch checkpoint have identical SHA-256 `8a788a192725d80a39c6ea4a5a4f74ade67cf4c259fa67cc943d9ede15c25092`. That Oracle result is qualified because it evaluated only **two locally staged validation samples**, not the full 71-sample fold.

## Scope and frozen evidence

The main review state was frozen before further inspection:

```text
HEAD:   6b315891a80fc2530d79ce29f8bff861b293e919
branch: gpu-sanity-gate-wave2-v2
```

The main worktree already contained unrelated user modifications and untracked files; none were changed. This review was written in an isolated worktree based on the canonical report commit.

Primary artifacts inspected:

- `C:\Users\hemas\Downloads\kaggle_train_run_v50_output\full_log.json`
- `C:\Users\hemas\Downloads\kaggle_train_run_v50_output\training_log.csv`
- `C:\Users\hemas\Downloads\kaggle_train_run_v50_output\training_progress.json`
- `C:\Users\hemas\Downloads\kaggle_train_run_v50_output\checkpoints\checkpoint_manifest.json`
- `C:\Users\hemas\Downloads\kaggle_train_run_v50_output\checkpoints\epoch_1_val_score_0.0020.pt`
- `C:\Users\hemas\Downloads\oracle_check_training_run.log`
- `C:\Users\hemas\Downloads\kaggle_probe_output_v3\gpu_learning_probe\training_run_checkpoint.pt`
- Git trees and history at `bc989ed`, the named fix commits, and canonical report commit `6b31589`
- Evaluator source at `bc989ed:src/evaluation.py` and `bc989ed:src/tracking_cellmot/metrics.py`
- Local staged GEFF metadata for `44b6_0113de3b` and `44b6_0b24845f`
- `kaggle_runs.db`, opened with SQLite read-only mode; registry notes were treated as secondary, post-hoc evidence

## Claim-by-claim classification

The following ledger contains 46 material claims. Counts are exact: **VERIFIED 15; VERIFIED_WITH_QUALIFICATION 8; UNSUPPORTED 8; CONTRADICTED 12; UNRESOLVED 3.**

| # | Material claim | Classification | Primary finding |
|---:|---|---|---|
| 1 | The reviewed report bytes match the expected SHA-256. | VERIFIED | File hash is exactly `c165bc3d...14ea7`. |
| 2 | The canonical report commit exists and is `6b315891...`. | VERIFIED | `git cat-file -t` returns `commit`; frozen HEAD matches it. |
| 3 | v50 declared deployed SHA `bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c`. | VERIFIED_WITH_QUALIFICATION | Two log markers, progress JSON, registry, and checkpoint manifest agree; the marker is generated provenance rather than a cryptographic archive of the mounted dataset. |
| 4 | The saved checkpoint records training code SHA `bc989ed...`. | VERIFIED | `checkpoint_manifest.json` records the exact SHA. |
| 5 | The epoch checkpoint has SHA-256 `8a788a19...c25092`. | VERIFIED | Manifest value equals an independently computed file hash. |
| 6 | The run was Kaggle kernel version 50. | VERIFIED_WITH_QUALIFICATION | Read-only registry row says version 50; the raw downloaded artifact itself does not embed the kernel version. |
| 7 | Validation covered all 71 configured validation samples. | VERIFIED | Manifest says 71/71 and `validation_is_full_fold=true`; log says full fold. |
| 8 | Validation loader contained 7,029 batches. | VERIFIED | Raw log: `Val dataset size: 7029` and `Val loader batches: 7029`. |
| 9 | v50 produced score/count fields reported in the CSV and progress JSON. | VERIFIED | Adjusted/score exact value is `0.0019855762417342647`; CSV rounds components to six decimals; nodes 554,366 and edges 388,859. |
| 10 | The execution was structurally healthy. | VERIFIED_WITH_QUALIFICATION | Full fold, checkpoint, zero enumerated fallback counters, and no ERROR/Traceback/CRITICAL; “healthy” does not establish scientific validity. |
| 11 | Short commit `76bf901` is a real ancestor of v50. | VERIFIED | It resolves to `76bf901126df7a70521be3b4923602a77188d565`; ancestry check succeeds. |
| 12 | Reported full SHA `76bf901abb836c5874d6107db431077ade57f8bf` is correct. | CONTRADICTED | It is a hybrid of `76bf901` and the suffix of `eb31af9`; no such commit is the loss fix. |
| 13 | Commit `76bf901` changed `src/train.py` with 93 insertions/23 deletions. | CONTRADICTED | It changed `src/targets.py` and `tests/test_targets.py`; 93/23 is the combined two-file stat. |
| 14 | `76bf901` itself introduced adaptive per-batch class weighting. | UNSUPPORTED | Adaptive weighting was introduced by predecessor `eef5700c24a9549579b248f6e07b178e371c2856`; `76bf901` changed normalization from `mean()` to weighted-sum normalization. |
| 15 | The quantile-normalization fix was present in v50. | VERIFIED_WITH_QUALIFICATION | Deployed ancestry is `ba1bdb4a434d925c9e54cc608c039d38e93cd4ef`, followed by mirror sync `bc989ed`; `2a263c2` is a parallel cherry-pick on the review branch, not the deployed ancestor. |
| 16 | Symmetric adaptive threshold fix `4a26f02` was present. | VERIFIED | Full SHA `4a26f021e1d9e549f3d2e5393d036d6a59726ddb`; ancestor of `bc989ed`. |
| 17 | Reference-aligned weighting fix `ab5fcc3` was present. | VERIFIED | Full SHA `ab5fcc305cc2aa360486a6b6bcf2c0cd346443a1`; ancestor of `bc989ed`. |
| 18 | The report's four-commit list is the complete relevant pre-v50 intervention set. | UNSUPPORTED | It omits at least LR `3e-3` (`872743646b...`) and warmup 300 (`1c5c50f16e...`), both active and invoked by the report's causal narrative. |
| 19 | The named fixes are proven to have solved historical collapse. | UNSUPPORTED | v50 combined multiple changes and has no ablation or controlled counterfactual. |
| 20 | v50 observationally escaped a zero-score state. | VERIFIED_WITH_QUALIFICATION | v50 score is nonzero, but “first ever” and causal attribution require a complete run inventory. |
| 21 | There are 1,626 raw severe-under-confidence warning records. | VERIFIED | Exact JSON count. |
| 22 | There are 1,626 adaptive-threshold invocations. | VERIFIED | Exact count of `using adaptive threshold=`; there were no high-positive-fraction adaptive warnings. |
| 23 | Those 1,626 invocations exhaust all validation detection calls. | CONTRADICTED | Source makes two calls for each of 7,029 batches: 14,058 total. |
| 24 | Adaptive fallback rate was 100%. | CONTRADICTED | `1626 / 14058 = 11.566368%`. |
| 25 | Fixed threshold `0.5` fired zero times. | CONTRADICTED | It was retained for 12,432 calls, 88.433632% of the semantic denominator. |
| 26 | Effective adaptive thresholds were not recoverable from the log. | CONTRADICTED | All 1,626 values parse; min `0.320821`, max `0.420600`, mean `0.383704879458793`. |
| 27 | Predicted node total was 554,366. | VERIFIED | CSV, progress JSON, manifest, and log agree. |
| 28 | Full-fold `T_true` was approximately 54–100K. | UNSUPPORTED | No exact all-71 metadata sum or per-sample evaluator rows were logged. |
| 29 | v50 overpredicted nodes by 5–10×. | UNSUPPORTED | The asserted denominator is estimated and must be rejected; two exact local `T_true` values already sum to 58,550. |
| 30 | Base edge Jaccard `0.001845` means 0.18% edge recall. | CONTRADICTED | It is Jaccard `TP/(TP+FP+FN)`, not recall `TP/(TP+FN)`. |
| 31 | Adjusted edge Jaccard/score was `0.0019855762417342647` (rounded `0.001986`). | VERIFIED | Exact progress/manifest value and rounded CSV/log values agree. |
| 32 | An overprediction penalty reduced base `0.001845` to adjusted `0.001986`. | CONTRADICTED | Adjusted is larger; per-sample underprediction produces a multiplier above one, and adjusted values are denominator-weighted. |
| 33 | Division Jaccard zero could mean perfect divisions. | CONTRADICTED | Perfect division Jaccard is 1. Zero means no TP with a positive denominator, or a NaN/no-division case sanitized to zero by `train.py`. |
| 34 | No Oracle result had been completed for the v50 checkpoint. | CONTRADICTED | `oracle_check_training_run.log` exists and its input checkpoint hash exactly matches the v50 epoch checkpoint. |
| 35 | The discovered Oracle run applies to the v50 checkpoint. | VERIFIED_WITH_QUALIFICATION | Exact identical SHA-256, but Oracle code ran from `8e7faa2` and only two local samples. |
| 36 | A full-71-sample Oracle ceiling exists for v50. | UNRESOLVED | Only two locally staged samples were evaluated. |
| 37 | Existing Oracle evidence proves extraction H4 is the primary bottleneck. | UNSUPPORTED | Two-sample mode results are low and useful diagnostically, but cannot establish a full-fold primary bottleneck. |
| 38 | EV-V50-001 records a nonzero result at scale. | VERIFIED_WITH_QUALIFICATION | Observation is valid; “validates fix” must be removed as causal overreach. |
| 39 | EV-V50-002 establishes universal fallback and a likely bottleneck. | CONTRADICTED | Rate is 11.57%, and no intervention test establishes bottleneck status. |
| 40 | EV-V50-003 establishes massive 5–10× overprediction. | UNSUPPORTED | Exact full-fold scoring denominator is missing; proposed effect and bottleneck are not demonstrated. |
| 41 | EV-V50-004 records a clean, completed execution. | VERIFIED_WITH_QUALIFICATION | Engineering health is supported, separate from scientific correctness. |
| 42 | Raw log contains 44 node-warning lines and 216 edge-warning lines. | VERIFIED | Exact substring counts. |
| 43 | Total warning count was 1,886. | CONTRADICTED | Raw JSON contains 1,890 records containing `WARNING`; four are unrelated library syntax/future warnings. |
| 44 | The 44/216 warning counts are unmatched individual nodes/edges and support percentage estimates. | UNSUPPORTED | `tracksdata` emits one warning when an entire match operation returns no IDs; stdout/stderr duplicate logger rendering also inflates raw-line counts. |
| 45 | v50 was the first nonzero real score in the project's complete history. | UNRESOLVED | Available registry rows support it but do not constitute a complete historical run inventory. |
| 46 | Exact all-71 `estimated_number_of_nodes` and per-sample adjustment factors can be reconstructed from saved v50 artifacts. | UNRESOLVED | They were not serialized; only two local GEFFs are available here. |

## Primary reconstruction

### 1. v50 execution and checkpoint identity

The raw log contains the deployed SHA twice and verifies imports from the mounted `st-act-src` dataset. The progress file and checkpoint manifest independently repeat `bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c`. The checkpoint manifest records:

```text
checkpoint_file:              epoch_1_val_score_0.0020.pt
checkpoint_sha256:            8a788a192725d80a39c6ea4a5a4f74ade67cf4c259fa67cc943d9ede15c25092
training_code_sha:            bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c
validation_is_full_fold:      true
validation_samples_evaluated: 71
validation_samples_total:     71
```

Independent hashing reproduced the manifest checkpoint hash. Identity is therefore strong artifact provenance. Qualification is still appropriate for the phrase “deployed tree”: the downloaded outputs do not contain a cryptographic archive/hash of every mounted source file, and `GIT_SHA.txt` is a generated marker.

### 2. Exact pre-v50 commit provenance

The report's materially relevant fix chain should be stated as follows:

| Change | Exact commit | Relationship to v50 |
|---|---|---|
| Adaptive per-batch class weighting introduced | `eef5700c24a9549579b248f6e07b178e371c2856` | Ancestor |
| DetectionLoss weighted-sum normalization | `76bf901126df7a70521be3b4923602a77188d565` | Ancestor |
| LR changed to `3e-3` | `872743646b646f9ec117d7d2b95c75fd98153917` | Ancestor |
| Warmup 300 enabled | `1c5c50f16efbb8c3c0ba4a482f50120925f37d44` | Ancestor |
| Symmetric low-confidence adaptive threshold | `4a26f021e1d9e549f3d2e5393d036d6a59726ddb` | Ancestor |
| Reference-aligned loss weighting + no-decay groups | `ab5fcc305cc2aa360486a6b6bcf2c0cd346443a1` | Ancestor |
| Deployed quantile-normalization fix | `ba1bdb4a434d925c9e54cc608c039d38e93cd4ef` | Direct parent ancestry |
| Kaggle source mirror sync | `bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c` | Declared v50 SHA |

`2a263c273bd62f2b60c709f7d8557c15c215378a` is the corresponding change on the review branch, not the commit in v50's deployed ancestry. The normalization commit changed `src/data_loader.py` and `tests/test_data_loader_real.py`, not `src/submission_pipeline.py`.

These ancestry facts prove only that code was present. They do not prove which change caused the nonzero score.

### 3. Severe-under-confidence denominator and fixed threshold

Primary source at `bc989ed:src/train.py` establishes:

```python
for batch in self.val_loader:
    ...
    peaks_t = self._peaks_for_channel(detection_probs, channel=0, t_idx=t_idx)
    peaks_t1 = self._peaks_for_channel(detection_probs, channel=1, t_idx=t_idx)
```

The v50 log establishes `Val loader batches: 7029`. Full-fold validation had no sample cap and completed 71/71 samples. Therefore:

```text
semantic detection calls = 7,029 batches × 2 channels = 14,058
zero-positive adaptive calls = 1,626
high-positive adaptive calls = 0
fixed-threshold calls = 14,058 − 1,626 = 12,432
zero-positive fallback rate = 1,626 / 14,058 = 11.566368%
fixed-threshold usage rate = 12,432 / 14,058 = 88.433632%
```

Channel distribution among the 1,626 severe warnings was 182 for channel 0 and 1,444 for channel 1. This asymmetry is another reason that `1,626 / 2 = 813` is not a valid reconstruction of unique two-channel validation windows.

The fixed path is also directly visible in validation log samples, e.g. logged sigmoid maximum `0.5927`. The report's claim that raw sigmoid never exceeded 0.5 anywhere is contradicted.

All adaptive threshold values are present in the raw messages:

```text
count: 1,626
min:   0.320821
max:   0.420600
mean:  0.383704879458793
```

### 4. GT population and the rejected 5–10× conclusion

The scoring denominator is per-sample `GeffMetadata.extra['estimated_number_of_nodes']`, not sparse labeled graph node count. v50 serialized neither the 71 per-sample values nor their sum. Therefore the report's `~54–100K` figure is not primary evidence.

The two locally staged validation samples demonstrate the magnitude of the denominator error:

| Sample | Sparse labeled nodes | Exact `estimated_number_of_nodes` | v50 predicted nodes in two-sample Oracle replay |
|---|---:|---:|---:|
| `44b6_0113de3b` | 52 | 25,755 | 1,645 |
| `44b6_0b24845f` | 51 | 32,795 | 25,354 |
| **Two-sample total** | **103** | **58,550** | **26,999** |

Thus two of 71 samples alone already exceed the report's 54K lower estimate. On both available samples, v50 predicts fewer nodes than the scoring target, not more. This does not prove the all-71 aggregate direction; it proves that the report's denominator and 5–10× conclusion are not established. The exact full-fold population remains scientifically unresolved from saved artifacts.

### 5. Evaluator semantics and why adjusted exceeds base

At `bc989ed`, `evaluate_submission()` calls the vendored evaluator for each sample, derives `n_total` from GEFF metadata, calls `per_sample_metrics()`, then calls `summarise()`.

For sample `i`:

```text
J_i = TP_i / (TP_i + FP_i + FN_i)
r_i = (N_pred_i − T_true_i) / T_true_i
J_adj_i = max(0, J_i × (1 − 0.1 × r_i))
w_i = TP_i + FP_i + FN_i
```

Aggregation is:

```text
base edge Jaccard = ΣTP_i / Σ(TP_i + FP_i + FN_i)
adjusted Jaccard  = Σ(w_i × J_adj_i) / Σw_i
```

When `N_pred_i < T_true_i`, `r_i` is negative and `(1 − 0.1r_i) > 1`, so that sample's adjusted value exceeds its base value. Because the base micro-average is also the `w_i`-weighted average of `J_i`, an aggregate adjusted score above base indicates a positive weighted adjustment among scoring samples. It is incompatible with the report's statement that a dominant overprediction penalty reduced the base metric.

The saved base `0.001845` is rounded to six decimals; the exact base value was not serialized. The exact adjusted value is `0.0019855762417342647`.

Division semantics are also misstated. The evaluator returns NaN and drops the division term when the total division denominator is zero; `train.py` sanitizes NaN to logged `0.0`. Otherwise, division Jaccard zero means zero TP with a positive denominator. Perfect division prediction would be 1.0, never 0.0.

### 6. Exact Oracle checkpoint identity and scope

The Oracle log begins with:

```text
Loading checkpoint: ..\kaggle_probe_output_v3\gpu_learning_probe\training_run_checkpoint.pt
epoch=1 training_code_sha=bc989ed6787d21ccb550cdc4b1faf19fda1e8b4c
Validation batches: 198 across samples ['44b6_0113de3b', '44b6_0b24845f']
```

Independent hashes:

```text
kaggle_probe_output_v3/.../training_run_checkpoint.pt
  8a788a192725d80a39c6ea4a5a4f74ade67cf4c259fa67cc943d9ede15c25092

kaggle_train_run_v50_output/checkpoints/epoch_1_val_score_0.0020.pt
  8a788a192725d80a39c6ea4a5a4f74ade67cf4c259fa67cc943d9ede15c25092
```

The Oracle result therefore applies to the exact v50 checkpoint. It used Oracle tooling from commit `8e7faa2036795e9d5ce6d02861792454630db88c`; the only `src/oracle_evaluation.py` change relative to the earlier committed implementation was `zip(..., strict=False)` to `strict=True` in a three-coordinate distance calculation.

Observed two-sample results:

| Sample | Mode | Score | Edge Jaccard |
|---|---|---:|---:|
| `44b6_0113de3b` | GT nodes + model edges | 0.0000 | 0.0000 |
| `44b6_0113de3b` | Model nodes + Oracle edges | 0.0000 | 0.0000 |
| `44b6_0b24845f` | GT nodes + model edges | 0.0000 | 0.0000 |
| `44b6_0b24845f` | Model nodes + Oracle edges | 0.0040 | 0.0039 |

These are valid observations for two samples. They are not a full-fold Oracle ceiling and cannot alone designate H4 as the primary leaderboard bottleneck.

## Observation versus causality

v50 observationally demonstrates:

- a completed 71-sample validation under declared SHA `bc989ed`;
- a nonzero adjusted edge score;
- nonzero predicted nodes and edges;
- a checkpoint with verified hash and full-fold manifest;
- 1,626 zero-positive adaptive fallbacks among 14,058 detection calls;
- 12,432 calls that retained the fixed threshold;
- zero enumerated training/evaluation fallback failures.

v50 does **not** causally demonstrate:

- which pre-v50 change produced the nonzero score;
- that quantile normalization, loss normalization, LR, warmup, or extraction independently “solved” historical collapse;
- that architecture limitations are ruled out;
- that extraction is the primary score bottleneck;
- that predicted nodes exceed the exact full-fold scoring target;
- that longer training will improve the score;
- that the two-sample Oracle pattern generalizes to all 71 validation samples.

## EV-V50 record audit: Observation → Defect → Effect → Bottleneck → Priority

| Record | Observation | Defect | Effect | Bottleneck | Priority / disposition |
|---|---|---|---|---|---|
| EV-V50-001 | Keep: score/counts/full-fold/checkpoint identity are verified. | Do not label as a validated fix. | Nonzero output and completed evaluation only. | Not established. | **KEEP WITH MAJOR QUALIFICATION**; remove causal “validates fix” language. |
| EV-V50-002 | Replace with 1,626/14,058 = 11.566368%; include 182 ch0 and 1,444 ch1. | Universal miscalibration is false. A minority zero-positive path is observed. | Fixed threshold remained active in 88.43% of calls. | Not established. | **REWRITE**; bottleneck status `UNRESOLVED`, not `LIKELY`. |
| EV-V50-003 | Keep only predicted total 554,366. Remove `~54–100K` and 5–10×. | No exact full-fold count defect is demonstrated. | No false-positive explosion can be inferred from aggregate nodes alone. | Not established. | **REJECT AS WRITTEN**; regenerate only after exact per-sample `T_true` extraction. |
| EV-V50-004 | Keep execution-health facts. | N/A. | Infrastructure completed without enumerated technical fallback. | Says nothing about scientific bottleneck. | **KEEP WITH QUALIFICATION**. |

## Exact correction list

The correction list has **5 BLOCKER** and **8 MAJOR** items.

| # | File section | Existing claim | Primary evidence | Required correction | Severity |
|---:|---|---|---|---|---|
| 1 | C.1, C.4, Final Decision 1 | Full loss-fix SHA is `76bf901abb...`; file is `src/train.py`; commit introduced adaptive weighting. | `git rev-parse 76bf901`; `git show --stat 76bf901`; predecessor `eef5700`. | Use `76bf901126df7a70521be3b4923602a77188d565`; state it changed `src/targets.py` + tests and fixed weighted-loss normalization; credit adaptive weighting to `eef5700...`. | MAJOR |
| 2 | C.2, C.4, Final Decision 1 | `2a263c2 / ba1bdb4` are interchangeable deployed SHAs; changed `src/submission_pipeline.py`. | Commit trees/parents and stats. | Identify deployed ancestor `ba1bdb4...`, mirror sync `bc989ed...`, parallel cherry-pick `2a263c2...`; correct files to `src/data_loader.py` + tests. | MAJOR |
| 3 | C.3–C.4, J.1, Final Decision 2 | Four listed fixes adequately define the pre-v50 causal intervention. | Git blame/history shows LR `87274364...`, warmup `1c5c50f1...`, and adaptive precursor `eef5700...`. | Expand provenance table and explicitly prohibit causal attribution without ablation. | MAJOR |
| 4 | F.4, H, I.1–I.4, M.2, T, V.2, X.3/X.7, Final Decisions 3–5/7/10 | 1,626 warnings cover every detection call; fallback 100%; fixed threshold never fired. | Log says 7,029 batches; source makes two calls/batch; 1,626 severe warnings; zero high-positive warnings. | Replace denominator with 14,058; report fallback 11.566368%, fixed use 12,432/14,058 = 88.433632%; remove all universal-miscalibration claims. | BLOCKER |
| 5 | I.2, H, X.6/X.7 | Adaptive thresholds are truncated/not recoverable. | All 1,626 log messages contain parseable values. | Report count/min/max/mean: 1,626 / 0.320821 / 0.420600 / 0.383704879458793. | MAJOR |
| 6 | J.2, K.2/K.4, M.3, N, V.3, Final Decisions 3–5 | `554K` is 5–10× GT `~54–100K`, proving systematic false-positive explosion. | Exact denominator absent; two local `T_true` values total 58,550 and both exceed local v50 prediction counts individually. | Withdraw 5–10× and overprediction/false-positive conclusions; mark exact all-71 `T_true` unresolved pending direct metadata extraction. | BLOCKER |
| 7 | J.3, N.1, Final Decision 4 | Adjustment penalty reduces base Jaccard and dominates loss. | `per_sample_metrics()` and `summarise()` at `bc989ed`; adjusted is numerically greater than base. | Document per-sample adjustment and weighted aggregation; explain underprediction multiplier >1; remove “dominant overprediction penalty.” | BLOCKER |
| 8 | J.3, N.1 | Division `0.0` could mean perfect divisions. | Metric code and NaN-to-zero sanitation in `train.py`. | State perfect is 1.0; zero is zero TP with denominator, or sanitized NaN/no-division case. | MAJOR |
| 9 | L.3, S.1, T, U, X.4/X.6, Final Decisions 5/8–10 | Oracle evaluation is only in progress/not completed for v50. | Oracle input hash equals v50 checkpoint hash exactly; log has completed two-sample results. | Incorporate the two-sample v50 Oracle result; explicitly state it is not full-fold and cannot by itself choose the primary bottleneck. | BLOCKER |
| 10 | A, J.1, M.1, Final Decision 2 | v50 proves the named fixes solved historical collapse and rules out a fundamental architecture issue. | Single combined run; no ablation/counterfactual; very low score and two-sample Oracle results. | Restrict to observational statements; classify individual causal effects and architecture ceiling as unresolved/unsupported. | BLOCKER |
| 11 | V.1–V.4 | Proposed EV records have correct defect/effect/bottleneck statuses. | Reconstructed denominator, absent exact `T_true`, and lack of interventions. | Apply the EV audit above: qualify 001/004, rewrite 002, reject 003 as written. | MAJOR |
| 12 | F.6, O.1–O.2, X.3 | 44/216 are individual unmatched nodes/edges, percentages are meaningful, and total warnings are 1,886. | `tracksdata` source warns once when a match operation returns no IDs; stdout/stderr duplicate renderings; raw WARNING count 1,890. | Describe them as raw log records (likely duplicated logical events), remove node/edge percentage estimates, and correct total raw WARNING records to 1,890. | MAJOR |
| 13 | B.3, P, Final Decision 10 | Registry notes independently verify universal fallback/full-fold details. | Read-only row shows post-hoc free text with the same false 1,626/1,626 calculation; structured validation fields are null/false. | Treat registry notes as secondary narrative only; use manifest/log/source for coverage and denominator. Correct or supersede the bad note in a separately authorized database operation, not as part of this review. | MAJOR |

## Report errors versus scientifically unresolved evidence

### Demonstrable report errors

- The full SHA and file scope for `76bf901` are wrong.
- The 1,626-warning semantic denominator is wrong.
- The 100% fallback rate and zero fixed-threshold activations are false.
- Effective thresholds are recoverable.
- Base Jaccard is mislabeled as recall.
- Adjusted-score direction and penalty semantics are reversed.
- “Perfect divisions” cannot yield division Jaccard zero.
- The Oracle status is stale/false: an exact-v50-checkpoint two-sample run completed.
- Warning totals and warning-unit interpretations are wrong.

### Evidence that remains scientifically unresolved

- Exact sum and distribution of all 71 samples' `estimated_number_of_nodes`.
- Exact unrounded base edge Jaccard and per-sample adjustment rows.
- Full-fold Oracle ceiling for the v50 checkpoint.
- Which pre-v50 intervention caused the transition to nonzero score.
- Whether extraction, linker, representation, training duration, or an interaction is the primary full-fold bottleneck.
- Whether v50 is literally the first nonzero real score across every historical run not present in the available registry/artifacts.

These unresolved questions are not licenses for the report's conclusions. Where the necessary denominator or controlled experiment is absent, the scientifically correct classification is `UNRESOLVED` or `UNSUPPORTED`, not a directional claim.

## Reproducible primary-evidence commands

Representative commands used in the independent reconstruction:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
  'docs/evidence/HSOM_V50_FORENSIC_RECONSTRUCTION_2026-08-15.md'

git rev-parse 76bf901
git show --stat --oneline 76bf901
git merge-base --is-ancestor 76bf901 bc989ed
git show --stat --oneline ba1bdb4 4a26f02 ab5fcc3 87274364 1c5c50f1 eef5700

$j = Get-Content -Raw `
  'C:\Users\hemas\Downloads\kaggle_train_run_v50_output\full_log.json' |
  ConvertFrom-Json
@($j | Where-Object { $_.data -like '*severe under-confidence*' }).Count
@($j | Where-Object { $_.data -like '*undertrained-model miscalibration*' }).Count
$j | Where-Object { $_.data -like '*Val loader batches:*' } |
  Select-Object -ExpandProperty data

git show bc989ed:src/train.py
git show bc989ed:src/evaluation.py
git show bc989ed:src/tracking_cellmot/metrics.py

Get-FileHash -Algorithm SHA256 `
  'C:\Users\hemas\Downloads\kaggle_probe_output_v3\gpu_learning_probe\training_run_checkpoint.pt', `
  'C:\Users\hemas\Downloads\kaggle_train_run_v50_output\checkpoints\epoch_1_val_score_0.0020.pt'

Select-String -LiteralPath `
  'C:\Users\hemas\Downloads\oracle_check_training_run.log' `
  -Pattern '^Loading checkpoint|training_code_sha|^Validation batches:|^===|^  gt_nodes|^  model_nodes'
```

No training, Kaggle execution, deployment, submission, commit, push, branch switch in the main worktree, source modification, test modification, canonical-report modification, or database mutation was performed. The project verification script was intentionally not run because the review phase was placed under a strict read-only main-worktree constraint and this deliverable changes documentation only in an isolated worktree.

## Review summary

```text
verdict:                         REJECT
blocker corrections:             5
major corrections:               8
VERIFIED:                        15
VERIFIED_WITH_QUALIFICATION:      8
UNSUPPORTED:                      8
CONTRADICTED:                    12
UNRESOLVED:                       3
```
