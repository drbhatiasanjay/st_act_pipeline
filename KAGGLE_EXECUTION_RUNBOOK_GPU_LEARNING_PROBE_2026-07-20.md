# Kaggle Execution Runbook — GPU Learning Probe (PR #7)

**Purpose:** a self-contained runbook so Codex (or any other agent with Kaggle CLI credentials)
can deploy and run the bounded GPU learning probe end-to-end **without Claude**. This mirrors,
step for step, the flow Claude already executed manually for the GPU First-Light Gate (PR #6,
kernel `drbhatiasanjay/st-act-gpu-first-light`) and verified PASS. Nothing here is speculative —
every command, field name, and known failure mode below was directly observed and confirmed
during that PR #6 run and during the PR #7 code/test review, not guessed.

Read this whole document before running anything. Do not improvise steps not listed here — if
something doesn't match (a missing file, a different field name), stop and report the mismatch
rather than working around it.

---

## Runtime requirement — read this before anything else, in every session that opens this file

This runbook requires an execution environment that has **both** of the following, right now, in
the actual process running these commands:

1. A valid, already-authenticated Kaggle API credential (a real `~/.kaggle/kaggle.json` token for
   the `drbhatiasanjay` account) — not just the `kaggle` CLI package installed.
2. Outbound network access to Kaggle's API/website (`kaggle.com` and its API endpoints) that is
   not blocked by a firewall, sandbox, or proxy.

**If either is missing — no Kaggle credentials, no authenticated Kaggle connector, or network
access to Kaggle is blocked — every command in §3 will fail immediately (an auth error or a
connection error), and there is no workaround available to this runbook.** Specifically:

- **Spawning a sub-agent does not help.** A sub-agent inherits the exact same sandbox and
  credential restrictions as the agent that spawned it. If the parent session has no Kaggle
  credentials and no network egress to Kaggle, every sub-agent it spawns has the same gap —
  don't retry this runbook by delegating it downward expecting a different result.
- **There is no browser/manual-automation fallback authorized by this runbook.** That path is
  explicitly out of scope here (see §0/§5) — it is not a substitute for real API access.
- **Retrying the same command, changing a flag, or trying a different Kaggle endpoint will not
  fix an absent credential or a blocked network path.** Don't loop on this.

**Confirmed-working environment:** the local Windows machine this repository already lives on
(the same machine/session where the PR #6 GPU First-Light Gate Kaggle run was actually executed
end-to-end and verified PASS) has an authenticated `kaggle` CLI and working network access to
Kaggle right now. If whatever environment is reading this file does **not** have both of those
things, **stop immediately, before attempting any command in §3, and say so explicitly** — state
which precondition is missing (e.g. "no file at `~/.kaggle/kaggle.json`", "`kaggle datasets list`
returns a 401/auth error", "a request to `kaggle.com` times out / connection refused / DNS
fails") — and hand this back to whoever is running from the environment that actually has both.
Do not substitute a different action, do not simulate or fabricate a result, and do not mark any
step in §6 as complete when it was not actually executed against the real Kaggle API.

---

## 0. Scope and authorization

This runbook covers **exactly one bounded action**: deploy the reviewed source to the
`drbhatiasanjay/st-act-src` Kaggle dataset, push and run the `kaggle_kernel_learning_probe`
kernel, retrieve its outputs, and verify the report. It does **not** cover:

- Merging PR #7 (must already be merged to `master` before this runbook applies — see §1)
- Any 512-batch *training* run beyond the probe's own bounded 512 batches (the probe itself caps
  at 512 train batches / 2 validation samples / 3600s — that's the whole job, not a preamble to
  something bigger)
- Checkpoint promotion, `write_checkpoint_manifest()`, or any deployment-manifest generation
  (the probe's checkpoint is permanently `probe_only=True, sanity_only=True,
  deployment_eligible=False` and must stay that way)
- Any real competition submission
- Updating `GIT_SHA.txt` via `git commit` (it gets written to disk by the sync script as part of
  normal operation — see §3 — but that file change must NOT be committed to git)

If any of the above seems to be needed to make this "work," that is a sign something is wrong —
stop and report, don't broaden scope.

---

## 1. Preconditions — confirm before doing anything

1. **PR #7 must be merged to `master`.** Get the exact merge commit SHA:
   ```
   git fetch origin master
   git log --oneline -1 origin/master
   ```
   Confirm this merge commit's tree contains all five PR #7 files:
   `src/gpu_learning_probe.py`, `kaggle_src_dataset/src/gpu_learning_probe.py`,
   `kaggle_kernel_learning_probe/gpu_learning_probe_kernel.py`,
   `kaggle_kernel_learning_probe/kernel-metadata.json`,
   `tests/test_p10_gpu_learning_probe.py`. If PR #7 is not yet merged, **stop** — this runbook
   does not apply to an unmerged branch head. (The last verified PR #7 head, fully test-passing
   and adversarially reviewed, was `76d3bd6eb1b0ed20ebc4287fe69c0349d82fdd8a`; the merge commit
   will be a different SHA once merged — use the real merged SHA, not this one.)

2. **Kaggle CLI is installed and authenticated** (`kaggle --version`; `~/.kaggle/kaggle.json`
   present). If not, this runbook cannot proceed — do not attempt to authenticate with anything
   other than the user's own existing Kaggle credentials.

3. **Use a fresh, isolated git worktree**, not whatever branch/directory has uncommitted work in
   it:
   ```
   git worktree add ../st_act_pipeline_kaggle_exec <exact-merge-sha>
   cd ../st_act_pipeline_kaggle_exec
   git status --porcelain   # must be empty (ignore only tool-generated noise like .claude-flow/,
                             # .claude/.proven-config-version, .claude/proven-config.json — delete
                             # those first if present, they are not real changes)
   ```
   `scripts/sync_kaggle_src.py --push` (§3) **refuses to run on a dirty tree** — this is a
   deliberate fail-closed guard (see the script's own docstring/`working_tree_is_clean()`), not
   a bug to work around. Get to a genuinely clean tree before proceeding, don't bypass the check.

4. **Check for any other in-flight Kaggle trigger.** Only one path (this runbook run) should ever
   push/trigger a run against `drbhatiasanjay/st-act-gpu-learning-probe` at a time — a prior
   session's runbook for PR #6 hit real problems when two triggers raced. Confirm no one else
   (a human, another agent, a browser tab) is about to push the same kernel right now.

---

## 2. What you are deploying/running

| Item | Value |
|---|---|
| Kaggle dataset | `drbhatiasanjay/st-act-src` |
| Kaggle kernel | `drbhatiasanjay/st-act-gpu-learning-probe` |
| Kernel metadata file | `kaggle_kernel_learning_probe/kernel-metadata.json` |
| Kernel entrypoint | `kaggle_kernel_learning_probe/gpu_learning_probe_kernel.py` |
| GPU shape | `NvidiaTeslaT4` (already set in `kernel-metadata.json` — confirmed present, do not need to pass `--accelerator`) |
| Kernel output dir (inside Kaggle) | `/kaggle/working/gpu_learning_probe/` |
| Report file | `gpu_learning_probe_report.json` |
| Checkpoint file | `learning_probe_checkpoint.pt` (permanently non-deployable) |
| Bounded budget | 512 train batches, exactly 2 validation samples, 3600 seconds wall clock |

---

## 3. Step-by-step commands, in order

Run every command from the fresh worktree root (§1.3) unless noted. Redirect output to a real
file and check the exit code explicitly for every command — do not trust bare stdout in a
terminal, and do not trust a command "looked like it worked" without checking `$?`.

### 3.1 Sync and push the source dataset

```
py scripts/sync_kaggle_src.py --push -m "GPU learning probe source at <exact-merge-sha>"
```

Expected: prints `Current git SHA: <sha>`, `Synced src/ -> kaggle_src_dataset/src/`, `Synced 3
split file(s) -> kaggle_src_dataset/`, `Wrote kaggle_src_dataset\GIT_SHA.txt = <sha>`, `Verified:
kaggle_src_dataset/ matches src/ and the active split files exactly.`, then uploads 4 files
(`data_split.json`, `data_splits.zip`, `GIT_SHA.txt`, `src.zip`) and prints `Dataset version is
being created. Please check progress at https://www.kaggle.com/datasets/drbhatiasanjay/st-act-src`.
Exit code must be 0.

**Do not `git add`/`git commit` the `GIT_SHA.txt` change this writes to disk.** It's supposed to
exist on disk (that's how the sync verified), just not committed.

### 3.2 Confirm the dataset version actually finished processing

```
py -m kaggle datasets status drbhatiasanjay/st-act-src
```

Expected output: `ready`. If it says anything else (e.g. `error`), stop and report — do not
proceed to push the kernel against a dataset version that didn't finish.

### 3.3 Push the kernel

```
py -m kaggle kernels push -p kaggle_kernel_learning_probe
```

Expected: `Kernel version <N> successfully pushed.  Please check progress at
https://www.kaggle.com/code/drbhatiasanjay/st-act-gpu-learning-probe`. Exit code 0.

If this hits a persistent `409 Client Error: Conflict` that doesn't clear after closing tabs and
waiting ~90s: **do not keep blindly retrying.** This has happened before on this project and the
CLI-side fix was never found; the fallback is a manual browser paste-and-run, which is out of
scope for this runbook — stop and report the 409 instead of improvising a workaround.

### 3.4 Poll status until terminal (do not re-push while RUNNING)

Kaggle gives no live log streaming — polling `kernels status` is the only thing that tells you
anything mid-run, and it is legitimately expected to say `RUNNING` for a while. Do not interpret
a stale-looking status as a stuck run; do not push again while it says `RUNNING` or `QUEUED`.

```
py -m kaggle kernels status drbhatiasanjay/st-act-gpu-learning-probe
```

Poll this every ~60 seconds (a simple loop is fine) until it returns something other than
`RUNNING`/`QUEUED` (i.e. `COMPLETE` or `ERROR`). The probe's own internal budget is 3600 seconds;
allow real headroom beyond that for Kaggle queueing before treating a long run as suspicious.

### 3.5 Retrieve outputs and full logs once terminal

```
py -m kaggle kernels output drbhatiasanjay/st-act-gpu-learning-probe -p <fresh-output-dir>
```

Expect three files under `<fresh-output-dir>/gpu_learning_probe/`: `gpu_learning_probe_report.json`,
`learning_probe_checkpoint.pt`, `unused_training_log.csv`. (A trailing `'charmap' codec can't
encode characters...` message on Windows consoles is a known benign encoding quirk of the CLI's
own progress-bar text — it does not mean the files failed to download; confirm the three files
actually exist on disk before worrying about it.)

```
py scripts/kaggle_check_run.py drbhatiasanjay/st-act-gpu-learning-probe --save-log <fresh-log-path>
```

This prints a status summary (deployed SHA, GPU, first traceback if any, last batch progress) and
saves the full raw log as JSON to `<fresh-log-path>`. If for any reason this script's summary is
insufficient, the raw fallback is `kaggle kernels logs drbhatiasanjay/st-act-gpu-learning-probe`
(**not** `kernels output`, which only returns `/kaggle/working/` files, not the execution trace).
On Windows, if you need to parse the saved log JSON yourself in Python, set
`PYTHONIOENCODING=utf-8` first — the raw log contains non-ASCII characters that crash the default
Windows console codec otherwise.

---

## 4. Required PASS criteria — parse `gpu_learning_probe_report.json` against every one of these

Do not eyeball the report and declare success — check every field below explicitly. These are
taken directly from `evaluate_learning_probe_report()` in `src/gpu_learning_probe.py` at the
verified PR #7 head; if the report's own top-level `"verdict"` field already says `"PASS"`, that
means this function found zero failure reasons when run *inside* the kernel — but independently
re-check the fields yourself rather than trusting the verdict string alone, the same way Claude
did for PR #6 (cross-checked the report against the raw log, not just the report's own claims).

- `verdict` == `"PASS"` and `failure_reasons` == `[]`
- `schema_version` == `1`; `probe_name` == `"GPU-LEARNING-PROBE-01"`; `probe_scope` ==
  `"bounded_learning_signal_not_model_quality"`
- `deployed_sha` == the exact merge SHA from §1.1 (40-char lowercase hex)
- `probe_entrypoint_sha256` and `split_membership_sha256` present, 64-char lowercase hex
- `import_origins_verified` == `true`; `import_origins` non-empty, and every path resolves under
  `/kaggle/input/datasets/drbhatiasanjay/st-act-src/...` (proves the freshly-pushed dataset was
  actually what got imported, not some stale cached mount)
- `cuda_available` == `true`; `device_type` == `"cuda"`; `cuda_arch_compatible` == `true`;
  `gpu_name` is a non-empty string (e.g. `"Tesla T4"`)
- `requested_train_batches` == `completed_train_batches` == `512`
- `train_dataset_pair_count` >= `512`
- `average_train_loss`, `last_unet_gradient_norm`, `last_transformer_gradient_norm`: all finite
  and **strictly positive**
- `expected_train_sample_ids` and `successfully_opened_train_sample_ids` are the same set,
  non-empty, no duplicates (full strict-coverage train split)
- `requested_validation_samples` == `2`
- `selected_validation_sample_ids`: exactly 2 unique IDs
- `successfully_opened_validation_sample_ids`: **exactly equal as a set** to
  `selected_validation_sample_ids` — not merely a superset/subset. If this run ever shows any
  extra opened validation sample beyond the selected 2, that is the exact regression Codex's own
  remediation (head `76d3bd6`) fixed — treat it as a hard FAIL, not a rounding issue.
- `source_validation_fold_sample_count` >= `2` (this is the *original* full validation fold size,
  reported separately — expect it around 71 based on the current `data_split.json`, but don't
  hardcode 71, just check it's a sane positive integer >= 2)
- `full_fold_validation_performed` == `false` (explicit, must be exactly `false`, not absent)
- `validation_metrics` dict, all of:
  - `evaluation_completed_successfully` == `true`
  - `validation_samples_evaluated` == `2`
  - `validation_sample_cap` == `2`
  - `validation_samples_total` == `2` (bounded to the allowlisted subset, not the full fold — this
    is the post-remediation behavior; if you see `71` here instead of `2`, the fix regressed)
  - `validation_is_full_fold` == `true` (relative to the 2-sample bounded subset)
  - `predicted_nodes_total` — positive integer (zero here means a structural-zero execution
    failure, a different and more severe problem than a genuine zero quality score)
  - `predicted_edges_total` — non-negative integer
  - `is_structural_zero` == `false`
  - `edge_jaccard`, `adjusted_edge_jaccard`, `division_jaccard`, `score` — all finite and
    non-negative (exactly `0.0` is allowed and must NOT be treated as a failure by itself — a
    genuine zero quality score is a legitimate, honestly-reportable outcome per this probe's
    contract, as long as `predicted_nodes_total` is still positive)
- `learning_signal_observed` — must be present as a boolean. **Its value (true or false) does not
  by itself determine technical PASS/FAIL** — do not treat `learning_signal_observed: false` as a
  reason to fail this check. Report its actual value either way in your final write-up.
- `train_fallback_counts` and `post_validation_fallback_counts` — every key in both dicts must be
  exactly `0`
- `train_biological_counts.edge_supervised_batches_total` > 0 and
  `train_biological_counts.edge_supervised_batches_with_nonzero_transformer_grad` > 0
- `elapsed_seconds` finite, positive, and <= `time_budget_seconds` (3600.0)
- `training_elapsed_seconds` finite, positive, and <= `elapsed_seconds`
- `peak_gpu_memory_allocated_bytes` and `peak_gpu_memory_reserved_bytes` — both positive integers
- `deployment_manifest_generated` == `false`
- `probe_checkpoint_saved` == `true`; `probe_checkpoint_sha256` present, 64-char lowercase hex
  (the round-trip verification that produces this SHA is a hard code-level guarantee inside
  `_save_and_verify_probe_checkpoint()` — it reloads the checkpoint and raises `RuntimeError` if
  `probe_only`/`sanity_only`/`deployment_eligible` or provenance don't survive the round trip; if
  `probe_checkpoint_saved` is `true` and there's no such `RuntimeError` anywhere in the raw log,
  the round-trip passed)

Also scan the full raw log (not just the report) for `Traceback`, `ERROR`, or `FAIL` outside
known-benign noise (e.g. `skimage`/`dask` `FutureWarning`, `mistune`/`nbconvert`
`SyntaxWarning` during the notebook-viewer HTML conversion at the very end — these appeared in the
PR #6 run and are not real problems).

---

## 5. If anything fails

Stop. Do not retry with a bigger batch count, a different sample count, or any other parameter
change to "make it pass" — a genuine failure here means something is actually broken, not that
the bounds were too tight. Report:

1. Which numbered step in §3 you were on
2. The exact command and its exit code
3. The first complete traceback (from the raw log, not a truncated grep) or the exact report
   field(s) that violated §4
4. Do not broaden scope trying to fix it yourself in this runbook's execution — that requires a
   separate code-review/remediation pass, not more Kaggle runs.

---

## 6. If everything passes — final report format

Post (wherever this project's coordination thread lives — the PR, or wherever Codex normally
reports) something in this shape, filled in with real values, not placeholders:

```
[<YOUR_AGENT>_HANDOFF] master=<exact-merge-sha> kernel=drbhatiasanjay/st-act-gpu-learning-probe v<N> verdict=PASS status=<next-status>

Executed the Kaggle Execution Runbook (KAGGLE_EXECUTION_RUNBOOK_GPU_LEARNING_PROBE_2026-07-20.md) steps 1-6.

1. Dataset push: drbhatiasanjay/st-act-src -> ready
2. Kernel push: drbhatiasanjay/st-act-gpu-learning-probe v<N>
3. Poll: <N> checks, terminal status COMPLETE
4. Retrieved outputs + full log
5. Parsed gpu_learning_probe_report.json against every §4 criterion -- all satisfied:
   - verdict: PASS, deployed_sha matches
   - train: 512/512 batches, loss=<x>, unet_grad=<x>, transformer_grad=<x>
   - validation: 2/2 samples (<ids>), source_validation_fold_sample_count=<x>,
     full_fold_validation_performed=false, validation_samples_total=2,
     validation_is_full_fold=true, score=<x>, learning_signal_observed=<true/false>
   - fallback_counts: all zero (both train and post-validation)
   - CUDA: <gpu_name>, elapsed=<x>s / budget 3600s
   - probe_checkpoint_saved=true, round-trip verified (no RuntimeError in log)
   - no Traceback/ERROR outside known-benign warnings

No merge, checkpoint promotion, or submission performed. Outputs at <fresh-output-dir>, log at <fresh-log-path>.
```

Do not claim PASS unless every single §4 line item was actually checked against the real
downloaded report — not summarized from memory, not inferred from the kernel's own log line
saying "verdict: PASS" alone.
