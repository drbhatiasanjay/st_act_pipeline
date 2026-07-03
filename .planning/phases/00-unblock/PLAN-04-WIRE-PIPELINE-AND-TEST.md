---
wave: 3
depends_on:
  - PLAN-00-VENDOR-SCORING-CODE
  - PLAN-01-FIX-DATA-LOADER-CONFIG
  - PLAN-02-BUILD-EVAL-HARNESS
  - PLAN-03-IMPLEMENT-SUBMISSION-EXPORTER
files_modified:
  - run_pipeline.py
autonomous: false
---

# Phase 0, Plan 04: Wire Pipeline and End-to-End Test

**Goal:** Integrate the data loader, tracker, exporter, and evaluation harness into `run_pipeline.py`, then execute an end-to-end test on real staged data to generate a schema-valid, scoreable submission.

**Rationale:** This is the integration point. Plans 00-03 provide the building blocks; this plan orchestrates them and validates the complete flow. The exit criterion for Phase 0 requires a real submission CSV generated from staged data and scored locally using the exact competition metric.

**Must-haves:**
- [ ] `run_pipeline.py` reads ALL dataset folders in `data/staging/` (not hardcoded single path)
- [ ] For each dataset: load image data (via updated loader), run tracker, export node/edge rows
- [ ] Combine all dataset blocks into single submission CSV
- [ ] CSV is validated (schema-compliant per Plan 03)
- [ ] Evaluation harness loads corresponding .geff ground truth and scores the submission
- [ ] Score is reported: `adjusted_edge_jaccard`, `division_jaccard`, combined `score`
- [ ] All 4 staged datasets processed: 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1
- [ ] Submission is above classical baseline (0.763) — stretch goal, not hard requirement for Phase 0
- [ ] Submission file saved to `submissions/phase_0_baseline_submission.csv` (or configured output path)
- [ ] Execution time logged; ensure it runs in reasonable time (< 30 min on local hardware for Phase 0 scale)

## Tasks

### Task 04-01: Refactor run_pipeline.py for Multi-Dataset Iteration

```xml
<task id="04-01" title="Update run_pipeline.py to iterate over all datasets, keeping test (submission) and train (scoring) distinct">
  <description>
    **Real staged layout (confirmed, flat, not nested):**
    ```
    data/staging/train/{id}.zarr + data/staging/train/{id}.geff   (4 ids, HAS ground truth)
    data/staging/test/{id}.zarr                                    (SAME 4 ids, NO ground truth)
    ```
    The same 4 embryo IDs (44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1) appear in
    BOTH folders. This is intentional (train/ lets you validate against real GT; test/ mirrors what
    the actual hidden test set looks like). **Do not scan `data_dir` as one flat pool of 8 "datasets"
    -- that double-counts the same 4 embryos and would produce a submission with duplicate/wrong
    dataset blocks.**

    Refactor the orchestrator to keep these as two distinct runs sharing the same detect+track code:
    1. Add `--test-dir` (default: `data/staging/test/`) and `--train-dir` (default: `data/staging/train/`) arguments.
    2. `run_dataset(zarr_path, dataset_id)` helper: loads via AnisotropicZarrLoader, runs
       STHypergraphTracker.solve_lineage(), returns the output graph. Used by both passes below.
    3. **Submission pass** (real deliverable): glob `{test_dir}/*.zarr`, dataset_id = zarr folder's
       stem (e.g. `44b6_0113de3b.zarr` -> `44b6_0113de3b`), run `run_dataset` on each, collect into
       `{dataset_id: graph}`, pass to `submission_exporter.export_submission()`. This produces
       exactly 4 dataset blocks, matching the real test set.
    4. **Local scoring pass** (validation only, NOT part of the submission file): glob
       `{train_dir}/*.zarr`, run `run_dataset` on each (same detect+track code, different input),
       load the matching `{train_dir}/{dataset_id}.geff` via tracksdata, call
       `evaluate_submission()`/`evaluate_datasets()`. This is how Phase 0 proves the eval harness
       and the pipeline agree with each other -- it validates the SAME code path used for the
       submission, just pointed at data where we can actually check the answer.
    5. Return path to generated submission CSV plus the local scoring results (separately).

    Details:
    - Graceful error handling: log warnings for zarr stores that can't be loaded (continue to next)
    - Progress logging: print dataset_id, which pass (submission/scoring), and step (load, track, export/evaluate)
    - Do NOT modify tracker or loader logic -- only orchestration
  </description>
  <files>
    <read>
      - run_pipeline.py (current version)
      - src/data_loader.py (to understand loader interface)
      - src/tracker.py (to understand tracker interface)
      - data/staging/README.md (confirms the train/ vs test/ layout and why both exist)
    </read>
    <write>
      - run_pipeline.py (refactored for multi-dataset, two distinct passes)
    </write>
  </files>
  <verification>
    - Script runs without syntax errors: `python run_pipeline.py --help`
    - Accepts --test-dir and --train-dir arguments
    - Submission pass produces exactly 4 dataset blocks (from test/), not 8
    - Local scoring pass runs against train/ + matching .geff, separately from the submission file
    - Logs output at each step (dataset_id, pass, load, track, export/evaluate)
  </verification>
</task>
```

### Task 04-02: Integrate Submission Exporter into run_pipeline.py

```xml
<task id="04-02" title="Wire export_submission() call at end of pipeline">
  <description>
    At the end of run_pipeline():
    1. After all datasets are tracked, call export_submission(graphs_dict, output_path)
    2. Add --output-path argument (default: submissions/phase_0_baseline_submission.csv)
    3. Create output directory if it doesn't exist
    4. Ensure export_submission returns a DataFrame or path
    5. Call validate_submission() on the output CSV
    6. Log: "Submission CSV generated: <path>, <num_rows> rows, schema valid"
    7. If validation fails: raise exception with detailed error (don't proceed to evaluation)
  </description>
  <files>
    <write>
      - run_pipeline.py (add export_submission call and validation)
    </write>
  </files>
  <verification>
    - Import statement works: `from src.submission_exporter import export_submission, validate_submission`
    - Function is called before evaluation
    - Output CSV exists after run_pipeline completes
    - Validation passes (no exception)
  </verification>
</task>
```

### Task 04-03: Integrate Evaluation Harness into run_pipeline.py

```xml
<task id="04-03" title="Wire evaluate_submission() call for the local scoring pass (train/, not the submission)">
  <description>
    This scores the **local scoring pass** from Task 04-01 (run against `--train-dir`, which has
    matching `.geff` files) -- it does NOT score the submission CSV itself, since the real test/
    data has no ground truth (exactly like the actual hidden Kaggle test set).
    1. For each dataset_id processed from `--train-dir`, load `{train_dir}/{dataset_id}.geff` via
       tracksdata's `IndexedRXGraph.from_geff()` (remember: returns a `(graph, GeffMetadata)` tuple)
    2. Pair each predicted graph (from the train/ pass) with its GT graph
    3. Call evaluate_submission(pred_graphs_dict, gt_graphs_dict) -- vendored tracksdata-based harness from Plan 02
    4. Log results:
       - "Edge Jaccard: 0.XXX"
       - "Division Jaccard: 0.XXX"
       - "Adjusted Edge Jaccard: 0.XXX"
       - "Combined Score: 0.XXX"
    5. Compare score to baseline (0.763): "Above baseline: <yes/no>"
    6. Return score as main output (or as part of log) -- separate from the submission CSV path

    Note: If a `.geff` is unexpectedly missing for a train/ dataset_id, skip scoring for that one
    with a warning rather than failing the whole run.
  </description>
  <files>
    <write>
      - run_pipeline.py (add evaluate_submission call and results logging)
    </write>
  </files>
  <verification>
    - Import statement works: `from src.evaluation import evaluate_submission, load_geff_ground_truth`
    - Function is called after export and validation
    - Results are logged (check stdout or log file)
    - Script exits gracefully even if .geff is missing (with warning)
  </verification>
</task>
```

### Task 04-04: End-to-End Test on All Staged Datasets

```xml
<task id="04-04" title="Run run_pipeline.py on all 4 staged datasets and verify output">
  <description>
    Execute: python run_pipeline.py --test-dir data/staging/test/ --train-dir data/staging/train/ --output-path submissions/phase_0_test.csv

    Verify:
    1. Submission pass processes exactly the 4 test/ datasets (not 8, not train/ duplicates):
       - Logs show: "Processing 44b6_0113de3b", "Processing 44b6_0b24845f", "Processing 6bba_05b6850b", "Processing 6bba_05db0fb1" under the submission pass
    2. Submission CSV is created and has correct structure:
       - Read CSV, check header and row count (should be > 4, at least 1 node per dataset)
       - Exactly 4 distinct `dataset` values in the CSV -- not 8
       - Call validate_submission programmatically and assert True
    3. Local scoring pass (against train/) produces results:
       - Logs show 4 edge/division Jaccard scores (one per train/ embryo), 1 combined score
       - Combined score is numeric (not NaN, Inf)
    4. No exceptions or errors (script exits with code 0)
    5. Output files created:
       - submissions/phase_0_test.csv (submission, from test/ only)
       - Optional: submissions/phase_0_test_scores.json (local scoring results, from train/ only)
    
    If score is above 0.763: bonus validation — note this in test report
    If score is below 0.763: document why (e.g., placeholder detector is weak, expected for Phase 0)
  </description>
  <files>
    <write>
      - tests/test_e2e_pipeline.py (test that runs end-to-end and validates output)
    </write>
  </files>
  <verification>
    - Test runs without exception: `python -m pytest tests/test_e2e_pipeline.py::test_full_pipeline -v`
    - Submission CSV is created and readable
    - validate_submission() passes
    - Evaluation scores are reported and numeric
    - All 4 datasets are processed
  </verification>
</task>
```

### Task 04-05: Spot-Check Submission Against sample_submission.csv

```xml
<task id="04-05" title="Manual inspection: compare generated submission to sample structure">
  <description>
    Spot-check the generated submission CSV:
    1. Open submissions/phase_0_test.csv and sample_submission.csv (from Kaggle or data/staging/)
    2. Compare:
       - Header matches exactly
       - First 5 rows: check that ids are sequential, node_ids reset per dataset, coords are integers
       - Count of rows: should match expected number of nodes + edges across 4 datasets
       - Random row from middle and end: spot-check format
    3. Document findings in a brief report or as comments in the test file
    
    This is a manual sanity check to catch any subtle schema issues before submission
  </description>
  <files>
    <read>
      - submissions/phase_0_test.csv (generated in Task 04-04)
      - data/staging/sample_submission.csv (reference, if available)
    </read>
    <write>
      - Document comparison (comment in test file, or separate VERIFICATION.md file)
    </write>
  </files>
  <verification>
    - Spot-check completed and documented
    - No schema mismatches found (or noted and explained)
  </verification>
</task>
```

### Task 04-06: Measure Runtime and Identify Bottlenecks

```xml
<task id="04-06" title="Profile pipeline execution time and log bottlenecks">
  <description>
    During Task 04-04 execution:
    1. Add timing instrumentation to run_pipeline.py:
       - Time per dataset: load, track, export
       - Total time end-to-end
       - Log format: "Dataset 44b6_0113de3b: load=1.2s, track=5.3s, export=0.1s, total=6.6s"
    2. After execution, analyze:
       - Which step is slowest? (Likely: tracker, due to ILP solving)
       - Is total time < 30 minutes for 4 datasets? (Yes: Phase 0 passes, No: note for Phase 2/3)
    3. Log results: "Total pipeline time: 26.4s for 4 datasets"
    4. If tracker takes > 10s per dataset, note in report (ILP may be slow at scale, concern for Phase 3)
    
    This is informational only — Phase 0 success doesn't depend on timing (just functional correctness)
  </description>
  <files>
    <write>
      - run_pipeline.py (add timing instrumentation)
      - Document runtime report (stdout log or separate file)
    </write>
  </files>
  <verification>
    - Timing information is logged at each step
    - Total runtime is reasonable (< 5 minutes on typical hardware for 4 small staged datasets)
    - No timeout exceptions
  </verification>
</task>
```

## Verification Criteria

- [ ] All 4 staged datasets are processed (40+ nodes total, 30+ edges expected)
- [ ] Submission CSV is generated and schema-valid
- [ ] Evaluation harness produces 4 sets of Jaccard scores + 1 combined score
- [ ] Combined score is reported and numeric
- [ ] Script exits with status 0 (no exceptions)
- [ ] Output files are created: submission CSV + optional score report
- [ ] Spot-check against sample_submission.csv passes
- [ ] Runtime is < 30 minutes (likely < 5 minutes for staged data)

## Exit Criteria (Phase 0)

This plan is complete when:
1. `run_pipeline.py` is refactored and integrated (Tasks 04-01..04-03)
2. End-to-end test runs successfully on all 4 datasets (Task 04-04 passes)
3. Submission CSV is valid and scores are computed (Tasks 04-04, 04-05)
4. Spot-check against schema reference passes (Task 04-05)

## Phase 0 Exit Criterion (Overall)

**Phase 0 is successfully completed when:**
- [ ] Submission CSV `submissions/phase_0_baseline_submission.csv` is generated
- [ ] CSV has correct schema: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
- [ ] Separate node/edge rows, per-dataset node_id resets, integer coords
- [ ] Generated from real Zarr stores (data/staging/) + real .geff ground truth
- [ ] Scored locally via tracksdata + vendored metrics
- [ ] Combined score is reported: `adjusted_edge_jaccard + 0.1 · division_jaccard`
- [ ] All 4 staged datasets processed: 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1
- [ ] Score passes validation (no NaN/Inf, structurally valid per exact formula)
- [ ] Submission is ready for Kaggle (schema valid, field lengths/types correct, no spurious chars)

**Nice-to-have (not blocking, but valuable if achieved):**
- Score > 0.763 (classical baseline)
- Score is competitive (aim for top 50% of leaderboard, but Phase 0 is about *correctness*, not *competition*)
