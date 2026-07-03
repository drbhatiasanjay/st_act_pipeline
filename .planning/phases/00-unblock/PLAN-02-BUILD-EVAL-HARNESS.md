---
wave: 2
depends_on:
  - PLAN-00-VENDOR-SCORING-CODE
files_modified:
  - src/evaluation.py
autonomous: true
---

# Phase 0, Plan 02: Build Local Evaluation Harness

**Goal:** Implement a local evaluation module that computes the exact same score as Kaggle's official scorer using the vendored `metrics.py`/`division_metrics.py`.

**Rationale:** EVAL-01..04 in REQUIREMENTS.md. The local harness is critical for every submission — it validates that predictions are scoreable before spending a precious Kaggle submission slot. This plan integrates the vendored scoring code with a clean Python API for Plan 04 (wire pipeline) to call.

**Must-haves:**
- [ ] `src/evaluation.py` provides `evaluate_submission()` or similar function that takes predicted graph(s) and GT graph(s)
- [ ] Uses tracksdata's `evaluate()`/`evaluate_datasets()` for edge Jaccard (7.0 µm gated DistanceMatching)
- [ ] Uses vendored `division_metrics.evaluate_divisions()` for division Jaccard (real algorithm, not out-degree check)
- [ ] Computes adjusted edge Jaccard: `max(0, j · (1 - 0.1 · (T_pred - T_true) / T_true))`
- [ ] Returns combined score: `adjusted_edge_jaccard + 0.1 · division_jaccard`
- [ ] Handles zero-division GT case: drops division term entirely (not +0) when no GT divisions
- [ ] Unit tests on synthetic/real GT data (staged data has 4 samples with .geff annotations)
- [ ] Passes exact metric computation against a known baseline (e.g., a submission CSV hand-scored)

## Tasks

### Task 02-01: Implement Evaluation Module

```xml
<task id="02-01" title="Create src/evaluation.py with evaluate_submission() and helpers">
  <description>
    Implement a clean evaluation API:
    1. Function signature: evaluate_submission(pred_graphs, gt_graphs, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)
       CORRECTED 2026-07-03: original draft used (4.0,1.0,1.0), the anisotropy RATIO, as the
       default scale. That's wrong for a physical-distance gate -- it inflates every computed
       distance by ~2.46x (1/0.40625), corrupting the 7.0um matching threshold. The real physical
       voxel scale in micrometers is (1.625, 0.40625, 0.40625), confirmed against io.py's own
       DEFAULT_SCALE (vendored in Plan 00) and the .geff/zarr metadata read directly this session.
       - pred_graphs: list of tracksdata graphs (or dict of dataset_id -> graph)
       - gt_graphs: list of .geff-loaded ground-truth graphs (same structure)
       - Returns: dict with keys: edge_jaccard, adjusted_edge_jaccard, division_jaccard, score, num_pred_nodes_total
    
    2. Implement helper to load .geff files into tracksdata graphs (use tracksdata.graph.IndexedRXGraph.from_geff())
    3. Compute micro-averaged metrics using tracksdata.evaluate_datasets()
    4. Apply adjustment formula for edge Jaccard (ADJUSTMENT_ALPHA=0.1, uses T_true from .geff metadata)
    5. Handle division_jaccard:
       - If no GT divisions anywhere, division_jaccard = 0 and division term is dropped (score = adjusted_edge_jaccard)
       - Otherwise, include SCORE_DIVISION_WEIGHT=0.1 multiplier
    
    6. Robustness:
       - Validate that pred_graphs and gt_graphs are aligned (same dataset IDs/ordering)
       - Handle empty graph lists gracefully
       - Provide informative error messages (e.g., "GT graph has no nodes")
  </description>
  <files>
    <create>
      - src/evaluation.py
    </create>
  </files>
  <verification>
    - Module imports without errors: `from src.evaluation import evaluate_submission`
    - Function has docstring explaining parameters and return values
    - Constants match vendored metrics: ADJUSTMENT_ALPHA=0.1, SCORE_DIVISION_WEIGHT=0.1
    - No hardcoded paths or test-data assumptions
  </verification>
</task>
```

### Task 02-02: Implement .geff Loader and Graph Construction

```xml
<task id="02-02" title="Add helpers to load .geff annotations into tracksdata graphs">
  <description>
    Implement functions:
    1. load_geff_ground_truth(geff_path) -> (graph, metadata)
       - Uses tracksdata.graph.IndexedRXGraph.from_geff(geff_path)
       - Note: Returns a TUPLE (graph, GeffMetadata), not a bare graph
       - Extract T_true = metadata.estimated_number_of_nodes (used for adjustment)
    
    2. load_gt_for_dataset(dataset_id, geff_dir) -> graph
       - Locates and loads <dataset_id>.geff from the directory
       - Handles file-not-found gracefully (optional GT in sparse-annotation case)
    
    3. Validation:
       - Check that loaded graph has nodes and edges
       - Warn if graph is empty or suspiciously small
    
    These helpers go into src/evaluation.py or a dedicated io module used by evaluation.py
  </description>
  <files>
    <write>
      - src/evaluation.py (add load_geff_ground_truth, load_gt_for_dataset)
    </write>
  </files>
  <verification>
    - Functions exist and are callable
    - Return types are correct (tuple for from_geff, graph for load_gt_for_dataset)
    - No ModuleNotFoundError for tracksdata imports
    - Docstrings explain tuple-return behavior
  </verification>
</task>
```

### Task 02-03: Unit Tests on Staged GT Data

```xml
<task id="02-03" title="Write tests: evaluate_submission on staged .geff annotations">
  <description>
    Create tests/test_evaluation_harness.py with:
    1. Test: load_geff_ground_truth on real staged .geff file
       - Assert: returned graph has >= 1 nodes (real staged samples have 52 nodes)
       - Assert: metadata.estimated_number_of_nodes is a positive integer
    
    2. Test: evaluate_submission with identical pred/gt graphs
       - Provide a GT graph, then pass the same graph as prediction
       - Assert: edge_jaccard == 1.0 (perfect match)
       - Assert: division_jaccard in [0, 1]
       - Assert: score == 1.0 + 0.1*division_jaccard (or == 1.0 if no divisions)
    
    3. Test: evaluate_submission with empty prediction graph
       - Provide empty pred graph, real GT
       - Assert: edge_jaccard == 0.0
       - Assert: score >= 0
    
    4. Test: micro-averaging across multiple datasets
       - Provide 2+ samples with different metrics, verify counts are summed before ratio
       - Spot-check against hand calculation
    
    Use real staged data from data/staging/train/44b6_0113de3b.zarr + data/staging/train/44b6_0113de3b.geff (has 52-node .geff)
  </description>
  <files>
    <write>
      - tests/test_evaluation_harness.py
    </write>
  </files>
  <verification>
    - All tests pass: `python -m pytest tests/test_evaluation_harness.py -v`
    - No ModuleNotFoundError or import errors
    - Tests use real staged .geff files (not synthetic)
    - Test assertions are specific (e.g., == 1.0, not >= 0.9)
  </verification>
</task>
```

### Task 02-04: Sanity Check Against Reference Submission

```xml
<task id="02-04" title="Validate harness against a known baseline (optional but high-confidence)">
  <description>
    If a reference submission CSV or known-good predictions are available:
    1. Parse the reference submission into a graph
    2. Load the corresponding .geff ground truth
    3. Call evaluate_submission()
    4. Compare result against the published Kaggle leaderboard score (or a hand-calculated baseline)
    5. Assert: difference < 0.001 (due to floating-point rounding)
    
    If no reference submission is available, skip this task (tests above still provide confidence).
    This is a "nice-to-have" for extra validation, not a blocker.
  </description>
  <files>
    <write>
      - tests/test_evaluation_reference.py (if reference data is available)
    </write>
  </files>
  <verification>
    - If test exists: passes with published baseline
    - If no reference: document why (reference submission not yet obtained) — acceptable for Phase 0
  </verification>
</task>
```

## Verification Criteria

- [ ] `evaluate_submission()` exists and is callable
- [ ] Uses tracksdata.evaluate_datasets() for edges (7.0 µm matching)
- [ ] Uses vendored division_metrics.evaluate_divisions() for divisions
- [ ] Adjustment formula is correct: `max(0, j · (1 - 0.1 · (T_pred - T_true) / T_true))`
- [ ] Division term is dropped (not +0) when no GT divisions
- [ ] Unit tests pass on staged .geff data
- [ ] No hardcoded paths or dataset-specific assumptions

## Exit Criteria (Phase 0)

This plan is complete when:
1. Evaluation module is implemented (Task 02-01, 02-02)
2. Unit tests pass on real staged data (Task 02-03)
3. Plan 04 (wire pipeline) can import and call `evaluate_submission()` without modification
