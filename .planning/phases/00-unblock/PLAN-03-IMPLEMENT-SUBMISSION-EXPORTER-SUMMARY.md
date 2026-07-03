---
phase: 0
plan: 03
subsystem: submission-pipeline
tags:
  - csv-export
  - tracksdata
  - kaggle-schema
requires:
  - Phase 0, Plans 00-02 (evaluation harness, tracker)
provides:
  - Kaggle-compliant submission exporter
  - CSV schema validation
affects:
  - Phase 0, Plan 04 (pipeline wiring)
tech-stack:
  added:
    - pandas (CSV handling)
  patterns:
    - Dict-based row building for DataFrame creation
    - Per-dataset state reset for node_id mapping
    - Comprehensive validation with detailed error messages
file-tracking:
  key-files:
    created:
      - src/submission_exporter.py (288 lines)
      - src/submission_schema_comparison.py (106 lines)
      - tests/test_submission_exporter.py (464 lines)
  decisions:
    - Used pandas DataFrame for CSV construction and output
    - Implemented validate_submission() with 9 distinct checks
    - Used tracksdata.graph.edge_list() instead of iterating over graph.edges property
    - Node attributes accessed via dict-like interface (graph.nodes[node_id]['attr'])
  metrics:
    duration: 45 minutes
    completed: 2026-07-03
    tasks: 4/4
    commits: 4

# Phase 0 Plan 03: Implement Submission Exporter

## Summary

Successfully implemented `export_submission()` and `validate_submission()` functions that convert tracksdata graph objects to Kaggle-compliant submission CSVs. Schema is **exact match** with competition's sample_submission.csv.

## What Was Built

### 1. export_submission() Function
**File:** `src/submission_exporter.py`

- **Input:** Dictionary of dataset_id → tracksdata.graph.BaseGraph
- **Output:** CSV file with exact Kaggle schema

**Key Features:**
- Global id counter (0, 1, 2, ...) spanning all datasets
- Per-dataset node_id reset (1, 2, 3 per dataset) - CRITICAL REQUIREMENT
- Separate rows for nodes (coordinates + -1 for edge fields)
- Separate rows for edges (node_ids + -1 for coordinate fields)
- Integer coordinate conversion (no float decimals)
- Deterministic processing (sorted dataset iteration)

**Schema (exact):**
```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

### 2. validate_submission() Function
**File:** `src/submission_exporter.py`

Comprehensive schema validation with 9 checks:
1. File exists and readable
2. Header is exactly as expected
3. All rows have 10 columns (no missing values)
4. `id` column is globally sequential (0, 1, 2, ..., len-1)
5. `row_type` is either 'node' or 'edge'
6. For 'node' rows: source_id=-1, target_id=-1, coordinates integers
7. For 'edge' rows: node_id=-1, t/z/y/x all=-1, source_id/target_id positive integers
8. `node_id` resets per dataset (verified sequential per dataset)
9. No duplicate (dataset, node_id) pairs within node rows

**Raises:** ValueError with detailed message on any violation

### 3. Comprehensive Unit Tests
**File:** `tests/test_submission_exporter.py`

18 tests covering:
- Single node export
- Multiple nodes with edges
- Multiple datasets (node_id reset verification)
- Integer coordinate handling
- Validation with valid/invalid CSVs
- Edge cases: header mismatches, non-sequential ids, invalid row types
- Node row constraints (source/target=-1)
- Edge row constraints (node_id/coords=-1)
- Integration: export → validate cycle

**All 18 tests pass** with real tracksdata.IndexedRXGraph objects.

### 4. Schema Comparison
**File:** `src/submission_schema_comparison.py`

Documented verification against Kaggle's sample_submission.csv:
- Column order: Exact match
- Data types: All int64 except dataset (string)
- Row structure: Node and edge rows exactly as Kaggle expects
- Sample inspection: 20 rows across 4 datasets ✓

## Test Execution Results

```
python3 -m pytest tests/test_submission_exporter.py -v

============================= 18 passed in 8.05s ==============================

TestExportSubmission::test_export_single_node_no_edges PASSED
TestExportSubmission::test_export_nodes_and_edges PASSED
TestExportSubmission::test_export_multiple_datasets PASSED
TestExportSubmission::test_coordinates_are_integers PASSED
TestValidateSubmission::test_validate_valid_submission PASSED
TestValidateSubmission::test_validate_header_mismatch PASSED
TestValidateSubmission::test_validate_non_sequential_ids PASSED
TestValidateSubmission::test_validate_invalid_row_type PASSED
TestValidateSubmission::test_validate_node_row_source_not_minus_one PASSED
TestValidateSubmission::test_validate_node_row_target_not_minus_one PASSED
TestValidateSubmission::test_validate_edge_row_node_id_not_minus_one PASSED
TestValidateSubmission::test_validate_edge_row_coordinates_not_minus_one PASSED
TestValidateSubmission::test_validate_edge_row_source_negative PASSED
TestValidateSubmission::test_validate_node_id_per_dataset_reset PASSED
TestValidateSubmission::test_validate_no_duplicate_node_ids_per_dataset PASSED
TestValidateSubmission::test_validate_file_not_found PASSED
TestIntegrationExportAndValidate::test_export_and_validate_cycle PASSED
TestIntegrationExportAndValidate::test_export_multiple_datasets_and_validate PASSED
```

## Implementation Details

### Key API Discoveries (tracksdata)
- `graph.node_ids()` - Get list of node identifiers (integers)
- `graph.nodes[node_id]` - Access NodeInterface with dict-like [] access
- `graph.edge_list()` - Get [(source, target), ...] tuples (NOT graph.edges iteration)
- Node attributes: `graph.nodes[node_id]['t']`, `graph.nodes[node_id]['z']` etc.
- Edge attributes added with empty dict: `graph.add_edge(src, tgt, {})`

### Critical Schema Details
- **node_id per-dataset reset:** Not globally unique, resets to 1 for each new dataset
- **Global id:** Always sequential across entire file (0, 1, 2, ...)
- **Integer coordinates:** Must cast to int, no float decimals
- **-1 placeholders:** Used for "not applicable" in node/edge rows

## Files Modified/Created

### New Files
- `src/submission_exporter.py` - Main exporter (288 lines)
  - `export_submission(graphs_dict, output_path)` 
  - `validate_submission(csv_path)`
- `tests/test_submission_exporter.py` - Unit tests (464 lines, 18 tests)
- `src/submission_schema_comparison.py` - Schema documentation (106 lines)

## Deviations from Plan

### None

Plan executed exactly as specified. No bugs encountered, no critical functionality missing, no blocking issues.

## Verification Criteria - ALL MET

- ✅ `export_submission()` produces CSV with exact schema
- ✅ `id` is globally sequential (0, 1, 2, ...)
- ✅ `node_id` is per-dataset-LOCAL and resets for each new dataset
- ✅ Separate rows for nodes (source_id=-1, target_id=-1) and edges (node_id=-1, coords=-1)
- ✅ Coordinates are integers (no float rounding)
- ✅ Unit tests pass on synthetic graphs (18/18)
- ✅ Validation function catches schema violations
- ✅ Schema matches sample_submission.csv exactly

## Next Phase Readiness

**Plan 04 (Wire Pipeline) can now:**
- Call `export_submission(pred_graphs, output_path)` to generate submission CSVs
- Call `validate_submission(csv_path)` to verify before marking "submit-ready"
- Confidently pass exports to Kaggle without schema mismatch risk

**Confidence Level:** HIGH - Schema is verified against real sample, tests cover all requirements, API is stable.

## Exit Criteria Met

All exit criteria from PLAN.md satisfied:
1. ✅ Export and validation functions implemented
2. ✅ Unit tests pass on synthetic graphs  
3. ✅ Schema matches sample_submission.csv
4. ✅ Plan 04 can call these functions
