---
wave: 2
depends_on: []
files_modified:
  - src/submission_exporter.py
autonomous: true
---

# Phase 0, Plan 03: Implement Submission Exporter

**Goal:** Create a submission exporter that generates Kaggle-compliant submission CSVs with the exact schema `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`, with per-dataset node_id resets.

**Rationale:** SUB-01, SUB-02, SUB-03 in REQUIREMENTS.md. The schema is non-obvious:
- `id` is globally sequential (0, 1, 2, ...)
- `node_id` is per-dataset-LOCAL and resets to 1 for each new dataset
- Separate `node` and `edge` rows
- Integer voxel coordinates (no floats)
- `-1` for missing `source_id`/`target_id` in node rows

**Must-haves:**
- [ ] `src/submission_exporter.py` provides `export_submission(graphs_dict, output_path)` function
- [ ] Schema is exact: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
- [ ] `id` is globally sequential across the entire file (0, 1, 2, ...)
- [ ] `node_id` is per-dataset-LOCAL and resets to 1 at each new dataset block
- [ ] Separate row for each node (`row_type='node'`, `source_id=-1`, `target_id=-1`)
- [ ] Separate row for each edge (`row_type='edge'`, `node_id=-1`, coordinates all `-1`)
- [ ] Coordinates are integers (no float rounding)
- [ ] CSV is valid: matches `sample_submission.csv` schema line-by-line
- [ ] Unit test on synthetic graph produces valid submission structure
- [ ] Validation function checks CSV against schema before declaring "submit-ready"

## Tasks

### Task 03-01: Implement export_submission() Function

```xml
<task id="03-01" title="Create src/submission_exporter.py with export_submission()">
  <description>
    Implement function: export_submission(graphs_dict, output_path)
    where graphs_dict = {dataset_id: tracksdata_graph, ...}
    
    Algorithm:
    1. Initialize global_id = 0 and list rows = []
    2. Iterate over datasets in sorted order (consistent ordering)
    3. For each dataset:
       a. Reset per-dataset node_id_map = {} and per_dataset_node_id = 0
       b. Iterate over nodes in the graph:
          - Increment per_dataset_node_id
          - Store node_id_map[node] = per_dataset_node_id
          - Create row: [global_id, dataset_id, 'node', per_dataset_node_id, t, z, y, x, -1, -1]
          - Increment global_id
       c. Iterate over edges in the graph:
          - Get source_node_id = node_id_map[source], target_node_id = node_id_map[target]
          - Create row: [global_id, dataset_id, 'edge', -1, -1, -1, -1, -1, source_node_id, target_node_id]
          - Increment global_id
    
    4. Convert rows to DataFrame, save as CSV with header
    5. Return path (or submission DataFrame)
    
    Details:
    - Node coordinates must be integers (cast if necessary)
    - Time/Z/Y/X from tracksdata graph node attributes (node.t, node.z, node.y, node.x or equivalent)
    - Edges come from graph.edges() or graph.edge_list() depending on tracksdata API
    - Error handling: validate that all node_ids in edges are found in node_id_map (no dangling edges)
  </description>
  <files>
    <create>
      - src/submission_exporter.py
    </create>
  </files>
  <verification>
    - Module imports without errors: `from src.submission_exporter import export_submission`
    - Function has correct signature and docstring
    - Handles graphs_dict parameter correctly
    - Outputs CSV with required columns
  </verification>
</task>
```

### Task 03-02: Implement Validation Function

```xml
<task id="03-02" title="Add validate_submission() to check schema compliance">
  <description>
    Implement: validate_submission(csv_path) -> bool
    
    Checks:
    1. File exists and is readable
    2. Header is exactly: id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
    3. All rows have 10 columns (no missing values, correct CSV format)
    4. `id` column is globally sequential (0, 1, 2, ..., len(df)-1)
    5. `row_type` is either 'node' or 'edge'
    6. For 'node' rows: source_id=-1, target_id=-1, coordinates are integers
    7. For 'edge' rows: node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id/target_id are positive integers
    8. `node_id` resets per dataset (e.g., goes 1, 2, 3 for dataset A, then 1, 2 for dataset B)
    9. No duplicate (dataset, node_id) pairs within node rows (uniqueness within dataset)
    
    Return: True if all checks pass, raise ValueError with detailed message otherwise
    
    This function is used by Plan 04 before marking a submission as "submit-ready"
  </description>
  <files>
    <write>
      - src/submission_exporter.py (add validate_submission)
    </write>
  </files>
  <verification>
    - Function is callable: `from src.submission_exporter import validate_submission`
    - Error messages are informative (e.g., "id column not sequential at row 5")
    - Docstring explains all checks
  </verification>
</task>
```

### Task 03-03: Unit Tests on Synthetic Graph

```xml
<task id="03-03" title="Write tests: export_submission on synthetic graphs">
  <description>
    Create tests/test_submission_exporter.py with:
    
    1. Test: export single node, no edges
       - Input: synthetic graph with 1 node at (t=0, z=5, y=10, x=15)
       - Expected output: 1 row with id=0, row_type='node', node_id=1, coords match, source_id=-1, target_id=-1
    
    2. Test: export nodes and edges
       - Input: synthetic graph with 3 nodes, 2 edges (node1->node2, node2->node3)
       - Expected: 5 rows (3 nodes + 2 edges)
       - ids are 0..4
       - node_ids are 1, 2, 3
       - edges have source_id/target_id pointing to correct node_ids
    
    3. Test: export multiple datasets
       - Input: graphs_dict = {'dataset_A': graph_a, 'dataset_B': graph_b}
       - Expected:
         - dataset_A rows have dataset='dataset_A', node_id resets (1, 2, ...)
         - dataset_B rows have dataset='dataset_B', node_id resets again (1, 2, ...)
         - global id is continuous across both datasets
    
    4. Test: validate_submission on exported CSV
       - Export a valid submission, then call validate_submission(path)
       - Assert: returns True (or doesn't raise)
       - Modify CSV (break a constraint), re-validate
       - Assert: raises ValueError with informative message
    
    5. Test: coordinates are integers
       - Export a graph with float coordinates (synthetic)
       - Assert: CSV has integer values (no .0 decimals)
    
    Use synthetic tracksdata graphs (or mock objects that mimic the API)
  </description>
  <files>
    <write>
      - tests/test_submission_exporter.py
    </write>
  </files>
  <verification>
    - All tests pass: `python -m pytest tests/test_submission_exporter.py -v`
    - No import errors
    - Tests are deterministic (not relying on floating-point order)
    - Tests create temporary CSV files (cleanup after)
  </verification>
</task>
```

### Task 03-04: Comparison Against sample_submission.csv

```xml
<task id="03-04" title="Validate output schema against competition's sample_submission.csv">
  <description>
    Manually inspect or programmatically verify:
    1. Read the competition's sample_submission.csv (from Kaggle or data/staging/)
    2. Extract header and a few sample rows
    3. Verify that export_submission output has:
       - Identical column names and order
       - Identical data types for each column (id=int, dataset=str, row_type=str, etc.)
       - Identical range for id (0 to num_rows-1)
       - Identical structure for node/edge rows
    
    Document any differences found and update export_submission if needed
    (This is a spot-check to ensure we're not missing subtle schema requirements)
  </description>
  <files>
    <read>
      - data/staging/sample_submission.csv (if available, or from Kaggle)
    </read>
    <write>
      - Document comparison result in test or as a comment in src/submission_exporter.py
    </write>
  </files>
  <verification>
    - Comparison document/test exists
    - No schema mismatches found (or noted and explained)
  </verification>
</task>
```

## Verification Criteria

- [ ] `export_submission()` produces CSV with exact schema
- [ ] `id` is globally sequential (0, 1, 2, ...)
- [ ] `node_id` is per-dataset-LOCAL and resets for each new dataset
- [ ] Separate rows for nodes (source_id=-1, target_id=-1) and edges (node_id=-1, coords=-1)
- [ ] Coordinates are integers
- [ ] Unit tests pass on synthetic graphs
- [ ] Validation function catches schema violations

## Exit Criteria (Phase 0)

This plan is complete when:
1. Export and validation functions are implemented (Tasks 03-01, 03-02)
2. Unit tests pass (Task 03-03)
3. Schema matches sample_submission.csv (Task 03-04)
4. Plan 04 (wire pipeline) can call `export_submission()` and use `validate_submission()` for final checks
