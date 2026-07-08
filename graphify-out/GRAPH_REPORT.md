# Graph Report - .  (2026-07-08)

## Corpus Check
- 131 files · ~58,841 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 803 nodes · 1316 edges · 47 communities (36 shown, 11 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 104 edges (avg confidence: 0.79)
- Token cost: 155,094 input · 6,388 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Evaluation Harness|Evaluation Harness]]
- [[_COMMUNITY_Submission Exporter|Submission Exporter]]
- [[_COMMUNITY_Vendored Scoring Code (tracking_cellmot)|Vendored Scoring Code (tracking_cellmot)]]
- [[_COMMUNITY_ILP Hypergraph Tracker|ILP Hypergraph Tracker]]
- [[_COMMUNITY_GEFF Metadata Schema (sample 1)|GEFF Metadata Schema (sample 1)]]
- [[_COMMUNITY_GEFF Metadata Schema (sample 2)|GEFF Metadata Schema (sample 2)]]
- [[_COMMUNITY_GEFF Metadata Schema (sample 3)|GEFF Metadata Schema (sample 3)]]
- [[_COMMUNITY_GEFF Metadata Schema (sample 4)|GEFF Metadata Schema (sample 4)]]
- [[_COMMUNITY_Pipeline Orchestration & Threshold Sweep|Pipeline Orchestration & Threshold Sweep]]
- [[_COMMUNITY_Zarr Store Metadata (test set)|Zarr Store Metadata (test set)]]
- [[_COMMUNITY_Zarr ChunkCodec Configuration|Zarr Chunk/Codec Configuration]]
- [[_COMMUNITY_Zarr Store Metadata (train set)|Zarr Store Metadata (train set)]]
- [[_COMMUNITY_Vendored IO & Dataset Loading|Vendored I/O & Dataset Loading]]
- [[_COMMUNITY_Image Statistics & Quantiles (sample 1)|Image Statistics & Quantiles (sample 1)]]
- [[_COMMUNITY_Image Statistics & Quantiles (sample 2)|Image Statistics & Quantiles (sample 2)]]
- [[_COMMUNITY_Image Statistics & Quantiles (sample 3)|Image Statistics & Quantiles (sample 3)]]
- [[_COMMUNITY_Image Statistics & Quantiles (sample 4)|Image Statistics & Quantiles (sample 4)]]
- [[_COMMUNITY_Phase 0 Integration Summary|Phase 0 Integration Summary]]
- [[_COMMUNITY_Real Data Loader Tests|Real Data Loader Tests]]
- [[_COMMUNITY_Zarr Store Metadata|Zarr Store Metadata]]
- [[_COMMUNITY_Detection Model (CNN)|Detection Model (CNN)]]
- [[_COMMUNITY_Zarr Data Loader|Zarr Data Loader]]
- [[_COMMUNITY_GSD Project Config|GSD Project Config]]
- [[_COMMUNITY_Simulated Data Fallback|Simulated Data Fallback]]
- [[_COMMUNITY_Zarr Store Metadata (per-sample a)|Zarr Store Metadata (per-sample a)]]
- [[_COMMUNITY_Zarr Store Metadata (per-sample b)|Zarr Store Metadata (per-sample b)]]
- [[_COMMUNITY_Zarr Store Metadata (per-sample c)|Zarr Store Metadata (per-sample c)]]
- [[_COMMUNITY_Zarr Store Metadata (per-sample d)|Zarr Store Metadata (per-sample d)]]
- [[_COMMUNITY_Zarr Store Metadata (per-sample e)|Zarr Store Metadata (per-sample e)]]
- [[_COMMUNITY_Zarr Store Metadata (per-sample f)|Zarr Store Metadata (per-sample f)]]
- [[_COMMUNITY_Run Comparison Script|Run Comparison Script]]
- [[_COMMUNITY_Peak-Finding & Host Reference Notes|Peak-Finding & Host Reference Notes]]
- [[_COMMUNITY_End-to-End Pipeline Tests|End-to-End Pipeline Tests]]
- [[_COMMUNITY_Submission Spot-Check Tests|Submission Spot-Check Tests]]
- [[_COMMUNITY_System Diagnostics Script|System Diagnostics Script]]
- [[_COMMUNITY_Submission Schema Comparison|Submission Schema Comparison]]
- [[_COMMUNITY_Quantile Normalization Logic|Quantile Normalization Logic]]
- [[_COMMUNITY_Submission Coordinate Tests|Submission Coordinate Tests]]
- [[_COMMUNITY_Simulate-False Default Test|Simulate-False Default Test]]
- [[_COMMUNITY_Quantile Params Test|Quantile Params Test]]
- [[_COMMUNITY_Multi-Timepoint Load Test|Multi-Timepoint Load Test]]
- [[_COMMUNITY_Nonexistent Path Test|Nonexistent Path Test]]
- [[_COMMUNITY_Normalization Range Test|Normalization Range Test]]
- [[_COMMUNITY_Train Data Path Constant|Train Data Path Constant]]
- [[_COMMUNITY_Test Data Path Constant|Test Data Path Constant]]
- [[_COMMUNITY_Temp CSV Dir Fixture (a)|Temp CSV Dir Fixture (a)]]
- [[_COMMUNITY_Temp CSV Dir Fixture (b)|Temp CSV Dir Fixture (b)]]

## God Nodes (most connected - your core abstractions)
1. `AnisotropicZarrLoader` - 27 edges
2. `STHypergraphTracker` - 25 edges
3. `load_geff_ground_truth()` - 19 edges
4. `validate_submission()` - 19 edges
5. `chunk_grid` - 18 edges
6. `chunk_key_encoding` - 18 edges
7. `shape` - 16 edges
8. `data_type` - 16 edges
9. `fill_value` - 16 edges
10. `codecs` - 16 edges

## Surprising Connections (you probably didn't know these)
- `main` --implements--> `PRD: ST-ACT`  [INFERRED]
  run_pipeline.py → PRD.md
- `run_dataset()` --calls--> `load_dataset_checkpoint()`  [INFERRED]
  run_pipeline.py → src/run_tracker.py
- `run_dataset()` --calls--> `AnisotropicZarrLoader`  [INFERRED]
  run_pipeline.py → src/data_loader.py
- `run_dataset()` --calls--> `load_detection_cache()`  [INFERRED]
  run_pipeline.py → src/run_tracker.py
- `run_dataset()` --calls--> `save_detection_cache()`  [INFERRED]
  run_pipeline.py → src/run_tracker.py

## Communities (47 total, 11 thin omitted)

### Community 0 - "Evaluation Harness"
Cohesion: 0.05
Nodes (49): PRD: ST-ACT, main, evaluate_submission(), load_geff_ground_truth(), load_gt_for_dataset(), Local evaluation harness for the Kaggle cell tracking competition.  Provides a, Evaluate predicted tracking graphs against ground-truth graphs.      Computes, Load a .geff ground-truth file into a tracksdata graph.      Parameters     - (+41 more)

### Community 1 - "Submission Exporter"
Cohesion: 0.06
Nodes (31): Submission exporter for the Kaggle cell tracking competition.  Provides functi, Validate a submission CSV against the schema.      Parameters     ----------, validate_submission(), Unit tests for submission exporter.  Tests export_submission() and validate_subm, Test export with 3 nodes and 2 edges., Test export with multiple datasets (node_id reset per dataset)., Test that exported coordinates are integers (no floats)., Tests for export_submission() function. (+23 more)

### Community 2 - "Vendored Scoring Code (tracking_cellmot)"
Cohesion: 0.05
Nodes (50): NamedTuple, Baseline tests for the vendored scoring code (tracking_cellmot).  Verifies that:, Empty-dataset edge case: evaluate_datasets([]) should return a DatasetsResult, test_evaluate_datasets_empty_list(), _bipartite_max_matching(), count_matched_pred_divisions(), DivisionCounts, evaluate_divisions() (+42 more)

### Community 3 - "ILP Hypergraph Tracker"
Cohesion: 0.06
Nodes (29): Mitosis Backward-Smoothing (Temporal Window Align):         Backtracks division, Anisotropic Velocity Edge Pruning: Inspects coordinates and discards         co, Constructs and solves ILP for cell centroids.         Supports multi-frame look, Spatio-Temporal Hypergraph Lineage Solver (Grandmaster Tier).     Models tracki, STHypergraphTracker, create_small_test_zarr(), Integration test for the Phase 0 pipeline.  This test verifies that: 1. The pipe, Create a small test zarr store (3 timepoints, small volumes). (+21 more)

### Community 4 - "GEFF Metadata Schema (sample 1)"
Cohesion: 0.04
Nodes (44): attributes, geff, estimated_number_of_nodes, axes, directed, display_hints, edge_props_metadata, ellipsoid (+36 more)

### Community 5 - "GEFF Metadata Schema (sample 2)"
Cohesion: 0.04
Nodes (44): attributes, geff, estimated_number_of_nodes, axes, directed, display_hints, edge_props_metadata, ellipsoid (+36 more)

### Community 6 - "GEFF Metadata Schema (sample 3)"
Cohesion: 0.04
Nodes (44): attributes, geff, estimated_number_of_nodes, axes, directed, display_hints, edge_props_metadata, ellipsoid (+36 more)

### Community 7 - "GEFF Metadata Schema (sample 4)"
Cohesion: 0.04
Nodes (44): attributes, geff, estimated_number_of_nodes, axes, directed, display_hints, edge_props_metadata, ellipsoid (+36 more)

### Community 8 - "Pipeline Orchestration & Threshold Sweep"
Cohesion: 0.07
Nodes (30): main(), Phase 1 threshold-calibration sweep for the new max_pool3d/maximum_filter-based, dataset_checkpoint_key(), detection_cache_key(), _format_duration(), _git_commit_hash(), load_dataset_checkpoint(), load_detection_cache() (+22 more)

### Community 9 - "Zarr Store Metadata (test set)"
Cohesion: 0.33
Nodes (16): attributes, chunk_grid, configuration, name, chunk_key_encoding, configuration, name, codecs (+8 more)

### Community 10 - "Zarr Chunk/Codec Configuration"
Cohesion: 0.31
Nodes (16): attributes, chunk_grid, configuration, name, chunk_key_encoding, configuration, name, codecs (+8 more)

### Community 11 - "Zarr Store Metadata (train set)"
Cohesion: 0.31
Nodes (16): attributes, chunk_grid, configuration, name, chunk_key_encoding, configuration, name, codecs (+8 more)

### Community 12 - "Vendored I/O & Dataset Loading"
Cohesion: 0.12
Nodes (18): Dataset, invert_time_graph(), list_datasets(), _lookup_precomputed_quantile(), open_dataset(), _parse_scale(), _process_on_gpu(), I/O utilities for tracking challenge datasets. (+10 more)

### Community 13 - "Image Statistics & Quantiles (sample 1)"
Cohesion: 0.14
Nodes (15): attributes, image_statistics, multiscales, consolidated_metadata, quantiles, node_type, 0.0, 0.001 (+7 more)

### Community 14 - "Image Statistics & Quantiles (sample 2)"
Cohesion: 0.14
Nodes (15): attributes, image_statistics, multiscales, consolidated_metadata, quantiles, node_type, 0.0, 0.001 (+7 more)

### Community 15 - "Image Statistics & Quantiles (sample 3)"
Cohesion: 0.14
Nodes (15): attributes, image_statistics, multiscales, consolidated_metadata, quantiles, node_type, 0.0, 0.001 (+7 more)

### Community 16 - "Image Statistics & Quantiles (sample 4)"
Cohesion: 0.14
Nodes (15): attributes, image_statistics, multiscales, consolidated_metadata, quantiles, node_type, 0.0, 0.001 (+7 more)

### Community 17 - "Phase 0 Integration Summary"
Cohesion: 0.13
Nodes (16): AnisotropicZarrLoader, evaluate_submission, load_geff_ground_truth, Phase 0 Integration Rationale, RunTracker, export_submission, validate_submission, Run System Diagnostics (+8 more)

### Community 18 - "Real Data Loader Tests"
Cohesion: 0.14
Nodes (7): Unit tests for AnisotropicZarrLoader with real staged data. Tests validate corre, Test AnisotropicZarrLoader with real staged Zarr v3 data., Test that loader correctly initializes with real Zarr v3 OME-NGFF store., Test that loaded data has correct 4D shape (T, Z, Y, X)., Test that raw real data is uint16 as expected., Test that default anisotropy is correctly set to (4.0, 1.0, 1.0)., TestAnisotropicZarrLoaderReal

### Community 19 - "Zarr Store Metadata"
Cohesion: 0.48
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 20 - "Detection Model (CNN)"
Cohesion: 0.22
Nodes (6): AnisotropicCoordinateTransformer, Args:             voxel_coords (torch.Tensor): Shape (B, N, 3) representing (Z,, Fully Convolutional 3D Network that inputs anisotropic timepoint blocks     and, Args:             x (torch.Tensor): Volume tensor of shape (B, 1, Z, Y, X), Natively maps 3D coordinate tensors from anisotropic voxel space (Z, Y, X), STACTCentroidPredictor

### Community 21 - "Zarr Data Loader"
Cohesion: 0.18
Nodes (6): AnisotropicZarrLoader, Returns the (T, Z, Y, X) dimensions of the 4D dataset.          Returns:, ST-ACT Memory-Safe 4D Anisotropic Zarr v3 Ingestor.     Loads and decompresses, A memory-efficient generator yielding spatial sub-chunks (Z, Y, X) of a single t, Test that first timepoint can be loaded without errors., Test that quantile normalization is correctly applied when available.

### Community 22 - "GSD Project Config"
Cohesion: 0.20
Nodes (9): commit_docs, depth, mode, model_profile, parallelization, workflow, plan_check, research (+1 more)

### Community 23 - "Simulated Data Fallback"
Cohesion: 0.20
Nodes (5): Generates a mock Zarr store mimicking a 4D anisotropic microscopy volume., Initialize the Anisotropic Zarr Loader.          Args:             store_path, Helper method to render a Gaussian cellular signal in anisotropic space., Initializes connection to the Zarr v3 store.         For real competition data:, Extract quantile normalization parameters from Zarr metadata.         Looks for

### Community 24 - "Zarr Store Metadata (per-sample a)"
Cohesion: 0.57
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 25 - "Zarr Store Metadata (per-sample b)"
Cohesion: 0.57
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 26 - "Zarr Store Metadata (per-sample c)"
Cohesion: 0.57
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 27 - "Zarr Store Metadata (per-sample d)"
Cohesion: 0.57
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 28 - "Zarr Store Metadata (per-sample e)"
Cohesion: 0.57
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 29 - "Zarr Store Metadata (per-sample f)"
Cohesion: 0.57
Nodes (4): attributes, consolidated_metadata, node_type, zarr_format

### Community 30 - "Run Comparison Script"
Cohesion: 0.52
Nodes (6): load_run(), main(), print_run_detail(), print_summary_table(), Compare pipeline run statistics across logs/runs/*.jsonl.  Usage:     py scripts, summarize()

### Community 31 - "Peak-Finding & Host Reference Notes"
Cohesion: 0.33
Nodes (6): Host Reference Implementation, convert_nx_to_tracksdata, ensemble_consensus_centroids, extract_peaks_from_volume, pool_kernel_from_um, run_dataset

### Community 32 - "End-to-End Pipeline Tests"
Cohesion: 0.33
Nodes (5): End-to-end pipeline test for Phase 0.  Tests: 1. Pipeline runs without exception, Test the CSV file structure independently., Test the complete pipeline: detect, track, export, evaluate., test_full_pipeline(), test_submission_csv_structure()

### Community 33 - "Submission Spot-Check Tests"
Cohesion: 0.33
Nodes (5): Task 04-05: Spot-check generated submission against sample_submission.csv  Verif, Verify submission CSV schema matches sample_submission.csv, Verify coordinate values in sample are reasonable, test_submission_coordinates_are_valid(), test_submission_schema_matches_sample()

### Community 34 - "System Diagnostics Script"
Cohesion: 0.50
Nodes (4): Colors, get_ram_info(), Retrieves system RAM size in GB. Uses psutil if installed,     falls back to Wi, run_diagnostics()

### Community 35 - "Submission Schema Comparison"
Cohesion: 0.50
Nodes (3): compare_with_sample_submission(), Schema comparison: export_submission output vs Kaggle's sample_submission.csv, Compare exported submission CSV with Kaggle's sample_submission.csv.      Para

## Knowledge Gaps
- **216 isolated node(s):** `Colors`, `mode`, `depth`, `parallelization`, `commit_docs` (+211 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `evaluate_submission()` connect `Evaluation Harness` to `Vendored Scoring Code (tracking_cellmot)`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `evaluate_datasets()` connect `Vendored Scoring Code (tracking_cellmot)` to `Evaluation Harness`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `main()` connect `Evaluation Harness` to `Pipeline Orchestration & Threshold Sweep`, `Submission Exporter`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `AnisotropicZarrLoader` (e.g. with `TestAnisotropicZarrLoaderReal` and `run_dataset()`) actually correct?**
  _`AnisotropicZarrLoader` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `STHypergraphTracker` (e.g. with `TestPositiveLinking` and `TestNegativeCasesAndRegressions`) actually correct?**
  _`STHypergraphTracker` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `load_geff_ground_truth()` (e.g. with `main()` and `test_load_geff_real_staged_file()`) actually correct?**
  _`load_geff_ground_truth()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `validate_submission()` (e.g. with `main()` and `test_pipeline_structure()`) actually correct?**
  _`validate_submission()` has 16 INFERRED edges - model-reasoned connections that need verification._