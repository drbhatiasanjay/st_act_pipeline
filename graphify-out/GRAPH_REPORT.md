# Graph Report - st_act_pipeline  (2026-08-14)

## Corpus Check
- 50 files · ~294,479 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2554 nodes · 5229 edges · 161 communities (125 shown, 36 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 326 edges (avg confidence: 0.52)
- Token cost: 164,116 input · 6,705 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Community 119
- Community 120
- Community 121
- Community 122
- Community 123
- Community 124
- Community 125
- Community 126
- Community 127
- Community 128
- Community 129
- Community 130
- Community 131
- Community 132
- Community 133
- Community 134
- Community 135
- Community 136
- Community 137
- Community 138
- Community 139
- Community 140
- Community 141
- Community 142
- Community 143
- Community 144
- Community 145
- Community 147
- Community 148
- Community 149
- Community 150
- Community 151
- Community 152
- Community 154
- Community 155
- Community 156
- Community 158
- Community 159
- Community 160

## God Nodes (most connected - your core abstractions)
1. `TrainingLoop` - 100 edges
2. `_make_small_models()` - 58 edges
3. `CompetitionDataset` - 56 edges
4. `load_verified_checkpoint()` - 52 edges
5. `AnisotropicZarrLoader` - 51 edges
6. `validate_submission()` - 47 edges
7. `evaluate_submission()` - 47 edges
8. `load_and_validate_split()` - 47 edges
9. `CompetitionDataset` - 46 edges
10. `UNet3D` - 45 edges

## Surprising Connections (you probably didn't know these)
- `HSOM Self-Learning and Meta-Learning` --references--> `Oracle Decomposition Runner`  [INFERRED]
  META_LEARNING.md → scripts/oracle_decomposition.py
- `TestF5CheckpointWritingAndEligibility` --uses--> `SimpleNodeTransformer`  [INFERRED]
  tests/test_p06_submission_deployment.py → kaggle_src_dataset/src/model.py
- `TestF5CheckpointWritingAndEligibility` --uses--> `UNet3D`  [INFERRED]
  tests/test_p06_submission_deployment.py → kaggle_src_dataset/src/model.py
- `TestF5CheckpointWritingAndEligibility` --uses--> `PredictionGraphAssembler`  [INFERRED]
  tests/test_p06_submission_deployment.py → kaggle_src_dataset/src/prediction_graph.py
- `TestF5CheckpointWritingAndEligibility` --uses--> `TrainingLoop`  [INFERRED]
  tests/test_p06_submission_deployment.py → kaggle_src_dataset/src/train.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **HSOM Specification v1.0** — hsom_prd_package_readme, hsom_north_star_constitution, hsom_prd_document, hsom_architecture_spec, hsom_experiment_protocol, hsom_hypothesis_registry [EXTRACTED 1.00]
- **Oracle Evaluation Flow** — scripts_oracle_decomposition_script, src_oracle_evaluation, tests_test_oracle_evaluation [EXTRACTED 1.00]
- **HSOM Meta-Learning Principles** — meta_learning, scripts_diagnose_detection_loss_recipe, scripts_oracle_decomposition_script [INFERRED 0.90]
- **Architecture Audit & Gap Analysis** — gap_analysis_codex_md, src_model_unet3d, src_model_simplenodetransformer [EXTRACTED 0.90]
- **Evaluation & Scoring Flow** — src_tracking_cellmot_metrics, src_evaluation_evaluate_submission, src_submission_exporter_export_submission [EXTRACTED 0.90]
- **Kaggle Deployment Integrity Pattern** — src_deployment_provenance, src_checkpoint_manifest, kaggle_kernel_train_kernel, kaggle_kernel_inference_kernel [EXTRACTED 0.95]
- **Learned Detection & Tracking Pipeline** — src_model_unet3d, src_model_simplenodetransformer, src_inference_greedy_edge_assignment, src_inference_tta_inference [EXTRACTED 0.95]
- **Embryo-Disjoint Split Validation Flow** — src_split_utils_extract_embryo_id, src_split_utils_compute_membership_sha256, src_split_utils_load_and_validate_split, scripts_build_train_val_split_main [EXTRACTED 1.00]
- **Evaluation Metrics Suite** — src_tracking_cellmot_evaluate, src_tracking_cellmot_division_metrics_evaluate_divisions [EXTRACTED 1.00]
- **Phase 2 Learned Detection Flow** — src_dataset, src_model, src_inference, src_prediction_graph, src_evaluation [EXTRACTED 1.00]
- **Production Inference Stack** — src_submission_pipeline, src_prediction_graph, src_submission_exporter, src_checkpoint_manifest [EXTRACTED 1.00]
- **Target Generation Flow** — src_targets_load_geff_cached, src_targets_generate_heatmap_targets, src_targets_generate_edge_targets [EXTRACTED 1.00]
- **Training Integrity & Split Validation** — src_split_utils, src_dataset, scripts_local_smoke_train, src_checkpoint_manifest [EXTRACTED 1.00]
- **Metric and Scoring Validation** — tests_test_p07a_metric_parity_testp07arequireddifferentials, tests_test_targets_testdetectionloss, src_tracking_cellmot_division_metrics [INFERRED 0.85]
- **Kaggle Submission Generation Flow** — src_submission_pipeline_run_submission_inference, src_submission_exporter_export_submission, src_tracker_sthypergraphtracker [INFERRED 0.85]
- **Training Verification Suite** — test_train_fixes_verification [INFERRED 0.85]
- **Pipeline Integrity & Deployment Verification** — tests_test_p06_submission_deployment_testf1productionsubmissionpath, tests_test_p08_gpu_sanity_gate_infrastructure_testsampleidallowlist, tests_test_sync_kaggle_src_testverifysync [INFERRED 0.90]
- **Segfault Investigation Flow** — debug_segfault_repro_run_repro, diagnose_iteration, test_zarr_loop_main, test_zarr_reopen_main, verify_eval_fixed_run_evaluation [INFERRED 0.90]

## Communities (161 total, 36 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (42): main(), deployment_eligibility_errors(), Return every reason (human-readable) checkpoint is NOT eligible for manifest…, Transactionally create/replace checkpoint_manifest.json beside checkpoint_path…, write_checkpoint_manifest(), _make_eligible_checkpoint(), _make_small_models(), fixture (+34 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (46): ST-ACT Gap Analysis - Independent Codex Review, main(), 3D U-Net for cell detection in anisotropic volumetric data. Architecture: -…, Forward pass. Args: x: (B, 2, 64, 256, 256) two consecutive frames concatenated…, Cross-attention Transformer for pairwise edge probability prediction.…, Generate sinusoidal positional encoding., Forward pass for edge logit prediction. Args: nodes_t: (n_t, 3) node…, SimpleNodeTransformer (+38 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (49): 3D U-Net for cell detection in anisotropic volumetric data. Architecture: -…, Forward pass. Args: x: (B, 2, 64, 256, 256) two consecutive frames concatenated…, Cross-attention Transformer for pairwise edge probability prediction.…, Generate sinusoidal positional encoding., Forward pass for edge logit prediction. Args: nodes_t: (n_t, 3) node…, SimpleNodeTransformer, UNet3D, DivisionLoss (+41 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (30): _count_peaks_for_channel_calls(), _FakeAlwaysDetectingUNet3D, _FakeDatasetWithPairs, _FakeEdgeTransformer, _FakeValLoaderWithPairs, make_bare_training_loop(), _make_fake_val_batch(), _make_loop() (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (19): load_verified_checkpoint(), Fail-closed checkpoint load (B7): every verification step below must pass, IN…, Streaming SHA-256 hex digest (lowercase) of a file's exact bytes., sha256_file(), F4.17: checkpoint's OWN model_contract field disagrees with the (schema-…, F4.19: checkpoint's real split differs from manifest's declared split -- caught…, F4.21: evaluated != total in the manifest itself., F4.29: a checkpoint 'file' that isn't even a real torch archive must fail with… (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (36): extract_peaks(), find_latest_local_checkpoint(), get_nodes_and_features(), device, Path, Tensor, Local evaluation script for Task 3.4. Evaluates the downloaded Kaggle sanity-…, Find the most-recently-modified epoch_*.pt anywhere under search_root. Doesn't… (+28 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (28): Embryo-Disjoint Split, AugmentedCompetitionDataset, CompetitionDataset, AnisotropicZarrLoader, Any, Dataset, Path, Return this instance's cached loader for sample_id, opening it on first use. (+20 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (30): find_all_kaggle_input_dirs(), find_exactly_one_kaggle_input_dir(), Path, Kaggle inference kernel for ST-ACT -- Code Competition compliant submission.…, Part C2: after importing every production module this kernel depends on, verify…, verify_import_origins(), Shared production submission inference pipeline (P0-6, Part A). Owns the actual…, Training loop for ST-ACT model with end-to-end correctness. Handles: - Data… (+22 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (32): extract_inference_peaks(), extract_peaks_from_volume(), nodes_and_features_at_peaks(), pool_kernel_from_um(), _post_ntfy_heartbeat(), Any, DataLoader, device (+24 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (39): evaluate_submission(), load_geff_ground_truth(), load_gt_for_dataset(), BaseGraph, Local evaluation harness for the Kaggle cell tracking competition. Provides a…, Evaluate predicted tracking graphs against ground-truth graphs. Computes the…, Load a .geff ground-truth file into a tracksdata graph. Parameters ----------…, Load ground-truth .geff file for a specific dataset by ID. Parameters… (+31 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (24): git_sha(), main(), Path, Sync src/ into kaggle_src_dataset/src/, embed a git SHA marker, and verify the…, Copy every file in SPLIT_FILES from repo_root to dataset_dir, preserving the…, Return a list of mismatched/missing relative paths across BOTH the Python…, True iff `git status --porcelain` reports no staged, unstaged, or untracked…, Write the current git HEAD SHA to sha_file, but ONLY when the working tree is… (+16 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (26): main(), pick_train_and_window_timepoints(), Task 2.3 empirical benchmark: point vs. dilated-Gaussian heatmap targets.…, First 6 labeled timepoints for training; a WINDOW_SIZE-frame consecutive window…, run_one_config(), Spatio-Temporal Hypergraph Lineage Solver (Grandmaster Tier). Models tracking…, STHypergraphTracker, Positive/negative test suite for STHypergraphTracker (src/tracker.py). Written… (+18 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (35): load_geff_ground_truth(), load_gt_for_dataset(), BaseGraph, Local evaluation harness for the Kaggle cell tracking competition. Provides a…, Load a .geff ground-truth file into a tracksdata graph. Parameters ----------…, Load ground-truth .geff file for a specific dataset by ID. Parameters…, assert_true(), main() (+27 more)

### Community 13 - "Community 13"
Cohesion: 0.10
Nodes (22): convert_nx_to_tracksdata(), linefit_smooth_coordinates(), prune_short_tracks(), BaseGraph, DiGraph, Component-based short-track pruning, with division protection. Drops entire…, Topology-preserving line-fit coordinate smoothing. Mutates only node 'coords'…, Convert networkx DiGraph to tracksdata BaseGraph format. Args: nx_graph:… (+14 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (20): AnisotropicCoordinateTransformer, Tensor, Natively maps 3D coordinate tensors from anisotropic voxel space (Z, Y, X) to…, Args: voxel_coords (torch.Tensor): Shape (B, N, 3) representing (Z, Y, X)…, Fully Convolutional 3D Network that inputs anisotropic timepoint blocks and…, Args: x (torch.Tensor): Volume tensor of shape (B, 1, Z, Y, X) Returns:…, Double convolution block with anisotropic kernels. GroupNorm after every conv…, STACTCentroidPredictor (+12 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (22): Run one validation epoch. Uses full inference pipeline: NMS peak-finding ->…, greedy_edge_assignment(), Module, Tensor, Test-time augmentation: average detection logits across 4 views. Applies flip…, Greedy edge assignment respecting cardinality constraints. Sorts candidate…, tta_inference(), Fix 3: Verify edges are actually added to prediction graph. Original bug:… (+14 more)

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (14): Validate a submission CSV against the schema. Parameters ---------- csv_path :…, validate_submission(), P0-6 (Part F3/E2): validate_submission()'s required_dataset_ids structural…, One dataset, 2 nodes (t=0 -> t=1), 1 valid edge between them., F3.7: header-only passes generic mode but must fail required mode., F3.5/F3.6: a required dataset with zero node rows can only occur as total…, F3.11: an edge in ds_a referencing node_id=2 must be treated as a missing…, F3.15: a node with 3 outgoing edges (only 0/1/2 children are physically valid)… (+6 more)

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (17): InMemoryGraph, DivisionCounts, evaluate_divisions(), Compute TP, FN, and FP counts for division events. - **TP**: GT divisions…, Counts for division event evaluation., evaluate, _build_graph(), _make_gt() (+9 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (30): _bipartite_max_matching(), _branch_component_evidence(), count_matched_pred_divisions(), evaluate_divisions(), extract_divisions(), _gt_weak_component_ids(), _is_strongly_connected_division(), match_divisions() (+22 more)

### Community 19 - "Community 19"
Cohesion: 0.09
Nodes (19): AugmentedCompetitionDataset, CompetitionDataset, AnisotropicZarrLoader, Any, Dataset, Path, Return this instance's cached loader for sample_id, opening it on first use., Return {timepoint: gt_node_count} for sample_id, parsed from its .geff once and… (+11 more)

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (13): dataset_checkpoint_key(), detection_cache_key(), _format_duration(), _git_commit_hash(), Any, Path, Run execution tracking: fine-grained stage timing, live progress/ETA,…, Times named stages within one unit (e.g. one dataset's load/detect/track/...). (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.14
Nodes (27): deployment_eligibility_errors(), find_single_manifest(), _is_exact_schema_version(), _is_lowercase_hex(), load_verified_checkpoint(), _parse_manifest_bytes(), Any, Path (+19 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (20): find_single_manifest(), _is_exact_schema_version(), _is_lowercase_hex(), _parse_manifest_bytes(), Any, Path, Verified checkpoint manifest: schema, discovery, deployment eligibility,…, True only if value is a genuine int equal to expected -- Python's `==` treats… (+12 more)

### Community 23 - "Community 23"
Cohesion: 0.10
Nodes (12): _make_edge_loss_harness(), Build a minimal TrainingLoop (via __new__ bypass) whose train_epoch() exercises…, Reviewer-required (P0-7 v1 review, Defect 2): a technical exception from…, The FIRST _get_gt_nodes call (t_idx) succeeds; the SECOND (t_idx+1) fails --…, Reviewer-required (P0-7 v1 review, Defect 1): a non-empty all-negative edge…, Defensive regression: generate_edge_targets() (src/targets.py) is verified to…, TestDefect2GtNodeLoadFailureCounted, TestRuleATechnicalGtLoadFailurePropagates (+4 more)

### Community 24 - "Community 24"
Cohesion: 0.10
Nodes (15): Tests for validate_submission() function., Helper to create a test CSV file., Test validation of a valid submission., Test validation fails on header mismatch., Test validation fails on non-sequential global ids., Test validation fails on invalid row_type., Test validation fails when node row has source_id != -1., Test validation fails when node row has target_id != -1. (+7 more)

### Community 25 - "Community 25"
Cohesion: 0.09
Nodes (13): PredictionGraphAssembler, IndexedRXGraph, Shared production helper for sequential predicted-graph assembly across…, Enforce the chronological ordering contract: the first window observed for a…, Register one window's raw NMS peaks and apply the canonical graph identity…, edges: list of (src_idx, tgt_idx, prob) where src_idx/tgt_idx are indices INTO…, The per-sample IndexedRXGraphs, exactly the shape evaluate_submission() expects…, Aggregate, run-wide counters. predicted_nodes_total/ predicted_edges_total… (+5 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (25): tracking_cellmot: Vendored scoring and data loading from royerlab's kaggle-…, _compute_score(), DatasetsResult, _evaluate(), evaluate_datasets(), _evaluate_matched_graph(), EvaluationResult, _jaccard() (+17 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (14): build_and_validate_targets(), Path, Tensor, Build the (1, 2, Z, Y, X) target tensor for one smoke-test step and fail loudly…, P0-2 checkpoint/split-identity fix (2026-07-16, round 2): resuming this…, validate_resume_split_identity(), make_heatmap(), Tensor (+6 more)

### Community 28 - "Community 28"
Cohesion: 0.11
Nodes (18): Any, DataLoader, device, IndexedRXGraph, Module, PredictionGraphAssembler, Production submission inference entry point (Part A). Both…, Run one sample's complete (t, t+1) window inference, feeding every window… (+10 more)

### Community 29 - "Community 29"
Cohesion: 0.12
Nodes (12): _fake_loader(), _FakeAcceptAllEdgeTransformer, _FakePeakUNet3D, Real end-to-end validate_epoch() call (hermetic: fake val_loader, deterministic…, Real end-to-end validate_epoch() call where evaluate_submission() raises -- the…, P0-7 (COUNTED_THEN_FATAL): a missing .geff raises FileNotFoundError, which…, Deterministic detection model: on call number `call_idx` (0-indexed,…, Deterministic edge transformer: returns a fixed positive logit for every… (+4 more)

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (16): Kaggle Inference Kernel, find_all_kaggle_input_dirs(), find_exactly_one_kaggle_input_dir(), Shared source-provenance verification for Kaggle kernels and the GPU sanity…, Return every directory beneath /kaggle/input containing marker_relpath., Exact-one discovery: never silently select the first directory, directory…, _make_grad_harness(), _patch_fake_kaggle_input() (+8 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (15): evaluate_submission(), Evaluate predicted tracking graphs against ground-truth graphs. Computes the…, Test: provide empty prediction graph, real GT. Expected: edge_jaccard should be…, Test: micro-averaging across 2+ datasets. Verify that metrics are summed before…, Verify that mismatched lengths raise ValueError., Verify that empty input lists raise ValueError., Test evaluate_submission with dict-based inputs., Test the main evaluate_submission() function. (+7 more)

### Community 32 - "Community 32"
Cohesion: 0.08
Nodes (14): _post_ntfy_heartbeat(), Run one validation epoch. Uses full inference pipeline: NMS peak-finding ->…, Train model for specified number of epochs. If max_wall_clock_seconds is set,…, Unconditional checkpoint of the current weights, independent of validation…, Save model checkpoint., Log epoch results to CSV., Overwrite (not append) a small JSON heartbeat after each epoch. Unlike the full…, Mid-epoch heartbeat, same atomic-overwrite mechanism as… (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.10
Nodes (15): Test export of a single node with no edges., Test export with 3 nodes and 2 edges., Test export with multiple datasets (node_id reset per dataset)., Tests for export_submission() function., REGRESSION GUARD: in GENERIC mode (required_dataset_ids=None), a submission…, P0-6 (Part E1.6): required_dataset_ids mode must REJECT (not silently accept,…, Test that exported coordinates are integers (no floats)., Helper to create a synthetic tracksdata graph. Parameters ---------- nodes_data… (+7 more)

### Community 34 - "Community 34"
Cohesion: 0.11
Nodes (23): Dataset, invert_time_graph(), list_datasets(), _lookup_precomputed_quantile(), open_dataset(), _parse_scale(), _process_on_gpu(), BaseGraph (+15 more)

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (13): skipif, generate_edge_targets(), Generate edge probability targets from .geff ground truth. Candidate nodes are…, Unit tests for src/targets.py: load_geff_cached, generate_heatmap_targets,…, t=0 in the real geff has exactly 1 real GT centroid -- point targets must be…, REGRESSION GUARD for bug 1.2: two generate_heatmap_targets() calls sharing a…, REGRESSION GUARD, gap found by adversarial review: this file's other…, REGRESSION GUARD for bug 1.2 at the actual call-site: two calls with the same… (+5 more)

### Community 36 - "Community 36"
Cohesion: 0.14
Nodes (15): export_submission(), BaseGraph, DataFrame, Path, Export tracksdata graphs to a Kaggle-compliant submission CSV. Parameters…, _make_graph(), Unit tests for submission exporter. Tests export_submission() and…, Module-level synthetic-graph helper (P0-6, Part F3) -- identical pattern to the… (+7 more)

### Community 37 - "Community 37"
Cohesion: 0.16
Nodes (24): _branch_component_evidence(), count_matched_pred_divisions(), extract_divisions(), _gt_weak_component_ids(), match_divisions(), _match_full(), _matched_division_nodes(), _matched_node_attrs() (+16 more)

### Community 38 - "Community 38"
Cohesion: 0.11
Nodes (23): Dataset, invert_time_graph(), list_datasets(), _lookup_precomputed_quantile(), open_dataset(), _parse_scale(), _process_on_gpu(), BaseGraph (+15 more)

### Community 39 - "Community 39"
Cohesion: 0.14
Nodes (16): _count_exact_sigmoid_assignment(), _first_positional_arg_name(), _get_class_method_source(), _get_function_source(), _get_module_source(), _has_call(), _is_source_or_source_dot_float(), AST (+8 more)

### Community 40 - "Community 40"
Cohesion: 0.11
Nodes (13): frame_counts(), PredictionGraphAssembler, Test C: two distinct channel-1 detections within the SAME newly owned frame…, Test D: with multiple cells per frame, persistent node IDs must correctly map…, Test F: interleaving two samples' windows must not create cross-sample node…, Test A: the exact P0-3 reproducer shape -- window A's channel-1 output for…, Test B: proves canonical-frame reuse is NOT coordinate-based. Window B's…, TestChronologicalContract (+5 more)

### Community 41 - "Community 41"
Cohesion: 0.14
Nodes (17): _count_exact_sigmoid_assignment(), _first_positional_arg_name(), _get_function_source(), _get_module_source(), _has_call(), _is_source_or_source_dot_float(), AST, P0-6 tests: shared submission inference pipeline, verified checkpoint manifest,… (+9 more)

### Community 42 - "Community 42"
Cohesion: 0.13
Nodes (12): AnisotropicZarrLoader, ndarray, Extract quantile normalization parameters from Zarr metadata. Looks for…, Returns the (T, Z, Y, X) dimensions of the 4D dataset. Returns: Tuple[int, int,…, Generates a mock Zarr store mimicking a 4D anisotropic microscopy volume.…, Helper method to render a Gaussian cellular signal in anisotropic space., Apply quantile normalization to raw data if normalization parameters are…, ST-ACT Memory-Safe 4D Anisotropic Zarr v3 Ingestor. Loads and decompresses 3D… (+4 more)

### Community 43 - "Community 43"
Cohesion: 0.09
Nodes (13): DivisionLoss, Weighted BCE loss for edge prediction with division event upweighting. Division…, Initialize division loss. Args: weight_division: Loss weight multiplier for…, Initialize detection loss. Args: weight_pos: Weight for positive (cell) voxels…, Fix 2: Verify real generate_edge_targets() is used, not fake placeholder.…, test_fix_2_real_edge_targets(), _make_small_unet_and_transformer(), Verify DivisionLoss signature expects logits parameter. (+5 more)

### Community 44 - "Community 44"
Cohesion: 0.09
Nodes (12): The cache must not silently return stale data for a different t., Test AnisotropicZarrLoader with real staged Zarr v3 data., Test that normalized data values are in expected [0, 4] range (reference…, Test that simulate=False is the default (requires real data)., Test that loader correctly initializes with real Zarr v3 OME-NGFF store., Test that loaded data has correct 4D shape (T, Z, Y, X)., Test that raw real data is uint16 as expected., Test that default anisotropy is correctly set to (4.0, 1.0, 1.0). (+4 more)

### Community 45 - "Community 45"
Cohesion: 0.13
Nodes (12): make_dataset(), Unit tests for src/dataset.py: CompetitionDataset, against REAL local staged…, A sample_id with no matching .zarr on disk must be silently skipped (logged,…, Build a CompetitionDataset without needing a real data_split.json file, using…, Test 12 (real-data integration): confirms the filtering behaves correctly…, REGRESSION GUARD for bug 2.1: CompetitionDataset used to construct a fresh…, The loader opened during _build_pair_index() (to read num_frames) must be the…, REGRESSION GUARD for the Phase 2 Wave 1 bug already documented in CLAUDE.md:… (+4 more)

### Community 46 - "Community 46"
Cohesion: 0.16
Nodes (13): build_leave_one_embryo_out_folds(), extract_embryo_id(), Return the embryo ID for a sample ID, per Kaggle's own naming convention…, Build one leave-one-embryo-out fold per distinct embryo present in `samples`.…, extract_embryo_id(), Return the embryo ID for a sample ID, per Kaggle's own naming convention…, make_synthetic_inventory(), Unit tests for the P0-2 fix: leave-one-embryo-out split generation… (+5 more)

### Community 47 - "Community 47"
Cohesion: 0.24
Nodes (6): _find_duplicates(), load_and_validate_split(), Load a split JSON file and validate embryo-disjointness. Raises RuntimeError --…, P0-2 fix (2026-07-16): load_and_validate_split() now cross-checks…, TestLoadAndValidateSplitMembershipSha256, TestLoadAndValidateSplitMetadataConsistency

### Community 48 - "Community 48"
Cohesion: 0.11
Nodes (13): _FakeMetadata, _make_graph(), Verify evaluate_submission() sums TP/FP/FN across samples BEFORE computing…, division_jaccard must be computed from POOLED division TP/FP/FN across samples…, Verify fallback to GT node count when estimated_number_of_nodes is unavailable,…, Duck-types tracksdata's GeffMetadata just enough for evaluate_submission(): an…, evaluate_submission() must reject mixed pred/gt/metadata container types with a…, One failing sample must fail the ENTIRE evaluate_submission() call -- it must… (+5 more)

### Community 49 - "Community 49"
Cohesion: 0.13
Nodes (15): DetectionLoss, generate_edge_targets(), generate_heatmap_targets(), load_geff_cached(), Path, Tensor, Target generation for detection (heatmaps) and edge prediction. Supports two…, Generate edge probability targets from .geff ground truth. Candidate nodes are… (+7 more)

### Community 50 - "Community 50"
Cohesion: 0.13
Nodes (14): extract_peaks_from_volume(), pool_kernel_from_um(), ndarray, Convert physical microns to voxel kernel size., Real 3D non-max suppression via maximum_filter with centroid collapsing. Sub-…, Fix 4: Verify NMS peak-finding is used, not raw thresholding. Original bug:…, test_fix_4_nms_not_just_threshold(), A `vol == pooled` tied plateau is mathematically guaranteed to be uniform-… (+6 more)

### Community 51 - "Community 51"
Cohesion: 0.17
Nodes (9): _make_training_loop_for_checkpoint_save(), Bare TrainingLoop instantiation (bypassing __init__, which needs real…, Real production-path (TrainingLoop.save_checkpoint()) regressions: a…, Establish a real, valid manifest + checkpoint via the actual production…, Positive control: an uncorrupted active manifest lets save_checkpoint() proceed…, Corrects a confirmed transactional defect: the derived checkpoint filename…, Required regression 1: same derived filename, INELIGIBLE new metrics -- must…, TestActiveManifestProtectionFailsClosed (+1 more)

### Community 52 - "Community 52"
Cohesion: 0.16
Nodes (15): BaseGraph, HSOM Self-Learning and Meta-Learning, Oracle Decomposition Runner, _build_oracle_graph(), compute_detection_metrics(), evaluate_oracle_modes(), OracleDecomposition, OracleModeResult (+7 more)

### Community 53 - "Community 53"
Cohesion: 0.17
Nodes (18): compute_membership_sha256(), extract_embryo_id(), _find_duplicates(), get_split_identity(), load_and_validate_split(), Any, Path, Split-file resolution and embryo-disjointness validation for CompetitionDataset… (+10 more)

### Community 54 - "Community 54"
Cohesion: 0.25
Nodes (6): parametrize, Fail-closed validation of the hyperparameters production inference needs…, validate_inference_hyperparams(), F8.6: every load_state_dict(...) call in the two production submission callers…, TestF7HyperparameterValidation, _valid_hyperparams()

### Community 55 - "Community 55"
Cohesion: 0.20
Nodes (17): _dataset_full_config(), main(), Phase 0 Pipeline Orchestrator: Multi-Dataset Integration and End-to-End Testing…, Everything that affects a dataset's output -- feeds both cache-key functions.…, Load a single Zarr dataset, run detector+tracker, return lineage graph + timing…, run_dataset(), dataset_checkpoint_key(), detection_cache_key() (+9 more)

### Community 56 - "Community 56"
Cohesion: 0.12
Nodes (10): _FakeLoader, _new_bare_dataset(), The .geff for a sample must be parsed exactly once while building the pair…, Case 2, genuine integration test: a sample with an EXISTING Zarr directory but…, Case 3: .geff exists but IndexedRXGraph.from_geff() itself raises (e.g. corrupt…, Case 4: .geff parses successfully but contains zero GT nodes -- must raise…, split_type == "validation" with filter_unannotated_pairs left at its DEFAULT…, When filter_unannotated_pairs is False, _get_gt_counts_by_time must never be… (+2 more)

### Community 57 - "Community 57"
Cohesion: 0.16
Nodes (4): Positive case: a genuinely embryo-disjoint split must NOT raise., P0-2 direct regression test (2026-07-16): exercises the embryo-disjointness…, TestGetSplitIdentity, TestLoadAndValidateSplitGuard

### Community 58 - "Community 58"
Cohesion: 0.18
Nodes (6): _format_duration(), _git_commit_hash(), Times named stages within one unit (e.g. one dataset's load/detect/track/...)., Tracks one pipeline invocation across multiple units (e.g. datasets). Writes an…, RunTracker, UnitTimer

### Community 59 - "Community 59"
Cohesion: 0.14
Nodes (14): extract_inference_peaks(), nodes_and_features_at_peaks(), Any, DataLoader, device, Module, Path, Tensor (+6 more)

### Community 60 - "Community 60"
Cohesion: 0.16
Nodes (9): make_dataset_with_gt_counts(), Build a CompetitionDataset entirely from synthetic/mocked pieces -- no real…, P0-1 fix (2026-07-16): CompetitionDataset used to build (t, t+1) pairs from…, GT timepoints = {2, 3, 5, 6, 7} (each with >=1 node, all other t implicitly…, candidate_pairs must equal the sum of retained + all three excluded categories,…, Filtering which pairs are BUILT must not change what __getitem__ RETURNS for a…, The core regression this follow-up fix exists for: split_type == "train" ALONE…, annotation_pair_stats must be the dict when filter_unannotated_pairs is True,… (+1 more)

### Community 61 - "Community 61"
Cohesion: 0.19
Nodes (10): _extract_discovery_functions(), _extract_function_source(), _extract_sha_validation_block(), _FakeModule, _patch_fake_kaggle_input(), Path, P0-7 (2026-07-19) training-integrity regression tests. Dedicated test file for…, Redirect the literal '/kaggle/input' path (hardcoded in production, by design… (+2 more)

### Community 62 - "Community 62"
Cohesion: 0.12
Nodes (13): Phase 2: Learned Detection — Complete Planning Index, Diagnostic script to understand which sample/timepoint corresponds to step 34., find_all_kaggle_input_dirs(), find_exactly_one_kaggle_input_dir(), Path, Kaggle training kernel for ST-ACT (Spatio-Temporal Anisotropic Cell Tracker).…, P0-7 (2026-07-19): after importing every production module this kernel actually…, # NOTE: KAGGLE_MODE's exact input subdirectory structure under INPUT_DIR (+5 more)

### Community 63 - "Community 63"
Cohesion: 0.15
Nodes (13): DivisionCounts, DivisionScores, NamedTuple, Result of :func:`score_divisions`. Attributes ---------- scores : dict[int,…, Counts for division event evaluation., DivisionScores, NamedTuple, Result of :func:`score_divisions`. Attributes ---------- scores : dict[int,… (+5 more)

### Community 64 - "Community 64"
Cohesion: 0.14
Nodes (9): DiGraph, ndarray, Mitosis Backward-Smoothing (Temporal Window Align): Backtracks division nodes…, Anisotropic Velocity Edge Pruning: Inspects coordinates and discards…, Constructs and solves ILP for cell centroids. Supports multi-frame lookahead…, Spatio-Temporal Hypergraph Lineage Solver (Grandmaster Tier). Models tracking…, STHypergraphTracker, 2-3 layer 3D UNet, single frame in/out. Task 2.3 benchmark only -- not Task… (+1 more)

### Community 65 - "Community 65"
Cohesion: 0.17
Nodes (14): ensemble_consensus_centroids(), extract_peaks_from_volume(), pool_kernel_from_um(), ndarray, Ensemble Consensus Centroid Clustering (DBSCAN): Combines cell centroid…, Convert a physical-micron NMS radius into an odd per-axis pooling kernel size…, Real 3D non-max suppression via max_pool3d, kernel sized from physical…, build_candidate_edges() (+6 more)

### Community 66 - "Community 66"
Cohesion: 0.14
Nodes (8): AnisotropicCoordinateTransformer, Tensor, Natively maps 3D coordinate tensors from anisotropic voxel space (Z, Y, X) to…, Args: voxel_coords (torch.Tensor): Shape (B, N, 3) representing (Z, Y, X)…, Fully Convolutional 3D Network that inputs anisotropic timepoint blocks and…, Args: x (torch.Tensor): Volume tensor of shape (B, 1, Z, Y, X) Returns:…, Double convolution block with anisotropic kernels. GroupNorm after every conv…, STACTCentroidPredictor

### Community 67 - "Community 67"
Cohesion: 0.24
Nodes (5): P0-2 checkpoint/split-identity fix (2026-07-16), round 2: last_checkpoint.pt…, Build a minimal, real torch.save()'d checkpoint (matching the real shape…, Test 6: when THIS TrainingLoop itself has no configured split identity (not the…, TestLoadCheckpointSplitIdentity, TestSaveLastCheckpointSplitIdentity

### Community 68 - "Community 68"
Cohesion: 0.20
Nodes (6): _FakeEdgeGraph, _patch_geff_load(), Minimal stand-in for the loaded .geff ground-truth graph -- implements only…, 2x2 candidate grid engineered to produce all three categories at once: (A,C)…, A scenario with no both-matched-but-unconnected pairs at all --…, TestEdgeHardEasyNegativeSplit

### Community 69 - "Community 69"
Cohesion: 0.15
Nodes (8): _FakeEdgeTransformer, _FakeLateBloomerUNet3D, Unit tests for src/train.py's TrainingLoop helper methods, using the same…, Returns zero-detection logits for its first `zero_calls` forward() calls, then…, Minimal stand-in for SimpleNodeTransformer: returns a fixed high-probability…, The exact P0-3 regression case: a model whose first several chronological…, Test double for threading.Thread that runs the target synchronously on .start()…, _SyncThread

### Community 70 - "Community 70"
Cohesion: 0.19
Nodes (11): main(), Path, Local submission generator (P0-6, Part D). Runs a verified, manifest-referenced…, _read_source_sha(), Submission exporter for the Kaggle cell tracking competition. Provides…, create_small_test_zarr(), Path, Integration test for the Phase 0 pipeline. This test verifies that: 1. The… (+3 more)

### Community 71 - "Community 71"
Cohesion: 0.21
Nodes (14): build_test_dataset(), Any, CompetitionDataset, DataLoader, device, IndexedRXGraph, Module, Path (+6 more)

### Community 72 - "Community 72"
Cohesion: 0.24
Nodes (6): _fake_evaluate_submission(), _make_validate_harness(), P0-7 COUNTED_THEN_FATAL: missing GEFF must be counted then raised in strict…, TestMissingGeffHandling, TestStrictValidationIntegrity, TestValidationAccounting

### Community 73 - "Community 73"
Cohesion: 0.24
Nodes (5): _FakeLoader, _new_bare_dataset(), Reports a controlled frame count, or raises on get_shape() to simulate an…, P0-7 DATASET COVERAGE CONTRACT: strict_sample_coverage semantics., TestDatasetCoverageContract

### Community 74 - "Community 74"
Cohesion: 0.19
Nodes (7): _make_fake_val_batch(), P0-1 fix (2026-07-16), Section 6 training-side invariant: CompetitionDataset's…, The exact regression this invariant guards against: heatmap generation SUCCEEDS…, Independent of the both-channels-zero test: channel 1 (t_idx+1) alone having…, Mirror of the channel-0 case: a real, nonzero channel 0 must NOT mask channel 1…, Positive case: a real (nonzero) heatmap for both channels must pass through…, TestGenerateAndValidateHeatmapTarget

### Community 75 - "Community 75"
Cohesion: 0.18
Nodes (9): IndexedRXGraph, Test detection precision/recall computation., Compute metrics when all GT nodes have matching predictions., Compute metrics when detector finds nothing., Compute metrics with false-positive predictions., Regression test: mode 2 (gt_nodes_model_edges) must not assume pred_graph's…, pred_graph built from scratch (own node-ID space, not a GT copy) with nodes at…, TestDetectionMetrics (+1 more)

### Community 76 - "Community 76"
Cohesion: 0.21
Nodes (8): PyTorch Dataset class for ST-ACT competition. CompetitionDataset loads Zarr v3…, # TODO: Wave 3 augmentations, AnisotropicZarrLoader, Returns the (T, Z, Y, X) dimensions of the 4D dataset. Returns: Tuple[int, int,…, ST-ACT Memory-Safe 4D Anisotropic Zarr v3 Ingestor. Loads and decompresses 3D…, main(), Minimal Zarr decompression stress test -- NO torch/training, just pure Zarr…, Unit tests for AnisotropicZarrLoader with real staged data. Tests validate…

### Community 77 - "Community 77"
Cohesion: 0.24
Nodes (5): Strict, unconditional GIT_SHA.txt validation for GPU-SANITY-GATE-01 and any…, validate_git_sha_file(), No allow_unknown / allow-missing escape hatch exists on this function at all --…, Structural guard: fails loud (TypeError) if a future edit reintroduces an…, TestValidateGitShaFile

### Community 78 - "Community 78"
Cohesion: 0.19
Nodes (6): make_bare_training_loop(), Bypass __init__ (which needs real models/loaders) and set only the attributes…, An out-of-bounds peak (possible from NMS on a small/edge volume) must be…, Regression coverage for the live ntfy.sh progress channel added to close the…, TestNodesAndFeaturesAtPeaks, TestPostNtfyHeartbeat

### Community 79 - "Community 79"
Cohesion: 0.27
Nodes (11): CompletedProcess, build_context_blocks(), fetch_leaderboard(), list_top_kernels(), main(), pull_kernel_text(), Competitor research via the Kaggle CLI + Gemini API. Pulls real public kernels…, Real public kernels for this competition, sorted by vote count. (+3 more)

### Community 80 - "Community 80"
Cohesion: 0.20
Nodes (11): find_all_kaggle_input_dirs(), find_exactly_one_kaggle_input_dir(), ModuleType, Path, Shared source-provenance verification for Kaggle kernels and the GPU sanity…, Return every directory beneath /kaggle/input containing marker_relpath., Exact-one discovery: never silently select the first directory, directory…, Strict, unconditional GIT_SHA.txt validation for GPU-SANITY-GATE-01 and any… (+3 more)

### Community 81 - "Community 81"
Cohesion: 0.24
Nodes (7): ModuleType, Path, Part C2: after importing every production module the caller depends on, verify…, verify_import_origins(), ModuleType, Codex review, PR #4, 2026-07-19: an empty modules list must not silently 'pass'…, TestVerifyImportOrigins

### Community 82 - "Community 82"
Cohesion: 0.27
Nodes (9): get_memory_usage_mb(), main(), patch_checkpoint_disabled(), PROCESS_MEMORY_COUNTERS, Module, Minimal, deterministic reproduction script for investigating native Windows…, replace_group_norm_with_identity(), run_repro() (+1 more)

### Community 83 - "Community 83"
Cohesion: 0.20
Nodes (8): _find_score_divisions_callers_in_repo(), _find_score_divisions_calls_in_source(), Path, Walk every *.py file under repo_root on disk (rglob -- plain filesystem…, Negative control, required by the P0-7A v2 review: prove the AST scanner…, The function definition line itself must never be mistaken for a call -- def…, Pure filesystem + AST repository scan (see…, Parse source with ast and return the line number of every ast.Call node whose…

### Community 84 - "Community 84"
Cohesion: 0.27
Nodes (9): main(), Path, Write each fold to output_dir/{fold_name}.json. Returns the list of written…, write_folds(), enumerate_dataset_from_zip(), main(), Spot-check n random samples to verify format., Enumerate all train samples from the competition zip file. (+1 more)

### Community 85 - "Community 85"
Cohesion: 0.20
Nodes (5): Extract quantile normalization parameters from Zarr metadata. Looks for…, Generates a mock Zarr store mimicking a 4D anisotropic microscopy volume.…, Helper method to render a Gaussian cellular signal in anisotropic space., Initialize the Anisotropic Zarr Loader. Args: store_path (str): File system…, Initializes connection to the Zarr v3 store. For real competition data: reads…

### Community 86 - "Community 86"
Cohesion: 0.24
Nodes (7): _find_nearest_gt_node(), Find nearest GT node to a predicted node within max_distance_um. Returns…, Test nearest GT node matching within distance threshold., Find GT node at exact same coordinates., Find nearest among multiple GT nodes., Return None if nearest GT node is beyond max_distance., TestFindNearestGtNode

### Community 88 - "Community 88"
Cohesion: 0.25
Nodes (8): export_submission(), BaseGraph, DataFrame, Path, Submission exporter for the Kaggle cell tracking competition. Provides…, Validate a submission CSV against the schema. Parameters ---------- csv_path :…, Export tracksdata graphs to a Kaggle-compliant submission CSV. Parameters…, validate_submission()

### Community 89 - "Community 89"
Cohesion: 0.33
Nodes (4): compute_membership_sha256(), Canonical fingerprint of exactly which samples are in train vs. validation.…, P0-2 checkpoint/split-identity fix (2026-07-16): compute_membership_sha256() is…, TestComputeMembershipSha256

### Community 90 - "Community 90"
Cohesion: 0.28
Nodes (6): _bipartite_max_matching(), _is_strongly_connected_division(), Check a predicted division's local directed topology. The prediction window…, Maximum-cardinality bipartite matching via DFS augmenting paths. *edges* maps…, These functions did not exist in our old vendored file at all -- their mere…, TestUpstreamPortedStronglyConnectedDivision

### Community 91 - "Community 91"
Cohesion: 0.33
Nodes (5): Path, REGRESSION GUARD for bug 1.2: _get_gt_nodes() at the real call site…, P0-7 (2026-07-19) Rule A: a missing .geff is a TECHNICAL GT-load failure and…, Real geff has GT nodes only at specific t values (e.g. t=0..2, 27-33, ...) -- a…, TestGetGtNodesGeffCache

### Community 92 - "Community 92"
Cohesion: 0.29
Nodes (7): greedy_edge_assignment(), Module, Tensor, Inference utilities for detection and edge assignment. Includes test-time…, Test-time augmentation: average detection logits across 4 views. Applies flip…, Greedy edge assignment respecting cardinality constraints. Sorts candidate…, tta_inference()

### Community 93 - "Community 93"
Cohesion: 0.43
Nodes (7): load_run(), main(), print_run_detail(), print_summary_table(), Path, Compare pipeline run statistics across logs/runs/*.jsonl. Usage: py…, summarize()

### Community 94 - "Community 94"
Cohesion: 0.32
Nodes (6): Diagnose Detection Loss Recipe, main(), Single-frame overfit diagnostic: does the reference implementation's exact…, Same architecture as scripts/benchmark_heatmap_targets.py's throwaway model --…, run_config(), ThrowawayTinyUNet

### Community 95 - "Community 95"
Cohesion: 0.39
Nodes (7): main(), pull_log_lines(), Path, Check a Kaggle kernel's real status and, if it's finished, pull and summarize…, Fetch the real log and return it as a flat list of clean text lines. NOTE:…, run_status(), summarize()

### Community 96 - "Community 96"
Cohesion: 0.29
Nodes (6): _compute_physical_distance(), Compute physical distance in micrometers between two coordinates (z, y, x)., Test physical distance computation., Distance from a point to itself should be 0., Test distance with anisotropic scale., TestPhysicalDistance

### Community 97 - "Community 97"
Cohesion: 0.25
Nodes (5): DiGraph, ndarray, Mitosis Backward-Smoothing (Temporal Window Align): Backtracks division nodes…, Anisotropic Velocity Edge Pruning: Inspects coordinates and discards…, Constructs and solves ILP for cell centroids. Supports multi-frame lookahead…

### Community 99 - "Community 99"
Cohesion: 0.25
Nodes (3): Regression tests for the adaptive per-batch class-imbalance weighting. UPDATED…, REGRESSION GUARD for the real confirmed bug: the original adaptive…, TestDetectionLoss

### Community 100 - "Community 100"
Cohesion: 0.29
Nodes (5): _FakeDegenerateUNet3D, Always returns deeply-negative logits (sigmoid~0 everywhere) regardless of…, REGRESSION GUARD for the exact incident this test class is named after: a real…, Proves the check is now a genuine post-pass check, not a disguised early abort…, TestValidateEpochCircuitBreaker

### Community 101 - "Community 101"
Cohesion: 0.29
Nodes (5): Load model checkpoint. P0-2 checkpoint/split-identity fix (2026-07-16):…, Any, Stricter, DELIBERATELY SEPARATE counterpart to…, validate_resume_checkpoint_split_identity(), Load model checkpoint. P0-2 checkpoint/split-identity fix (2026-07-16):…

### Community 102 - "Community 102"
Cohesion: 0.33
Nodes (4): ndarray, Apply quantile normalization to raw data if normalization parameters are…, Loads and decompresses a single 3D timepoint volume (Z, Y, X) into memory.…, A memory-efficient generator yielding spatial sub-chunks (Z, Y, X) of a single…

### Community 103 - "Community 103"
Cohesion: 0.33
Nodes (6): load_timepoint_no_cache(), main(), ndarray, Load a timepoint WITHOUT caching the zarr.open() call. This reopens the zarr…, Run the dataset iteration with or without zarr caching., test_pattern()

### Community 104 - "Community 104"
Cohesion: 0.29
Nodes (3): _FakeFrameDataset, Dataset, Deterministic, hermetic stand-in for CompetitionDataset's test-mode behavior --…

### Community 105 - "Community 105"
Cohesion: 0.29
Nodes (3): REGRESSION GUARD: an earlier version of this ramp used global_step/warmup_steps…, The critical regression case: global_step=9 is the LAST call made under…, TestComputeWarmupLr

### Community 106 - "Community 106"
Cohesion: 0.40
Nodes (6): Any, Path, find_latest_checkpoint(), Find the most recent usable checkpoint (by modification time)., Run oracle score decomposition on validation split. Args: checkpoint_path: Path…, run_oracle_decomposition()

### Community 107 - "Community 107"
Cohesion: 0.40
Nodes (6): HSOM Architecture Specification, HSOM Experiment Protocol, HSOM North Star / Constitution, HSOM Product Requirements Document, HSOM Specification Package README, HSOM v1.1 Adversarial Review Reconciliation

### Community 108 - "Community 108"
Cohesion: 0.33
Nodes (5): End-to-end pipeline test for Phase 0. Tests: 1. Pipeline runs without…, Test the CSV file structure independently., Test the complete pipeline: detect, track, export, evaluate., test_full_pipeline(), test_submission_csv_structure()

### Community 109 - "Community 109"
Cohesion: 0.33
Nodes (4): Test the four oracle modes on real data., Run all four oracle modes on a real staged sample., Verify oracle modes return expected structure., TestOracleModes

### Community 110 - "Community 110"
Cohesion: 0.33
Nodes (3): Confirms the architecture the P0-3 audit relied on to declare this path safe is…, Test H: run_pipeline.py (submission/inference) must remain completely unrouted…, TestSubmissionPathIsolation

### Community 111 - "Community 111"
Cohesion: 0.33
Nodes (5): Task 04-05: Spot-check generated submission against sample_submission.csv…, Verify submission CSV schema matches sample_submission.csv, Verify coordinate values in sample are reasonable, test_submission_coordinates_are_valid(), test_submission_schema_matches_sample()

### Community 113 - "Community 113"
Cohesion: 0.47
Nodes (3): A volume where very few voxels exceed the threshold (well-calibrated, trained…, REGRESSION-relevant: an undertrained model's near-uniform sigmoid output (all…, TestPeaksForChannel

### Community 114 - "Community 114"
Cohesion: 0.33
Nodes (3): P0-2 checkpoint/split-identity fix (2026-07-16): save_checkpoint() must embed…, unknown" (TrainingLoop's default) must NOT be written as a literal placeholder…, TestSaveCheckpointSplitIdentity

### Community 115 - "Community 115"
Cohesion: 0.40
Nodes (3): fixture, Path to real training data., Path to real test data (no ground truth).

### Community 116 - "Community 116"
Cohesion: 0.40
Nodes (4): build_test_dataset(), CompetitionDataset, Path, Construct a CompetitionDataset covering exactly one test sample (Part A1's…

### Community 117 - "Community 117"
Cohesion: 0.40
Nodes (3): Tensor, Compute weighted BCE loss. Args: logits: (n_candidates,) edge logits targets:…, Compute weighted BCE loss for detection. Args: logits: (B, 1, Z, Y, X)…

### Community 118 - "Community 118"
Cohesion: 0.50
Nodes (4): Colors, get_ram_info(), Retrieves system RAM size in GB. Uses psutil if installed, falls back to…, run_diagnostics()

### Community 121 - "Community 121"
Cohesion: 0.40
Nodes (3): evaluate_checkpoint.py must call the shared, corrected evaluate_submission()…, verify_eval_fixed.py is a diagnostic/crash-verification script (memory +…, TestCallerRegression

### Community 122 - "Community 122"
Cohesion: 0.50
Nodes (3): compare_with_sample_submission(), Schema comparison: export_submission output vs Kaggle's sample_submission.csv…, Compare exported submission CSV with Kaggle's sample_submission.csv. Parameters…

### Community 123 - "Community 123"
Cohesion: 0.67
Nodes (3): get_diff(), main(), Adversarial code review via the Gemini API. Sends a diff and/or specific files…

### Community 124 - "Community 124"
Cohesion: 0.50
Nodes (3): compare_with_sample_submission(), Schema comparison: export_submission output vs Kaggle's sample_submission.csv…, Compare exported submission CSV with Kaggle's sample_submission.csv. Parameters…

### Community 126 - "Community 126"
Cohesion: 0.50
Nodes (3): Build oracle graph when all predicted nodes match GT nodes., Test oracle graph construction by GT matching., TestOracleGraphBuilding

### Community 127 - "Community 127"
Cohesion: 0.50
Nodes (3): Test logical relationships between oracle modes., Verify that oracle modes form a sensible scoring hierarchy: - Mode 1 (GT+GT)…, TestOracleModeComparisons

## Knowledge Gaps
- **27 isolated node(s):** `Colors`, `test_submission_coordinates_are_valid`, `test_submission_schema_matches_sample`, `Phase 2: Learned Detection — Planning Complete`, `ST-ACT Pipeline Configuration` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **36 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrainingLoop` connect `Community 2` to `Community 0`, `Community 129`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 15`, `Community 19`, `Community 22`, `Community 23`, `Community 25`, `Community 28`, `Community 29`, `Community 30`, `Community 31`, `Community 41`, `Community 48`, `Community 49`, `Community 50`, `Community 51`, `Community 54`, `Community 61`, `Community 67`, `Community 68`, `Community 69`, `Community 72`, `Community 73`, `Community 74`, `Community 77`, `Community 78`, `Community 81`, `Community 87`, `Community 91`, `Community 100`, `Community 101`, `Community 104`, `Community 105`, `Community 113`, `Community 114`, `Community 121`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `evaluate_submission()` connect `Community 31` to `Community 32`, `Community 3`, `Community 5`, `Community 37`, `Community 7`, `Community 70`, `Community 9`, `Community 11`, `Community 12`, `Community 15`, `Community 48`, `Community 17`, `Community 55`, `Community 63`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `TrainingLoop` connect `Community 1` to `Community 32`, `Community 2`, `Community 35`, `Community 3`, `Community 101`, `Community 5`, `Community 7`, `Community 69`, `Community 41`, `Community 91`, `Community 43`, `Community 49`, `Community 17`, `Community 30`, `Community 25`, `Community 59`, `Community 61`, `Community 62`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 81 inferred relationships involving `TrainingLoop` (e.g. with `PredictionGraphAssembler` and `DetectionLoss`) actually correct?**
  _`TrainingLoop` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `AnisotropicZarrLoader` (e.g. with `ThrowawayTinyUNet` and `TestAnisotropicZarrLoaderReal`) actually correct?**
  _`AnisotropicZarrLoader` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Colors`, `test_submission_coordinates_are_valid`, `test_submission_schema_matches_sample` to the rest of the system?**
  _27 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.062317429406037 - nodes in this community are weakly interconnected._