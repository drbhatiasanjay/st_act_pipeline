# Graph Report - .  (2026-07-03)

## Corpus Check
- Corpus is ~4,459 words - fits in a single context window. You may not need a graph.

## Summary
- 58 nodes · 63 edges · 10 communities (6 shown, 4 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.8)
- Token cost: 13,550 input · 1,772 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Coordinate Transform & Centroid Model|Coordinate Transform & Centroid Model]]
- [[_COMMUNITY_Hypergraph Lineage Tracker|Hypergraph Lineage Tracker]]
- [[_COMMUNITY_Pipeline Modules Overview|Pipeline Modules Overview]]
- [[_COMMUNITY_Peak Detection & Consensus Clustering|Peak Detection & Consensus Clustering]]
- [[_COMMUNITY_Anisotropic Zarr Data Loader|Anisotropic Zarr Data Loader]]
- [[_COMMUNITY_System Diagnostics|System Diagnostics]]
- [[_COMMUNITY_Zarr Store Initialization|Zarr Store Initialization]]
- [[_COMMUNITY_Simulated Zarr Store Generator|Simulated Zarr Store Generator]]
- [[_COMMUNITY_Dataset Shape Accessor|Dataset Shape Accessor]]
- [[_COMMUNITY_Chunked Volume Streaming|Chunked Volume Streaming]]

## God Nodes (most connected - your core abstractions)
1. `AnisotropicZarrLoader` - 10 edges
2. `STHypergraphTracker` - 7 edges
3. `Run ST-ACT Pipeline` - 6 edges
4. `run_st_act_pipeline()` - 5 edges
5. `AnisotropicCoordinateTransformer` - 5 edges
6. `STACTCentroidPredictor` - 4 edges
7. `ensemble_consensus_centroids()` - 3 edges
8. `extract_peaks_from_volume()` - 3 edges
9. `get_ram_info()` - 3 edges
10. `run_diagnostics()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `Run ST-ACT Pipeline` --references--> `Hyperparameters Configuration`  [INFERRED]
  run_pipeline.py → config/hyperparams.yaml
- `run_st_act_pipeline()` --calls--> `AnisotropicZarrLoader`  [INFERRED]
  run_pipeline.py → src/data_loader.py
- `run_st_act_pipeline()` --calls--> `STHypergraphTracker`  [INFERRED]
  run_pipeline.py → src/tracker.py
- `Run ST-ACT Pipeline` --conceptually_related_to--> `ST-ACT Centroid Predictor`  [INFERRED]
  run_pipeline.py → src/model.py
- `Run ST-ACT Pipeline` --calls--> `Anisotropic Zarr Loader`  [EXTRACTED]
  run_pipeline.py → src/data_loader.py

## Communities (10 total, 4 thin omitted)

### Community 0 - "Coordinate Transform & Centroid Model"
Cohesion: 0.22
Nodes (6): AnisotropicCoordinateTransformer, Args:             voxel_coords (torch.Tensor): Shape (B, N, 3) representing (Z,, Fully Convolutional 3D Network that inputs anisotropic timepoint blocks      an, Args:             x (torch.Tensor): Volume tensor of shape (B, 1, Z, Y, X), Natively maps 3D coordinate tensors from anisotropic voxel space (Z, Y, X), STACTCentroidPredictor

### Community 1 - "Hypergraph Lineage Tracker"
Cohesion: 0.22
Nodes (5): Mitosis Backward-Smoothing (Temporal Window Align):         Backtracks division, Anisotropic Velocity Edge Pruning: Inspects coordinates and discards          c, Constructs and solves ILP for cell centroids.         Supports multi-frame look, Spatio-Temporal Hypergraph Lineage Solver (Grandmaster Tier).     Models tracki, STHypergraphTracker

### Community 2 - "Pipeline Modules Overview"
Cohesion: 0.22
Nodes (9): Anisotropic Zarr Loader, Hyperparameters Configuration, Anisotropic Coordinate Transformer, ST-ACT Centroid Predictor, Ensemble Consensus Centroids, Extract Peaks From Volume, Run ST-ACT Pipeline, Run System Diagnostics (+1 more)

### Community 3 - "Peak Detection & Consensus Clustering"
Cohesion: 0.47
Nodes (5): ensemble_consensus_centroids(), extract_peaks_from_volume(), Ensemble Consensus Centroid Clustering (DBSCAN):     Applies spatial density-ba, Simulates CNN/U-Net heatmap thresholding and peak local max finding.     Return, run_st_act_pipeline()

### Community 4 - "Anisotropic Zarr Data Loader"
Cohesion: 0.40
Nodes (3): AnisotropicZarrLoader, ST-ACT Memory-Safe 4D Anisotropic Zarr v3 Ingestor.     Loads and decompresses, Loads and decompresses a single 3D timepoint volume (Z, Y, X) into memory.

### Community 5 - "System Diagnostics"
Cohesion: 0.50
Nodes (4): Colors, get_ram_info(), Retrieves system RAM size in GB. Uses psutil if installed,     falls back to Wi, run_diagnostics()

## Knowledge Gaps
- **7 isolated node(s):** `Colors`, `Ensemble Consensus Centroids`, `Extract Peaks From Volume`, `ST-Hypergraph Tracker`, `Anisotropic Coordinate Transformer` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AnisotropicZarrLoader` connect `Anisotropic Zarr Data Loader` to `Peak Detection & Consensus Clustering`, `Zarr Store Initialization`, `Simulated Zarr Store Generator`, `Dataset Shape Accessor`, `Chunked Volume Streaming`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `run_st_act_pipeline()` connect `Peak Detection & Consensus Clustering` to `Hypergraph Lineage Tracker`, `Anisotropic Zarr Data Loader`?**
  _High betweenness centrality (0.192) - this node is a cross-community bridge._
- **Why does `STHypergraphTracker` connect `Hypergraph Lineage Tracker` to `Peak Detection & Consensus Clustering`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Run ST-ACT Pipeline` (e.g. with `ST-ACT Centroid Predictor` and `Hyperparameters Configuration`) actually correct?**
  _`Run ST-ACT Pipeline` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_st_act_pipeline()` (e.g. with `AnisotropicZarrLoader` and `STHypergraphTracker`) actually correct?**
  _`run_st_act_pipeline()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Ensemble Consensus Centroid Clustering (DBSCAN):     Applies spatial density-ba`, `Simulates CNN/U-Net heatmap thresholding and peak local max finding.     Return`, `Colors` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._