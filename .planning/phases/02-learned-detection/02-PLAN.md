---
wave: 2
depends_on:
  - 01-PLAN.md (Wave 1 complete)
files_modified:
  - src/model.py (new/major refactor)
  - src/targets.py (new)
  - run_pipeline.py (updated for inference)
autonomous: true
---

# Phase 2 Wave 2: Model Architecture & Target Generation

## Summary
Wave 2 implements the full host-style learned detection architecture and target generation pipeline. Two parallel tracks:

**Track A (Model):** Implement 3D UNet + SimpleNodeTransformer following REFERENCE_IMPLEMENTATION.md S5 and RESEARCH.md S1.1–1.2 exactly. Hyperparameters locked from host's documented values.

**Track B (Targets):** Generate heatmap (detection) and edge probability targets from .geff ground truth. Decide empirically between point vs. dilated Gaussian heatmaps, implement division loss weighting, scaffold test-time augmentation.

Exit criterion: Model architecture matches host paradigm, target generation produces correct per-voxel heatmaps + edge labels, both verified on a subset of real competition data.

## Must-Haves (Goal-Backward Verification)
- [ ] 3D UNet backbone produces per-voxel detection logits (1x1 Conv3d final layer)
- [ ] SimpleNodeTransformer produces pairwise edge probabilities between consecutive-frame node sets
- [ ] Heatmap targets generated from .geff centroids (point or dilated, empirically benchmarked)
- [ ] Edge targets correctly label all GT edges as 1, non-GT pairs as 0
- [ ] Division loss weighting upweights edges where parent has >1 children
- [ ] Test-time augmentation (4-view averaging) scaffolded and testable
- [ ] Greedy edge assignment (not ILP at inference) implemented and testable
- [ ] All components verified end-to-end on 10 real competition samples

---

## Tasks

### Task 2.1: Implement 3D UNet Backbone
**Depends:** Wave 1 complete  
**Owner:** Executor  
**Autonomous:** true

Build 3D UNet detection head following RESEARCH.md S1.1–1.2 (host's exact hyperparameters).

**Architecture (from REFERENCE_IMPLEMENTATION.md, verified exact):**
- Input: (T=2, Z=64, Y=256, X=256) — two consecutive frames
- Channels: [32, 64, 128] (from host's documented values)
- Downsample strides: (1, 4, 4) — Z untouched, Y/X downsampled 4x
- Backbone: standard 3D UNet with skip connections
- Output: per-voxel binary logits via 1x1 Conv3d final layer → shape (1, 64, 256, 256) → sigmoid → [0,1]

**Execution Steps:**
1. Create src/model.py with class UNet3D(nn.Module):
   - Constructor with configurable channel list (default [32, 64, 128])
   - forward(x) → logits, features
   - Returns both detection logits (for BCE loss) and feature maps (for transformer input)
   
2. Implement down/up blocks respecting anisotropic stride (1,4,4):
   - Conv3d with kernel (1,3,3) or (3,3,3) depending on layer
   - AvgPool3d / Upsample with strides (1,4,4)

3. Verify input/output shapes:
   - Input: (B, 1, 2, 64, 256, 256) [batch, channels=1, time=2, z, y, x]
   - Output: (B, 1, 64, 256, 256) [per-voxel logits, no Z down]
   - Features: (B, 128, 64, 256, 256) [for transformer, dense representations at full resolution]

4. Test on real competition data:
   - Load 3 samples from src/dataset.py
   - Forward pass with dummy input (B=2, shape (2,1,2,64,256,256))
   - Check output shapes and dtype (logits should be float32 [0,1], features float32)

5. Document architecture in model docstring with exact channel/stride values

**Verification:**
- [ ] src/model.py contains UNet3D class
- [ ] forward() produces logits (B, 1, 64, 256, 256) and features (B, 128, 64, 256, 256)
- [ ] No Z-axis downsampling (all Z operations have stride 1)
- [ ] Forward pass on real data produces valid float32 tensors
- [ ] Architecture matches host hyperparameters: [32, 64, 128] channels, (1,4,4) strides

---

### Task 2.2: Implement SimpleNodeTransformer (Edge Prediction)
**Depends:** Task 2.1  
**Owner:** Executor  
**Autonomous:** true

Build cross-attention Transformer predicting pairwise edge probabilities between detected nodes in consecutive frames (following RESEARCH.md S1.1–1.2, host's exact hyperparameters).

**Architecture (from REFERENCE_IMPLEMENTATION.md, verified exact):**
- Inputs: detected node sets from frame_t and frame_t+1 (variable cardinality, dynamically detected)
- Transformer hidden dim: 128
- Attention heads: 4
- Transformer blocks: 4
- Dropout: 0.3
- Output: pairwise edge probabilities (1 per candidate edge)
- Greedy assignment: sort by probability descending, accept respecting cardinality (1 parent, 2 children per node), halt when exhausted

**Execution Steps:**
1. Create src/model.py with class SimpleNodeTransformer(nn.Module):
   - Constructor: hidden_dim=128, num_heads=4, num_blocks=4, dropout=0.3
   - forward(nodes_t, nodes_t1, features_t, features_t1) → edge_probabilities
   - Nodes input: (n_t, 3) [z, y, x centroids] and (n_t1, 3)
   - Features input: extracted from UNet at node positions
   - Output: (num_candidate_edges,) with probabilities [0,1]

2. Implement cross-attention architecture:
   - Embed nodes with sinusoidal positional encodings
   - Cross-attention between source (frame_t) and target (frame_t+1) node sets
   - Feed-forward layers with given hidden_dim
   - Repeat for num_blocks layers

3. Candidate edge generation:
   - All (node_t, node_t1) pairs within spatial distance threshold (~3-5 frames of motion expected)
   - Return sorted candidate list for greedy assignment downstream

4. Test on real data:
   - Load 3 competition samples with GT nodes extracted
   - Forward pass with dummy nodes (n_t=10, n_t1=15)
   - Check output shape (num_candidates,) with probabilities [0,1]

5. Document architecture and candidate-generation strategy in model docstring

**Verification:**
- [ ] SimpleNodeTransformer class exists with correct hyperparameters
- [ ] forward() accepts nodes_t, nodes_t1, features_t, features_t1
- [ ] Returns edge probabilities as (num_candidates,) tensor [0,1]
- [ ] Sinusoidal positional encoding applied
- [ ] Cross-attention implemented (not self-attention)
- [ ] Test on real data produces valid probability outputs

---

### Task 2.3: Benchmark & Implement Heatmap Target Generation
**Depends:** Wave 1 complete (data split)  
**Owner:** Executor  
**Autonomous:** true

**Dependency clarification (resolves ambiguity found during plan verification):** the
"tiny UNet" used for benchmarking below is a **separate, throwaway 2-3 layer model built
inline within this task**, NOT Task 2.1's real `UNet3D`. This task does not depend on
Task 2.1 and can run in parallel with Track A (Tasks 2.1/2.2), matching this wave's
stated Track A/B parallelism. Do not import or wait for `UNet3D` here.

Resolve RESEARCH.md S4's second empirical decision point: point vs. dilated Gaussian heatmap targets.

**Decision Rule:** Whichever approach measurably improves local evaluation score on a held-out benchmark set wins. Test both on 10 held-out samples, measure edge_jaccard + adjusted_edge_jaccard, document and proceed with winner.

**Option A (Point targets):** Single voxel per centroid, sparse (~0.1% positive voxels).  
**Option B (Dilated Gaussian):** Anisotropic Gaussian around centroid (sigma_z=1.0, sigma_yx=2.0 voxels suggested), softer targets (~1-2% positive).

**Execution Steps:**
1. Create src/targets.py with function generate_heatmap_targets(sample_id, geff_path, zarr_volume, volume_shape, anisotropy, target_type='gaussian'):
   - Load .geff ground truth via tracksdata.graph.IndexedRXGraph.from_geff()
   - For each GT centroid (x, y, z, t) in each frame:
     - **Option A:** Set single voxel [z, y, x, t] = 1.0
     - **Option B:** Create anisotropic Gaussian: sigma=(1.0, 2.0, 2.0) for (z, y, x), normalize to [0,1]
   - Return heatmap of same shape as volume (1, Z, Y, X) for each frame

2. Implement empirical benchmark on held-out set:
   - Select 10 random samples from validation split (data_split.json)
   - For each sample:
     - Generate both Option A and Option B heatmaps
     - Train the throwaway tiny UNet (2-3 layers, defined inline in this task,
       not Task 2.1's UNet3D -- see dependency clarification above) for 2-3
       epochs on each
     - Evaluate on the same held-out set using src/evaluation.py's local metrics
     - Log edge_jaccard + adjusted_edge_jaccard for both options
   - Calculate mean scores across 10 samples for each option
   - Pick winner (higher adjusted_edge_jaccard)

3. Document decision in 02-SUMMARY.md:
   - Option A mean score: X
   - Option B mean score: Y
   - Winner and reasoning

4. Implement full heatmap generation:
   - Create src/targets.py function generate_heatmap_targets(sample_id, geff_path, volume_shape, anisotropy, target_type=WINNER)
   - Also include GT metadata extraction: num_divisions, edge counts per timepoint, etc.

5. Test on 3 real samples (44b6 and 6bba prefix each):
   - Verify heatmap shapes match volume
   - Verify heatmap values in [0,1]
   - Spot-check: visualize heatmap overlay on raw volume for one sample (qualitative check that centroids align)

**Verification:**
- [ ] generate_heatmap_targets() produces (1, Z, Y, X) heatmaps with values [0,1]
- [ ] Empirical benchmark completed on 10 held-out samples
- [ ] Option A and Option B scores documented in 02-SUMMARY.md
- [ ] Winner chosen with clear reasoning
- [ ] Heatmap targets spot-checked on 3 real samples for centroid alignment
- [ ] target_type parameter locked to winner for all downstream training

---

### Task 2.4: Implement Edge Probability Target Generation & Division Loss
**Depends:** Wave 1 complete (data split)  
**Owner:** Executor  
**Autonomous:** true

Generate edge probability targets from .geff ground truth and implement division loss weighting.

**Execution Steps:**
1. Create src/targets.py function generate_edge_targets(sample_id, geff_path, frame_t_nodes, frame_t1_nodes, num_candidate_edges):
   - For each candidate edge (node_t → node_t1):
     - Label = 1 if GT edge exists between these nodes (from .geff), 0 otherwise
   - Return edge labels as (num_candidate_edges,) binary tensor
   - Handle class imbalance (likely <<1% positive edges) with inverse-frequency weighting

2. Implement division event detection:
   - For each GT node with >1 outgoing edges → mark as division event
   - Generate edge_division_mask: (num_candidate_edges,) boolean indicating which edges are division-related

3. Add division loss weighting function:
   - Implement weighted BCE loss: edge_loss = bce(logits, targets) * edge_loss_weights
   - edge_loss_weights = 1.0 for normal edges, weight_division (suggested 2.0-3.0x) for division edges
   - Allow weight_division to be tuned (add as hyperparameter in training config)

4. Integrate with detection loss:
   - Heatmap detection loss: BCE with inverse-frequency weighting (weight_pos=1/n_pos, weight_neg=0.01/n_neg)
   - Combined loss: total_loss = edge_loss + det_loss_weight(1.0) * det_loss
   - Allow both weights to be configurable hyperparameters

5. Test on 5 real samples:
   - Generate edge targets for all (t, t+1) pairs in sample
   - Verify edge labels are binary (0/1)
   - Spot-check: verify at least one positive edge per sample (sanity check)
   - Measure class imbalance ratio (positive/total edges)

**Verification:**
- [ ] generate_edge_targets() produces binary (num_candidates,) tensor
- [ ] Division edges correctly identified
- [ ] Edge loss weighting applied (division edges get weight_division)
- [ ] Combined loss (edge + detection) implemented
- [ ] Edge class imbalance ratios documented (expect <1% positive)
- [ ] All 5 test samples have ≥1 positive edge

---

### Task 2.5: Implement Test-Time Augmentation (TTA) Pipeline
**Depends:** Task 2.1 (UNet)  
**Owner:** Executor  
**Autonomous:** true

Scaffold test-time augmentation: average detection logits across 4 views (original, flip Y, flip X, flip Y+X) before peak detection. Per CONTEXT.md, include TTA from the start (not deferred).

**Execution Steps:**
1. Create inference utility in src/inference.py:
   - Function tta_inference(model, frame_t, frame_t1, views=['original', 'flip_y', 'flip_x', 'flip_yx']):
     - For each view: apply flip transformations to input
     - Forward pass through UNet → get logits
     - Reverse flips on output logits to match original coordinate system
     - Average logits across all 4 views
     - Return averaged logits (B, 1, Z, Y, X)

2. Implement flip transformations:
   - Flip Y: torch.flip(x, dims=[2]) (flip Y axis)
   - Flip X: torch.flip(x, dims=[3]) (flip X axis)
   - Flip YX: torch.flip(x, dims=[2, 3])
   - Reverse transformations post-inference

3. Integrate with NMS peak detection:
   - TTA inference → averaged logits → NMS peak extraction (reuse Phase 1's pool_kernel_from_um + extract_peaks_from_volume from run_pipeline.py)
   - Verify peaks extracted at same positions regardless of view

4. Test on 3 real samples:
   - Run TTA inference, verify 4 intermediate logits + averaged logit
   - Check that averaged logits are smooth/symmetric (sanity check)
   - Verify peak positions consistent across individual views + averaged

**Verification:**
- [ ] tta_inference() implemented in src/inference.py
- [ ] 4 views processed correctly (flips applied/reversed)
- [ ] Averaged logits have same shape as single-pass logits
- [ ] TTA test on 3 samples shows consistent peak positions
- [ ] Average logits are symmetric (averaged flipped view ≈ original view)

---

### Task 2.6: Implement Greedy Edge Assignment
**Depends:** Task 2.2 (Transformer)  
**Owner:** Executor  
**Autonomous:** true

Implement greedy edge assignment (replaces ILP at inference, per CONTEXT.md's locked decision to use greedy + keep ILP as fallback).

**Execution Steps:**
1. Create src/inference.py function greedy_edge_assignment(edge_probs, nodes_t, nodes_t1, candidate_edges, max_children=2, max_parents=1):
   - Input: edge probabilities from Transformer, node sets, candidate edge list
   - Sort candidate edges by probability descending
   - Greedily accept edges respecting cardinality:
     - Each node in frame_t can have ≤max_parents (=1) incoming edge
     - Each node in frame_t can have ≤max_children (=2) outgoing edges (to account for divisions)
   - Output: directed graph (node_t → node_t1 edges)

2. Handle edge cases:
   - Empty node sets (no detections in a frame) → return empty graph
   - All edges below threshold → return empty graph
   - Nodes with >max_children predictions → accept top N by probability

3. Build output graph structure:
   - Return dict: {(node_t_id, node_t1_id): probability, ...}
   - Or return NetworkX DiGraph for compatibility with existing graph infrastructure

4. Test on real data:
   - Generate 10 samples with predicted edge probs + nodes
   - Run greedy assignment, verify output respects cardinality constraints
   - Compare to Phase 1's ILP tracker on same sample (sanity check: greedy should produce ≥90% of ILP's edges for high-prob predictions)

5. Document cardinality constraints in function docstring

**Verification:**
- [ ] greedy_edge_assignment() implemented
- [ ] Output respects max_parents=1 and max_children=2 constraints
- [ ] Greedy output on 10 test samples: ≥90% edge overlap with ILP baseline (sanity check)
- [ ] Empty input handling tested (no detections, all low-prob edges)

---

## Verification Criteria

All tasks must pass before proceeding to Wave 3:

- [ ] **Model architecture complete:** UNet3D produces logits (B,1,64,256,256), SimpleNodeTransformer produces edge probs
- [ ] **Model verified on real data:** Forward passes on 10 competition samples without error
- [ ] **Heatmap targets empirically benchmarked:** Option A vs Option B tested, winner documented with scores
- [ ] **Edge targets & division loss:** Generated and tested on 5 samples, class imbalance quantified
- [ ] **TTA implemented & tested:** Averaged logits symmetric, consistent peaks across views
- [ ] **Greedy assignment working:** Output respects cardinality, ≥90% overlap with ILP baseline on test samples

---

## Output Artifacts
- `02-SUMMARY.md` — Heatmap benchmark results (Option A vs B scores), division loss weight, architecture notes
- `src/model.py` — UNet3D, SimpleNodeTransformer classes
- `src/targets.py` — generate_heatmap_targets(), generate_edge_targets() functions
- `src/inference.py` — tta_inference(), greedy_edge_assignment() functions

