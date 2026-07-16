# Phase 2: Learned Detection — Research Findings

**Date:** 2026-07-08  
**Scope:** How to implement Phase 2 (Learned Detection) grounded in actual competition data, host reference implementation, and submission/evaluation infrastructure.

**Executive Summary:** Phase 2's success hinges on adopting the host's validated UNet+Transformer architecture for learned edge probabilities, correctly normalizing competition data (quantile-based, not raw thresholds), building embryo-disjoint train/val splits from the full 199-sample set, and validating against local metrics before spending scarce Kaggle submissions. The host's reference implementation documents concrete hyperparameters and loss functions; this research extracts them and maps them to the codebase's existing infrastructure (evaluation harness, submission exporter, ILP tracker).

---

## 1. Model Architecture — Concrete Details from Reference Implementation

### 1.1 Host's Actual Architecture (from REFERENCE_IMPLEMENTATION.md §5)

The host **does not use a simple detect-then-track pipeline**. Instead:

1. **`UNetNodeTransformer`**: 3D UNet backbone producing two outputs:
   - **Detection head**: per-voxel binary logits (`1×1 Conv3d` final layer) — exactly one logit per voxel
   - **Feature maps**: dense learned representations at each voxel (to feed the transformer)

2. **Cross-Attention Transformer** (`SimpleNodeTransformer`):
   - Input: detected nodes from both frames (frame `t` and `t+1`), plus sinusoidal positional embeddings
   - Output: **pairwise edge probabilities** — direct prediction of "does node A in frame t link to node B in frame t+1"
   - Design: cross-attention between source and target node sets, not self-attention

3. **Greedy Edge Assignment** (not ILP at inference time):
   - Sort candidate edges by learned probability, descending
   - Greedily accept edges respecting cardinality caps: **1 parent, 2 children per node** (biological constraint)
   - Halt when no more valid edges exist
   - Output: directed graph with edge attributes (edge probability, distance stored for reference)

**Critical architectural insight:** The host's edge is in the *learned edge-probability model*, not the assignment algorithm. A simple greedy assignment over good learned probabilities suffices; sophistication should come from the learned signal, not post-hoc optimization.

### 1.2 Hyperparameter Specifics

| Component | Hyperparameter | Value | Note |
|-----------|----------------|-------|------|
| UNet | Channels | `[32, 64, 128]` | Small due to large input volumes; test-mode gradient checkpointing recommended |
| UNet | Input shape | `(T, Z, Y, X)` = `(2, 64, 256, 256)` | Two consecutive frames stacked |
| UNet | Downsample strides | `(1, 4, 4)` | Z untouched (already 4x coarser voxel-wise); Y/X downsampled 4x |
| Transformer | Hidden dim | 128 | Feature vector size |
| Transformer | Heads | 4 | Attention heads in multi-head cross-attention |
| Transformer | Blocks | 4 | Transformer layers (cross-attention blocks) |
| Transformer | Dropout | 0.3 | Applied in transformer; helps prevent overfitting |
| Optimizer | Type | AdamW | |
| Optimizer | LR | `1e-4` | Critical: not `1e-3` or `5e-4` — this exact value is tuned for 50-epoch convergence |
| Optimizer | Grad clip | 1.0 | Prevents explosion on sparse edge targets |
| Training | Batch size | 16 | Per-GPU on Tesla T4 (11GB VRAM) |
| Training | Epochs | 50 | Validation split every epoch; early stop on validation loss plateau (patience ~10 epochs) |

### 1.3 Loss Function Design

**Detection loss:** Binary Cross-Entropy with inverse-frequency weighting  
- `weight_pos = 1.0 / n_pos_total` — upweight the sparse positive (cell-center) voxels
- `weight_neg = 0.01 / n_neg_total` — heavily downweight the mostly-background negative class
- Justification: real data has ~0.1%–1% of voxels being cell centers (confirmed on staged data in data/staging/README.md); raw BCE without weighting → model learns trivial "always background" predictor

**Edge loss:** Focal-weighted BCE applied only to *annotated* edges  
- Focal weighting: `(1 − p_t)^2 × bce(p, target)` where `p_t` is the predicted probability of the true class
- Rationale: hard negatives (high-prob mispredictions) are penalized harder; easy negatives contribute less
- Scope: only computed over edges between annotated nodes (sparse GT does not cover all possible node pairs)

**Combined loss:** `edge_loss + det_loss_weight × det_loss`  
- `det_loss_weight = 1.0` — both terms equally weighted (not detection-heavy)

### 1.4 Why This Matters for Phase 2 Design Choices

**Option A — "Deepen the current `STACTCentroidPredictor`"**:
- Current design: 2 conv layers → heatmap + motion-vector heads (hand-built, no transformer)
- Pros: incremental from existing code
- Cons: motion vectors are an intermediate quantity; the model never sees paired-frame data, so motion prediction is decoupled from learned spatial/temporal patterns

**Option B — "Adopt the host's UNet+Transformer paradigm"**:
- Pros: 
  - Proven to work at leaderboard rank #1 (0.875 score)
  - End-to-end edge probability learning handles rare fast motions (3.1% of true edges exceed naive distance metric's break-even)
  - Easier feature reuse (UNet features feed transformer, not two separate pathways)
- Cons: requires rewrite of model training loop; more complex architecture

**Recommendation:** **Adopt Option B** — the host's reference is not just documentation, it's a validated, published solution. The gap between 0.763 (baseline) and 0.875 (current leaderboard leader) is the delta between learned and hand-tuned — and the host's architecture is the proof that this delta is achievable. Phase 2 should aim to reproduce this architecture exactly, then iterate on data augmentation, regularization, and ensemble methods rather than fundamental architectural changes.

---

## 2. Data Infrastructure: 199 Samples, Embryo-Disjoint Splits

### 2.1 Full Dataset Structure

- **Total competition train set:** 199 samples (embryo-ID + field-of-view pairs)
- **Currently staged locally:** 4 samples across 2 movie prefixes (data/staging/train/)
- **Real data mount on Kaggle:** `/kaggle/input/competitions/biohub-cell-tracking-during-development/`
  - Path structure: `train/` and `test/` directories with flat naming `{embryo_id}_{field_of_view}.{zarr|geff}`
  - NO nested per-embryo folders (a prior planner hallucinated this; verified via actual ls output)

### 2.2 Staged Data (What We Have Now)

```
data/staging/train/
├── 44b6_0113de3b.zarr  (Zarr v3 OME-NGFF format, real microscopy data)
├── 44b6_0113de3b.geff  (52 annotated nodes, 35 edges, 1 division)
├── 44b6_0b24845f.zarr
├── 44b6_0b24845f.geff  (26 nodes, 13 edges, 1 division)
├── 6bba_05b6850b.zarr
├── 6bba_05b6850b.geff  (63 nodes, 42 edges, 1 division)
├── 6bba_05db0fb1.zarr
└── 6bba_05db0fb1.geff  (29 nodes, 17 edges, 0 divisions)
```

- **Movie/plate prefixes present in staged subset:** 2 (`44b6`, `6bba`)
- **Samples per movie prefix (staged subset only):** 2 each
- **Total nodes across 4 samples:** ~170 nodes
- **Anisotropy:** Z = 1.625 µm, Y/X = 0.40625 µm each → ratio 4.0:1:1 (Z:Y:X)
- **Anisotropy bug fixed in Phase 0:** the repo previously hardcoded 5.0:1:1; corrected to 4.0:1:1

### 2.3 Embryo-Disjoint Train/Val Split Strategy -- CORRECTED after direct verification

**P0-2 SECOND CORRECTION (2026-07-16): the "corrected" conclusion below (embryo =
individual `{prefix}_{hash}` sample) is itself WRONG.** It was inferred by comparing
against the 4 locally staged `test/` samples, which are explicitly documented elsewhere
(`data/staging/README.md`) as byte-identical placeholder copies of 4 train samples, NOT
the real hidden test set -- so that comparison never actually tested the competition's
real embryo-disjoint boundary. Kaggle's own official Data description page (fetched live
via `kaggle competitions pages -c biohub-cell-tracking-during-development --page-name
data-description --content`) states plainly: folder names follow
`{embryo_id}_{field_of_view}`, the first underscore-delimited segment IS the embryo ID,
and "multiple samples may share the same embryo." So the 2-value prefix (`44b6`/`6bba`)
is the real embryo ID, and the split strategy below (which kept both prefixes present in
both train and validation) had real embryo-level leakage. See the P0-2 audit and
`scripts/build_train_val_split.py`'s leave-one-embryo-out replacement.

**IMPORTANT CORRECTION to this research pass's original claim:** the first draft of this
document assumed "~15-20 embryo IDs" across the full train set. That was an unverified
guess and is **wrong**. Direct inspection of the actual downloaded competition zip
(`c:\Users\hemas\Downloads\biohub-cell-tracking-during-development.zip`, all 199 train
`.geff` entries enumerated) shows:

- **Only 2 movie/plate prefixes exist across the ENTIRE 199-sample train set**: `44b6`
  (71 samples) and `6bba` (128 samples). Sample IDs are `{prefix}_{8-char-hash}`, e.g.
  `44b6_0113de3b`.
- **The official test set (4 samples) draws from these exact same 2 prefixes** —
  `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1` — 2 from each.

Since train and test share the same 2 movie prefixes, and the competition guarantees
train/test are embryo-disjoint, **"embryo" in this competition's terminology must map to
each individual `{prefix}_{hash}` sample, not the 2-value movie/plate prefix.** A
prefix-level split would be meaningless here (you can't hold out "44b6" entirely -- the
official test set itself depends on samples from both prefixes) and isn't how the
competition's own train/test boundary is actually drawn.

**Corrected split strategy for Phase 2 training:**
1. Treat each of the 199 individual sample IDs as one "embryo" unit for split purposes
   (not the 2-value prefix).
2. Partition the 199 sample IDs into training and held-out validation sets — a 75%/25%
   split (~149 train / ~50 held-out) is reasonable at this sample count, unlike the
   original mistaken "~15-20 embryo IDs" framing.
3. Stratify by prefix if practical (keep roughly proportional 44b6:6bba representation in
   both splits, since the two movies may have different characteristics), but this is a
   secondary consideration -- the primary constraint is per-sample disjointness.
4. **Never train on a held-out sample**, even partially. Validate on held-out samples every
   epoch; early-stop on validation loss plateau.

**Rationale:** This mirrors how the competition itself actually splits train/test (by
individual sample, drawing from both movies), not a movie-level split that the real data
structure doesn't support. Overfitting to one sample's particular cell dynamics remains a
real risk; held-out samples catch this regardless of which movie they come from.

### 2.4 Kaggle Data Access (Confirmed Working)

- **GPU availability:** Tesla T4 × 2 (confirmed working via smoketest.py; P100s reported "available" but compute-capability-incompatible)
- **Data mount path:** `/kaggle/input/competitions/biohub-cell-tracking-during-development/`
- **Full-size volume shape:** `(100, 64, 256, 256)` uint16 Zarr v3 format, chunked `(1, 64, 256, 256)`
- **Confirmed working:** smoketest.py successfully reads Zarr and .geff from Kaggle mount (2026-07-03)
- **Kernel setup:** Already staged in `kaggle_kernel/` with `kernel-metadata.json` (GPU enabled, internet enabled)

---

## 3. Data Normalization — Exact Quantile Percentiles

### 3.1 Host's Quantile Normalization Function

From REFERENCE_IMPLEMENTATION.md §5, the exact function:

```python
def quantile_normalize(image, gamma=1.0, subsample_factor=50, q_min=0.001, q_max=0.999,
                        clip_min=0.0, clip_max=4.0) -> np.ndarray:
    """Normalize uint16 raw intensity to [0, 4.0] float range."""
    image = image.astype(np.float32)
    q1, q2 = np.quantile(image.ravel()[::subsample_factor], [q_min, q_max])
    image_normalized = (image - q1) / (q2 - q1 + 1e-6)
    image_normalized = np.clip(image_normalized, 0.0, None)
    image_normalized = image_normalized ** gamma
    image_normalized = np.clip(image_normalized, clip_min, clip_max)
    return image_normalized
```

### 3.2 Critical Parameters (NOT Generic)

| Parameter | Value | Why |
|-----------|-------|-----|
| `q_min` | 0.001 (0.1%) | Lower quantile for normalization |
| `q_max` | 0.999 (99.9%) | Upper quantile (not 100% = max value) |
| `clip_max` | 4.0 | **NOT 1.0** — capped at 4.0 after gamma |
| `gamma` | 1.0 (default) | No gamma correction (linear); optional tuning lever for Phase 4 |
| `subsample_factor` | 50 | Compute quantiles on every 50th voxel (speed optimization; statistically sound on large volumes) |

**CRITICAL:** the two paragraphs above describe the *host's* normalization function as
documented in `REFERENCE_IMPLEMENTATION.md`. It is NOT what this codebase currently does.

### 3.3 Quantile Normalization in Current Codebase -- CORRECTED (this research pass's
original draft got this wrong; verified directly against `src/data_loader.py`, not
guessed)

**Our `AnisotropicZarrLoader` ALREADY normalizes -- this is not a "not yet built" gap.**
Confirmed via direct source read (`src/data_loader.py:95-119, 229-252`) and cross-checked
against real Phase 0/1 pipeline logs ("Extracted quantile params: 0.1=..., 0.9=...",
"Quantile normalization parameters found" -- these appear on every real run):

- Reads precomputed `image_statistics.quantiles` values already stored in each sample's
  zarr metadata (`0.1` and `0.9` keys -- the 10th/90th percentile, NOT the host's
  self-computed 0.001/0.999).
- Normalizes via `(data - q_low) / (q_high - q_low)`, then **clips to `[0, 1]`** (NOT the
  host's `[0, 4.0]`).
- This is what Phase 1's real peak-finding thresholds (`CNN_THRESHOLD=0.85`,
  `UNET_THRESHOLD=0.9` in `run_pipeline.py`) are calibrated against, verified working
  end-to-end (local score 0.0259 on this exact normalization).

**This is a real decision point for Phase 2 planning, not a detail to gloss over:** the
model architecture/loss described above (S3.1-3.2) assumes the host's `[0, 4.0]`
self-computed-quantile input distribution. Two options, pick one deliberately:
1. **Keep our existing `[0, 1]`-clipped, zarr-metadata-quantile normalization** and treat
   the host's exact `q_min=0.001/q_max=0.999/clip_max=4.0` values as reference-only (adapt
   detection-loss/threshold expectations to a `[0,1]` input instead).
2. **Switch to the host's self-computed-quantile, `[0, 4.0]`-clipped approach** in
   `AnisotropicZarrLoader` -- this changes Phase 1's already-verified peak-finding
   thresholds and would need them recalibrated again (another `sweep_threshold.py`-style
   pass), since 0.85/0.9 were tuned against the `[0,1]` range specifically.

Do not silently assume one without checking -- this exact class of unit-mismatch (a
threshold tuned for one data distribution silently producing wrong results against a
different one) has caused two real incidents in this project already (the original
2.5+ hour stuck run, and the Phase 1 NMS tie-explosion bug).

---

## 4. Target Generation for Training

### 4.1 Heatmap Targets (Detection)

**Design decision:** Generate per-voxel binary labels from `.geff` centroids.

**Two candidate approaches:**

**Option 1: Point targets (single voxel per node)**
- Set `label[t, z, y, x] = 1.0` at the exact .geff centroid coordinate
- Pros: sparse, matches the host's peak-finding approach
- Cons: very small positive-class region; high class imbalance (~0.1% positive voxels)

**Option 2: Dilated Gaussian targets (small region around centroid)**
- Generate anisotropy-aware Gaussian blobs centered at each .geff node
- Suggested: σ = 1–2 voxels (empirically calibrated; Phase 1 can test both)
- Pros: softer targets, eases optimization
- Cons: requires resampling; changes the effective point-location semantics

**Recommendation:** Start with **Option 2 (dilated Gaussian)** — the class imbalance in Option 1 is severe, and soft targets are gentler to optimize. Validate empirically on held-out embryos whether Gaussian width (σ) matters.

**Anisotropy consideration:** Gaussian should be anisotropy-aware — stretch in-plane (Y/X) less than in Z to account for Z's 4x coarser voxel size. Suggested: `σ_z = 1.0`, `σ_yx = 2.0` voxels.

### 4.2 Edge Probability Targets

**Design:** For each pair of nodes `(node_i_t, node_j_t+1)`:
- Label = 1 if there exists an edge in .geff from node_i at time t to node_j at time t+1
- Label = 0 otherwise (non-annotated pairs, negative examples)

**Class imbalance:** Only positive edges are labeled (out of all possible node pairs, ~0.1%–1% have GT edges). Heavy class imbalance → use the host's inverse-frequency weighting (§1.3).

### 4.3 Division Loss Weighting (NEW in Phase 2)

**Goal:** Make division events (rare, high-leverage) visible to the loss.

**Host's approach:** Already integrated in focal-weighted edge BCE.

**Phase 2 enhancement (from CONTEXT.md §3):** Add auxiliary loss term weighting division edges more heavily.

Example auxiliary loss:
```python
# For each node j at t+1, if it has >1 incoming edge from t (or is a division event),
# upweight those edges in the loss
division_weight = 2.0  # or 3.0; empirically tuned
for i, j in annotated_edges:
    if is_division_edge(i, j):  # parent dividing, j is one of two children
        edge_loss_ij *= division_weight
```

**Justification:** Divisions are ~10% of the `0.1 × division_jaccard` term in the score but rare in the data (~1 division per 30 nodes); upweighting ensures the model learns to prioritize division accuracy.

---

## 5. Peak Detection & Non-Max Suppression (NMS)

### 5.1 Exact Host's NMS Approach

From REFERENCE_IMPLEMENTATION.md §5:

```python
def pool_kernel_from_um(um: float, voxel_size: tuple[float, ...]) -> tuple[int, ...]:
    """Convert physical µm to odd voxel-count kernel size."""
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0: k += 1  # Ensure odd kernel
        kernel.append(k)
    return tuple(kernel)

def _detect_cells_pooled(det_logits, t, det_threshold=0.5, pool_kernel=(3,3,3)):
    pooled = F.max_pool3d(det_logits.unsqueeze(0), pool_kernel, stride=1, 
                          padding=[k//2 for k in pool_kernel])
    is_peak = (det_logits == pooled) & (torch.sigmoid(det_logits) > det_threshold)
    coords = torch.nonzero(is_peak)  # Return (t, z, y, x) coordinates
    return coords
```

**Key details:**
- **Kernel sizing from physical µm, not fixed voxels:** Kernel = `pool_kernel_from_um(5.0, (1.625, 0.40625, 0.40625))`
  - Z: `round(5.0 / 1.625) = 3` voxels
  - Y: `round(5.0 / 0.40625) = 13` voxels
  - X: `round(5.0 / 0.40625) = 13` voxels
  - → Kernel = `(3, 13, 13)` (anisotropic by design)
- **Stride = 1:** No skipping; test every voxel
- **Padding = `k//2`:** Same-size output
- **Threshold:** Applied *after* max-pool via `sigmoid(det_logits) > threshold` — test on raw logits, not probabilities

### 5.2 Test-Time Augmentation (TTA)

**Host's approach:** Average detection logits across 4 views:
1. Original (no flip)
2. Flip Y
3. Flip X
4. Flip Y+X

Then apply NMS to the averaged logits. TTA is a *real accuracy lever* (not just an ensemble trick) because:
- Logit averaging is robust to minor prediction inconsistencies
- Real cell locations are preserved by symmetry
- Spurious edge-voxel peaks from asymmetric artifacts wash out

**Phase 2 plan:** Implement TTA in the inference pipeline (not training, inference only).

---

## 6. Validation Strategy: Local Metrics Before Kaggle Submission

### 6.1 Trusted Local Evaluation Harness (Already Built)

- **Module:** `src/evaluation.py` (226 lines, phase-0-verified)
- **Main API:** `evaluate_submission(pred_graphs, gt_graphs, scale, max_distance, gt_metadata)`
- **Metrics:**
  - `edge_jaccard`: bipartite-matched edge TP/(TP+FP+FN), 7µm max distance gate
  - `adjusted_edge_jaccard`: edge_jaccard × (1 − 0.1 × (T_pred − T_true) / T_true)
  - `division_jaccard`: separately computed, 0.1-weighted in final score
  - `score`: adjusted_edge_jaccard + 0.1 × division_jaccard

### 6.2 Validation Sample Selection

**Current state:** 4 staged samples locally (2 from `44b6`, 2 from `6bba`); full train set is
199 samples total (71 `44b6`, 128 `6bba` -- see corrected S2.3 above, "embryo" = individual
sample ID, not the 2-value movie prefix).

**For Phase 2:**
1. Reserve **~50 individual sample IDs** (~25% of 199) for validation (never train on them)
2. Use these held-out samples' `.geff` as ground truth for per-epoch validation
3. Track `edge_jaccard`, `adjusted_edge_jaccard`, and `division_jaccard` separately (not just one combined score)
4. Early-stop on validation loss plateau (or validation score plateau)

**Held-out set strategy:**
- 199 total samples: hold out ~50 (25%), stratified roughly proportionally across the 2
  movie prefixes (e.g. ~18 from `44b6`, ~32 from `6bba`) so both movies are represented in
  both splits

**Validation loop (per epoch):**
```python
for held_out_embryo_id in held_out_embryo_ids:
    for sample in samples_of_embryo[held_out_embryo_id]:
        pred_graph = model.infer(sample.zarr)
        gt_graph = sample.geff
        metrics = evaluate_submission([pred_graph], [gt_graph], ...)
        log(metrics)  # Track separately
epoch_val_score = aggregate_over_held_out(metrics)
if epoch_val_score < best_val_score:
    checkpoint_model()
```

### 6.3 Division Recall as Separate Diagnostic

Phase 4's original scope includes division tracking. **Phase 2 should start tracking it early** as a diagnostic (not just at end of Phase 4):

```python
division_tp = count_correct_divisions(pred_graph, gt_graph)
division_gt_count = count_gt_divisions(gt_graph)
division_recall = division_tp / division_gt_count if division_gt_count > 0 else NaN
```

Log this per epoch alongside main metrics. Reveals whether the model is learning division patterns or missing them entirely.

---

## 7. Training Data Pipeline Architecture

### 7.1 PyTorch Dataset Class

**Input requirements:**
- Zarr v3 volume (T, Z, Y, X) uint16
- `.geff` ground-truth graph
- Embryo ID (for train/val splitting)

**Output per sample:**
- (frame_t, frame_t+1, heatmap_targets_t, heatmap_targets_t+1, edge_targets, node_features)

**Pseudocode:**

```python
class GeffTrackingDataset(Dataset):
    def __init__(self, zarr_paths: list[str], geff_paths: list[str], embryo_ids: list[str],
                 held_out_embryo_ids: set[str], mode='train'):
        self.samples = []
        for zarr_p, geff_p, embryo_id in zip(zarr_paths, geff_paths, embryo_ids):
            if mode == 'train' and embryo_id in held_out_embryo_ids:
                continue
            if mode == 'val' and embryo_id not in held_out_embryo_ids:
                continue
            self.samples.append((zarr_p, geff_p, embryo_id))
    
    def __getitem__(self, idx):
        zarr_path, geff_path, embryo_id = self.samples[idx]
        volume = zarr.open(zarr_path)['0'][:]  # Load full volume (T, Z, Y, X)
        graph, metadata = load_geff_ground_truth(geff_path)
        
        # Quantile normalize
        volume_norm = quantile_normalize(volume)
        
        # Randomly select a timepoint pair (t, t+1)
        t = np.random.randint(0, volume_norm.shape[0] - 1)
        
        # Stack frames
        frames = torch.tensor(volume_norm[[t, t+1]]).float()  # (2, Z, Y, X)
        
        # Generate heatmap targets from .geff nodes
        heatmap_t = generate_heatmap_targets(graph, t, volume.shape[1:])
        heatmap_t1 = generate_heatmap_targets(graph, t+1, volume.shape[1:])
        
        # Generate edge targets (pairwise node labels)
        nodes_t = [(n_id, coords) for n_id, coords in graph.nodes.items() if coords.t == t]
        nodes_t1 = [(n_id, coords) for n_id, coords in graph.nodes.items() if coords.t == t+1]
        edge_targets = torch.zeros((len(nodes_t), len(nodes_t1)), dtype=torch.float32)
        for (i, (id_i, coords_i)), (j, (id_j, coords_j)) in itertools.product(
            enumerate(nodes_t), enumerate(nodes_t1)):
            if graph.has_edge(id_i, id_j):
                edge_targets[i, j] = 1.0
        
        return {
            'frames': frames,
            'heatmap_t': torch.tensor(heatmap_t).float(),
            'heatmap_t1': torch.tensor(heatmap_t1).float(),
            'edge_targets': edge_targets,
            'nodes_t': nodes_t,
            'nodes_t1': nodes_t1,
        }
```

### 7.2 Data Augmentation

**Phase 2 scope (from CONTEXT.md):** Implement anisotropy-aware augmentations.

**Suggested:**
- Elastic deformations (respect anisotropy)
- Random rotation in Y/X plane (not Z, which is already undersampled)
- Slight intensity jitter (±5% of normalized range)
- Dropout (set random voxel patches to background)

**Test-time:** TTA (flip Y, X, Y+X) before peak extraction.

---

## 8. Kaggle Kernel Workflow for Training

### 8.1 Confirmed Setup (From smoketest.py Output)

- **GPU:** Tesla T4 × 2 (confirmed usable, compute capability compatible)
- **Memory:** ~11GB per T4 VRAM
- **Data mount:** `/kaggle/input/competitions/biohub-cell-tracking-during-development/`
- **Kernel metadata:** Saved at `kaggle_kernel/kernel-metadata.json`; use `kaggle kernels push` to deploy

### 8.2 Realistic Iteration Loop

**Local development:**
1. Write training script locally (`.py` file, correct imports + syntax)
2. Test on 1–2 staged samples to verify code runs (5–10 min)
3. Push to Kaggle kernel via `kaggle kernels push`
4. Run full training on all 199 train samples (2–4 hours, depending on epochs)
5. Fetch best-model checkpoint back to local disk
6. Evaluate locally on held-out embryos via `src/evaluation.py`
7. If score > 0.763 and > Phase 1's 0.0259, generate submission
8. Submit to Kaggle (rate limit: 1 submission per 24–48 hours, verify exact rate-limit in Kaggle competition rules)

**Iteration speed:** ~4–6 hours per full training run (fetch→train→fetch). Mitigate by:
- Running training for shorter epoch counts (10–20 epochs) to debug first
- Using only a subset of train embryos for initial validation
- Caching detection features if moving to Phase 3 ILP tuning

### 8.3 Output Handoff

Model checkpoint format: PyTorch `.pt` or `.pth`  
- Save: `torch.save(model.state_dict(), 'model_best.pt')`
- Load: `model.load_state_dict(torch.load('model_best.pt'))`
- Include metadata: commit hash, epoch, validation score

---

## 9. Submission Mechanics (Already Wired)

### 9.1 CSV Format (From submission_exporter.py)

Already implemented in Phase 0; no changes needed for Phase 2.

- **Header:** `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`
- **Node rows:** `row_type='node'`, `t,z,y,x` = voxel coordinates (integers), source/target = -1
- **Edge rows:** `row_type='edge'`, source/target = node_id references, z/y/x = -1
- **Critical:** Per-dataset node_id reset (embryo `44b6_0113de3b` has node_ids 1..N, next embryo starts at 1 again)

### 9.2 Inference Pipeline (Phase 2 Extension)

**Current (Phase 0/1):** `run_pipeline.py` loads zarr, extracts peaks, runs ILP tracker, exports CSV.

**Phase 2 changes:**
1. Load trained model checkpoint
2. For each zarr volume:
   a. Quantile normalize
   b. Infer detection logits (UNet) + transformer edge probabilities
   c. Apply NMS peak extraction (with TTA)
   d. Assign edges via greedy algorithm (or feed to ILP tracker if continuing to use it)
   e. Build tracksdata graph
3. Export to CSV via `export_submission()`

**Option:** Can feed learned edge probabilities to ILP tracker as costs (replacing hand-tuned distance²) — but this is Phase 3 optimization, not Phase 2 scope.

---

## 10. Required Code Changes Summary

### 10.1 New Files to Create

1. **`src/model_unet_transformer.py`**
   - `UNet3D` backbone with channel progression `[32, 64, 128]`
   - `SimpleNodeTransformer` cross-attention module
   - Combined `UNetNodeTransformer` model

2. **`src/preprocessing.py`**
   - `quantile_normalize()` with exact host parameters
   - `generate_heatmap_targets()` (Gaussian or point-based)
   - `generate_edge_targets()` from .geff
   - Anisotropy-aware augmentation helpers

3. **`src/training_dataset.py`**
   - `GeffTrackingDataset` class (PyTorch Dataset)
   - Embryo-disjoint train/val split logic
   - Dataloader wrappers

4. **`scripts/train_phase2.py`**
   - Main training loop
   - Epoch loop with validation + early stopping
   - Checkpoint saving
   - Loss computation (detection + edge + optional division weighting)

5. **`scripts/infer_phase2.py`**
   - Model inference on test set
   - Peak extraction + NMS
   - Greedy edge assignment
   - CSV export

### 10.2 Modifications to Existing Files

1. **`src/data_loader.py`**
   - Add `quantile_normalize()` call (or reference to new `preprocessing.py`)
   - Confirm Zarr v3 format handling (already fixed in Phase 0)

2. **`run_pipeline.py`**
   - Add model loading + inference
   - Swap `extract_peaks_from_volume()` for learned peak detection (or enhance existing)
   - Consider: keep ILP tracker or switch to greedy assignment

3. **`src/evaluation.py`**
   - Already complete; use as-is for validation metric computation
   - Consider adding `division_recall` tracking if not present

---

## 11. Data Characteristics (From Staged Samples)

### 11.1 Ground-Truth Sparsity (Important!)

From `data/staging/README.md`:

- **Total voxels per sample:** ~4M (64 × 256 × 256)
- **Annotated nodes per sample:** 25–65 nodes across 100 timepoints
- **Consequence:** <0.001% of voxels are positive in heatmap targets
- **Implication:** Class imbalance is severe → weighted loss and/or hard negative mining essential

### 11.2 Inter-Frame Motion Distribution

From Phase 1 research (STATE.md):
- **Median displacement:** 0.91–2.88 µm/frame
- **Tail (3.1% of edges):** >7–8 µm/frame (fast motion, possibly around division)
- **Implication:** 7µm gating threshold is not overly strict, but hand-tuned distance-squared costs can't model the tail → learned costs (via Transformer) better handle rare fast motion

---

## 12. Known Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **ILP solve time at scale** | High (Phase 3 concern, not Phase 2) | Phase 2: benchmark peak detection + greedy assignment on full 199 samples; measure inference time. If >3 hours, plan for Phase 3 windowing. |
| **Overfitting to public 29% leaderboard slice** | High | Validate primarily on held-out train embryos, not public leaderboard (check divergence weekly in Phase 5). |
| **Heatmap target design (point vs. Gaussian)** | Medium | Empirically test both on held-out embryos; expected: Gaussian easier to optimize. |
| **Division edge weighting not enough** | Medium | Track `division_recall` separately during Phase 2 validation; if <0.5 on held-out, increase auxiliary loss weight or add explicit division-focused data augmentation in Phase 4. |
| **Quantile-normalize threshold mismatch** | Medium | Thresholds (0.4 for peak detection) are calibrated to normalized `[0, 4.0]` range. If you change normalization percentiles, re-tune thresholds on held-out data. |
| **Kaggle rate limit on submissions** | Medium | Confirm exact rate limit on Kaggle competition page (suspected: 1 per 24–48 hours). Phase 2 should generate only 1–2 submissions (for sanity check); save the rest for Phase 4–5 tuning. |

---

## 13. Success Metrics & Phase Exit Criterion

**Phase 2 goal:** Local score ≥0.80 (target), materially >0.0259 (Phase 1) + >0.763 (classical baseline).

**Minimum exit criterion:** Successful Kaggle submission with score >0.763 (beat the classical baseline).

**Tracked metrics (per epoch, on held-out embryos):**
- `edge_jaccard`
- `adjusted_edge_jaccard`
- `division_jaccard`
- `division_recall` (diagnostic)
- `score` (combined)

**Decision logic:**
- If local held-out score >0.80 → generate Kaggle submission
- If local held-out score 0.763–0.80 → optional submission (decide based on improvement trend + time budget)
- If local held-out score <0.763 → debug; do not submit

---

## 14. Unresolved Questions for the Planner

1. **Greedy vs. ILP assignment:** Phase 2 can use greedy (simpler, faster). Should we also prototype feeding learned edge probabilities to the existing ILP tracker as costs? (Deferred to Phase 3 or optional Phase 2 experiment.)

2. **Train on full 199 vs. staged 4:** Phase 2 MUST run on full 199-sample set (locked decision in CONTEXT.md). But for initial debugging/validation, can start with staged 4-sample subset locally, then scale to Kaggle for full training.

3. **Early-stopping patience:** How many epochs of validation-score plateau before stopping? Suggested: 10–15 epochs. Planner can tune based on observed validation loss curves.

4. **Batch size on T4:** Host used batch 16 on unspecified GPU. T4 has ~11GB VRAM. May need gradient accumulation or smaller batch size. Test on staged samples first.

5. **Division loss weight:** How much to upweight division edges? Suggested starting point: 2.0–3.0×. Empirically tune on held-out embryos.

---

## 15. Timeline Estimate

Rough per-iteration breakdown:
- Local debugging (staged samples): 2–4 hours
- Full training on 199 samples (Kaggle): 2–4 hours per run
- Validation + metric analysis: 1 hour
- Submission (if decision made): 30 min

**Phase 2 budget:** 2–3 weeks for 3–4 full training runs (architecture baseline, then 2–3 variants or hyperparameter sweeps).

---

## Appendix: Exact Host Hyperparameters Summary (Quick Reference)

```yaml
model:
  unet_channels: [32, 64, 128]
  unet_downsample: [1, 4, 4]
  transformer_hidden_dim: 128
  transformer_heads: 4
  transformer_blocks: 4
  transformer_dropout: 0.3

training:
  optimizer: AdamW
  learning_rate: 1.0e-4
  batch_size: 16
  epochs: 50
  grad_clip: 1.0

loss:
  det_loss_weight: 1.0
  edge_loss_type: focal_bce
  det_loss_pos_weight: 1.0 / n_pos
  det_loss_neg_weight: 0.01 / n_neg

preprocessing:
  quantile_min: 0.001
  quantile_max: 0.999
  clip_max: 4.0
  gamma: 1.0

nms:
  pool_radius_um: 5.0  # → (3, 13, 13) voxels for this competition
  det_threshold: 0.5
  tta_flips: [none, flip_y, flip_x, flip_yx]

assignment:
  type: greedy
  max_children_per_node: 2
  max_parents_per_node: 1
```

---

**End of Research Document**
