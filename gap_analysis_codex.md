# ST-ACT Gap Analysis — Independent Codex Review

Date: 2026-07-23  
Reviewed branch: `gpu-sanity-gate-wave2-v2` at `417aa73`  
Target: first clear the classical baseline (`0.763`), then reach at least `0.80` locally on embryo-disjoint validation.

## Executive conclusion

The project should not proceed directly to a longer Kaggle training run. The strongest blockers are:

1. Sparse `.geff` annotations are converted into dense targets whose unannotated voxels are treated as background.
2. `UNet3D` does not mix information across Z slices.
3. The edge model trains on perfect ground-truth nodes but infers on noisy model detections.
4. There is no oracle decomposition showing whether detection or linking limits the attainable score.

These are higher priority than replacing the tracker, adding an ensemble, or tuning thresholds. The first three mechanisms are confirmed in the current executable code. Their exact score effect remains an empirical question, so every proposed fix below has a falsifiable validation gate.

## Review method

- Inspected the current model, target generation, training, validation, inference, submission, tracker, configuration, and relevant tests.
- Compared implementation claims with the actual call paths.
- Re-ran focused tests:

  ```text
  python -m pytest tests/test_model.py tests/test_targets.py tests/test_train.py \
    tests/test_p07_training_integrity.py \
    tests/test_p08_gpu_sanity_gate_infrastructure.py \
    -q -p no:cacheprovider --import-mode=importlib

  Result: 149 passed, 1 skipped
  ```

- Cross-checked mechanisms against primary sources. External sources support the proposed experiments; they do not prove a particular score improvement in this repository.

## Evidence scale

- **Confirmed defect:** Current code contradicts the data contract or its stated interface.
- **Confirmed gap:** Current code lacks a capability required to test or control an important risk.
- **High-confidence design risk:** The mechanism is confirmed and supported by research, but score impact requires an ablation.
- **Hypothesis:** A plausible improvement that must not be treated as a defect until an experiment demonstrates benefit.

## Priority order

| Priority | Issue | Classification | Reason |
|---|---|---|---|
| P0.1 | Sparse annotations are treated as dense negatives | Confirmed defect | Corrupts the detector’s learning target |
| P0.2 | Missing oracle score decomposition | Confirmed gap | Prevents locating the active score ceiling |
| P0.3 | No cross-Z feature mixing in `UNet3D` | High-confidence design risk | Volumetric task, slice-independent features |
| P0.4 | Linker train/inference node mismatch | Confirmed gap | Trains on perfect nodes, deploys on noisy peaks |
| P0.5 | Current GPU run remains a probe | Confirmed gap | Longer training is unjustified before P0 fixes |
| P1.1 | Ordinal positional encoding and unnormalized coordinates | Confirmed defect/risk | Encoding depends on arbitrary node ordering |
| P1.2 | Quadratic edge candidates with fixed weighting | High-confidence design risk | Poor class balance and unnecessary runtime |
| P1.3 | Fixed thresholds are not metric-calibrated | Confirmed gap | `0.5` is unrelated to the competition optimum |
| P2.1 | “Cross-attention” claim does not match implementation | Interface mismatch; improvement is hypothetical | Rename now; ablate true cross-attention later |
| P2.2 | Duplicated live configuration and dead YAML | Confirmed architecture gap | Reproducibility and deployment-drift risk |
| P2.3 | Tracker replacement | Deferred hypothesis | Accuracy priority depends on oracle results |

---

## P0.1 — Sparse annotations are treated as dense negatives

### Validation

**Confirmed defect.**

`PRD.md:65` identifies `.geff` annotations as sparse and records a separate full-embryo `estimated_number_of_nodes`. In `src/targets.py:135-181`, every requested heatmap begins as an all-zero volume and only annotated centroids are painted positive. `DetectionLoss` then computes BCE over every voxel and gives every zero target a negative weight (`src/targets.py:503-550`).

An unannotated real cell is therefore trained as background. Reweighting changes the strength of this error but not its meaning.

Primary-source cross-check:

- The original 3D U-Net sparse-annotation method sets unlabeled-pixel weights to zero instead of treating them as background: [Çiçek et al., 3D U-Net](https://doi.org/10.1007/978-3-319-46723-8_49).
- Incomplete cell annotations are a positive-unlabeled learning problem because unlabeled locations may contain real cells: [Zhao et al., Positive-unlabeled learning for cell detection](https://arxiv.org/abs/2106.15918).
- Ultrack’s zebrafish validation data is explicitly generated through sparse labeling and used to assess dense tracking: [Ultrack, Nature Methods](https://www.nature.com/articles/s41592-025-02778-0).

### Exact code changes

1. Add a target container to `src/targets.py`:

   ```python
   @dataclass(frozen=True)
   class DetectionTargets:
       heatmap: torch.Tensor
       supervised_mask: torch.Tensor
       positive_mask: torch.Tensor
       trusted_negative_mask: torch.Tensor
   ```

2. Change `generate_heatmap_targets` to return `DetectionTargets` for each timepoint.
3. Preserve Gaussian regions around every `.geff` centroid as supervised positives.
4. Make unannotated regions ignored by default: `supervised_mask=False`.
5. Add conservative trusted negatives only outside a configurable physical exclusion radius from all annotated and pseudo-labeled cells.
6. Keep label provenance: `gt_positive`, `pseudo_positive`, or `trusted_negative`.
7. Change `DetectionLoss.forward`:

   ```python
   def forward(self, logits, targets, supervised_mask, positive_mask):
       raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
       weights = compute_class_weights(targets, positive_mask)
       active = supervised_mask.bool()
       if not active.any():
           raise RuntimeError("Detection batch has zero supervised voxels")
       return (
           (raw[active] * weights[active]).sum()
           / weights[active].sum().clamp_min(1e-12)
       )
   ```

8. Update `TrainingLoop._generate_and_validate_heatmap_target` and `train_epoch` to pass the masks.
9. Add tests proving:

   - Changes in ignored regions cannot change the loss.
   - GT-positive voxels contribute positive loss.
   - Trusted negatives contribute negative loss.
   - Zero supervised voxels fail closed.

10. Generate pseudo-labels offline using a versioned classical or Ultrack pipeline. Never regenerate them implicitly inside the loss.

### Acceptance gate

- Single-sample overfit produces clear GT-center/background separation without suppressing unlabeled cell-like peaks.
- Held-out recall@7µm improves without catastrophic precision loss.
- Compare existing dense-negative BCE, masked sparse supervision, and masked supervision plus pseudo-labels.

---

## P0.2 — Missing oracle score decomposition

### Validation

**Confirmed gap.**

The repository scores complete predicted graphs but lacks a standard experiment separating detector and linker ceilings. A low end-to-end score therefore cannot identify which module should change.

### Exact code changes

1. Add `src/oracle_evaluation.py`:

   ```python
   @dataclass(frozen=True)
   class OracleDecomposition:
       gt_nodes_gt_edges: dict
       gt_nodes_model_edges: dict
       model_nodes_oracle_edges: dict
       model_nodes_model_edges: dict
       detection_precision_7um: float
       detection_recall_7um: float
       localization_p50_um: float
       localization_p95_um: float
   ```

2. Add `scripts/oracle_decomposition.py`.
3. Implement four modes:

   - GT nodes + GT edges: metric and graph-assembly control.
   - GT nodes + model edges: linker ceiling.
   - Model nodes + oracle edges induced through unique GT matching: detector ceiling.
   - Model nodes + model edges: deployed system.

4. Reuse `DEFAULT_SCALE`, `DEFAULT_MAX_DISTANCE`, official graph conversion, and `evaluate_submission`. Do not implement a second metric.
5. Write per-sample and aggregate JSON containing checkpoint SHA, split identity, and inference configuration.
6. Add tests for perfect graphs, localization misses, false-positive nodes, missed edges, and divisions.

### Acceptance gate

- GT nodes + GT edges reproduces the expected control score.
- All modes are deterministic and use the same embryo-disjoint validation set.
- The next engineering task is selected by measured ceiling.

---

## P0.3 — `UNet3D` does not mix Z slices

### Validation

**Mechanism confirmed; score impact is a high-confidence design risk.**

Every encoder, bottleneck, and decoder convolution uses `(1,3,3)` kernels. Pooling and upsampling use `(1,4,4)` and leave Z untouched (`src/model.py:42-170`). A prediction at slice Z cannot use intensity from adjacent Z slices.

The data is anisotropic, so preserving Z resolution is reasonable. Eliminating cross-Z context everywhere is a separate decision. The original 3D U-Net uses 3D operations for volumetric context, while nnU-Net adapts architecture and preprocessing to voxel spacing and anisotropy: [3D U-Net](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/), [nnU-Net official repository](https://github.com/MIC-DKFZ/nnUNet).

This evidence supports an ablation; it does not prove that every block should use `(3,3,3)`.

### Exact code changes

1. Make kernels explicit configuration:

   ```python
   class UNet3D(nn.Module):
       def __init__(
           self,
           in_channels=2,
           channels=(32, 64, 128),
           kernels=((1, 3, 3), (3, 3, 3), (3, 3, 3)),
           pool_strides=((1, 4, 4), (1, 4, 4)),
       ):
   ```

2. First ablation:

   - `enc0`: `(1,3,3)`
   - `enc1`, `enc2`, bottleneck and `dec2`: `(3,3,3)`
   - `dec1`: test `(1,3,3)` and `(3,3,3)`
   - Keep Z pooling stride at 1.

3. Compute padding from the kernel:

   ```python
   padding = tuple(k // 2 for k in kernel)
   ```

4. Add a receptive-field regression test: changing only slice `z-1` must change output at slice `z` for the mixed-3D configuration.
5. Store kernels and pool strides in the checkpoint manifest. Reject incompatible inference architecture.
6. Benchmark GPU memory and throughput.

### Acceptance gate

- Better held-out recall@7µm or localization error than the slice-independent control.
- No unacceptable precision or runtime regression.
- Retain the current architecture if the official local score does not improve.

---

## P0.4 — Linker trains on GT nodes but infers on predicted nodes

### Validation

**Confirmed gap.**

`TrainingLoop.train_epoch` uses exact GT coordinates, samples features at those locations, and constructs edge targets from them (`src/train.py:772-980`). Validation and submission use NMS peaks from model probabilities (`src/train.py:1167+`, `src/submission_pipeline.py:130-166`).

The linker therefore never trains on localization errors, false-positive detections, missed detections, or variable predicted candidate counts.

Trackastra is a relevant primary reference because it learns pairwise associations over the spatiotemporal context of detected objects and supports divisions: [Trackastra paper](https://arxiv.org/abs/2405.15700).

### Exact code changes

1. Add `src/link_training_examples.py`:

   ```python
   @dataclass(frozen=True)
   class LinkTrainingExample:
       source_nodes: torch.Tensor
       target_nodes: torch.Tensor
       source_features: torch.Tensor
       target_features: torch.Tensor
       candidate_edges: torch.Tensor
       edge_labels: torch.Tensor
       division_mask: torch.Tensor
       hard_negative_mask: torch.Tensor
       provenance: dict
   ```

2. Implement three modes:

   - `gt`: existing teacher-forced nodes.
   - `perturbed_gt`: bounded coordinate noise plus injected false positives.
   - `predicted`: NMS peaks matched one-to-one to GT within 7µm.

3. Use scheduled mixing. Start primarily with GT/perturbed GT and gradually increase predicted-node examples after the detector passes its gate.
4. For predicted nodes:

   - Match detections uniquely to GT.
   - Label an edge positive only when both predicted endpoints match a real GT edge.
   - Treat unmatched predicted nodes as explicit hard negatives.
   - Count missing GT nodes diagnostically rather than fabricating detections.

5. Replace duplicated GT-node logic in `train_epoch` with this module.
6. Test localization errors, false positives, all-negative biological batches, and division edges.

### Acceptance gate

- GT-node linker establishes a viable score ceiling.
- Predicted-node training improves on the same detector outputs versus GT-only training.
- Hard-negative edge AP and end-to-end edge Jaccard improve.

---

## P0.5 — Current GPU configuration remains a probe

### Validation

**Confirmed gap, but not a reason to train longer immediately.**

`kaggle_kernel/train_kernel.py` uses `max_batches_per_epoch=5000` while documenting roughly 14,751 pairs per epoch, and `num_epochs=1`. The learning rate is an exploratory point after failed runs. `SESSION_HANDOFF_2026-07-19.md:101-109` reports no subsequent full GPU run proving that detector collapse was reversed.

### Exact code changes

1. Split configuration into explicit profiles:

   ```python
   RUN_PROFILES = {
       "sanity_gate": {...},
       "learning_probe": {...},
       "full_train": {
           "max_batches_per_epoch": None,
           "num_epochs": 3,
       },
   }
   ```

2. Require a profile argument. Do not leave temporary caps in production defaults.
3. Require a passing formal GPU sanity report before `full_train`.
4. Save the best checkpoint by held-out official score and a last checkpoint for recovery.
5. Treat three epochs as the first learning-curve experiment, not a permanent constant; wall-clock gating remains authoritative.

### Acceptance gate

- P0.1–P0.4 are implemented and their local gates pass.
- Detector and transformer gradient/supervision gate passes.
- A full run completes an uncapped epoch or exits cleanly with resumable state.

---

## P1.1 — Ordinal positional encoding and coordinate scaling

### Validation

**Ordinal encoding is a confirmed defect relative to the stated spatial-encoding intent.**

`SimpleNodeTransformer` calls `sinusoidal_positional_encoding(n_t, ...)`, encoding row positions `0..n_t-1`, then concatenates it with node coordinates (`src/model.py:251-296`). NMS list order is not a stable biological coordinate. Raw voxel coordinates are also unnormalized.

### Exact code changes

1. Delete ordinal positional encoding.
2. Convert coordinates to physical microns and normalize per window:

   ```python
   coords_um = nodes * physical_voxel_size
   all_coords = torch.cat([source_um, target_um])
   center = all_coords.mean(dim=0, keepdim=True)
   scale = all_coords.std(dim=0, keepdim=True).clamp_min(1e-3)
   normalized = (coords_um - center) / scale
   ```

3. Optionally add Fourier features behind a configuration flag.
4. Add a permutation-equivariance test: permuting node order and reversing that permutation must preserve corresponding edge logits.

### Acceptance gate

- Permutation test passes.
- Ablation improves edge AP or end-to-end edge Jaccard.

---

## P1.2 — Quadratic candidate edges and fixed class weighting

### Validation

**High-confidence design risk.**

The transformer creates every Cartesian pair (`src/model.py:303-316`), target generation labels every `n_t*n_t1` pair, and `DivisionLoss` uses fixed `pos_weight=10`. The negative-to-positive ratio therefore changes with cell density, while most distant pairs are trivial negatives.

### Exact code changes

1. Add one shared `build_candidate_edges` function for training and inference.
2. Build candidates in physical space:

   - Always retain GT-positive edges during training.
   - Include targets within a tunable physical radius.
   - Optionally cap at `k` nearest targets per source.
   - Add gap and division candidates deliberately.

3. Pass `candidate_edges` into `SimpleNodeTransformer.forward`.
4. Generate labels only for those candidates.
5. Compare capped adaptive `pos_weight` against focal BCE.
6. Preserve separate hard-negative and easy-negative metrics.

### Acceptance gate

- Candidate recall is 100% for GT-positive training edges.
- Candidate count and runtime fall materially.
- Hard-negative AP and official local score do not regress.

---

## P1.3 — Thresholds are not calibrated to the metric

### Validation

**Confirmed gap.**

Training configuration fixes detection and edge thresholds at `0.5`. The competition objective depends on node matching, false positives, false negatives, and graph topology. A probability threshold of `0.5` has no inherent relationship to the optimum.

### Exact code changes

1. Add `scripts/calibrate_inference.py`.
2. Jointly sweep:

   - detection threshold,
   - NMS radius in microns,
   - edge threshold,
   - candidate radius,
   - optional predicted-node-count tolerance.

3. Cache detector probability maps so sweeps do not rerun the network.
4. Optimize official held-out score while reporting detection precision/recall and predicted/estimated node-count ratio.
5. Write selected values into the checkpoint manifest. Production inference reads validated manifest values only.

### Acceptance gate

- Selected values improve official score over defaults on embryo-disjoint validation.
- A separate fold confirms the result is not specific to one validation embryo.

---

## P2.1 — Cross-attention claim versus implementation

### Validation

**Interface mismatch confirmed; benefit of cross-attention remains a hypothesis.**

The class describes cross-attention but contains two independent self-attention encoders followed by a pairwise MLP. No attention call uses queries from one frame and keys/values from the other.

### Exact code changes

1. Rename the existing implementation `SelfEncodedPairScorer` if retained.
2. First add explicit relative pair features:

   - physical `Δz, Δy, Δx`,
   - distance and squared distance,
   - feature difference and product,
   - optional motion residual.

3. Implement true bipartite cross-attention as a separate adapter:

   ```python
   source_context, _ = self.source_to_target(source_h, target_h, target_h)
   target_context, _ = self.target_to_source(target_h, source_h, source_h)
   ```

4. Compare the simpler pair scorer and cross-attention adapter on identical candidate sets.

Trackastra validates the plausibility of transformer association, but it does not prove this repository needs cross-attention specifically.

### Acceptance gate

- Promote cross-attention only if it improves held-out edge AP and official score enough to justify memory and runtime cost.

---

## P2.2 — Configuration is not authoritative

### Validation

**Confirmed architecture gap.**

`config/hyperparams.yaml` explicitly says it is not loaded. Live values are duplicated inside kernel dictionaries and module constants.

### Exact code changes

1. Add typed `TrainingConfig`, `ModelConfig`, `InferenceConfig`, and `TrackerConfig` dataclasses.
2. Load one versioned YAML or JSON document through a validated loader.
3. Remove duplicate live literals from kernel scripts.
4. Serialize the complete resolved configuration and schema version into every checkpoint.
5. Fail inference on unknown fields or architecture mismatch.

### Acceptance gate

- Every configuration value has one authoritative definition.
- Local training, Kaggle training, evaluation, and submission use the same serialized values.

---

## P2.3 — Tracker replacement is deferred

### Reassessment

The earlier recommendation to replace SCIP/ILP immediately was too strong.

Ultrack demonstrates that ILP-based joint segmentation and tracking can scale when candidate generation and implementation are designed accordingly. The repository’s measured ILP runtime risk is still real, but no current evidence shows that tracker replacement is the highest-value accuracy action for reaching `0.80`.

### Decision

- Do not replace the tracker before oracle decomposition.
- If GT nodes plus the current linker/tracker clear the required ceiling, retain it while fixing detection.
- If runtime fails a full hidden-set projection, benchmark windowed ILP and min-cost flow on identical candidate graphs.
- Preserve division semantics explicitly; a naive one-to-one flow formulation could damage division score.

---

## Implementation waves

### Wave 0 — Measurement

1. Implement oracle decomposition.
2. Freeze the validation split and artifact schema.
3. Record the four-mode baseline.

Exit: detector and linker ceilings are measured.

### Wave 1 — Detector correctness

1. Implement masked sparse/positive-unlabeled supervision.
2. Add mixed 3D kernels as an ablation.
3. Run single-sample and small-subset overfit tests.
4. Compare precision, recall, and localization at 7µm.

Exit: detector provides a credible end-to-end ceiling.

### Wave 2 — Linker robustness

1. Add physical candidate gating.
2. Replace ordinal encoding with physical coordinate features.
3. Train with GT, perturbed, and predicted nodes.
4. Add relative displacement and appearance features.
5. Evaluate true cross-attention only as an adapter ablation.

Exit: predicted-node linker materially improves edge Jaccard.

### Wave 3 — Calibration and full training

1. Calibrate inference hyperparameters against official local score.
2. Pass the formal GPU sanity gate.
3. Run complete wall-clock-budgeted training.
4. Select checkpoint by held-out official score.

Exit: local score exceeds `0.763`, then `0.80`.

### Wave 4 — Scale and competitive ceiling

1. Project runtime to the full hidden test set.
2. Benchmark windowed ILP versus min-cost flow only if required.
3. Add pseudo-label expansion, folds, or ensemble only after single-model ablations plateau.

## What not to do

- Do not launch a long GPU run with the current dense-negative interpretation.
- Do not assume “3D” in the class name means the network has a cross-Z receptive field.
- Do not replace the tracker before oracle tests identify it as limiting.
- Do not add cross-attention merely to match a label.
- Do not tune thresholds on the public leaderboard or a single validation embryo.
- Do not treat speculative score estimates in planning documents as evidence.

## Primary sources

1. Çiçek et al., **3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation** — [publisher](https://doi.org/10.1007/978-3-319-46723-8_49), [University of Freiburg](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/).
2. Bragantini et al., **Ultrack: pushing the limits of cell tracking across biological scales** — [Nature Methods](https://www.nature.com/articles/s41592-025-02778-0).
3. Gallusser and Weigert, **Trackastra: Transformer-based cell tracking for live-cell microscopy** — [paper](https://arxiv.org/abs/2405.15700), [official repository](https://github.com/weigertlab/trackastra).
4. Zhao et al., **Positive-unlabeled Learning for Cell Detection in Histopathology Images with Incomplete Annotations** — [paper](https://arxiv.org/abs/2106.15918).
5. MIC-DKFZ, **nnU-Net** — [official repository](https://github.com/MIC-DKFZ/nnUNet).

## Final recommendation

The immediate sequence should be:

1. Oracle decomposition.
2. Sparse-label masking/positive-unlabeled target semantics.
3. Cross-Z model ablation.
4. Predicted-node linker training.
5. Physical candidate gating and spatial pair features.
6. Metric-based calibration.
7. Formal sanity gate and full training.

This order maximizes information gained per unit of engineering and GPU time. It prevents expensive tracker or architecture work from masking the more fundamental target-label error.
