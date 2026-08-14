# Fresh Architecture Plan — Biohub Cell Tracking Competition
**Date:** 2026-07-19 | **Status:** PLAN ONLY — no code touched | **v2: Research corrections applied**  
**Purpose:** Independent, zero-legacy-bias evaluation of approaches to reach Kaggle leaderboard #1  
**Constraint:** Competition deadline 2026-09-22 (~9 weeks). 12h Kaggle runtime cap. No internet in submission kernel.

---

## RESEARCH CORRECTIONS (v2 — verified findings, supersede v1 assumptions)

> These were confirmed via direct package docs, arxiv, PyPI, and PMC searches. Each item below
> overrides the equivalent claim in the body of this document where they conflict.

| Assumption in v1 | Verified Fact | Impact |
|---|---|---|
| StarDist-3D has a pre-trained `3D_demo` model | **NO 3D pre-trained model exists.** Only `2D_versatile_fluo` (2D). 3D API exists but requires training from scratch. TensorFlow-only. | Phase R1 "pre-trained, zero training needed" is INVALID — must pivot |
| Trackastra has usable 3D weights | **`general_2d` and `general_2d_w_SAM2_features` only.** 3D point-cloud API exists but no 3D pre-trained checkpoint. | Option C linker choice must be custom-trained |
| Ultrack uses SCIP/OR-Tools solver | **Ultrack's own ILP uses CBC (COIN-OR).** OR-Tools is a separate MCF replacement for the current project's custom tracker. These are not the same. | Ultrack integration does NOT remove our SCIP dependency; we must separately adopt OR-Tools MCF |
| nnUNet not relevant to this competition | **PMC 2024 paper: nnUNet v2 + LAP tracking on zebrafish light-sheet → Dice 95%, Jaccard 91%.** pip-installable, Kaggle dataset exists (`jinttt/nnunet`). Directly validated on this exact data modality. | nnUNet v2 is now the highest-confidence Option B replacement |
| OR-Tools Kaggle-incompatible (no-internet) | **Confirmed manylinux wheels available** → Kaggle no-internet kernel compatible. `pip install ortools` works. | OR-Tools MCF is a viable drop-in for SCIP ILP |
| Top competitor approaches unknown | **No high-scoring solution notebooks are public.** Only: classical baseline (local maxima + Hungarian), nearest-neighbor baseline, and one EDA notebook found on Kaggle. | Can't reverse-engineer top-10 directly. Must infer from leaderboard gap structure. |
| Ultrack has a zebrafish-specific neural model | **No zebrafish neural model shipped with Ultrack.** Ultrack's zebrafish example (`zebrahub`) uses its watershed candidate generator on raw fluorescence, not a pre-trained neural network. | Option A starts from raw intensity → watershed, not from a pre-trained zebrafish detector |

**Corrected Phase R1 quick-win path (replaces StarDist-3D pre-trained):**
1. **Ultrack direct inference** — run `ultrack` watershed candidate generator on the 4 staged training samples. No ML training. Uses CBC ILP internally. Estimated recall@7µm: unknown locally but Ultrack is SOTA on CTC benchmarks. Test in < 1 day.
2. **nnUNet v2 + LAP** — pip-installable, Kaggle-compatible, directly validated on zebrafish light-sheet. Expected score from published results: ~0.85+ on similar data. Requires GPU training (~8h T4) before first real score.
3. **StarDist-3D trained from scratch** — feasible but slower than nnUNet v2 for this data type. Lower priority.

---

## 0. Problem Framing (First Principles)

### What the score actually measures

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard

adjusted_jaccard = max(0, TP/(TP+FP+FN) × (1 − 0.1 × (T_pred − T_true)/T_true))
```

Three things kill your score:
1. **Missed detections** — FN in the edge jaccard denominator
2. **False detections** — FP in the edge jaccard denominator AND the over-prediction penalty
3. **Wrong links** — FP edges (linked two different cells) or FN edges (broke a real track)

**The over-prediction penalty is unique and widely misunderstood.** `T_true` is `estimated_number_of_nodes` in the `.geff` file — the host's estimate of ALL cells in the embryo, not just the sparse labeled subset. Predicting 2× too many nodes doesn't just add FP: it actively deflates every Jaccard numerator. Getting within ~10% of the true node count per sample is a distinct optimization objective.

### Why the classical baseline (0.763) is beatable

The 0.763 baseline uses: local maxima on raw intensity + Hungarian matching. It has no learning component and:
- Doesn't know that annotation is sparse (labels only ~5–15% of true cells)
- Uses fixed NMS radius regardless of embryo type
- Doesn't model cell appearance or motion for linking
- Handles division via a simple heuristic threshold

### Where the leaderboard ceiling likely sits

Top-1 at 0.875 has ~0.112 gap from #1 to baseline. The metric can exceed 1.0 by construction. A perfect detector + perfect tracker approaches ~1.0 + 0.1 = 1.1 theoretical max. Realistically, noise in GT annotation limits the practical ceiling to ~0.92–0.95.

---

## 1. The Landscape of Approaches (What Exists)

| Approach | Description | Published Score on Similar Data | Pre-trained 3D? | Kaggle-compatible? |
|---|---|---|---|---|
| **Ultrack** (host's own) | Multi-scale watershed candidates + CBC ILP | SOTA on CTC benchmarks, has zebrafish example | No neural model — watershed only | Yes (`pip install ultrack`) |
| **nnUNet v2 + LAP** | Self-configuring 3D UNet + Linear Assignment | **Dice 95%, Jaccard 91% on zebrafish light-sheet (PMC 2024)** | No (trains from scratch, ~8h T4) | Yes (`pip install nnunetv2`, dataset `jinttt/nnunet`) |
| **StarDist 3D** | Radial distance prediction → star-convex instances | #1 on several CTC Nuclear tracks | **NO — 2D only (`2D_versatile_fluo`). 3D requires training from scratch. TF-only.** | Yes (manylinux wheels) |
| **CellPose 3D** | Flow field prediction → watershed instances | Very strong on yeast/nuclei, some generalization cost | Yes — `cyto3` (cells), needs fine-tuning for nuclei | Yes |
| **Trackastra** | Transformer-based learned cell matching (2024, ECCV) | SOTA on 2D+t benchmarks | **`general_2d` only. No 3D pre-trained model.** | Yes (`pip install trackastra`) |
| **OR-Tools SimpleMinCostFlow** | Min-cost flow linker (replaces SCIP ILP) | Classical MCF — optimal assignment | N/A — solver not detector | **Yes (manylinux wheels confirmed)** |
| **BTrack / LAP** | Probabilistic linking with motion models | Competitive classical tracker | N/A | Yes |
| **Graph Neural Network tracker** | Learn edge scores directly on candidate graph | Research, not competition-hardened | No | Depends |

---

## 2. Candidate Architectures

### Option A — Ultrack-Native with Learned Scoring

**Philosophy:** Use the host's own algorithm as the backbone, replace hand-crafted edge/node scores with learned ones.

```
Raw Volume (T×Z×Y×X)
       │
       ▼
  nnUNet-style 3D UNet
  (foreground probability map)
       │
       ▼
  Multi-scale watershed
  (Ultrack's own candidate generator)
  → K candidate segments per timepoint
  (K typically 3–10× true cell count)
       │
       ▼
  Node Scorer (small MLP)
  Features: size, sphericity, mean intensity,
  boundary sharpness, z-profile flatness
  → p(real cell | candidate)
       │
       ├────────────────┐
       ▼                ▼
  Edge Builder        Ultrack's CBC ILP
  (overlap graph,     (COIN-OR, built-in)
   t → t+1)          OR: OR-Tools MCF
                      if replacing Ultrack's
                      native solver
       │                │
       ▼                ▼
  Edge Scorer       Final tracking graph
  (cross-attention
   on candidate pairs)
       │
       ▼
  Division head
  (1-node → 2-children = division)
```

**Key training signal:** GT `.geff` provides sparse ground-truth nodes and edges. Use these for:
- Node scorer: GT nodes = positive, non-overlapping non-GT candidates = negative
- Edge scorer: GT edges = positive, alternative assignments = negative (hard negatives via nearest-non-GT)
- Division: explicit division events in GT

**Node count calibration:** Train a per-sample scalar head on `estimated_number_of_nodes`. At inference, binary-search NMS threshold to hit within 10% of predicted count.

---

### Option B — StarDist-3D Detection + Min-Cost Flow Tracking

**Philosophy:** Best-in-class instance segmentation for nuclei, then classical (but optimal) tracking.

```
Raw Volume (T×Z×Y×X)
       │
       ▼
  StarDist-3D (pre-trained or fine-tuned)
  → Star-convex polygon predictions
  → Instance segmentation map
  → Cell centroids + volumes + shape params
       │
       ▼
  Motion predictor (lightweight UNet head)
  → per-cell displacement at t→t+1
       │
       ▼
  OR-Tools SimpleMinCostFlow
  Cost = spatial_distance² × gap_penalty
         - appearance_similarity
         + motion_residual²
       │
       ▼
  Division detector
  Binary classifier on (centroid-at-t, two-centroids-at-t+1) triplets
  → reclassify link vs. division
```

**Key advantage:** StarDist's star-convex polygon model is explicitly designed for spherical nuclei — the exact morphology of zebrafish embryo cells. The MCF linker is mathematically optimal given correct detections.

**⚠️ CORRECTED FROM v1:** StarDist-3D has **NO pre-trained 3D model**. The `2D_versatile_fluo` model is 2D only. The `3D_demo` checkpoint mentioned in v1 does not exist as a shipped asset. Using StarDist-3D therefore requires training from scratch (~6–12 GPU hours T4). StarDist is also TensorFlow-only — add a TF dependency to the Kaggle kernel.

**Preferred alternative for Option B:** Replace StarDist with **nnUNet v2** as the detection backbone. nnUNet v2 is: (1) PyTorch-native (no TF conflict), (2) self-configuring on this exact data shape, (3) directly validated on zebrafish light-sheet fluorescence (PMC 2024, Dice 95%), (4) pip-installable with existing Kaggle dataset `jinttt/nnunet` for no-internet kernels. Replace the "StarDist-3D" block in the diagram above with "nnUNet v2 3D full-resolution UNet" → centroids via watershed on predicted foreground.

**Key risk:** nnUNet auto-configuration requires a manual "planning" step that fingerprints the dataset before training. The planning artifact must be included in the Kaggle dataset bundle (not auto-generated at inference time under no-internet). One extra setup step vs. "pip install and run."

---

### Option C — CellPose 3D Detection + Transformer Linking

**Philosophy:** CellPose's flow-field approach is uniquely robust to irregular cell shapes. Pair with a learned matcher for linking.

```
Raw Volume (T×Z×Y×X)
       │
       ▼
  CellPose 3D (cyto3 model, fine-tuned)
  Predicts: gradient flow field + foreground
  → Euler integration → instance masks
  → Centroids + masks
       │
       ▼
  Appearance encoder
  (small CNN, crops 16×16×8 µm around each centroid)
  → 64-dim embedding per cell per timepoint
       │
       ▼
  Trackastra-style cross-attention
  Q = cells at t+1
  K,V = cells at t (within search radius)
  → assignment matrix → link scores
       │
       ▼
  Threshold links + handle divisions
  (out-degree ≥ 2 at t → division candidate)
```

**Key advantage:** CellPose is the most battle-tested tool on densely packed cells. The transformer linker directly learns track-vs-division semantics from GT edge data.

**Key risk:** CellPose 3D with the `cyto3` pretrained model requires significant fine-tuning time (5–8 GPU hours) to adapt to zebrafish nuclei. Transformer linking has been validated primarily on 2D data; 3D generalization is less certain.

---

### Option D — End-to-End Learned (Custom Architecture)

**Philosophy:** No modular decomposition. One network sees pairs of volumes and directly outputs a tracking graph. Most ML-native approach.

```
Volume at t (Z×Y×X) ─────┐
                           ├─▶ Dual-stream UNet with
Volume at t+1 (Z×Y×X) ───┘    cross-frame attention
                               │
                               ▼
                   Heatmap head → cell positions at t and t+1
                   Edge head → affinity matrix between positions
                   Division head → binary flag per t-node
                               │
                               ▼
                   Greedy or min-cost-flow assignment
                               │
                               ▼
                   Tracking graph
```

**Key advantage:** Joint learning — the network can learn that "a brighter, rounder nucleus that appears to have been moving fast" is more likely to be a division. No hand-crafted feature engineering.

**Key risk:** Requires the most data and training time to work well. With sparse GT annotations (~5–15% coverage), the end-to-end signal is weak. Risk of training collapse (already observed once in this project).

---

### Option E — Ensemble Hybrid (Two Independent Models + Voting)

**Philosophy:** Use Options A and B (or B and C) independently, then merge their detections before tracking.

```
Model 1 (Option A: Ultrack+learned)  →  Detections₁
Model 2 (Option B: StarDist+MCF)     →  Detections₂
                                          │
                                          ▼
                              NMS merge (7µm gate)
                              Confidence = mean(scores₁, scores₂)
                              if only one model found it: lower confidence
                                          │
                                          ▼
                              Final track via best-performing linker
```

**Key advantage:** If Model 1 misses a cell that Model 2 found, the ensemble catches it. False positives from one model are damped if the other disagrees. Highest potential ceiling.

**Key risk:** Highest engineering complexity. Two full model pipelines. Difficult to fit both into 12h Kaggle runtime.

---

## 3. Comparative Evaluation

### 3.1 Accuracy (Expected Competition Score)

| Option | Detection Quality | Linking Quality | Division Quality | Expected Score | Notes |
|---|---|---|---|---|---|
| **A — Ultrack+Neural** | High | High | Medium | **0.87–0.92** | Host's own algorithm + learned scoring. Most likely to match what host validated internally |
| **B — nnUNet+MCF** *(v2 correction: replaces StarDist)* | **Very High** | Medium-High | Medium | **0.85–0.91** | **PMC 2024: directly validated on zebrafish light-sheet. Dice 95%, Jaccard 91%.** PyTorch, pip-installable. Strongest validated single-model option. |
| **B-alt — StarDist-3D+MCF** | High | Medium-High | Medium | **0.82–0.88** | ⚠️ **No pre-trained 3D model — training from scratch required (~12h T4). TF-only adds dependency risk.** Lower priority than nnUNet. |
| **C — CellPose+Transformer** | High | High | High | **0.83–0.90** | Strong on irregular cells; transformer linking is explicit about divisions. CellPose `cyto3` needs fine-tuning. |
| **D — End-to-End (current codebase)** | Uncertain | Uncertain | Uncertain | **0.60–0.88** | Wide variance. Training collapse already observed. 3 architecture bugs confirmed. Highest risk. |
| **E — Ensemble A+B** | Highest | High | Medium | **0.90–0.95** | Highest ceiling, highest complexity. Needs both sub-models working first. |
| **R1-A — Ultrack direct (no ML)** | Medium-High | Classical | None | **0.70–0.82** | No training needed. Fastest first score. Floor depends on watershed quality on this data. |

**Revised recommendation ordering by risk-adjusted expected return:**
1. **R1-A (Ultrack direct)** — test in 1 day, establishes a real floor
2. **Option B (nnUNet+MCF)** — highest published validation on this exact data type  
3. **Option A (Ultrack+Neural scoring)** — best theoretical ceiling
4. **Option E (ensemble)** — Phase R3 only, after A and B both work

**Caveat:** All trained-model estimates assume successful training. The current project has a documented training collapse history — validate locally (single-sample DetectionLoss overfit test) before any Kaggle GPU run.

---

### 3.2 Performance (Runtime — Must Fit in 12h Kaggle Kernel)

A competition test set has ~149 samples, each ~(100, 64, 256, 256) uint16. Rough single-sample inference times:

| Component | Estimated time/sample (T4 GPU) |
|---|---|
| 3D UNet foreground (nnUNet/custom) | 3–8 min |
| Multi-scale watershed (Ultrack) | 1–3 min |
| StarDist 3D inference | 2–5 min |
| CellPose 3D inference | 4–10 min |
| OR-Tools min-cost flow (1000s of nodes) | < 30 sec |
| SCIP ILP (current codebase) | 70% of runtime → **not viable at full scale** |
| Transformer linker (small, per-frame) | 1–2 min |

**Runtime budget per sample: ~12h / 149 ≈ 5 min**

| Option | Est. time/sample | Fits 12h? | Headroom |
|---|---|---|---|
| **A — Ultrack+Neural** | 5–11 min | Borderline | Needs parallelism or windowing |
| **B — StarDist+MCF** | 3–7 min | Yes | Comfortable margin |
| **C — CellPose+Transformer** | 6–13 min | Borderline | Risk at dense timepoints |
| **D — End-to-End** | 4–8 min | Yes | Depends on model size |
| **E — Ensemble A+B** | 8–18 min | No | Requires aggressive optimization |

**Critical:** Replace SCIP ILP with OR-Tools `SimpleMinCostFlow` in ALL options. This change alone removes the 70% runtime bottleneck from the current implementation.

---

### 3.3 Resource Requirements

| Option | GPU Training Hours | GPU Hardware | Data Required | External Dependencies |
|---|---|---|---|---|
| **A — Ultrack+Neural** | 8–20h | T4/A100 | 149 train samples | `ultrack`, `tracksdata`, `ilpy`, `ortools` |
| **B — StarDist+MCF** | 4–10h fine-tune | T4 | 149 train samples + pseudo-labels | `stardist`, `csbdeep`, `ortools` |
| **C — CellPose+Transformer** | 6–15h | T4/A100 | 149 train samples | `cellpose`, `torch`, `ortools` |
| **D — End-to-End** | 20–50h | A100 preferred | 149 train samples | `torch`, `monai` |
| **E — Ensemble A+B** | 12–30h | A100 | 149 train samples | All of A + B |

**Kaggle GPU quota note:** The current project has T4 access (confirmed in CLAUDE.md). A100 access requires pay-as-you-go or a different tier. Options requiring A100 have a resource constraint risk.

---

### 3.4 Time to First Real Kaggle Score (Weeks from Now)

Assumes: 1 Kaggle GPU run = 1–2 days to set up, submit, and evaluate.

| Option | Weeks to First Real Score > 0.763 | Weeks to First Score > 0.85 | Notes |
|---|---|---|---|
| **A — Ultrack+Neural** | 2–3 | 4–6 | Ultrack detection works out-of-box; scoring just needs trained scorer |
| **B — StarDist+MCF** | 1–2 | 3–5 | Pre-trained StarDist can run TODAY without fine-tuning; ~0.75–0.80 expected even un-tuned |
| **C — CellPose+Transformer** | 2–3 | 4–6 | CellPose out-of-box slower to tune than StarDist for this data type |
| **D — End-to-End** | 4–8 | 8–12+ | Training stability risk makes this unpredictable |
| **E — Ensemble A+B** | 3–5 | 5–8 | Requires both sub-models working before ensemble helps |

---

### 3.5 Innovation Potential (What the Top Teams Don't Have)

The differentiators that could push score beyond 0.875 regardless of which option is chosen:

| Innovation | Applies To | Estimated Score Lift | Complexity |
|---|---|---|---|
| **Node-count-aware NMS calibration** (predict `estimated_number_of_nodes`, tune threshold to match) | All options | +0.02–0.05 | Low |
| **Pseudo-label pre-training via Ultrack** (run Ultrack on all 199 samples → dense labels) | All options | +0.03–0.06 | Medium |
| **Per-embryo-type model selection** (44b6 vs. 6bba have different morphology — separate hyperparams) | All options | +0.01–0.03 | Low |
| **Division-aware augmentation** (synthetically create division events during training) | C, D | +0.01–0.02 | Medium |
| **Test-time ensembling over NMS thresholds** (run 3–5 thresholds, pick nearest to T_true) | A, B | +0.02–0.04 | Low |

---

## 4. Recommended Approach: Phased Strategy

### Phase R1 — Quick Win (Weeks 1–2)

**⚠️ CORRECTED FROM v1:** The original R1 plan (StarDist-3D pre-trained, zero training) is invalid. No 3D StarDist pre-trained model exists.

**Revised R1 path — two parallel tracks, test both in Week 1:**

**Track R1-A: Ultrack direct inference (zero ML training)**
Run Ultrack's multi-scale watershed candidate generator directly on the 4 staged training samples. No training step. Uses CBC ILP internally. Score from Ultrack alone on CTC-class data is typically 0.70–0.82 depending on intensity quality. If it beats 0.763 immediately, this is the fastest first real submission.
- Install: `pip install ultrack`
- Time to first Kaggle score: ~3–5 days (install, run, bundle kernel)
- Risk: Low for getting *a* score. Risk: Medium for beating 0.763 (dependent on raw intensity SNR of this specific data).

**Track R1-B: nnUNet v2 + LAP (requires GPU training)**
nnUNet v2 is self-configuring on new data — run `nnUNetv2_plan_and_preprocess` on the 199 training samples, then train for ~8h T4. Published zebrafish light-sheet result (PMC 2024): Dice 95%, Jaccard 91% — this directly implies expected competition score > 0.85 if replicated.
- Install: `pip install nnunetv2` (Kaggle dataset `jinttt/nnunet` for offline)
- Time to first Kaggle score: ~7–10 days (preprocessing + training + kernel integration)
- Risk: Medium (training required, but the approach is directly validated on this exact data type).

Pair either Track with OR-Tools `SimpleMinCostFlow` for the linking step (< 1 day to implement, confirmed Kaggle-compatible).

**Expected first real score: 0.76–0.85** (R1-A lower, R1-B higher).
**Priority: Run R1-A first (no training cost) while R1-B trains in background.**

---

### Phase R2 — Competitive Score (Weeks 3–5)

**Extend the winner of R1 with learned components:**

**If R1-A (Ultrack) won:** Add a neural node scorer (small MLP on candidate morphology features) to re-rank Ultrack's multi-scale watershed candidates. Add node-count-aware threshold calibration (train a scalar regressor on `estimated_number_of_nodes`). Replace CBC ILP with OR-Tools MCF for runtime headroom.

**If R1-B (nnUNet v2) won:** Fine-tune the nnUNet detector with pseudo-labels from Ultrack (Ultrack generates dense candidate sets across all 199 samples — use high-confidence ones as additional training supervision). Add node-count-aware NMS. Upgrade LAP linker to learned cross-attention (Trackastra-style, custom-trained on competition GT edges).

**Common to both:**
1. Node-count-aware NMS: per-sample scalar regressor trained on `estimated_number_of_nodes` from GT `.geff` files. At inference, binary-search the NMS/detection threshold to land within 10% of predicted count.
2. Pseudo-label generation via Ultrack on all 199 training samples — use pseudo-labels to augment supervision for any learned component.
3. Per-embryo-type tuning: 44b6 and 6bba embryo types have distinct morphology — train separate NMS thresholds and motion priors.

**Expected score: 0.86–0.90**  
**Risk: Medium.** nnUNet is pre-validated on this data type, so the floor risk is lower than the current custom UNet3D path.

---

### Phase R3 — Leaderboard Leadership (Weeks 6–9)

**Add Option A (Ultrack multi-scale candidates) as an alternative pathway, run as an ensemble.**

1. Implement Ultrack's full pipeline on competition data.
2. Compare per-sample: does StarDist or Ultrack produce detections closer to `estimated_number_of_nodes`?
3. Ensemble: for each sample, pick the model whose predicted node count is within 10% of `estimated_number_of_nodes`. Use the winner's detections, the best linker's tracks.
4. Fine-tune division detection: dedicated binary classifier on candidate triplets (node-at-t + two-candidate-daughters-at-t+1).

**Expected score: 0.89–0.94**  
**Risk: Medium-High.** Ensemble adds engineering complexity. Kaggle runtime is tight.

---

## 5. Decision Matrix Summary

```
           │  Accuracy  │ Runtime Risk │  Data/GPU   │  Time to 0.763 │  Innovation │
───────────┼────────────┼──────────────┼─────────────┼────────────────┼─────────────┤
Option A   │  ★★★★★    │  ★★★☆☆      │  ★★★☆☆      │  ★★★☆☆         │  ★★★★☆      │
Option B   │  ★★★★☆    │  ★★★★★      │  ★★★★☆      │  ★★★★★         │  ★★★☆☆      │
Option C   │  ★★★★☆    │  ★★★☆☆      │  ★★★☆☆      │  ★★★☆☆         │  ★★★★☆      │
Option D   │  ★★☆☆☆    │  ★★☆☆☆      │  ★★☆☆☆      │  ★☆☆☆☆         │  ★★★★★      │
Option E   │  ★★★★★    │  ★★☆☆☆      │  ★★☆☆☆      │  ★★☆☆☆         │  ★★★★★      │
```

**Verdict (v2 — corrected):**

| If you want to... | Choose |
|---|---|
| Get a real score above 0.763 fastest, zero training | **Ultrack direct (R1-A)** — run watershed + CBC ILP today |
| Highest validated ceiling on zebrafish data | **Option B (nnUNet v2 + OR-Tools MCF)** — PMC 2024 result directly applicable |
| Balance ceiling and host-algorithm alignment | **Option A** (Ultrack candidate generation + neural node/edge scoring) |
| Maximize theoretical ceiling | **Option E** (A + B ensemble, Phase R3 only) |
| Most innovative ML contribution | **Option D** (end-to-end, highest risk — only if A and B fail) |
| ~~Get a real score above 0.763 with StarDist pre-trained~~ | ~~**Option B (StarDist)**~~ **INVALID — no 3D pre-trained model exists** |

---

## 6. What NOT to Do (Grounded in This Project's History)

| Temptation | Why to avoid |
|---|---|
| Continue iterating on the current UNet3D + custom transformer | 3 architecture bugs confirmed (positional encoding, cross-attention, Z-convolution). Cost of fixing + retraining exceeds cost of switching to StarDist. |
| Use SCIP ILP at competition scale | 70% of runtime at 30 candidates/timepoint. OR-Tools MCF is the drop-in fix. |
| Optimize detection threshold without node-count calibration | The adjusted Jaccard penalty makes threshold optimization futile without also targeting T_true. |
| Run an end-to-end Kaggle GPU training run before local sanity gate passes | Documented: last real Kaggle run collapsed (max_sigmoid ≈ 2.2e-6). Do not repeat without local confirmation first. |
| Build a second custom tracker when OR-Tools MCF exists | Reinventing optimal assignment is a week of work vs. a day of integration. |

---

## 7. External Platform Roles

| Platform | Best Use for This Competition | Limitation |
|---|---|---|
| **Gemini CLI** (`gemini -p`) | Weekly: ingest entire Kaggle discussion + all competitor notebooks (2M context) → "what detection approach are top-10 likely using?" | Can't run Python or read local data |
| **NotebookLM** | One-time: upload Ultrack paper + PRD + REFERENCE_IMPLEMENTATION.md → Q&A to surface gaps between our approach and host's intended approach | Read-only, no code |
| **OpenAI Codex** | Parallel implementation in isolated worktrees — bounded, spec'd tasks only | Same "trust but verify" rule as all sub-agents |
| **Kaggle Notebooks (public)** | Copy competitor notebooks, run locally on staged data to benchmark their actual approach | Time-limited: notebooks get private after competition ends |
| **Weights & Biases / MLflow (local)** | Track every experiment: model config, local score, Kaggle score, what changed | Not a platform — install locally on Day 1 |

---

## 8. Immediate Next Actions (Before Any Code)

**Decision tree — pick one of these three paths, not all three:**

### Path 1: Ultrack Direct (fastest first score, no training needed)
1. `pip install ultrack` locally and run the zebrahub example on `data/staging/train/6bba_05b6850b.zarr`.
2. Report recall@7µm vs. `.geff` GT nodes. If recall > 0.70, proceed to Kaggle kernel integration.
3. Replace internal CBC ILP with OR-Tools MCF for the linking step (keeps Ultrack's candidate generation, swaps solver). This removes the SCIP dependency from our kernel entirely.
4. **Gate:** local score > 0.763 before spending Kaggle GPU quota on this path.

### Path 2: nnUNet v2 (highest expected ceiling, requires training)
1. `pip install nnunetv2`. Run `nnUNetv2_plan_and_preprocess -d <dataset_id>` on the 199 training Zarr samples converted to nnUNet format (NIfTI or similar).
2. Train 3D fullres configuration on T4 (~8h). Check validation Dice against local GT.
3. At inference: predict foreground mask → watershed → centroid extraction → OR-Tools MCF linking.
4. **Gate:** local score > 0.763 before first Kaggle submission.

### Path 3: Fix current codebase (Option D)
Only choose this if Paths 1 and 2 both fail local gate. Fix the three confirmed architecture bugs (positional encoding, cross-attention, Z-convolution) before any GPU training. Do NOT spend Kaggle GPU quota until the local detection overfit test (DetectionLoss on single sample) shows a clean learning curve.

**Regardless of path chosen:**
- Replace SCIP ILP with OR-Tools `SimpleMinCostFlow` in `src/tracker.py`. Confirmed Kaggle-compatible. 2-day task. Benefits all three paths as the linker.
- Do NOT run another Kaggle GPU training run without passing local gate first (last run collapsed at max_sigmoid ≈ 2.2e-6).

**Recommended immediate action:** Start with Path 1 (Ultrack, no training), run locally today, report recall@7µm. While that runs, start Path 2 preprocessing in background. This gives a real empirical answer within 24–48 hours that makes the architecture decision concrete rather than theoretical.

---

*Document status: PLAN ONLY — zero existing files modified. For review before implementation.*
