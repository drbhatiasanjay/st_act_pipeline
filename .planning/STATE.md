# Project State: ST-ACT — Spatio-Temporal Anisotropic Cell Tracker

**Last Updated:** 2026-07-03

---

## Project Reference

**Core Value:** A Kaggle competition submission that is schema-valid, scores above the classical baseline (0.763), and is engineered to compete for the top of the leaderboard.

**Target Leaderboard:** Biohub – Cell Tracking During Development (Kaggle)
- **Current #1 Score:** 0.875 (Kaushik Ramayya Chikkala)
- **Entry Deadline:** 2026-09-22 (~11 weeks from 2026-07-03)
- **Public Test Data Coverage:** ~29% revealed; full hidden test set used for final ranking

**Key Constraints:**
- Kaggle notebook runtime limit: 12 hours
- PyTorch + MONAI (not Keras 3/JAX) for 3D biomedical volume handling
- Competition is embryo-disjoint train/test (no data leakage)
- Submissions are rate-limited and precious

**Core Dependencies/Context:**
- Kaggle API + remote-GPU-kernel setup in parallel worktree (`../st_act_pipeline-kaggle-setup`, branch `kaggle-setup`) — Phase 2 training depends on this; Phase 0/1 do not
- ILP tracker flow-conservation bug already fixed this session; isolated nodes now resolve correctly instead of causing Infeasible

---

## Current Position

**Roadmap Status:** Phase 0 (Unblock) — awaiting approval and planning

**Phase Structure:**
```
Phase 0 (Unblock)
  └─ Phase 1 (Baseline parity)
       └─ Phase 2 (Learned detection)
            └─ Phase 3 (Scale & correctness)
                 └─ Phase 4 (Metric-directed tuning)
                      └─ Phase 5 (Competitive iteration loop)
```

**Current Sprint:** Roadmap review; awaiting approval to begin Phase 0 planning

**v1 Requirement Coverage:**
- Total v1 requirements: 19
- Phase 0: 12 requirements (DATA-01..05, SUB-01..03, EVAL-01..04)
- Phase 1: 0 requirements (validation only)
- Phase 2: 5 requirements (MODEL-01..04, TRACK-01)
- Phase 3: 2 requirements (TRACK-02, TRACK-03)
- Coverage: 19/19 ✓

---

## Performance Metrics

**Primary Success Signal (Competition):**
- Leaderboard rank and score, updated weekly through 2026-09-22
- Target: Rank #1 by final deadline

**Leading Indicator (Local):**
- `edge_jaccard + 0.1 × division_jaccard` on held-out train embryos (with real `.geff` ground truth)
- Must correlate with public leaderboard before trusted as decision signal
- Current baseline (classical non-learned): 0.763
- Phase 2 target: ≥0.80
- Phase 4 target: ≥0.875

**Guardrails:**
- Local metric on held-out embryos must not diverge from public leaderboard by >0.05 points (signals overfitting to visible 29% slice)
- Division recall tracked separately (0.1-weighted but high-leverage for top-tier separation)
- Full-size `(100, 64, 256, 256)` volumes must process end-to-end in <12 hours by Phase 3

---

## Accumulated Context

### Decisions

1. **Phase structure mirrors PRD § 8 exactly.** PRD's 6-phase roadmap is well-formed and aligns with requirements; no reordering or restructuring needed. All phases 0–5 included in v1 roadmap.

2. **PyTorch + MONAI chosen over Keras 3/JAX.** Framework selection rationale: ecosystem maturity for 3D biomedical volumes (sliding-window inference, anisotropic augmentations) outweighs JAX's framework-purity; GPU (not TPU) is the compute plan. Noted in REQUIREMENTS.md Out of Scope section.

3. **Local evaluation harness is critical to competition success.** Every model/tracker change must be validated against local harness (edge Jaccard + division Jaccard on held-out embryos) before a Kaggle submission is spent; submissions are scarce and rate-limited.

4. **Overfitting to ~29% public slice is a known risk.** Mitigated by always validating primarily against held-out train embryos with real `.geff` ground truth (not just public leaderboard feedback). Phase 5 includes monitoring for local-vs-public divergence.

5. **Kaggle setup (API + GPU kernel) is parallel work.** Phase 0 and Phase 1 do NOT depend on this (local development). Phase 2 (model training) DOES depend on it being complete. This is tracked separately in `../st_act_pipeline-kaggle-setup` branch; no blocking dependency for immediate Phase 0 work.

### Pending Todos

- [ ] **Phase 0 planning:** Decompose Phase 0 into executable tasks (data loading, `.geff` reader, submission writer, local metric)
- [ ] **Phase 1 setup:** Reserve 2–3 held-out train embryos for validation (never touched during Phase 2 training)
- [ ] **Confirm Kaggle rules:** Full `/rules` page (team size, external-data policy, compute/runtime caps) not fully retrievable at PRD time; re-verify before Phase 2 investment
- [x] **Confirm reference metric implementation — RESOLVED 2026-07-03:** host publishes a full
  reference implementation, `royerlab/kaggle-cell-tracking-competition` (linked directly from
  `/overview/evaluation`), built on a real `tracksdata` library (`.from_geff()`/`.to_geff()`/
  `DistanceMatching`). Exact formulas, the division-matching algorithm, and vendored source are in
  `REFERENCE_IMPLEMENTATION.md` at repo root. **Scope change for Phase 0 planning:** EVAL-01..04
  should vendor/wrap `tracksdata`'s `metrics.py` + `division_metrics.py` directly rather than
  reimplementing from prose — this removes most of the risk in that task. Also confirmed: the
  `adjusted_edge_jaccard` penalty formula is `max(0, jaccard·(1 − 0.1·(T_pred−T_true)/T_true))`
  where `T_true` = the `.geff`'s `estimated_number_of_nodes` field, and unmatched predicted nodes
  are structurally excluded from the FP count (sparse-GT fairness confirmed, not just implied).
  **Reconcile with this session's separate `geff`/`traccuracy` decision below:** `tracksdata` wraps
  `geff` and is the same-family tool that implements this competition's *exact bespoke* score;
  `traccuracy` computes generic CTC metrics (TRA/DET) which are a *different* formula — treat
  `tracksdata` as primary for FR-5, keep `traccuracy` only as an optional secondary sanity check.

### Blockers/Concerns

1. **ILP solve time at scale (Phase 3).** Current CBC-based ILP may not scale to thousands of cells × 100 timepoints × gap-closing within Kaggle's 12-hour budget. Mitigation in Phase 3: windowed/rolling-horizon solving or OR-Tools `SimpleMinCostFlow` reformulation (already verified to fit flow-conservation constraints). Must benchmark on realistic synthetic scale early in Phase 2.

2. **Detection model under-capacity (Phase 2).** Current 2-conv-layer FCN may be too shallow for `256×256×64` volumes. Treat as placeholder for Phase 1 validation; budget architecture experimentation in Phase 2 (deeper ResNet-style backbone, 3D convolutions, anisotropy-aware augmentations).

3. **Division events are rare (Phase 4).** Micro-averaged over samples; only 0.1-weighted in overall score, but top-tier separation may hinge on division recall. Phase 4 includes dedicated tracking of division_recall as independent metric.

4. **Hidden test set will reveal different patterns (Phase 5).** Leaderboard is ~29% public; final ranking uses full hidden set. Weekly cycle in Phase 5 monitors for overfitting; if local-vs-public divergence > 0.05, revert to last best submission and investigate.

---

## Session Continuity

**Session Start:** 2026-07-03 (roadmap creation)

**What We Know:**
- Project structure validated against PRD § 6 (Functional Requirements)
- v1 requirements extracted and categorized (19 total: 12 Data/Submission/Eval, 5 Model/Tracking, 2 Tracking scale)
- 6-phase roadmap derived from PRD § 8, mapped to requirements, success criteria defined

**What Happens Next:**
1. User reviews and approves ROADMAP.md
2. Roadmap locked; STATE.md initialized (this file)
3. REQUIREMENTS.md traceability section verified/updated
4. Run `/gsd:plan-phase 0` to break Phase 0 into executable plans (data loader, `.geff` reader, submission writer, local metric)
5. Execute Phase 0 plans
6. Validate Phase 1 execution (no new work; just wiring/testing Phase 0's infrastructure)
7. Depending on Phase 1 outcome: proceed to Phase 2 planning or loop back to debug Phase 0

**Risk to Track:**
- Kaggle setup (parallel work) must complete before Phase 2 starts
- Local metric implementation must exactly match Kaggle's scoring formula (Phase 0 EVAL-01..04) — RESOLVED: host reference implementation found and vendored, see `REFERENCE_IMPLEMENTATION.md`; remaining risk is just correct integration, not formula uncertainty
- ILP scale-up must be validated early (Phase 2/Phase 3 boundary)

---

**State initialized:** 2026-07-03  
**Next step:** Approve ROADMAP.md and proceed to `/gsd:plan-phase 0`
