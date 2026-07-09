"""
Task 2.6 real comparison: greedy edge assignment vs. Phase 1's real ILP tracker.

02-PLAN.md requires: "Compare to Phase 1's ILP tracker on same sample (sanity
check: greedy should produce >=90% of ILP's edges for high-prob predictions)."
Wave 2 substituted synthetic random 10/15-node data for this -- never run
against real detections or the real ILP tracker. This script does the real
comparison.

No trained edge-prediction Transformer exists yet (that's Wave 3+), so both
algorithms are given the same real input: candidate nodes from Phase 1's real
detector (extract_peaks_from_volume + ensemble_consensus_centroids on real
staged volumes) and the same edge probability signal, derived monotonically
from the identical anisotropic physical distance the ILP tracker itself
optimizes on (cost = distance^2, gap=1; probability = exp(-distance/scale)).
This isolates the one real variable Task 2.6 cares about -- does greedy
assignment on a shared cost signal recover what the ILP's optimal solution
finds -- from the (still separately unresolved) question of Transformer
quality.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_pipeline import (
    CNN_THRESHOLD,
    MAX_CANDIDATES_PER_TIMEPOINT,
    UNET_THRESHOLD,
    ensemble_consensus_centroids,
    extract_peaks_from_volume,
)
from src.data_loader import AnisotropicZarrLoader
from src.evaluation import DEFAULT_SCALE
from src.inference import greedy_edge_assignment
from src.tracker import STHypergraphTracker

ANISOTROPY = np.array([4.0, 1.0, 1.0])
DATA_DIR = Path("data/staging/train")
SAMPLES = ["44b6_0113de3b", "6bba_05b6850b"]  # one per prefix, both real staged
N_FRAMES = 5  # -> 4 consecutive pairs per sample
DIST_DECAY_SCALE_UM = 10.0  # edge_prob = exp(-distance_um / this)


def detect_centroids(loader, t):
    vol = loader.load_timepoint_block(t)
    cnn_c = extract_peaks_from_volume(vol, threshold=CNN_THRESHOLD, voxel_size=DEFAULT_SCALE)
    unet_c = extract_peaks_from_volume(vol, threshold=UNET_THRESHOLD, voxel_size=DEFAULT_SCALE)
    consensus = ensemble_consensus_centroids(cnn_c, unet_c, ANISOTROPY)
    if len(consensus) > MAX_CANDIDATES_PER_TIMEPOINT:
        consensus = consensus[:MAX_CANDIDATES_PER_TIMEPOINT]
    return consensus


def build_candidate_edges(centroids_t, centroids_t1, tracker):
    """Mirror the ILP tracker's own candidate-edge generation exactly (zero
    motion, gap=1, 40um search radius, same anisotropic pruning) so both
    algorithms see the identical edge universe."""
    candidate_edges = []
    edge_probs = []
    for i, u in enumerate(centroids_t):
        u_arr = np.array(u)
        for j, v in enumerate(centroids_t1):
            v_arr = np.array(v)
            if tracker.prune_unphysical_edges(u_arr, v_arr, gap=1, anisotropy=ANISOTROPY):
                continue
            distance = np.linalg.norm((u_arr - v_arr) * ANISOTROPY)
            if distance >= 40.0:
                continue
            candidate_edges.append((i, j))
            edge_probs.append(math.exp(-distance / DIST_DECAY_SCALE_UM))
    return candidate_edges, edge_probs


def main():
    total_ilp_edges = 0
    total_matched = 0
    per_sample_results = []

    for sample_id in SAMPLES:
        t0 = time.time()
        zarr_path = DATA_DIR / f"{sample_id}.zarr"
        loader = AnisotropicZarrLoader(str(zarr_path), simulate=False)

        centroids_by_t = {t: detect_centroids(loader, t) for t in range(N_FRAMES)}
        motion_vectors_by_t = {t: [[0.0, 0.0, 0.0] for _ in c] for t, c in centroids_by_t.items()}
        counts = {t: len(c) for t, c in centroids_by_t.items()}
        print(f"[{sample_id}] candidate counts per frame: {counts}")

        tracker = STHypergraphTracker(birth_cost=15.0, death_cost=15.0, division_reward=-8.0)
        # max_gap_frames=0: consecutive-only, so ILP's per-pair edges are directly
        # comparable to greedy_edge_assignment (which only handles one t->t+1 pair).
        lineage_graph = tracker.solve_lineage(centroids_by_t, motion_vectors_by_t, anisotropy=ANISOTROPY, max_gap_frames=0)

        sample_ilp_edges = 0
        sample_matched = 0
        for t in range(N_FRAMES - 1):
            centroids_t = centroids_by_t[t]
            centroids_t1 = centroids_by_t[t + 1]
            if not centroids_t or not centroids_t1:
                continue

            candidate_edges, edge_probs = build_candidate_edges(centroids_t, centroids_t1, tracker)
            ilp_edges_this_pair = {
                (u[1], v[1]) for u, v in lineage_graph.edges() if u[0] == t and v[0] == t + 1
            }

            if not candidate_edges:
                if ilp_edges_this_pair:
                    print(f"  WARNING [{sample_id} t={t}]: ILP found {len(ilp_edges_this_pair)} edges "
                          f"but greedy's candidate-edge builder found none -- pruning mismatch")
                continue

            nodes_t = torch.tensor(centroids_t, dtype=torch.float32)
            nodes_t1 = torch.tensor(centroids_t1, dtype=torch.float32)
            candidate_edges_t = torch.tensor(candidate_edges, dtype=torch.long)
            edge_probs_t = torch.tensor(edge_probs, dtype=torch.float32)

            greedy_result = greedy_edge_assignment(
                edge_probs_t, nodes_t, nodes_t1, candidate_edges=candidate_edges_t, threshold=0.0
            )
            greedy_edges_this_pair = {(i, j) for i, j, _ in greedy_result["edges"]}

            matched = len(greedy_edges_this_pair & ilp_edges_this_pair)
            sample_ilp_edges += len(ilp_edges_this_pair)
            sample_matched += matched

            print(f"  t={t}->{t+1}: ILP={len(ilp_edges_this_pair)} edges, "
                  f"greedy={len(greedy_edges_this_pair)} edges, matched={matched}")

        pct = (sample_matched / sample_ilp_edges * 100) if sample_ilp_edges > 0 else float("nan")
        print(f"[{sample_id}] {sample_matched}/{sample_ilp_edges} ILP edges recovered by greedy "
              f"({pct:.1f}%) in {time.time()-t0:.1f}s")
        per_sample_results.append({"sample_id": sample_id, "ilp_edges": sample_ilp_edges, "matched": sample_matched, "pct": pct})
        total_ilp_edges += sample_ilp_edges
        total_matched += sample_matched

    overall_pct = (total_matched / total_ilp_edges * 100) if total_ilp_edges > 0 else float("nan")
    print(f"\nOVERALL: {total_matched}/{total_ilp_edges} ILP edges recovered by greedy ({overall_pct:.1f}%)")
    print(f"Plan bar: >=90%. {'PASS' if overall_pct >= 90 else 'BELOW BAR'}")


if __name__ == "__main__":
    main()
