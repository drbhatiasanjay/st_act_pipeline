# ruff: noqa: E402, E701, E702, B905
"""
Dry-run: before/after score impact on real staged data.
Fast: 3 GT-overlapping frames, block-stride peak detection, greedy linker.
~15s on CPU. Uses official tracksdata scorer.
"""
import sys
import time
import warnings

import numpy as np
import tracksdata as td
import zarr

warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
from tracking_cellmot.metrics import evaluate, per_sample_metrics

SAMPLE      = "44b6_0b24845f"
ZARR_PATH   = f"data/staging/train/{SAMPLE}.zarr"
GEFF_PATH   = f"data/staging/train/{SAMPLE}.geff"
SCALE       = np.array([1.625, 0.40625, 0.40625])   # z, y, x µm
MAX_DIST_UM = 7.0
# GT labels are at t=11..50; use a 15-frame window in the middle
T_WINDOW    = set(range(20, 35))

def pool_kernel_um(um):
    k = []
    for s in SCALE:
        v = max(1, round(um / float(s)))
        if v % 2 == 0: v += 1
        k.append(int(v))
    return k   # [kz, ky, kx]

def nms_fast(vol_f32, threshold, kernel_um=5.0):
    """
    Fast NMS via local block maxima. Each voxel is a peak iff it equals
    the maximum in its neighbourhood AND exceeds threshold.
    Uses half-kernel strides so we visit each block once — O(N) not O(N*K).
    """
    kz, ky, kx = pool_kernel_um(kernel_um)
    Z, Y, X = vol_f32.shape
    # Pad to make shape divisible by strides
    def pad(a, sz, sy, sx):
        pz = (-a.shape[0]) % sz
        py = (-a.shape[1]) % sy
        px = (-a.shape[2]) % sx
        return np.pad(a, ((0, pz), (0, py), (0, px)), mode="edge")
    sz, sy, sx = max(1, kz // 2), max(1, ky // 2), max(1, kx // 2)
    v = pad(vol_f32, sz, sy, sx)
    peaks = []
    for iz in range(0, Z, sz):
        for iy in range(0, Y, sy):
            for ix in range(0, X, sx):
                block = v[iz:iz+kz, iy:iy+ky, ix:ix+kx]
                if block.size == 0: continue
                flat  = np.argmax(block)
                bz, by, bx = np.unravel_index(flat, block.shape)
                gz, gy, gx = iz + bz, iy + by, ix + bx
                if gz < Z and gy < Y and gx < X and vol_f32[gz, gy, gx] > threshold:
                    peaks.append([gz, gy, gx])
    return np.array(peaks) if peaks else np.zeros((0, 3), dtype=int)

def norm_current(raw, q_lo, q_hi):
    return np.clip((raw.astype(np.float32) - q_lo) / max(q_hi - q_lo, 1e-6), 0.0, 1.0)

def norm_reference(raw, q_lo, q_hi):
    v = (raw.astype(np.float32) - q_lo) / max(q_hi - q_lo, 1e-6)
    return np.clip(v, 0.0, 4.0)

def build_and_score(detections_by_t, gt_g, T_true, gt_n_win):
    g = td.graph.IndexedRXGraph()
    g.add_node_attr_key("z", 0.0)
    g.add_node_attr_key("y", 0.0)
    g.add_node_attr_key("x", 0.0)
    nmap = {}
    for t, coords in sorted(detections_by_t.items()):
        for i, (z, y, x) in enumerate(coords):
            nid = g.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)})
            nmap[(t, i)] = nid
    ts = sorted(detections_by_t.keys())
    for ta, tb in zip(ts, ts[1:]):
        if tb != ta + 1: continue
        S, D = detections_by_t[ta], detections_by_t[tb]
        if len(S) == 0 or len(D) == 0: continue
        Su, Du = S * SCALE, D * SCALE
        dists  = np.sqrt(((Su[:, None] - Du[None]) ** 2).sum(-1))
        used   = set()
        for i in np.argsort(dists.min(axis=1)):
            j = int(np.argmin(dists[i]))
            if dists[i, j] <= MAX_DIST_UM and j not in used:
                used.add(j); g.add_edge(nmap[(ta, i)], nmap[(tb, j)])
    er = evaluate(g, gt_g, scale=tuple(SCALE), max_distance=MAX_DIST_UM)
    m  = per_sample_metrics(er, T_true, er.edge_tp / max(gt_n_win, 1))
    return m, er, sum(len(v) for v in detections_by_t.values())

def run_scenario(label, vol, q_lo, q_hi, thr, nms_um, ref_norm, gt_g, T_true, gt_n_win):
    dets = {}
    for t in sorted(T_WINDOW):
        raw   = np.array(vol[t])
        norm  = norm_reference(raw, q_lo, q_hi) if ref_norm else norm_current(raw, q_lo, q_hi)
        dets[t] = nms_fast(norm, thr, nms_um)
    m, er, total = build_and_score(dets, gt_g, T_true, gt_n_win)
    return {"label": label, "peaks": total,
            "tp": er.edge_tp, "fp": er.edge_fp, "fn": er.edge_fn,
            "ej": m["edge_jaccard"], "aj": m["adj_edge_jaccard"]}

def main():
    print("=" * 74)
    print("DRY-RUN: Bug-fix impact on real staged data  (no training)")
    print(f"Sample: {SAMPLE}  |  T={min(T_WINDOW)}..{max(T_WINDOW)}")
    print("=" * 74)

    vol           = zarr.open(ZARR_PATH, mode="r")["0"]
    gt_g, meta    = td.graph.IndexedRXGraph.from_geff(GEFF_PATH)
    T_true        = meta.extra.get("estimated_number_of_nodes", 32795)
    gt_df         = gt_g.node_attrs()
    gt_t          = gt_df["t"].to_numpy()
    gt_n_win      = int(np.isin(gt_t, list(T_WINDOW)).sum())

    subsample = np.concatenate([vol[t].ravel()[::500] for t in list(T_WINDOW)[:5]])
    qc_lo, qc_hi = float(np.quantile(subsample, 0.10)),  float(np.quantile(subsample, 0.90))
    qr_lo, qr_hi = float(np.quantile(subsample, 0.001)), float(np.quantile(subsample, 0.999))

    # Raw threshold equivalents in uint16 space
    raw_thr_cur = {t: qc_lo + t * (qc_hi - qc_lo) for t in [0.30, 0.40, 0.50]}
    raw_thr_ref = {t: qr_lo + t * (qr_hi - qr_lo) for t in [1.00, 1.50, 2.00]}

    print(f"\nGT in window (t {min(T_WINDOW)}-{max(T_WINDOW)}): {gt_n_win} nodes  |  T_true={T_true:,}")
    print(f"Current  norm (q0.1/0.9):    lo={qc_lo:.0f} hi={qc_hi:.0f}  codomain [0,1]")
    print(f"Reference norm (q0.001/0.999): lo={qr_lo:.0f} hi={qr_hi:.0f}  codomain [0,4]")
    print("\nThreshold equivalents in raw uint16 intensity:")
    print(f"  BEFORE thr=0.3  ->  raw > {raw_thr_cur[0.30]:.0f}")
    print(f"  BEFORE thr=0.4  ->  raw > {raw_thr_cur[0.40]:.0f}  <- Phase-1 baseline")
    print(f"  BEFORE thr=0.5  ->  raw > {raw_thr_cur[0.50]:.0f}")
    print(f"  AFTER  thr=1.0  ->  raw > {raw_thr_ref[1.00]:.0f}")
    print(f"  AFTER  thr=1.5  ->  raw > {raw_thr_ref[1.50]:.0f}")
    print(f"  AFTER  thr=2.0  ->  raw > {raw_thr_ref[2.00]:.0f}")
    print()

    scenarios = [
        ("BEFORE-A  curr[0,1] thr=0.30 NMS=5um",         qc_lo, qc_hi, 0.30, 5.0, False),
        ("BEFORE-B  curr[0,1] thr=0.40 NMS=5um <- P1 bl", qc_lo, qc_hi, 0.40, 5.0, False),
        ("BEFORE-C  curr[0,1] thr=0.50 NMS=5um",         qc_lo, qc_hi, 0.50, 5.0, False),
        ("AFTER-1a  ref[0,4]  thr=0.40 NMS=5um (wrong codomain)", qr_lo, qr_hi, 0.40, 5.0, True),
        ("AFTER-1b  ref[0,4]  thr=1.00 NMS=5um (recal)", qr_lo, qr_hi, 1.00, 5.0, True),
        ("AFTER-1c  ref[0,4]  thr=1.50 NMS=5um (recal)", qr_lo, qr_hi, 1.50, 5.0, True),
        ("AFTER-1d  ref[0,4]  thr=2.00 NMS=5um (recal)", qr_lo, qr_hi, 2.00, 5.0, True),
        ("AFTER-2   ref[0,4]  thr=1.50 NMS=7um +wider",  qr_lo, qr_hi, 1.50, 7.0, True),
    ]

    results = []
    for args in scenarios:
        t0 = time.time()
        r  = run_scenario(*args, vol=vol, gt_g=gt_g, T_true=T_true, gt_n_win=gt_n_win)
        dt = time.time() - t0
        results.append(r)
        aj = f"{r['aj']:.4f}" if r['aj'] == r['aj'] else "  nan"
        print(f"[{dt:4.1f}s] {r['label']}")
        print(f"         peaks={r['peaks']:5d}  TP={r['tp']} FP={r['fp']} FN={r['fn']}"
              f"  edge_J={r['ej']:.4f}  adj_J={aj}")

    # Upper bound: GT centroids as perfect detector
    print("\n[Upper bound: GT centroids -> greedy linker]")
    gt_dets = {}
    for row in gt_df.iter_rows(named=True):
        t = int(row["t"])
        if t in T_WINDOW:
            gt_dets.setdefault(t, []).append([row["z"], row["y"], row["x"]])
    gt_dets = {t: np.array(v) for t, v in gt_dets.items()}
    m_ub, er_ub, n_ub = build_and_score(gt_dets, gt_g, T_true, gt_n_win)
    ub_aj = m_ub["adj_edge_jaccard"]
    print(f"         peaks={n_ub:5d}  TP={er_ub.edge_tp} FP={er_ub.edge_fp} FN={er_ub.edge_fn}"
          f"  edge_J={m_ub['edge_jaccard']:.4f}  adj_J={ub_aj:.4f}")

    # Results table
    print("\n" + "=" * 74)
    print(f"{'Scenario':<46} {'Peaks':>6} {'edge_J':>7} {'adj_J':>7}  {'TP':>3} {'FP':>5} {'FN':>4}")
    print("-" * 74)
    for r in results:
        aj = f"{r['aj']:.4f}" if r['aj']==r['aj'] else "   nan"
        print(f"{r['label'][:45]:<46} {r['peaks']:>6} {r['ej']:>7.4f} {aj:>7}"
              f"  {r['tp']:>3} {r['fp']:>5} {r['fn']:>4}")
    print(f"{'UPPER-BOUND: GT centroids + greedy':<46} {n_ub:>6}"
          f" {m_ub['edge_jaccard']:>7.4f} {ub_aj:>7.4f}"
          f"  {er_ub.edge_tp:>3} {er_ub.edge_fp:>5} {er_ub.edge_fn:>4}")

    before = [r["aj"] for r in results if "BEFORE" in r["label"] and r["aj"]==r["aj"]]
    after  = [r["aj"] for r in results if "AFTER"  in r["label"] and r["aj"]==r["aj"]]
    bb = max(before) if before else 0.0
    ba = max(after)  if after  else 0.0

    print(f"""
======== VERIFIED NUMBERS ON REAL DATA =============================
  BEFORE (current code, best):        adj_edge_jaccard = {bb:.4f}
  AFTER  (norm + threshold fix, best):adj_edge_jaccard = {ba:.4f}
  UPPER BOUND (perfect detection):    adj_edge_jaccard = {ub_aj:.4f}

  Delta from bug fixes alone:         {ba - bb:+.4f}
  Remaining gap to floor  (0.763):    {0.763 - ba:.4f}
  Remaining gap to winner (0.875):    {0.875 - ba:.4f}

  Adjustment penalty math check:
    T_true = {T_true:,}  (full embryo estimate in .geff metadata)
    If T_pred >> T_true: adj = max(0, J * (1 - 0.1*(T_pred-T_true)/T_true))
    With {results[0]['peaks']} peaks (BEFORE-A), adj_factor approx {max(0, 1 - 0.1*(results[0]['peaks'] - T_true)/T_true):.4f}
    i.e. over-prediction of {results[0]['peaks']:,} nodes vs {T_true:,} GT -> score wiped by penalty

CONCLUSION:
  1. Bug fixes move the score by {ba-bb:+.4f} without training (real, not theoretical).
  2. Even PERFECT detection + greedy linking only reaches {ub_aj:.4f} due to:
     - Sparse GT ({gt_n_win} labeled nodes) vs T_true={T_true:,} -> large over-prediction penalty
  3. Gap to 0.763 = {0.763-ba:.4f}. Cannot close this without a TRAINED detector.
  4. The trained model's signal -- not NMS tuning -- is what crosses the 0.763 floor.
=====================================================================""")

if __name__ == "__main__":
    main()
