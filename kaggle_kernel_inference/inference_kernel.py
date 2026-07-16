"""
Kaggle inference kernel for ST-ACT -- Code Competition compliant submission.

This is a SEPARATE kernel from train_kernel.py, built after discovering (from
the real /rules page, not guessed) that biohub-cell-tracking-during-development
is a Code Competition: the graded submission must be a committed Notebook with
"Internet access disabled", GPU/CPU runtime <= 12 hours, writing submission.csv.
train_kernel.py's live `pip install` from PyPI cannot run here -- every
dependency must already be present. This script installs from a pre-downloaded
wheels Dataset instead, via `pip install --no-index --find-links=...`.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Same self-discovery pattern as train_kernel.py, generalized to find any
# marker file under /kaggle/input rather than assuming a fixed slug/mount
# path -- confirmed necessary earlier this session (Kaggle nests attached
# datasets under /kaggle/input/datasets/<owner>/<slug>/, not flat).
MAX_SEARCH_DEPTH = 5


def find_kaggle_input_dir(marker_relpath: str) -> str | None:
    if not os.path.exists("/kaggle/input"):
        return None
    for dirpath, dirnames, _filenames in os.walk("/kaggle/input"):
        depth = dirpath[len("/kaggle/input"):].count(os.sep)
        if depth >= MAX_SEARCH_DEPTH:
            dirnames[:] = []
            continue
        if os.path.isfile(os.path.join(dirpath, marker_relpath)):
            return dirpath
    return None


def find_best_checkpoint() -> str | None:
    """Find the most-recently-modified epoch_*.pt under /kaggle/input.

    Deliberately does not assume a fixed filename: save_checkpoint() only
    writes a new file when val_score improves on the previous best, so the
    most-recently-modified epoch_*.pt IS the best one -- this lets inference
    pick up whatever checkpoint the latest training run produced (e.g.
    epoch_1_val_score_0.0000.pt from the earlier sanity run vs. a real
    epoch_N_val_score_X.XXXX.pt from the full run) without a code change.
    """
    if not os.path.exists("/kaggle/input"):
        return None
    candidates = []
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        depth = dirpath[len("/kaggle/input"):].count(os.sep)
        if depth >= MAX_SEARCH_DEPTH:
            dirnames[:] = []
            continue
        for name in filenames:
            if name.startswith("epoch_") and name.endswith(".pt"):
                full_path = os.path.join(dirpath, name)
                candidates.append((os.path.getmtime(full_path), full_path))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


KAGGLE_SRC_DATASET_DIR = find_kaggle_input_dir(os.path.join("src", "dataset.py"))
if KAGGLE_SRC_DATASET_DIR:
    sys.path.insert(0, KAGGLE_SRC_DATASET_DIR)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='[Kaggle Inference] %(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger("KaggleInference")

KAGGLE_MODE = os.path.exists("/kaggle/input")
if KAGGLE_MODE:
    try:
        logger.info(f"/kaggle/input contents: {os.listdir('/kaggle/input')}")
    except Exception as e:
        logger.warning(f"Could not list /kaggle/input: {e}")

# === ENVIRONMENT SETUP ===
# Deployed code identity: same mechanism as train_kernel.py (see
# DEFERRED_IMPROVEMENTS.md) -- read the SHA embedded at push time by
# scripts/sync_kaggle_src.py so "is this run using the code I think I
# committed" is a 2-second log check, not a post-mortem.
DEPLOYED_SHA = "unknown (GIT_SHA.txt not found -- was this pushed via scripts/sync_kaggle_src.py?)"
if KAGGLE_SRC_DATASET_DIR:
    _sha_file = Path(KAGGLE_SRC_DATASET_DIR) / "GIT_SHA.txt"
    if _sha_file.exists():
        DEPLOYED_SHA = _sha_file.read_text().strip()
logger.info(f"Deployed code SHA: {DEPLOYED_SHA}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    # Same fail-fast GPU compute-capability check as train_kernel.py -- fail
    # in milliseconds if an incompatible GPU (e.g. P100) was allocated,
    # instead of after loading test data.
    major, minor = torch.cuda.get_device_capability(device)
    cc_string = f"sm_{major}{minor}"
    supported_archs = torch.cuda.get_arch_list()
    if cc_string not in supported_archs:
        raise RuntimeError(
            f"CUDA capability mismatch: {torch.cuda.get_device_name(device)} is "
            f"{cc_string}, but this PyTorch build only supports: "
            f"{', '.join(supported_archs)}. Reselect a supported GPU (T4)."
        )
    logger.info(f"GPU compute capability {cc_string} verified compatible.")

# === OFFLINE DEPENDENCY INSTALL ===
# Same 17 packages as train_kernel.py's pip install list, but from a local
# wheels Dataset (drbhatiasanjay/st-act-wheels, pre-downloaded for
# manylinux2014_x86_64 cp312 -- matches Kaggle's real base image) instead of
# live PyPI -- required because Internet access is disabled for the real
# graded run. --no-deps stays load-bearing (see train_kernel.py's documented
# numpy-corruption lesson): the wheels list already includes every real
# transitive dependency, checked via the same audit method used earlier.
if KAGGLE_MODE:
    import subprocess

    WHEELS_DIR = find_kaggle_input_dir("tracksdata-0.1.0rc6-py3-none-any.whl")
    if WHEELS_DIR is None:
        raise RuntimeError(
            "st-act-wheels dataset not found under /kaggle/input -- attach it "
            "in the notebook's Input panel before running."
        )
    logger.info(f"Installing dependencies offline from {WHEELS_DIR}")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
            f"--find-links={WHEELS_DIR}",
            "tracksdata==0.1.0rc6", "zarr>=3.0.0", "numcodecs>=0.11.0",
            "geff>=1.0.0", "geff-spec",
            "dask", "bidict", "rustworkx", "psygnal", "donfig", "google-crc32c",
            "typer", "ilpy", "pyscipopt", "blosc2>=2.0.0",
        ],
        check=True,
    )

    # polars needs --force-reinstall, separately from the rest: a real run
    # (v22-27 of the training kernel) proved plain `pip install polars` --
    # offline OR from PyPI, doesn't matter -- leaves Kaggle's pre-existing
    # base-image polars untouched if it already satisfies the version bound,
    # even though that pre-existing install may be missing its compiled
    # extension entirely. This exact fix already exists in train_kernel.py;
    # omitting it here (a real bug, not a hypothetical) reproduced the
    # identical "Polars binary is missing!" failure on the very first real
    # run of this script.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
            "--force-reinstall", f"--find-links={WHEELS_DIR}",
            "polars>=1.36.0", "polars-runtime-32",
        ],
        check=True,
    )
    logger.info("Offline dependency installation complete.")

    import polars as _pl_check
    try:
        from polars._plr import PySeries as _PySeriesCheck  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"polars compiled extension (_plr) is broken after offline install: {e}"
        ) from e
    logger.info(f"polars {_pl_check.__version__} extension verified OK.")

import polars as pl  # noqa: E402
from tracksdata.graph import IndexedRXGraph  # noqa: E402

from src.dataset import CompetitionDataset  # noqa: E402
from src.evaluation import DEFAULT_SCALE  # noqa: E402
from src.inference import greedy_edge_assignment  # noqa: E402
from src.model import SimpleNodeTransformer, UNet3D  # noqa: E402
from src.submission_exporter import export_submission, validate_submission  # noqa: E402
from src.train import extract_peaks_from_volume  # noqa: E402


def peaks_for_channel(detection_probs: torch.Tensor, channel: int, hyperparams: dict) -> list:
    vol_np = detection_probs[0, channel].cpu().numpy()
    threshold = hyperparams['detection_threshold']
    positive_fraction = float((vol_np > threshold).mean())
    max_positive_fraction = hyperparams.get('max_positive_voxel_fraction', 0.005)
    if positive_fraction > max_positive_fraction:
        threshold = max(
            float(np.percentile(vol_np, 100 * (1 - max_positive_fraction))), threshold
        )
    elif positive_fraction == 0.0:
        # Opposite failure mode: raw confidence never crosses the fixed
        # threshold anywhere -- see src/train.py::_peaks_for_channel for the
        # full rationale (same duplicated logic). Without this, a real
        # submission run would silently emit zero detections forever.
        threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
    return extract_peaks_from_volume(
        vol_np, threshold=threshold, voxel_size=DEFAULT_SCALE,
        nms_radius_um=hyperparams['nms_radius_um']
    )


def nodes_and_features(features: torch.Tensor, peaks: list, device: torch.device):
    if len(peaks) == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, features.shape[1]), dtype=torch.float32, device=device),
        )
    nodes = torch.tensor(peaks, dtype=torch.float32, device=device)
    zc = torch.clamp(nodes[:, 0].long(), 0, features.shape[2] - 1)
    yc = torch.clamp(nodes[:, 1].long(), 0, features.shape[3] - 1)
    xc = torch.clamp(nodes[:, 2].long(), 0, features.shape[4] - 1)
    return nodes, features[0, :, zc, yc, xc].t()


def main():
    # === LOAD CHECKPOINT ===
    checkpoint_path_str = find_best_checkpoint()
    if checkpoint_path_str is None:
        raise RuntimeError("No epoch_*.pt checkpoint found under /kaggle/input.")
    checkpoint_path = Path(checkpoint_path_str)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128)).to(device)
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4).to(device)
    unet3d.load_state_dict(checkpoint['unet3d_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    unet3d.eval()
    transformer.eval()

    hyperparams = checkpoint.get('hyperparams', {
        'edge_threshold': 0.5, 'detection_threshold': 0.5, 'nms_radius_um': 5.0,
    })

    # === REAL TEST DATA (swapped in at grading time, per rules) ===
    test_dir = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test")
    test_zarrs = sorted(test_dir.glob("*.zarr"))
    required_dataset_ids = [z.stem for z in test_zarrs]
    logger.info(f"Found {len(test_zarrs)} real test samples: {required_dataset_ids}")

    all_pred_graphs = {}

    for zarr_path in test_zarrs:
        sample_id = zarr_path.stem
        dataset = CompetitionDataset.__new__(CompetitionDataset)
        dataset.data_dir = test_dir
        dataset.split_type = "test"
        dataset.normalize = True
        dataset.anisotropy = (4.0, 1.0, 1.0)
        dataset.physical_voxel_size = (1.625, 0.40625, 0.40625)
        dataset.zip_path = None
        dataset.sample_ids = [sample_id]
        dataset.pairs = []
        dataset._loader_cache = {}
        # P0-1 fix (2026-07-16): submission inference must see every real
        # consecutive pair -- test samples have no .geff at all, so GT-count
        # filtering isn't even meaningful here, let alone desirable.
        dataset.filter_unannotated_pairs = False
        dataset._gt_counts_by_time_cache = {}
        dataset.annotation_pair_stats = None
        dataset._build_pair_index()

        pred_graph = IndexedRXGraph()
        for key in ('t', 'x', 'y', 'z'):
            try:
                pred_graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass
        all_pred_graphs[sample_id] = pred_graph

        if len(dataset) == 0:
            logger.warning(f"No pairs built for {sample_id}, leaving graph empty")
            continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                frame_t = batch['frame_t'].to(device)
                frame_t1 = batch['frame_t1'].to(device)
                t_idx = int(batch.get('t_idx', [0])[0])

                x = torch.cat([frame_t, frame_t1], dim=1)
                logits, features = unet3d(x)
                detection_probs = torch.sigmoid(logits)

                peaks_t = peaks_for_channel(detection_probs, 0, hyperparams)
                peaks_t1 = peaks_for_channel(detection_probs, 1, hyperparams)
                nodes_t, features_t = nodes_and_features(features, peaks_t, device)
                nodes_t1, features_t1 = nodes_and_features(features, peaks_t1, device)

                if len(peaks_t) > 0 and len(peaks_t1) > 0:
                    edge_probs = transformer(nodes_t, nodes_t1, features_t, features_t1)
                    assignment = greedy_edge_assignment(
                        edge_probs, nodes_t.cpu(), nodes_t1.cpu(),
                        threshold=hyperparams['edge_threshold'], max_children=2, max_parents=1
                    )
                    edges = assignment['edges']
                else:
                    edges = []

                node_id_map_t = {}
                for i, (z, y, xc) in enumerate(peaks_t):
                    node_id_map_t[i] = pred_graph.add_node(
                        {'t': t_idx, 'x': int(round(xc)), 'y': int(round(y)), 'z': int(round(z))}
                    )
                node_id_map_t1 = {}
                for j, (z, y, xc) in enumerate(peaks_t1):
                    node_id_map_t1[j] = pred_graph.add_node(
                        {'t': t_idx + 1, 'x': int(round(xc)), 'y': int(round(y)), 'z': int(round(z))}
                    )
                for src_idx, tgt_idx, _prob in edges:
                    pred_graph.add_edge(node_id_map_t[src_idx], node_id_map_t1[tgt_idx], {})

        logger.info(f"{sample_id}: {pred_graph.num_nodes()} nodes, {pred_graph.num_edges()} edges")

    out_path = "/kaggle/working/submission.csv"
    logger.info(f"Exporting submission to {out_path}")
    csv_path = export_submission(all_pred_graphs, out_path, required_dataset_ids=required_dataset_ids)
    is_valid = validate_submission(csv_path)
    logger.info(f"validate_submission() result: {is_valid}")
    if not is_valid:
        raise RuntimeError("Generated submission.csv failed validate_submission() -- do not submit.")


if __name__ == "__main__":
    main()
