"""
Smoke-test submission generator for Task 3.5.

Runs the checkpoint's inference over the real TEST set (no ground truth --
test samples have no .geff) and writes a Kaggle-compliant submission CSV.
Reuses the exact inference pattern already verified in evaluate_checkpoint.py
(Task 3.4), minus the GT-comparison step.
"""

import logging
import sys
import time
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader
from tracksdata.graph import IndexedRXGraph

from evaluate_checkpoint import extract_peaks, find_latest_local_checkpoint, get_nodes_and_features
from src.dataset import CompetitionDataset
from src.inference import greedy_edge_assignment
from src.model import SimpleNodeTransformer, UNet3D
from src.submission_exporter import export_submission, validate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("generate_submission")


def main():
    device = torch.device("cpu")
    checkpoint_path = find_latest_local_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError("No epoch_*.pt checkpoint found anywhere under the project root")
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128))
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)
    unet3d.load_state_dict(checkpoint['unet3d_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    unet3d.eval()
    transformer.eval()

    hyperparams = checkpoint.get('hyperparams', {
        'edge_threshold': 0.5, 'detection_threshold': 0.5, 'nms_radius_um': 5.0,
    })

    test_dir = Path("data/staging/test")
    test_zarrs = sorted(test_dir.glob("*.zarr"))
    logger.info(f"Found {len(test_zarrs)} real test samples: {[z.stem for z in test_zarrs]}")

    all_pred_graphs = {}

    for zarr_path in test_zarrs:
        sample_id = zarr_path.stem
        logger.info(f"=== Running inference on TEST sample {sample_id} ===")

        # CompetitionDataset expects a data_dir + split_file/split_type; for
        # test data (no labels, no split membership) construct pairs directly
        # against this one sample instead of going through the split JSON.
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

        if len(dataset) == 0:
            logger.warning(f"No pairs built for {sample_id}, skipping")
            continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        pred_graph = IndexedRXGraph()
        for key in ('t', 'x', 'y', 'z'):
            try:
                pred_graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass
        all_pred_graphs[sample_id] = pred_graph

        t0 = time.time()
        total_nodes, total_edges = 0, 0
        with torch.no_grad():
            for batch in loader:
                frame_t = batch['frame_t'].to(device)
                frame_t1 = batch['frame_t1'].to(device)
                t_idx = int(batch.get('t_idx', [0])[0])

                x = torch.cat([frame_t, frame_t1], dim=1)
                logits, features = unet3d(x)
                detection_probs = torch.sigmoid(logits)

                peaks_t = extract_peaks(detection_probs, channel=0, t_idx=t_idx, hyperparams=hyperparams)
                peaks_t1 = extract_peaks(detection_probs, channel=1, t_idx=t_idx, hyperparams=hyperparams)
                nodes_t, features_t = get_nodes_and_features(features, peaks_t, device)
                nodes_t1, features_t1 = get_nodes_and_features(features, peaks_t1, device)

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
                    node_id_map_t[i] = pred_graph.add_node({'t': t_idx, 'x': int(round(xc)), 'y': int(round(y)), 'z': int(round(z))})
                node_id_map_t1 = {}
                for j, (z, y, xc) in enumerate(peaks_t1):
                    node_id_map_t1[j] = pred_graph.add_node({'t': t_idx + 1, 'x': int(round(xc)), 'y': int(round(y)), 'z': int(round(z))})
                for src_idx, tgt_idx, _prob in edges:
                    pred_graph.add_edge(node_id_map_t[src_idx], node_id_map_t1[tgt_idx], {})

                total_nodes += len(peaks_t) + len(peaks_t1)
                total_edges += len(edges)

        elapsed = time.time() - t0
        logger.info(
            f"{sample_id}: {len(loader)} batches in {elapsed:.1f}s -- "
            f"final graph: {pred_graph.num_nodes()} nodes, {pred_graph.num_edges()} edges "
            f"(raw totals across batches: {total_nodes} node-detections, {total_edges} edges)"
        )

    Path("submissions").mkdir(exist_ok=True)
    out_path = "submissions/smoke_test_submission.csv"
    logger.info(f"Exporting submission to {out_path}")
    csv_path = export_submission(all_pred_graphs, out_path)
    logger.info(f"export_submission returned: {csv_path}")

    logger.info("Running validate_submission()...")
    is_valid = validate_submission(csv_path)
    logger.info(f"validate_submission() result: {is_valid}")

    for sid, g in all_pred_graphs.items():
        logger.info(f"FINAL {sid}: {g.num_nodes()} nodes, {g.num_edges()} edges")


if __name__ == "__main__":
    main()
