"""
Local submission generator (P0-6, Part D).

Runs a verified, manifest-referenced checkpoint's inference over the real
TEST set (no ground truth -- test samples have no .geff) via the shared
production pipeline (src/submission_pipeline.py), and writes a Kaggle-
compliant submission CSV. Fail-closed: refuses to run against an unverified
checkpoint, a source/checkpoint SHA mismatch, or missing hyperparameters, and
refuses to accept an unmanifested, header-only, zero-node, or zero-edge
output.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from src.checkpoint_manifest import load_verified_checkpoint
from src.model import SimpleNodeTransformer, UNet3D
from src.submission_exporter import export_submission, validate_submission
from src.submission_pipeline import run_submission_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("generate_submission")

DEFAULT_SOURCE_SHA_FILE = "kaggle_src_dataset/GIT_SHA.txt"
DEFAULT_TEST_DIR = "data/staging/test"
DEFAULT_OUTPUT = "submissions/smoke_test_submission.csv"


def _read_source_sha(source_sha_file: Path) -> str:
    if not source_sha_file.exists():
        raise RuntimeError(f"Source SHA file not found: {source_sha_file}")
    raw = source_sha_file.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"Source SHA file {source_sha_file} is empty or whitespace-only.")
    if len(raw) != 40 or raw != raw.lower() or not all(c in "0123456789abcdef" for c in raw):
        raise RuntimeError(
            f"Source SHA file {source_sha_file} does not contain a 40-character "
            f"lowercase hex git SHA: {raw!r}"
        )
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a local submission from a verified checkpoint.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to checkpoint_manifest.json.")
    parser.add_argument(
        "--source-sha-file", type=Path, default=Path(DEFAULT_SOURCE_SHA_FILE),
        help=f"Path to a file containing the expected 40-char training source SHA (default: {DEFAULT_SOURCE_SHA_FILE}).",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path(DEFAULT_TEST_DIR),
        help=f"Directory containing test *.zarr samples (default: {DEFAULT_TEST_DIR}).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT),
        help=f"Output submission CSV path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    expected_source_sha = _read_source_sha(args.source_sha_file)
    logger.info(f"Expected source SHA (from {args.source_sha_file}): {expected_source_sha}")

    checkpoint, manifest, checkpoint_path = load_verified_checkpoint(
        args.manifest, expected_source_sha=expected_source_sha, map_location=device,
    )
    logger.info(f"Loaded verified checkpoint: {checkpoint_path}")
    logger.info(
        f"Manifest identity: training_code_sha={manifest['training_code_sha']} "
        f"split_membership_sha256={manifest['split_membership_sha256']} "
        f"model_contract={manifest['model_contract']} epoch={manifest['epoch']} "
        f"coverage={manifest['validation_samples_evaluated']}/{manifest['validation_samples_total']} "
        f"adjusted_edge_jaccard={manifest['adjusted_edge_jaccard']}"
    )

    hyperparams = checkpoint["hyperparams"]

    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128)).to(device)
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4).to(device)
    unet3d.load_state_dict(checkpoint["unet3d_state_dict"], strict=True)
    transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=True)
    unet3d.eval()
    transformer.eval()

    test_dir = args.test_dir
    test_zarrs = sorted(test_dir.glob("*.zarr"))
    logger.info(f"Found {len(test_zarrs)} real test sample(s) in {test_dir}: {[z.stem for z in test_zarrs]}")

    t0 = time.monotonic()
    pred_graphs, diagnostics = run_submission_inference(
        test_dir=test_dir,
        test_zarrs=test_zarrs,
        unet3d=unet3d,
        transformer=transformer,
        device=device,
        hyperparams=hyperparams,
    )
    elapsed = time.monotonic() - t0

    required_dataset_ids = diagnostics["required_dataset_ids"]
    logger.info(
        f"Inference complete in {elapsed:.1f}s: {diagnostics['total_unique_nodes']} nodes, "
        f"{diagnostics['total_unique_edges']} edges, "
        f"{diagnostics['total_accepted_edges']}/{diagnostics['total_candidate_edges']} edges accepted."
    )
    for sample_id in required_dataset_ids:
        sample_diag = diagnostics["per_sample"][sample_id]
        logger.info(
            f"  {sample_id}: pairs={sample_diag['processed_pair_count']}/"
            f"{sample_diag['expected_pair_count']} nodes={sample_diag['unique_node_count']} "
            f"edges={sample_diag['unique_edge_count']} elapsed={sample_diag['elapsed_seconds']:.1f}s"
        )
    if "cuda_max_memory_allocated_bytes" in diagnostics:
        logger.info(f"CUDA peak memory allocated: {diagnostics['cuda_max_memory_allocated_bytes']} bytes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = export_submission(pred_graphs, args.output, required_dataset_ids=required_dataset_ids)
    logger.info(f"export_submission returned: {csv_path}")

    is_valid = validate_submission(csv_path, required_dataset_ids=required_dataset_ids)
    logger.info(f"validate_submission() result: {is_valid}")
    if not is_valid:
        raise RuntimeError("Generated submission.csv failed validate_submission() -- do not submit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
