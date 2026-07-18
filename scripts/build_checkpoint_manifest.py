#!/usr/bin/env python3
"""
CLI to build/replace a verified checkpoint_manifest.json for one checkpoint
(P0-6, Part B9).

Calls production helpers only (src/checkpoint_manifest.py) -- never invents
metadata, and fails loudly for a legacy or deployment-ineligible checkpoint
rather than writing a manifest for it.

Usage:
    py scripts/build_checkpoint_manifest.py --checkpoint C:\\path\\to\\epoch_N_val_score_X.XXXX.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from src.checkpoint_manifest import write_checkpoint_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/replace a verified checkpoint_manifest.json beside one checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the epoch_N_val_score_X.XXXX.pt checkpoint to manifest.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    manifest_path = write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"Checkpoint path:          {checkpoint_path}")
    print(f"Manifest path:            {manifest_path}")
    print(f"Checkpoint SHA-256:       {manifest['checkpoint_sha256']}")
    print(f"Training code SHA:        {manifest['training_code_sha']}")
    print(f"Model contract:           {manifest['model_contract']}")
    print(f"Split membership SHA-256: {manifest['split_membership_sha256']}")
    print(f"Epoch:                    {manifest['epoch']}")
    print(
        f"Validation coverage:      {manifest['validation_samples_evaluated']}/"
        f"{manifest['validation_samples_total']} "
        f"(full_fold={manifest['validation_is_full_fold']})"
    )
    print(f"Datasets evaluated:       {manifest['num_datasets']}")
    print(f"Predicted nodes total:    {manifest['predicted_nodes_total']}")
    print(f"Predicted edges total:    {manifest['predicted_edges_total']}")
    print(f"Adjusted edge Jaccard:    {manifest['adjusted_edge_jaccard']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
