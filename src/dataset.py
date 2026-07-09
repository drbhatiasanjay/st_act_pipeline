"""
PyTorch Dataset class for ST-ACT competition.

CompetitionDataset loads Zarr v3 volumes and .geff ground truth.
Produces (frame_t, frame_t+1) pairs with anisotropic metadata.
Respects embryo-disjoint train/val split from data_split.json.
"""

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tracksdata
from torch.utils.data import Dataset

from data_loader import AnisotropicZarrLoader

logger = logging.getLogger(__name__)


class CompetitionDataset(Dataset):
    """
    PyTorch Dataset for ST-ACT competition.

    Loads Zarr v3 volumes + .geff ground truth from either:
    - Local staged directory (for development)
    - Kaggle-mounted competition path (for training)

    Produces (frame_t, frame_t+1) pairs with anisotropic metadata.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        split_type: str = "train",
        normalize: bool = True,
        anisotropy: tuple[float, float, float] = (4.0, 1.0, 1.0),
        zip_path: str | Path | None = None,
    ):
        """
        Initialize CompetitionDataset.

        Args:
            data_dir: Path to directory containing Zarr/geff data
                      (local staging or Kaggle-mounted competition path)
            split_file: Path to data_split.json
            split_type: "train" or "validation"
            normalize: Whether to apply normalization (default: True)
            anisotropy: Anisotropy ratio (Z:Y:X), default (4.0, 1.0, 1.0)
            zip_path: Optional path to zip file for extracting samples on-the-fly
        """
        self.data_dir = Path(data_dir)
        self.split_type = split_type
        self.normalize = normalize
        self.anisotropy = anisotropy
        self.physical_voxel_size = (1.625, 0.40625, 0.40625)  # um
        self.zip_path = Path(zip_path) if zip_path else None

        # Load split file
        split_file = Path(split_file)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            split_data = json.load(f)

        self.sample_ids = split_data[split_type]
        logger.info(
            f"Loaded {len(self.sample_ids)} samples for split '{split_type}' "
            f"from {split_file}"
        )

        # Build index of (frame_t, frame_t+1) pairs
        self.pairs = []
        self._build_pair_index()

    def _build_pair_index(self) -> None:
        """Build index of all (frame_t, frame_t+1) pairs."""
        for sample_id in self.sample_ids:
            try:
                # Check if sample exists locally
                zarr_path = self.data_dir / f"{sample_id}.zarr"
                if not zarr_path.exists():
                    logger.debug(
                        f"Sample {sample_id} not found locally (OK for local testing)"
                    )
                    continue

                # Load zarr to determine number of frames
                loader = AnisotropicZarrLoader(str(zarr_path))
                num_frames = loader.get_shape()[0]

                # Add all consecutive frame pairs
                for frame_idx in range(num_frames - 1):
                    self.pairs.append((sample_id, frame_idx))

                logger.debug(
                    f"Sample {sample_id}: {num_frames} frames → {num_frames - 1} pairs"
                )

            except Exception as e:
                logger.warning(f"Failed to index sample {sample_id}: {e}")

        logger.info(f"Built index: {len(self.pairs)} (frame_t, frame_t+1) pairs")

    def __len__(self) -> int:
        """Return total number of (frame_t, frame_t+1) pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load (frame_t, frame_t+1) pair by index.

        Returns:
            {
                "frame_t": (C, H, W) uint16 or float32,
                "frame_t1": (C, H, W) uint16 or float32,
                "sample_id": str,
                "t_idx": int (frame index for frame_t),
                "metadata": {
                    "sample_id": str,
                    "t_idx": int,
                    "volume_shape": tuple,
                    "physical_voxel_size": tuple,
                    "anisotropy_ratio": tuple,
                }
            }
        """
        sample_id, frame_idx = self.pairs[idx]

        # Load volume
        zarr_path = self.data_dir / f"{sample_id}.zarr"
        loader = AnisotropicZarrLoader(str(zarr_path))

        # Extract frame_t and frame_t+1
        frame_t = loader.load_timepoint_block(frame_idx, normalize=self.normalize)
        frame_t1 = loader.load_timepoint_block(frame_idx + 1, normalize=self.normalize)

        # Ensure float32
        frame_t = frame_t.astype(np.float32)
        frame_t1 = frame_t1.astype(np.float32)

        # Add channel dimension if missing (shape should be (Z, Y, X) -> (1, Y, X) by squeeze Z)
        # The shape (Z, Y, X) is from load_timepoint_block, but we want (C, H, W) format
        # For now, use as-is which will be (Z, Y, X)
        if frame_t.ndim == 3:
            # Squeeze Z dimension to get (Y, X), then add channel: (1, Y, X)
            frame_t = frame_t[0:1, :, :]  # Take first Z slice and add channel
            frame_t1 = frame_t1[0:1, :, :]

        # Convert to torch
        frame_t = torch.from_numpy(frame_t).float()
        frame_t1 = torch.from_numpy(frame_t1).float()

        volume_shape = loader.get_shape()
        metadata = {
            "sample_id": sample_id,
            "t_idx": frame_idx,
            "volume_shape": volume_shape,
            "physical_voxel_size": self.physical_voxel_size,
            "anisotropy_ratio": self.anisotropy,
        }

        return {
            "frame_t": frame_t,
            "frame_t1": frame_t1,
            "sample_id": sample_id,
            "t_idx": frame_idx,
            "metadata": metadata,
        }

    def load_geff_gt(self, sample_id: str) -> tuple[Any, Any]:
        """
        Load ground truth graph from .geff file for a sample.

        Returns:
            (graph, metadata) tuple from tracksdata.graph.IndexedRXGraph.from_geff()
        """
        geff_path = self.data_dir / f"{sample_id}.geff"
        if not geff_path.exists():
            if self.zip_path:
                # Extract from zip
                return self._load_geff_from_zip(sample_id)
            raise FileNotFoundError(f"GEFF file not found: {geff_path}")

        return tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))

    def _load_geff_from_zip(
        self, sample_id: str
    ) -> tuple[Any, Any]:
        """Load .geff file from zip archive."""
        if not self.zip_path:
            raise ValueError("zip_path not configured")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                # Extract .geff entry
                geff_entry = f"train/{sample_id}.geff/"
                members = [m for m in zf.namelist() if m.startswith(geff_entry)]

                if not members:
                    raise FileNotFoundError(f"No .geff entries for {sample_id}")

                # Extract to temp directory
                for member in members:
                    zf.extract(member, tmpdir)

                # Load from temp location
                geff_path = Path(tmpdir) / geff_entry
                return tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))


class AugmentedCompetitionDataset(CompetitionDataset):
    """
    CompetitionDataset with optional augmentations.

    Scaffolded for Wave 3. Currently no-op augmentations.
    """

    def __init__(self, *args, augment: bool = False, **kwargs):
        """
        Initialize with augmentation flag.

        Args:
            augment: Whether to apply augmentations (not yet implemented)
            *args, **kwargs: Passed to CompetitionDataset
        """
        super().__init__(*args, **kwargs)
        self.augment = augment

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and optionally augment."""
        item = super().__getitem__(idx)

        if not self.augment:
            return item

        # TODO: Wave 3 augmentations
        # - Elastic deformation (respect anisotropy)
        # - Y/X rotation (not Z)
        # - Intensity jitter
        # - Patch dropout

        return item
