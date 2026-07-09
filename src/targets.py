"""
Target generation for detection (heatmaps) and edge prediction.

Supports two heatmap target types:
- Point targets: single voxel per centroid (sparse, ~0.1% positive)
- Dilated Gaussian: anisotropic Gaussian around centroid (softer, ~1-2% positive)
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tracksdata
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def generate_heatmap_targets(
    sample_id: str,
    geff_path: str | Path,
    volume_shape: tuple[int, int, int, int],
    anisotropy: tuple[float, float, float] = (4.0, 1.0, 1.0),
    target_type: Literal['point', 'gaussian'] = 'gaussian',
    sigma_z: float = 1.0,
    sigma_yx: float = 2.0,
) -> tuple[dict, dict]:
    """
    Generate heatmap targets from .geff ground truth.

    Args:
        sample_id: Sample identifier
        geff_path: Path to .geff file
        volume_shape: (T, Z, Y, X) shape of the volume
        anisotropy: (z_ratio, y_ratio, x_ratio) for physical scaling
        target_type: 'point' or 'gaussian'
        sigma_z: Gaussian sigma for Z axis (voxels)
        sigma_yx: Gaussian sigma for Y/X axes (voxels)

    Returns:
        (heatmaps, metadata) tuple where:
        - heatmaps: dict mapping t -> (1, Z, Y, X) heatmap tensor [0,1]
        - metadata: dict with stats (num_centroids, num_timepoints, etc.)
    """
    # Load ground truth from .geff
    try:
        graph, geff_metadata = tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))
    except Exception as e:
        logger.error(f"Failed to load .geff for {sample_id}: {e}")
        raise

    T, Z, Y, X = volume_shape
    heatmaps = {}
    centroid_count = 0
    timepoint_counts = {}

    # Get all node attributes (returns polars DataFrame)
    try:
        node_attrs_df = graph.node_attrs()
        # Convert to dict of lists
        t_vals = node_attrs_df['t'].to_list()
        z_vals = node_attrs_df['z'].to_list()
        y_vals = node_attrs_df['y'].to_list()
        x_vals = node_attrs_df['x'].to_list()
    except Exception as e:
        logger.error(f"Failed to extract node attributes: {e}")
        raise

    # Group centroids by timepoint
    centroids_by_t = {}
    for t, z, y, x in zip(t_vals, z_vals, y_vals, x_vals, strict=True):
        if t not in centroids_by_t:
            centroids_by_t[t] = []
        centroids_by_t[t].append({'z': float(z), 'y': float(y), 'x': float(x)})

    # Extract centroids for each timepoint
    for t in range(T):
        heatmap = np.zeros((1, Z, Y, X), dtype=np.float32)
        nodes_at_t = centroids_by_t.get(t, [])
        timepoint_counts[t] = len(nodes_at_t)
        centroid_count += len(nodes_at_t)

        # Add centroids to heatmap
        if target_type == 'point':
            # Point targets: single voxel per centroid
            for node in nodes_at_t:
                z_idx = int(np.round(node['z']))
                y_idx = int(np.round(node['y']))
                x_idx = int(np.round(node['x']))

                # Bounds check
                if 0 <= z_idx < Z and 0 <= y_idx < Y and 0 <= x_idx < X:
                    heatmap[0, z_idx, y_idx, x_idx] = 1.0

        elif target_type == 'gaussian':
            # Dilated Gaussian targets
            for node in nodes_at_t:
                z_c, y_c, x_c = node['z'], node['y'], node['x']

                # Create Gaussian around centroid
                for z in range(max(0, int(z_c - 3 * sigma_z)), min(Z, int(z_c + 3 * sigma_z) + 1)):
                    for y in range(max(0, int(y_c - 3 * sigma_yx)), min(Y, int(y_c + 3 * sigma_yx) + 1)):
                        for x in range(max(0, int(x_c - 3 * sigma_yx)), min(X, int(x_c + 3 * sigma_yx) + 1)):
                            # Anisotropic Gaussian
                            dz = (z - z_c) / sigma_z
                            dy = (y - y_c) / sigma_yx
                            dx = (x - x_c) / sigma_yx
                            gauss_val = np.exp(-(dz**2 + dy**2 + dx**2) / 2)
                            heatmap[0, z, y, x] = max(heatmap[0, z, y, x], gauss_val)

        heatmaps[t] = torch.from_numpy(heatmap).float()

    # Compile metadata
    metadata = {
        'sample_id': sample_id,
        'volume_shape': volume_shape,
        'total_centroids': centroid_count,
        'centroids_per_frame': timepoint_counts,
        'target_type': target_type,
        'sigma_z': sigma_z if target_type == 'gaussian' else None,
        'sigma_yx': sigma_yx if target_type == 'gaussian' else None,
    }

    return heatmaps, metadata


def generate_edge_targets(
    sample_id: str,
    geff_path: str | Path,
    nodes_t: torch.Tensor,
    nodes_t1: torch.Tensor,
    t: int,
    max_distance: float = 7.0,
    physical_voxel_size: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> tuple[torch.Tensor, dict]:
    """
    Generate edge probability targets from .geff ground truth.

    Candidate nodes are matched to GT nodes by nearest-neighbor distance in
    physical (micron) space, independently at frame t and frame t+1. A
    candidate edge (i, j) is labeled positive only if BOTH nodes_t[i] and
    nodes_t1[j] have a GT match within max_distance AND a real GT edge exists
    between those two matched GT nodes.

    Args:
        sample_id: Sample identifier
        geff_path: Path to .geff file
        nodes_t: (n_t, 3) node coordinates at frame t [z, y, x], voxel units
        nodes_t1: (n_t1, 3) node coordinates at frame t+1 [z, y, x], voxel units
        t: Timepoint index of nodes_t (nodes_t1 is assumed to be frame t+1)
        max_distance: Maximum GT-match distance (um). Default 7.0 matches this
            competition's real scoring match gate (src/evaluation.py DEFAULT_MAX_DISTANCE).
        physical_voxel_size: (z, y, x) micrometers per voxel, for converting
            voxel-space distances to physical distances before gating.

    Returns:
        (edge_labels, metadata) tuple where:
        - edge_labels: (n_t * n_t1,) binary tensor [0, 1], row-major over (i, j)
        - metadata: dict with stats (num_candidates, num_matched_to_gt,
          num_positive_edges, class_imbalance_ratio, division_mask, etc.)
    """
    # Load ground truth
    try:
        graph, geff_metadata = tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))
    except Exception as e:
        logger.error(f"Failed to load .geff for {sample_id}: {e}")
        raise

    n_t = nodes_t.shape[0]
    n_t1 = nodes_t1.shape[0]

    if n_t == 0 or n_t1 == 0:
        # Return empty labels for empty node sets
        return torch.zeros(0, dtype=torch.long), {
            'sample_id': sample_id,
            't': t,
            'num_candidates': 0,
            'num_matched_to_gt': 0,
            'num_positive_edges': 0,
            'num_negative_edges': 0,
            'class_imbalance_ratio': 0.0,
            'division_mask': torch.zeros(0, dtype=torch.bool),
        }

    node_attrs_df = graph.node_attrs(attr_keys=['t', 'node_id', 'z', 'y', 'x'])
    gt_t = node_attrs_df.filter(node_attrs_df['t'] == t)
    gt_t1 = node_attrs_df.filter(node_attrs_df['t'] == t + 1)
    dividing = set(graph.dividing_nodes())
    scale = np.array(physical_voxel_size)  # (z, y, x) um per voxel

    def match_to_gt(candidate_coords: torch.Tensor, gt_frame) -> list[int | None]:
        """Nearest-neighbor match each candidate to a GT node id, gated by max_distance (um)."""
        if gt_frame.height == 0:
            return [None] * candidate_coords.shape[0]
        gt_coords = np.stack(
            [gt_frame['z'].to_numpy(), gt_frame['y'].to_numpy(), gt_frame['x'].to_numpy()],
            axis=1,
        )
        gt_ids = gt_frame['node_id'].to_list()
        cand = candidate_coords.numpy() if isinstance(candidate_coords, torch.Tensor) else np.asarray(candidate_coords)
        dists_um = cdist(cand * scale, gt_coords * scale)
        nearest_idx = dists_um.argmin(axis=1)
        nearest_dist = dists_um[np.arange(len(cand)), nearest_idx]
        return [
            gt_ids[idx] if dist <= max_distance else None
            for idx, dist in zip(nearest_idx, nearest_dist, strict=True)
        ]

    matched_t = match_to_gt(nodes_t, gt_t)
    matched_t1 = match_to_gt(nodes_t1, gt_t1)

    edge_labels = []
    division_mask = []
    num_matched_pairs = 0

    for i in range(n_t):
        gt_src = matched_t[i]
        for j in range(n_t1):
            gt_tgt = matched_t1[j]
            label = 0
            is_division = False
            if gt_src is not None and gt_tgt is not None:
                num_matched_pairs += 1
                if graph.has_edge(gt_src, gt_tgt):
                    label = 1
                    is_division = gt_src in dividing
            edge_labels.append(label)
            division_mask.append(is_division)

    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    division_mask = torch.tensor(division_mask, dtype=torch.bool)
    candidate_count = n_t * n_t1

    num_positive = int((edge_labels == 1).sum().item())
    num_negative = int((edge_labels == 0).sum().item())

    metadata = {
        'sample_id': sample_id,
        't': t,
        'num_candidates': candidate_count,
        'num_matched_to_gt': num_matched_pairs,
        'num_positive_edges': num_positive,
        'num_negative_edges': num_negative,
        'class_imbalance_ratio': num_positive / candidate_count if candidate_count > 0 else 0.0,
        'num_division_edges': int(division_mask.sum().item()),
        'division_mask': division_mask,
    }

    return edge_labels, metadata


class DivisionLoss(torch.nn.Module):
    """
    Weighted BCE loss for edge prediction with division event upweighting.

    Division edges (where parent has >1 children) get higher loss weight.
    """

    def __init__(self, weight_division: float = 2.0, pos_weight: float = 10.0):
        """
        Initialize division loss.

        Args:
            weight_division: Loss weight multiplier for division edges (default 2.0-3.0x)
            pos_weight: Weight for positive class in BCE (to handle class imbalance)
        """
        super().__init__()
        self.weight_division = weight_division
        self.pos_weight = pos_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                division_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: (n_candidates,) edge logits
            targets: (n_candidates,) binary edge labels [0, 1]
            division_mask: (n_candidates,) boolean mask for division edges

        Returns:
            Scalar loss (mean over candidates with weighting)
        """
        # Compute base BCE loss
        targets_float = targets.float() if targets.dtype == torch.bool else targets
        loss = self.bce_loss(logits, targets_float)

        # Apply class imbalance weighting
        loss = loss * (self.pos_weight * targets_float + (1.0 - targets_float))

        # Apply division edge weighting
        if division_mask is not None:
            division_float = division_mask.float() if division_mask.dtype == torch.bool else division_mask
            loss = loss * (self.weight_division * division_float + (1.0 - division_float))

        return loss.mean()


class DetectionLoss(torch.nn.Module):
    """
    Weighted BCE loss for heatmap detection with inverse-frequency weighting.

    Upweights rare positive voxels to handle extreme class imbalance.
    """

    def __init__(self, weight_pos: float = 1.0, weight_neg: float = 0.01):
        """
        Initialize detection loss.

        Args:
            weight_pos: Weight for positive (cell) voxels
            weight_neg: Weight for negative (background) voxels
        """
        super().__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss for detection.

        Args:
            logits: (B, 1, Z, Y, X) detection logits
            targets: (B, 1, Z, Y, X) binary heatmap targets [0, 1]

        Returns:
            Scalar loss (weighted mean)
        """
        # Compute base BCE loss
        loss = self.bce_loss(logits, targets.float())

        # Apply class imbalance weighting
        loss = loss * (self.weight_pos * targets.float() + self.weight_neg * (1.0 - targets.float()))

        return loss.mean()
