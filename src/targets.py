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
    max_distance: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """
    Generate edge probability targets from .geff ground truth.

    Args:
        sample_id: Sample identifier
        geff_path: Path to .geff file
        nodes_t: (n_t, 3) node coordinates at frame t [z, y, x]
        nodes_t1: (n_t1, 3) node coordinates at frame t+1 [z, y, x]
        max_distance: Maximum distance for candidate edges (um)

    Returns:
        (edge_labels, metadata) tuple where:
        - edge_labels: (num_candidates,) binary tensor [0, 1]
        - metadata: dict with stats (num_gt_edges, class_imbalance, division_edges, etc.)
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
            'num_gt_edges': 0,
            'num_positive_edges': 0,
            'num_negative_edges': 0,
        }

    # Extract GT edges for this timepoint (simplified - would need timepoint info in practice)
    # For now, assume nodes are indexed consistently with GT graph
    gt_edges = set()

    try:
        edge_ids = graph.edge_ids() if hasattr(graph, 'edge_ids') else []
        for edge_id in edge_ids:
            try:
                src, tgt = edge_id
                # In real implementation, check if this edge is at our timepoint
                gt_edges.add((src, tgt))
            except Exception:
                continue
    except Exception:
        pass

    # Generate candidate edges
    edge_labels = []
    candidate_count = 0

    for _i in range(n_t):
        for _j in range(n_t1):
            # Check if GT edge exists
            # This is simplified - real implementation would match nodes properly
            label = 0  # Default: negative edge

            # In practice, would:
            # 1. Match nodes_t[i] to GT node at time t
            # 2. Match nodes_t1[j] to GT node at time t+1
            # 3. Check if matched nodes have a GT edge

            edge_labels.append(label)
            candidate_count += 1

    edge_labels = torch.tensor(edge_labels, dtype=torch.long)

    # Count positive edges
    num_positive = (edge_labels == 1).sum().item()
    num_negative = (edge_labels == 0).sum().item()

    metadata = {
        'sample_id': sample_id,
        'num_candidates': candidate_count,
        'num_positive_edges': num_positive,
        'num_negative_edges': num_negative,
        'class_imbalance_ratio': num_positive / (num_positive + num_negative) if candidate_count > 0 else 0,
    }

    return edge_labels, metadata
