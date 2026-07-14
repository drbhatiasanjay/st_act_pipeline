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


def load_geff_cached(geff_path: str | Path, geff_cache: dict | None):
    """
    Load a .geff graph, reusing a previously-parsed result if geff_cache is
    given. Confirmed via a real Kaggle run + code audit: a single training
    batch calls IndexedRXGraph.from_geff() on the SAME sample's .geff file
    up to 4 times (twice in TrainingLoop._get_gt_nodes for t and t+1, once
    each here and in generate_edge_targets), ~600 redundant re-parses per
    epoch across only a handful of distinct files. Caller owns the cache's
    lifetime (e.g. TrainingLoop.__init__'s self._geff_cache) so it can be
    dropped/reset between runs; passing None preserves the original
    always-reparse behavior for standalone/test callers.
    """
    key = str(geff_path)
    if geff_cache is not None and key in geff_cache:
        return geff_cache[key]
    result = tracksdata.graph.IndexedRXGraph.from_geff(key)
    if geff_cache is not None:
        geff_cache[key] = result
    return result


def generate_heatmap_targets(
    sample_id: str,
    geff_path: str | Path,
    volume_shape: tuple[int, int, int, int],
    anisotropy: tuple[float, float, float] = (4.0, 1.0, 1.0),
    target_type: Literal['point', 'gaussian'] = 'gaussian',
    sigma_z: float = 1.0,
    sigma_yx: float = 2.0,
    target_ts: list[int] | None = None,
    geff_cache: dict | None = None,
) -> tuple[dict, dict]:
    """
    Generate heatmap targets from .geff ground truth.

    Args:
        sample_id: Sample identifier
        geff_path: Path to .geff file
        volume_shape: (T, Z, Y, X) shape of the volume. T is only used to
            validate bounds when target_ts is given; when target_ts is None
            (default), heatmaps are computed for every t in range(T).
        anisotropy: CURRENTLY UNUSED -- accepted for API compatibility with
            scripts/benchmark_heatmap_targets.py (which passes it believing
            it has an effect) but never referenced in this function body.
            The actual Gaussian shape is controlled entirely by sigma_z/
            sigma_yx below. Do not assume passing a different anisotropy
            here changes anything -- verified false 2026-07-14 (see
            CLAUDE.md's "Scientific/mathematical claims" section for the
            full incident: the original commit calls this an "Anisotropic
            Gaussian" and claims alignment with REFERENCE_IMPLEMENTATION.md,
            but that document contains no supporting content, and whether
            sigma_z=1.0/sigma_yx=2.0 -- physically 1.625um Z vs 0.8125um Y/X,
            a 2x mismatch either way you slice it -- is correct is a genuine
            open question, not yet empirically resolved. Two competing
            hypotheses: (a) should be geometrically isotropic in real
            microns (would need sigma_z=0.5), or (b) deliberately biased
            toward Z to model real light-sheet PSF anisotropy (worse axial
            resolution than lateral, independent of voxel sampling) -- (b)
            would argue the current values, or something like them, are
            reasonable. Do NOT change sigma_z/sigma_yx without empirical
            A/B validation against real local eval score -- see
            DEFERRED_IMPROVEMENTS.md.
        target_type: 'point' or 'gaussian'
        sigma_z: Gaussian sigma for Z axis (voxels)
        sigma_yx: Gaussian sigma for Y/X axes (voxels)
        target_ts: If given, only compute heatmaps for these absolute
            timepoints (real GT `t` values, not relative indices) instead of
            every t in range(T) -- avoids allocating/filling a full-size
            (1,Z,Y,X) array per timepoint when only a specific real t is
            needed (e.g. one training batch's t_idx out of a ~100-frame
            sample). Backward compatible: omitting it preserves the original
            all-timepoints behavior for existing callers.
        geff_cache: Optional dict for reusing a previously-parsed .geff
            graph across calls that share the same geff_path (see
            load_geff_cached). Omitting it re-parses every call.

    Returns:
        (heatmaps, metadata) tuple where:
        - heatmaps: dict mapping t -> (1, Z, Y, X) heatmap tensor [0,1]
        - metadata: dict with stats (num_centroids, num_timepoints, etc.)
    """
    # Load ground truth from .geff
    try:
        graph, geff_metadata = load_geff_cached(geff_path, geff_cache)
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

    # Extract centroids for each timepoint (or just target_ts, if given --
    # real absolute t values, not relative indices into range(T))
    for t in (target_ts if target_ts is not None else range(T)):
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
            # Dilated Gaussian targets. Vectorized over each node's local
            # bounding box with numpy broadcasting instead of a per-voxel
            # Python loop -- the previous triple-nested for-loop called
            # np.exp() individually per voxel (~1,200 voxels/node at
            # default sigmas), confirmed via a real Kaggle run to be the
            # dominant per-batch cost (generate_heatmap_targets runs once
            # per training batch). Same math, same per-node max-combine for
            # overlapping Gaussians, just computed as one array op per node.
            for node in nodes_at_t:
                z_c, y_c, x_c = node['z'], node['y'], node['x']

                z_lo, z_hi = max(0, int(z_c - 3 * sigma_z)), min(Z, int(z_c + 3 * sigma_z) + 1)
                y_lo, y_hi = max(0, int(y_c - 3 * sigma_yx)), min(Y, int(y_c + 3 * sigma_yx) + 1)
                x_lo, x_hi = max(0, int(x_c - 3 * sigma_yx)), min(X, int(x_c + 3 * sigma_yx) + 1)
                if z_lo >= z_hi or y_lo >= y_hi or x_lo >= x_hi:
                    continue

                zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
                dz = (zz - z_c) / sigma_z
                dy = (yy - y_c) / sigma_yx
                dx = (xx - x_c) / sigma_yx
                gauss_block = np.exp(-(dz**2 + dy**2 + dx**2) / 2).astype(np.float32)

                heatmap[0, z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = np.maximum(
                    heatmap[0, z_lo:z_hi, y_lo:y_hi, x_lo:x_hi], gauss_block
                )

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
    geff_cache: dict | None = None,
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
        geff_cache: Optional dict for reusing a previously-parsed .geff
            graph across calls that share the same geff_path (see
            load_geff_cached). Omitting it re-parses every call.

    Returns:
        (edge_labels, metadata) tuple where:
        - edge_labels: (n_t * n_t1,) binary tensor [0, 1], row-major over (i, j)
        - metadata: dict with stats (num_candidates, num_matched_to_gt,
          num_positive_edges, class_imbalance_ratio, division_mask, etc.)
    """
    # Load ground truth
    try:
        graph, geff_metadata = load_geff_cached(geff_path, geff_cache)
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
        # .cpu() before .numpy(): train.py moves nodes_t/nodes_t1 to the GPU
        # device (for feature-tensor indexing) right before calling
        # generate_edge_targets() with them -- a bare .numpy() on a CUDA
        # tensor raises here, caught by train.py's own try/except and
        # silently skipping edge loss for every batch with real GT at both
        # frames (confirmed via a real Kaggle run's repeated "Edge target
        # generation failed ... can't convert cuda:0 device type tensor to
        # numpy" warning). Matches the pattern already used correctly
        # elsewhere in train.py (e.g. nodes_t.cpu().numpy()).
        cand = (
            candidate_coords.detach().cpu().numpy()
            if isinstance(candidate_coords, torch.Tensor)
            else np.asarray(candidate_coords)
        )
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

    A fixed weight_neg cannot be correct across this dataset: empirically measured
    (real .geff ground truth, 2026-07-13) background/foreground loss contribution
    ratios range from ~67x (dense samples, ~7-10 cells/frame) to ~667x (sparse
    samples, 0-1 cells/frame) even after a 100x static compensation -- the sparse
    half of the dataset was under-compensated by up to ~6.7x, which is the
    confirmed root cause of a real full-epoch training run (14,751 batches)
    producing val_score=0.0 (zero detections above threshold everywhere, verified
    directly from the Kaggle log: all 182 validation filter calls used an empty
    node list). adaptive=True (default) computes weight_neg per-batch from the
    batch's own foreground/background voxel mass so positive and negative
    contributions balance regardless of local cell density, instead of assuming
    one dataset-wide ratio.

    CRITICAL, found by adversarial review (2026-07-13) of the first adaptive
    version and confirmed numerically before trusting it: the first
    implementation returned loss.mean() (dividing by targets_f.numel(), ~4.19M
    voxels). Since adaptive weighting makes BOTH the positive and negative
    contributions proportional to the tiny foreground pixel count for
    cell-containing batches, .mean() then shrinks the ENTIRE loss by
    ~pos_mass/numel on top of the balancing already done -- e.g. measured
    0.0000208 for a real cell-containing batch vs 0.0069 for an empty one, a
    333x GRADIENT SUPPRESSION on exactly the batches that matter, the opposite
    of the intended fix. A cheap 200-batch Kaggle verification run confirmed
    this empirically (sigmoid stuck at [0.0000, 0.0000] even with the "fix"
    active). Normalizing by the actual sum of applied weights instead keeps
    loss magnitude stable (~0.69, real BCE-for-p=0.5 scale) regardless of cell
    density, so gradient flows to positive voxels the way the balancing intends.
    """

    def __init__(self, weight_pos: float = 1.0, weight_neg: float = 0.01, adaptive: bool = True):
        """
        Initialize detection loss.

        Args:
            weight_pos: Weight for positive (cell) voxels
            weight_neg: Weight for negative (background) voxels. Used directly when
                adaptive=False, and as the fallback when adaptive=True but a batch
                has zero foreground mass (no GT cells in that batch at all, so
                there is nothing to balance against).
            adaptive: If True, compute weight_neg per-batch as
                weight_pos * pos_mass / neg_mass instead of using the fixed
                weight_neg -- see class docstring for why a fixed ratio fails.
        """
        super().__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.adaptive = adaptive
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
        targets_f = targets.float()

        # Compute base BCE loss
        loss = self.bce_loss(logits, targets_f)

        pos_mass = targets_f.sum()
        neg_mass = targets_f.numel() - pos_mass

        if self.adaptive:
            if pos_mass > 0 and neg_mass > 0:
                weight_neg = self.weight_pos * pos_mass / neg_mass
            else:
                # No GT cells in this batch (or, pathologically, no background) --
                # nothing to balance against, fall back to the fixed ratio.
                weight_neg = self.weight_neg
        else:
            weight_neg = self.weight_neg

        # Apply class imbalance weighting
        weights = self.weight_pos * targets_f + weight_neg * (1.0 - targets_f)
        loss = loss * weights

        # Normalize by the sum of APPLIED weights, not numel -- see class
        # docstring's CRITICAL note. loss.mean() would divide by numel
        # regardless of how the weights were balanced, silently re-introducing
        # a ~pos_mass/numel gradient suppression on cell-containing batches
        # even though the pos/neg CONTRIBUTIONS were correctly balanced above.
        sum_weights = weights.sum().clamp(min=1e-12)
        return loss.sum() / sum_weights
