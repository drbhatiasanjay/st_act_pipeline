"""
Training loop for ST-ACT model with end-to-end correctness.

Handles:
- Data loading with GT node coordinates (teacher forcing)
- Loss computation with real GT edge targets
- Validation with full inference pipeline (NMS + Transformer + greedy assignment)
- Checkpointing and early stopping
- Comprehensive logging with fallback tracking
"""

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tracksdata.graph import IndexedRXGraph

from src.evaluation import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_SCALE,
    evaluate_submission,
    load_geff_ground_truth,
)
from src.inference import greedy_edge_assignment
from src.targets import (
    DetectionLoss,
    DivisionLoss,
    generate_edge_targets,
    generate_heatmap_targets,
    load_geff_cached,
)

logger = logging.getLogger(__name__)


def pool_kernel_from_um(um: float, voxel_size: tuple) -> tuple:
    """Convert physical microns to voxel kernel size."""
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def extract_peaks_from_volume(
    vol: np.ndarray,
    threshold: float = 0.4,
    voxel_size: tuple = DEFAULT_SCALE,
    nms_radius_um: float = 5.0
) -> list:
    """
    Real 3D non-max suppression via maximum_filter with centroid collapsing.

    Returns list of [z, y, x] peak coordinates.
    """
    kernel = pool_kernel_from_um(nms_radius_um, voxel_size)
    pooled = ndimage.maximum_filter(vol, size=kernel, mode='constant', cval=-np.inf)
    is_peak = (vol == pooled) & (vol > threshold)

    labeled, num_labels = ndimage.label(is_peak)
    if num_labels == 0:
        return []

    centroids = ndimage.center_of_mass(is_peak, labeled, range(1, num_labels + 1))
    return [list(c) for c in centroids]


class TrainingLoop:
    """
    End-to-end training loop for ST-ACT model.

    Manages training/validation epochs, loss computation, early stopping,
    checkpointing, and logging with explicit error tracking.
    """

    def __init__(
        self,
        unet3d: nn.Module,
        transformer: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        data_dir: str | Path = "data/staging/train",
        checkpoint_dir: str = "checkpoints",
        log_file: str = "training_log.csv",
        hyperparams: dict[str, Any] | None = None,
    ):
        """
        Initialize training loop.

        Args:
            unet3d: UNet3D detection model
            transformer: SimpleNodeTransformer edge prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: torch.device (cpu or cuda)
            data_dir: Directory containing .geff ground truth files
            checkpoint_dir: Directory for saving model checkpoints
            log_file: Path for CSV training log
            hyperparams: Training hyperparameters
        """
        self.unet3d = unet3d
        self.transformer = transformer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = log_file

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters with defaults
        self.hyperparams = {
            'learning_rate': 1e-4,
            'grad_clip': 1.0,
            'weight_decay': 1e-4,
            'heatmap_loss_weight': 1.0,
            'division_loss_weight': 2.5,
            'early_stopping_patience': 10,
            'edge_threshold': 0.5,
            'detection_threshold': 0.5,
            'nms_radius_um': 5.0,
            'seed': 42,
            # None = unlimited (real full-training default). A real run
            # revealed the true per-epoch batch count is ~14,751 (149
            # samples x ~99 consecutive-frame pairs each), not the ~199
            # this session originally assumed -- 3 epochs at that scale is
            # ~17 real hours, not the ~30min a "sanity check" is meant to
            # take. Kaggle callers should set this explicitly to validate
            # the pipeline end-to-end fast instead.
            'max_batches_per_epoch': None,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        # Set random seed
        torch.manual_seed(self.hyperparams['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.hyperparams['seed'])
        np.random.seed(self.hyperparams['seed'])

        # Collect all model parameters for optimizer
        all_params = list(unet3d.parameters()) + list(transformer.parameters())

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            all_params,
            lr=self.hyperparams['learning_rate'],
            weight_decay=self.hyperparams['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Initialize loss functions
        # adaptive=True (default): weight_neg below is only the fallback used for
        # batches with zero GT cells -- real batches get a per-batch-computed
        # weight_neg instead (see DetectionLoss docstring, src/targets.py).
        self.detection_loss_fn = DetectionLoss(weight_pos=1.0, weight_neg=0.01)
        self.division_loss_fn = DivisionLoss(
            weight_division=self.hyperparams['division_loss_weight'],
            pos_weight=10.0
        )

        # Early stopping state
        self.best_val_score = -np.inf
        self.epochs_without_improvement = 0
        self.best_checkpoint_path = None

        # Error tracking (for silent fallback detection per CLAUDE.md lesson)
        self.epoch_fallback_counts = {
            'heatmap_generation_failure': 0,
            'edge_target_generation_failure': 0,
            'edge_loss_computation_failure': 0,
            'evaluation_failure': 0,
        }
        self.last_epoch_wall_clock_seconds = 0.0
        self.last_epoch_num_batches = 0

        # Cache of parsed .geff graphs, keyed by path -- a single training
        # batch calls IndexedRXGraph.from_geff() on the SAME sample's .geff
        # up to 4 times (twice in _get_gt_nodes for t/t+1, once each in
        # generate_heatmap_targets/generate_edge_targets), confirmed via a
        # real Kaggle run + code audit to be ~600 redundant re-parses per
        # epoch across only a handful of distinct files. Safe to reuse the
        # returned graph object across calls -- every caller only reads it
        # (node_attrs/dividing_nodes/has_edge), never mutates it.
        self._geff_cache: dict = {}

        # Logging
        self._init_log()

    def _init_log(self):
        """Initialize CSV log file with header."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'val_edge_jaccard',
                'val_adjusted_edge_jaccard',
                'val_division_jaccard',
                'val_score',
                'learning_rate',
                'heatmap_failures',
                'edge_target_failures',
                'edge_loss_failures',
                'eval_failures',
                'num_batches',
                'epoch_wall_clock_seconds',
            ])

    def _log_epoch(self, epoch: int, train_loss: float, val_metrics: dict[str, float]):
        """Log epoch results to CSV."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{train_loss:.6f}',
                f'{val_metrics.get("edge_jaccard", np.nan):.6f}',
                f'{val_metrics.get("adjusted_edge_jaccard", np.nan):.6f}',
                f'{val_metrics.get("division_jaccard", np.nan):.6f}',
                f'{val_metrics.get("score", np.nan):.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                self.epoch_fallback_counts['heatmap_generation_failure'],
                self.epoch_fallback_counts['edge_target_generation_failure'],
                self.epoch_fallback_counts['edge_loss_computation_failure'],
                self.epoch_fallback_counts['evaluation_failure'],
                self.last_epoch_num_batches,
                f'{self.last_epoch_wall_clock_seconds:.1f}',
            ])

    def _get_gt_nodes(self, sample_id: str, t_idx: int) -> torch.Tensor | None:
        """
        Extract GT node coordinates from .geff at a specific timepoint.

        Args:
            sample_id: Dataset sample ID
            t_idx: Timepoint index

        Returns:
            (n_nodes, 3) tensor of [z, y, x] coordinates, or None if load fails
        """
        try:
            geff_path = self.data_dir / f"{sample_id}.geff"
            if not geff_path.exists():
                return None

            graph, _ = load_geff_cached(geff_path, self._geff_cache)
            node_attrs = graph.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
            nodes_at_t = node_attrs.filter(node_attrs['t'] == t_idx)

            if nodes_at_t.height == 0:
                return torch.zeros((0, 3), dtype=torch.float32)

            coords = np.stack(
                [
                    nodes_at_t['z'].to_numpy(),
                    nodes_at_t['y'].to_numpy(),
                    nodes_at_t['x'].to_numpy(),
                ],
                axis=1
            )
            return torch.from_numpy(coords).float()
        except Exception as e:
            logger.warning(f"Failed to get GT nodes for {sample_id} at t={t_idx}: {e}")
            return None

    def train_epoch(self) -> float:
        """
        Run one training epoch.

        Uses real GT node coordinates (teacher forcing) for both detection
        loss and edge targets. Logs all fallback activations.

        Returns:
            Average training loss for the epoch
        """
        self.unet3d.train()
        self.transformer.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        # Reset fallback counters for this epoch
        for key in self.epoch_fallback_counts:
            self.epoch_fallback_counts[key] = 0

        max_batches = self.hyperparams.get('max_batches_per_epoch')

        for batch_idx, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                logger.info(f"Stopping train epoch early at {max_batches} batches (max_batches_per_epoch cap)")
                break

            # Move batch to device
            frame_t = batch['frame_t'].to(self.device)
            frame_t1 = batch['frame_t1'].to(self.device)
            sample_id = batch['sample_id'][0]  # Batch size 1
            t_idx = batch.get('t_idx', [0])[0]

            # Concatenate frames: (B, 2, Z, Y, X)
            x = torch.cat([frame_t, frame_t1], dim=1)

            # Forward pass through UNet3D
            logits, features = self.unet3d(x)

            # === DETECTION LOSS (teacher forcing) ===
            # Generate real GT heatmap targets for BOTH the batch's absolute
            # t_idx AND t_idx+1 -- UNet3D's detection head is 2-channel
            # (channel 0 = frame_t's own detections, channel 1 = frame_t1's),
            # so both real timepoints need real supervision from a single
            # forward pass. Single-channel-only supervision (the original
            # design) forced validate_epoch() into either reusing one
            # volume's peaks for both timepoints (corrupting edge scoring,
            # since peaks_t/peaks_t1 were then identical) or stitching
            # predictions from two separate forward passes together (a
            # confirmed train-test feature distribution mismatch, since this
            # loop's edge loss always slices features_t/features_t1 from the
            # SAME shared `features` tensor below -- never from two
            # different forward passes). target_ts=[t_idx] alone previously
            # also had a real bug: passing volume_shape=(1,...) made
            # generate_heatmap_targets() iterate `for t in range(1)` == only
            # t=0, so heatmap_targets_dict.get(t_idx, ...) silently missed
            # for any real t_idx != 0 and fell back to an all-zero target
            # with no fallback counted. target_ts=[...] makes the function
            # compute exactly (and only) the real timepoints needed.
            z, y, x = frame_t.shape[2:]
            volume_shape = (int(t_idx) + 2, z, y, x)  # T only used for bounds validation now
            try:
                heatmap_targets_dict, _ = generate_heatmap_targets(
                    sample_id,
                    str(self.data_dir / f"{sample_id}.geff"),
                    volume_shape,
                    target_type='gaussian',
                    target_ts=[int(t_idx), int(t_idx) + 1],
                    geff_cache=self._geff_cache,
                )
                # heatmaps[t] is (1, Z, Y, X) -- stack channel 0 (t_idx) and
                # channel 1 (t_idx+1), then add batch dim to match logits'
                # (B, 2, Z, Y, X) for DetectionLoss/BCEWithLogitsLoss.
                zero_channel = torch.zeros((1, z, y, x), dtype=torch.float32)
                heatmap_ch0 = heatmap_targets_dict.get(t_idx, zero_channel)
                heatmap_ch1 = heatmap_targets_dict.get(t_idx + 1, zero_channel)
                if not isinstance(heatmap_ch0, torch.Tensor):
                    heatmap_ch0 = torch.from_numpy(heatmap_ch0).float()
                if not isinstance(heatmap_ch1, torch.Tensor):
                    heatmap_ch1 = torch.from_numpy(heatmap_ch1).float()
                heatmap_target = torch.cat([heatmap_ch0, heatmap_ch1], dim=0).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.warning(f"Heatmap generation failed for {sample_id}: {e}, using zero targets")
                self.epoch_fallback_counts['heatmap_generation_failure'] += 1
                heatmap_target = torch.zeros((1, 2, z, y, x), dtype=torch.float32, device=self.device)

            # detection_loss expects (B, 2, Z, Y, X) for both logits and targets
            detection_loss = self.detection_loss_fn(logits, heatmap_target)

            # === EDGE LOSS (teacher forcing) ===
            edge_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Get GT nodes at frame t and t+1
            nodes_t = self._get_gt_nodes(sample_id, t_idx)
            nodes_t1 = self._get_gt_nodes(sample_id, t_idx + 1)

            if nodes_t is not None and nodes_t1 is not None and nodes_t.shape[0] > 0 and nodes_t1.shape[0] > 0:
                nodes_t = nodes_t.to(self.device)
                nodes_t1 = nodes_t1.to(self.device)

                # Extract features at GT node locations
                z_t = torch.clamp(nodes_t[:, 0].long(), 0, features.shape[2] - 1)
                y_t = torch.clamp(nodes_t[:, 1].long(), 0, features.shape[3] - 1)
                x_t = torch.clamp(nodes_t[:, 2].long(), 0, features.shape[4] - 1)
                features_t = features[0, :, z_t, y_t, x_t].t()  # (n_t, 128)

                z_t1 = torch.clamp(nodes_t1[:, 0].long(), 0, features.shape[2] - 1)
                y_t1 = torch.clamp(nodes_t1[:, 1].long(), 0, features.shape[3] - 1)
                x_t1 = torch.clamp(nodes_t1[:, 2].long(), 0, features.shape[4] - 1)
                features_t1 = features[0, :, z_t1, y_t1, x_t1].t()  # (n_t1, 128)

                # Generate real GT edge targets
                try:
                    edge_targets, edge_metadata = generate_edge_targets(
                        sample_id,
                        str(self.data_dir / f"{sample_id}.geff"),
                        nodes_t,
                        nodes_t1,
                        t=t_idx,
                        max_distance=DEFAULT_MAX_DISTANCE,
                        physical_voxel_size=DEFAULT_SCALE,
                        geff_cache=self._geff_cache,
                    )
                    edge_targets = edge_targets.to(self.device)
                    division_mask = edge_metadata.get('division_mask', torch.zeros_like(edge_targets, dtype=torch.bool))
                    division_mask = division_mask.to(self.device)
                except Exception as e:
                    logger.warning(f"Edge target generation failed for {sample_id}: {e}")
                    self.epoch_fallback_counts['edge_target_generation_failure'] += 1
                    edge_targets = None

                # Compute edge predictions and loss
                if edge_targets is not None:
                    try:
                        edge_probs = self.transformer(nodes_t, nodes_t1, features_t, features_t1)
                        if len(edge_probs) > 0:
                            # Convert targets to float for BCE
                            edge_targets_float = edge_targets.float()
                            edge_loss = self.division_loss_fn(
                                edge_probs.view(-1),
                                edge_targets_float,
                                division_mask
                            )
                        else:
                            edge_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    except Exception as e:
                        logger.warning(f"Edge loss computation failed for {sample_id}: {e}")
                        self.epoch_fallback_counts['edge_loss_computation_failure'] += 1
                        edge_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Total loss
            total_loss_item = (
                edge_loss +
                self.hyperparams['heatmap_loss_weight'] * detection_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_item.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.unet3d.parameters()) + list(self.transformer.parameters()),
                self.hyperparams['grad_clip']
            )

            self.optimizer.step()

            total_loss += total_loss_item.item()
            num_batches += 1

            # Every 5 batches (not 1/5th of the epoch, ~40 batches here) so
            # a genuinely slow epoch is visible within the first minute --
            # a real ~75min run was let continue for over 40min before
            # anyone had a rate/ETA signal to notice it was worth stopping.
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(self.train_loader):
                elapsed = time.time() - epoch_start_time
                rate = elapsed / (batch_idx + 1)
                eta_remaining = rate * (len(self.train_loader) - (batch_idx + 1))
                logger.info(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {total_loss_item.item():.6f}, "
                    f"elapsed={elapsed:.1f}s, {rate:.2f}s/batch, "
                    f"eta_remaining={eta_remaining:.0f}s"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.last_epoch_wall_clock_seconds = time.time() - epoch_start_time
        self.last_epoch_num_batches = num_batches

        # Log fallback counts
        for key, count in self.epoch_fallback_counts.items():
            if count > 0:
                logger.warning(f"Epoch had {count} {key} fallbacks")

        # Hard-fail if a majority of batches hit the same fallback: a real
        # sanity-check run's whole point is validating the pipeline works,
        # so silently completing on mostly-fallback data (as happened for a
        # full ~75min Kaggle run when polars._plr was silently broken --
        # every GT lookup fell back to all-zero targets with no crash) is
        # worse than failing loudly. Occasional real failures (a missing
        # .geff, an edge case) are expected and shouldn't abort a run.
        FALLBACK_RATE_THRESHOLD = 0.5
        for key, count in self.epoch_fallback_counts.items():
            rate = count / num_batches if num_batches > 0 else 0.0
            if rate > FALLBACK_RATE_THRESHOLD:
                raise RuntimeError(
                    f"Epoch aborted: {key} fired on {count}/{num_batches} batches "
                    f"({rate * 100:.1f}%, threshold {FALLBACK_RATE_THRESHOLD * 100:.0f}%). "
                    f"Training would silently produce a checkpoint from garbage data -- "
                    f"diagnose the root cause before retrying."
                )

        logger.info(f"Train epoch average loss: {avg_loss:.6f}")
        return avg_loss

    def _peaks_for_channel(self, detection_probs: torch.Tensor, channel: int, t_idx: int) -> list:
        """Extract NMS peaks from one channel of a (B, 2, Z, Y, X) detection map.

        An undertrained model's raw sigmoid output sits near 0.5 almost
        everywhere (near-zero logits), so a fixed threshold can flag a huge
        fraction of voxels as "peaks" -- ndimage.label() over that much noise
        then hangs/balloons memory. Same failure mode hit and fixed in
        scripts/benchmark_heatmap_targets.py this session; apply the same
        adaptive-threshold guard here.
        """
        vol_np = detection_probs[0, channel].cpu().numpy()
        threshold = self.hyperparams['detection_threshold']
        positive_fraction = float((vol_np > threshold).mean())
        max_positive_fraction = self.hyperparams.get('max_positive_voxel_fraction', 0.005)
        if positive_fraction > max_positive_fraction:
            adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
            logger.warning(
                f"Validation t_idx={t_idx} ch={channel}: threshold={threshold} flags "
                f"{positive_fraction*100:.2f}% of voxels (undertrained-model miscalibration) "
                f"-- using adaptive threshold={adaptive_threshold:.4f} instead"
            )
            threshold = max(adaptive_threshold, threshold)
        return extract_peaks_from_volume(
            vol_np,
            threshold=threshold,
            voxel_size=DEFAULT_SCALE,
            nms_radius_um=self.hyperparams['nms_radius_um']
        )

    def _nodes_and_features_at_peaks(
        self, features: torch.Tensor, peaks: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (n, 3) node coords and (n, C) feature vectors at given peak locations."""
        if len(peaks) == 0:
            return (
                torch.zeros((0, 3), dtype=torch.float32, device=self.device),
                torch.zeros((0, features.shape[1]), dtype=torch.float32, device=self.device),
            )
        nodes = torch.tensor(peaks, dtype=torch.float32, device=self.device)
        zc = torch.clamp(nodes[:, 0].long(), 0, features.shape[2] - 1)
        yc = torch.clamp(nodes[:, 1].long(), 0, features.shape[3] - 1)
        xc = torch.clamp(nodes[:, 2].long(), 0, features.shape[4] - 1)
        feats = features[0, :, zc, yc, xc].t()
        return nodes, feats

    def validate_epoch(self) -> dict[str, float]:
        """
        Run one validation epoch.

        Uses full inference pipeline: NMS peak-finding -> Transformer -> greedy assignment.
        No teacher forcing; simulates real inference.

        Returns:
            Dictionary with validation metrics
        """
        self.unet3d.eval()
        self.transformer.eval()

        import polars as pl

        all_pred_graphs: dict[str, IndexedRXGraph] = {}
        all_gt_graphs = {}
        all_gt_metadata = {}

        self.epoch_fallback_counts['evaluation_failure'] = 0

        # Each batch is now fully self-contained: UNet3D's detection head is
        # 2-channel (channel 0 = frame_t's own detections at t_idx, channel
        # 1 = frame_t1's own detections at t_idx+1, see model.py), so a
        # single forward pass gives real, DISTINCT peaks for both
        # timepoints, with features_t/features_t1 both sliced from that
        # SAME forward pass's shared `features` tensor -- matching exactly
        # how train_epoch's edge loss extracts features (see the comment
        # there), unlike an earlier attempt at this fix that cached
        # detections across batches and ended up pairing features from two
        # SEPARATE forward passes (a confirmed train-test distribution
        # mismatch, caught via an adversarial review). This also fixes a
        # second bug that fix had: the final frame of each sample is no
        # longer dropped, since channel 1 of the LAST batch gives real
        # detections for the sample's last timepoint (previously
        # unreachable -- no batch ever had it as an own/first-frame t_idx).
        #
        # Known, accepted limitation: this does NOT deduplicate nodes at
        # intermediate timepoints -- timepoint t_idx+1 gets predicted twice
        # (once as this batch's channel-1, once as the next batch's own
        # channel-0), both added as separate graph nodes rather than
        # merged. This inflates predicted node count roughly 2x at
        # non-boundary timepoints, which would need a proper nearest-
        # neighbor merge before this validation score is trustworthy for a
        # real Task 3.4 go/no-go decision -- acceptable for now to unblock
        # verifying training itself works, not to finalize scoring.
        max_batches = self.hyperparams.get('max_batches_per_epoch')

        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.val_loader):
                if max_batches is not None and _batch_idx >= max_batches:
                    logger.info(f"Stopping validation early at {max_batches} batches (max_batches_per_epoch cap)")
                    break

                frame_t = batch['frame_t'].to(self.device)
                frame_t1 = batch['frame_t1'].to(self.device)
                sample_id = batch['sample_id'][0]
                t_idx = int(batch.get('t_idx', [0])[0])

                # IndexedRXGraph.add_node_attr_key() requires an explicit
                # dtype/default (bare key-name-only raises "dtype is
                # required when not using AttrSchema"), add_node() takes a
                # single attrs dict and returns an auto-assigned int id,
                # add_edge() needs those returned int ids plus an attrs
                # dict. Mirrors the already-proven pattern in
                # run_pipeline.py:convert_nx_to_tracksdata().
                if sample_id not in all_pred_graphs:
                    pred_graph = IndexedRXGraph()
                    for key in ('t', 'x', 'y', 'z'):
                        try:
                            pred_graph.add_node_attr_key(key, pl.Int64, 0)
                        except ValueError:
                            pass  # key already exists
                    all_pred_graphs[sample_id] = pred_graph
                pred_graph = all_pred_graphs[sample_id]

                # Forward pass -- logits/detection_probs are (B, 2, Z, Y, X)
                x = torch.cat([frame_t, frame_t1], dim=1)
                logits, features = self.unet3d(x)
                detection_probs = torch.sigmoid(logits)

                peaks_t = self._peaks_for_channel(detection_probs, channel=0, t_idx=t_idx)
                peaks_t1 = self._peaks_for_channel(detection_probs, channel=1, t_idx=t_idx)
                nodes_t, features_t = self._nodes_and_features_at_peaks(features, peaks_t)
                nodes_t1, features_t1 = self._nodes_and_features_at_peaks(features, peaks_t1)

                if len(peaks_t) > 0 and len(peaks_t1) > 0:
                    edge_probs = self.transformer(nodes_t, nodes_t1, features_t, features_t1)
                    assignment = greedy_edge_assignment(
                        edge_probs,
                        nodes_t.cpu(),
                        nodes_t1.cpu(),
                        threshold=self.hyperparams['edge_threshold'],
                        max_children=2,
                        max_parents=1
                    )
                    edges = assignment['edges']
                else:
                    edges = []

                # Add both channels' detections as new nodes every batch
                # (see the known-limitation comment above the loop).
                # Coordinates cast to int to match the pl.Int64 schema above
                # (mirrors run_pipeline.py:convert_nx_to_tracksdata()) --
                # passing float values against an Int64 schema fails inside
                # evaluate_submission() with "unexpected value ... found
                # value of type Float64", silently caught by the
                # eval_failure fallback and masking the real validation
                # score with zeros.
                node_id_map_t = {}
                for i, (z, y, x) in enumerate(peaks_t):
                    node_id_map_t[i] = pred_graph.add_node({
                        't': t_idx, 'x': int(round(x)), 'y': int(round(y)), 'z': int(round(z)),
                    })
                node_id_map_t1 = {}
                for j, (z, y, x) in enumerate(peaks_t1):
                    node_id_map_t1[j] = pred_graph.add_node({
                        't': t_idx + 1, 'x': int(round(x)), 'y': int(round(y)), 'z': int(round(z)),
                    })

                for src_idx, tgt_idx, _prob in edges:
                    pred_graph.add_edge(node_id_map_t[src_idx], node_id_map_t1[tgt_idx], {})

        # Load each sample's GT graph once (not once per batch -- a sample
        # has many batches, all needing the same GT graph).
        for sample_id in all_pred_graphs:
            try:
                geff_path = self.data_dir / f"{sample_id}.geff"
                if geff_path.exists():
                    gt_graph, gt_metadata = load_geff_ground_truth(str(geff_path))
                    all_gt_graphs[sample_id] = gt_graph
                    all_gt_metadata[sample_id] = gt_metadata
            except Exception as e:
                logger.warning(f"Failed to load GT for {sample_id}: {e}")
                self.epoch_fallback_counts['evaluation_failure'] += 1

        # Hard-fail if most samples' GT couldn't load: train_epoch() already
        # has this protection (added after the polars bug ran silently for
        # ~75min producing a checkpoint from garbage data), but
        # validate_epoch() had none -- without it, a broken GT path would
        # just silently produce a meaningless near-zero val_score forever,
        # with early stopping/checkpoint selection quietly acting on noise
        # instead of a loud, fast failure.
        num_samples = len(all_pred_graphs)
        if num_samples > 0:
            eval_failure_rate = self.epoch_fallback_counts['evaluation_failure'] / num_samples
            if eval_failure_rate > 0.5:
                raise RuntimeError(
                    f"Validation aborted: GT loading failed on "
                    f"{self.epoch_fallback_counts['evaluation_failure']}/{num_samples} samples "
                    f"({eval_failure_rate * 100:.1f}%, threshold 50%). val_score would be "
                    f"meaningless -- diagnose the root cause before retrying."
                )

        # Evaluate if we have GT graphs
        if all_gt_graphs:
            try:
                val_metrics = evaluate_submission(
                    all_pred_graphs,
                    all_gt_graphs,
                    gt_metadata=all_gt_metadata
                )
                # Replace NaN with 0.0 for logging
                val_metrics_clean = {}
                for key, val in val_metrics.items():
                    if isinstance(val, float) and math.isnan(val):
                        val_metrics_clean[key] = 0.0
                    else:
                        val_metrics_clean[key] = val

                logger.info(f"Validation - Edge Jaccard: {val_metrics_clean['edge_jaccard']:.6f}, "
                          f"Adjusted: {val_metrics_clean['adjusted_edge_jaccard']:.6f}, "
                          f"Division: {val_metrics_clean['division_jaccard']:.6f}, "
                          f"Score: {val_metrics_clean['score']:.6f}")
                return val_metrics_clean
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                self.epoch_fallback_counts['evaluation_failure'] += 1
                val_metrics = {
                    'edge_jaccard': 0.0,
                    'adjusted_edge_jaccard': 0.0,
                    'division_jaccard': 0.0,
                    'score': 0.0,
                }
        else:
            val_metrics = {
                'edge_jaccard': 0.0,
                'adjusted_edge_jaccard': 0.0,
                'division_jaccard': 0.0,
                'score': 0.0,
            }

        return val_metrics

    def fit(self, num_epochs: int, max_wall_clock_seconds: float | None = None):
        """Train model for specified number of epochs.

        If max_wall_clock_seconds is set, stops cleanly (with whatever checkpoint
        already exists on disk) before starting an epoch projected to exceed the
        budget, rather than letting the platform kill the process mid-epoch. The
        projection uses the actual measured average epoch time so far, not a
        pre-run estimate -- this deliberately doesn't assume a specific
        per-batch rate, since that rate can change across code revisions (e.g.
        the Zarr per-item loader caching fix) and per-epoch cost may not be
        train-only (validate_epoch does real inference work once detections
        stop being trivially empty).
        """
        logger.info(f"Starting training for up to {num_epochs} epochs")
        logger.info(f"Hyperparameters: {json.dumps(self.hyperparams, indent=2)}")
        if max_wall_clock_seconds is not None:
            logger.info(f"Wall-clock budget: {max_wall_clock_seconds:.0f}s")
        fit_start_time = time.monotonic()

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Training
            train_loss = self.train_epoch()

            # Validation
            val_metrics = self.validate_epoch()

            # Log epoch
            self._log_epoch(epoch + 1, train_loss, val_metrics)

            # Update learning rate scheduler
            self.scheduler.step(val_metrics.get('adjusted_edge_jaccard', 0.0))

            # Early stopping check
            val_score = val_metrics.get('adjusted_edge_jaccard', -np.inf)
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, val_metrics)
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement}/"
                          f"{self.hyperparams['early_stopping_patience']} epochs")

                if self.epochs_without_improvement >= self.hyperparams['early_stopping_patience']:
                    logger.info("Early stopping triggered!")
                    break

            if max_wall_clock_seconds is not None:
                elapsed = time.monotonic() - fit_start_time
                avg_epoch_time = elapsed / (epoch + 1)
                if elapsed + avg_epoch_time > max_wall_clock_seconds:
                    logger.info(
                        f"Wall-clock budget ({max_wall_clock_seconds:.0f}s) would likely be "
                        f"exceeded by another epoch (elapsed={elapsed:.0f}s, "
                        f"avg_epoch={avg_epoch_time:.0f}s) -- stopping cleanly after "
                        f"{epoch + 1} epoch(s)."
                    )
                    break

        logger.info(f"\nTraining complete. Best val score: {self.best_val_score:.6f}")

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]):
        """Save model checkpoint."""
        val_score = metrics.get('adjusted_edge_jaccard', 0.0)
        checkpoint_name = f"epoch_{epoch}_val_score_{val_score:.4f}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            'epoch': epoch,
            'unet3d_state_dict': self.unet3d.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': metrics,
            'hyperparams': self.hyperparams,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        self.best_checkpoint_path = str(checkpoint_path)

        # Clean up old checkpoints (keep last 3)
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for old_checkpoint in checkpoints[3:]:
            old_checkpoint.unlink()
            logger.info(f"Deleted old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.unet3d.load_state_dict(checkpoint['unet3d_state_dict'])
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint.get('val_metrics', {})
