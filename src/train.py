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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import tracksdata
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

            graph, _ = tracksdata.graph.IndexedRXGraph.from_geff(str(geff_path))
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

        # Reset fallback counters for this epoch
        for key in self.epoch_fallback_counts:
            self.epoch_fallback_counts[key] = 0

        for batch_idx, batch in enumerate(self.train_loader):
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
            # Generate real GT heatmap targets
            # frame_t shape is (1, Z, Y, X), we need (T, Z, Y, X) for generate_heatmap_targets
            z, y, x = frame_t.shape[2:]
            volume_shape = (1, z, y, x)  # (T=1, Z, Y, X) - just one timepoint for this pair
            try:
                heatmap_targets_dict, _ = generate_heatmap_targets(
                    sample_id,
                    str(self.data_dir / f"{sample_id}.geff"),
                    volume_shape,
                    target_type='gaussian'
                )
                # Get frame 0 (frame_t) heatmap - it's (1, Z, Y, X)
                heatmap_target = heatmap_targets_dict.get(t_idx, torch.zeros((1, z, y, x), dtype=torch.float32))
                if not isinstance(heatmap_target, torch.Tensor):
                    heatmap_target = torch.from_numpy(heatmap_target).float()
                heatmap_target = heatmap_target.to(self.device)
            except Exception as e:
                logger.warning(f"Heatmap generation failed for {sample_id}: {e}, using zero targets")
                self.epoch_fallback_counts['heatmap_generation_failure'] += 1
                heatmap_target = torch.zeros((1, z, y, x), dtype=torch.float32, device=self.device)

            # detection_loss expects (B, 1, Z, Y, X) for both logits and targets
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

            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(self.train_loader)}, "
                          f"Loss: {total_loss_item.item():.6f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log fallback counts
        for key, count in self.epoch_fallback_counts.items():
            if count > 0:
                logger.warning(f"Epoch had {count} {key} fallbacks")

        logger.info(f"Train epoch average loss: {avg_loss:.6f}")
        return avg_loss

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

        all_pred_graphs = {}
        all_gt_graphs = {}
        all_gt_metadata = {}

        self.epoch_fallback_counts['evaluation_failure'] = 0

        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.val_loader):
                frame_t = batch['frame_t'].to(self.device)
                frame_t1 = batch['frame_t1'].to(self.device)
                sample_id = batch['sample_id'][0]
                t_idx = batch.get('t_idx', [0])[0]

                # Forward pass
                x = torch.cat([frame_t, frame_t1], dim=1)
                logits, features = self.unet3d(x)
                detection_probs = torch.sigmoid(logits)

                # Extract nodes via NMS peak-finding
                vol_np = detection_probs[0, 0].cpu().numpy()
                peaks_t = extract_peaks_from_volume(
                    vol_np,
                    threshold=self.hyperparams['detection_threshold'],
                    voxel_size=DEFAULT_SCALE,
                    nms_radius_um=self.hyperparams['nms_radius_um']
                )
                peaks_t1 = extract_peaks_from_volume(
                    vol_np,
                    threshold=self.hyperparams['detection_threshold'],
                    voxel_size=DEFAULT_SCALE,
                    nms_radius_um=self.hyperparams['nms_radius_um']
                )

                if len(peaks_t) > 0 and len(peaks_t1) > 0:
                    nodes_t = torch.tensor(peaks_t, dtype=torch.float32, device=self.device)
                    nodes_t1 = torch.tensor(peaks_t1, dtype=torch.float32, device=self.device)

                    # Extract features at peak locations
                    z_t = torch.clamp(nodes_t[:, 0].long(), 0, features.shape[2] - 1)
                    y_t = torch.clamp(nodes_t[:, 1].long(), 0, features.shape[3] - 1)
                    x_t = torch.clamp(nodes_t[:, 2].long(), 0, features.shape[4] - 1)
                    features_t = features[0, :, z_t, y_t, x_t].t()

                    z_t1 = torch.clamp(nodes_t1[:, 0].long(), 0, features.shape[2] - 1)
                    y_t1 = torch.clamp(nodes_t1[:, 1].long(), 0, features.shape[3] - 1)
                    x_t1 = torch.clamp(nodes_t1[:, 2].long(), 0, features.shape[4] - 1)
                    features_t1 = features[0, :, z_t1, y_t1, x_t1].t()

                    # Get edge predictions
                    edge_probs = self.transformer(nodes_t, nodes_t1, features_t, features_t1)

                    # Greedy edge assignment
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
                    nodes_t = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                    nodes_t1 = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                    edges = []

                # Build prediction graph
                pred_graph = IndexedRXGraph()
                pred_graph.add_node_attr_key('t')
                pred_graph.add_node_attr_key('x')
                pred_graph.add_node_attr_key('y')
                pred_graph.add_node_attr_key('z')

                # Add nodes
                for i, (z, y, x) in enumerate(nodes_t.cpu().numpy()):
                    pred_graph.add_node(
                        node_id=f"{sample_id}_t{t_idx}_n{i}",
                        t=int(t_idx),
                        x=float(x),
                        y=float(y),
                        z=float(z)
                    )

                for j, (z, y, x) in enumerate(nodes_t1.cpu().numpy()):
                    pred_graph.add_node(
                        node_id=f"{sample_id}_t{t_idx+1}_n{j}",
                        t=int(t_idx + 1),
                        x=float(x),
                        y=float(y),
                        z=float(z)
                    )

                # Add edges from greedy assignment
                for src_idx, tgt_idx, _prob in edges:
                    src_node_id = f"{sample_id}_t{t_idx}_n{src_idx}"
                    tgt_node_id = f"{sample_id}_t{t_idx+1}_n{tgt_idx}"
                    pred_graph.add_edge(src_node_id, tgt_node_id)

                all_pred_graphs[sample_id] = pred_graph

                # Load GT graph
                try:
                    geff_path = self.data_dir / f"{sample_id}.geff"
                    if geff_path.exists():
                        gt_graph, gt_metadata = load_geff_ground_truth(str(geff_path))
                        all_gt_graphs[sample_id] = gt_graph
                        all_gt_metadata[sample_id] = gt_metadata
                except Exception as e:
                    logger.warning(f"Failed to load GT for {sample_id}: {e}")
                    self.epoch_fallback_counts['evaluation_failure'] += 1

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

    def fit(self, num_epochs: int):
        """Train model for specified number of epochs."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Hyperparameters: {json.dumps(self.hyperparams, indent=2)}")

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
