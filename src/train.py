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
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
from scipy import ndimage
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src import checkpoint_manifest
from src.evaluation import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_SCALE,
    evaluate_submission,
    load_geff_ground_truth,
)
from src.inference import greedy_edge_assignment
from src.prediction_graph import PredictionGraphAssembler
from src.split_utils import validate_resume_checkpoint_split_identity
from src.targets import (
    DetectionLoss,
    DivisionLoss,
    generate_edge_targets,
    generate_heatmap_targets,
    load_geff_cached,
)

logger = logging.getLogger(__name__)

# Live mid-run progress channel, independent of `kaggle kernels output`/`kernels
# logs` (both confirmed unreliable/stale while a kernel is RUNNING -- see
# CLAUDE.md's Kaggle Training Run Monitoring Checklist). ntfy.sh needs no
# account/signup; the topic slug is the only "secret" (anyone who knows it can
# read the channel), so it's a random slug, not a guessable project name.
# Verified working from an actual Kaggle sandbox kernel (enable_internet=true
# in kernel-metadata.json) before wiring this in -- see the throwaway
# st-act-ntfy-verify-throwaway kernel.
NTFY_TOPIC = "st-act-train-23d0805beb57a749"


def _post_ntfy_heartbeat(payload: dict) -> None:
    """
    Fire-and-forget POST of a heartbeat payload to ntfy.sh, off the main
    thread. Must never be able to add wall-clock cost to real training: a
    silently-stalling (not just refused) connection would otherwise block
    the calling thread for the full timeout on every call -- at 1000+ batch
    heartbeats/epoch that's a real risk, not a theoretical one, so this runs
    in a daemon thread rather than inline with a bare try/except.
    """
    def _send():
        try:
            requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=json.dumps(payload), timeout=5)
        except Exception:
            pass  # network hiccups must never affect training -- this is a nice-to-have

    threading.Thread(target=_send, daemon=True).start()


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
    nms_radius_um: float = 5.0,
    subvoxel_refine_radius: int = 1,
    background_percentile: float = 20.0,
    max_shift_um: float = 2.8,
) -> list:
    """
    Real 3D non-max suppression via maximum_filter with centroid collapsing.

    Sub-voxel refinement: a `vol == pooled` tied plateau is, by construction,
    perfectly uniform-valued internally (any two adjacent True voxels are each
    other's local max, forcing equal value) -- weighting the centroid by `vol`
    restricted to just the plateau is mathematically identical to the plain
    geometric centroid (verified: 20k random-volume trials, zero
    counterexamples; also independently confirmed by two real competitor
    submissions per COMPETITOR_RESEARCH_2026-07-13.md item 1). Real sub-voxel
    information only exists in the falloff just OUTSIDE the plateau, so each
    plateau's centroid is instead computed over its bounding box padded by
    `subvoxel_refine_radius` voxels (excluding voxels claimed by a different
    peak's label, so nearby peaks don't bleed into each other), with the
    region's own `background_percentile`-th percentile subtracted first so
    refinement responds to the residual signal rather than an absolute
    intensity level that varies sample-to-sample. The refined position is
    discarded (falls back to the plain plateau centroid) if it would shift
    the peak by more than `max_shift_um` physical microns, as a safety bound
    against noise-driven refinement.

    Returns list of [z, y, x] peak coordinates.
    """
    kernel = pool_kernel_from_um(nms_radius_um, voxel_size)
    pooled = ndimage.maximum_filter(vol, size=kernel, mode='constant', cval=-np.inf)
    is_peak = (vol == pooled) & (vol > threshold)

    labeled, num_labels = ndimage.label(is_peak)
    if num_labels == 0:
        return []

    centroids = []
    for label_id, obj_slice in enumerate(ndimage.find_objects(labeled), start=1):
        if obj_slice is None:
            continue
        plateau_center = [
            c + s.start for c, s in
            zip(ndimage.center_of_mass(labeled[obj_slice] == label_id), obj_slice, strict=False)
        ]

        padded_slice = tuple(
            slice(max(0, s.start - subvoxel_refine_radius), min(dim, s.stop + subvoxel_refine_radius))
            for s, dim in zip(obj_slice, vol.shape, strict=False)
        )
        local_labels = labeled[padded_slice]
        local_vol = vol[padded_slice]
        # Include this peak's own plateau plus unclaimed background falloff;
        # exclude any voxel already claimed by a DIFFERENT peak's plateau.
        weight_mask = (local_labels == label_id) | (local_labels == 0)
        included_vals = local_vol[weight_mask]
        background = np.percentile(included_vals, background_percentile) if included_vals.size else 0.0
        residual = np.maximum(np.where(weight_mask, local_vol - background, 0.0), 0.0)

        if residual.sum() <= 0:
            centroids.append(plateau_center)
            continue

        local_center = ndimage.center_of_mass(residual)
        refined_center = [c + s.start for c, s in zip(local_center, padded_slice, strict=False)]
        shift_um = np.sqrt(sum(
            ((r - p) * s) ** 2 for r, p, s in zip(refined_center, plateau_center, voxel_size, strict=False)
        ))
        centroids.append(refined_center if shift_um <= max_shift_um else plateau_center)
    return centroids


def extract_inference_peaks(
    detection_probs: torch.Tensor,
    channel: int,
    t_idx: int,
    hyperparams: dict[str, Any],
) -> list:
    """Shared production top-level version of TrainingLoop._peaks_for_channel()
    (P0-6, Part A5) -- extracts NMS peaks from one channel of a (B, 2, Z, Y, X)
    detection map, with the same adaptive-threshold guard against an
    undertrained model's near-uniform sigmoid output (see
    TrainingLoop._peaks_for_channel()'s docstring for the full rationale).
    TrainingLoop._peaks_for_channel() and src/submission_pipeline.py both
    delegate to this single implementation so validation and submission
    inference can never silently diverge."""
    vol_np = detection_probs[0, channel].cpu().numpy()
    threshold = hyperparams['detection_threshold']
    positive_fraction = float((vol_np > threshold).mean())
    max_positive_fraction = hyperparams.get('max_positive_voxel_fraction', 0.005)
    if positive_fraction > max_positive_fraction:
        adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
        logger.warning(
            f"t_idx={t_idx} ch={channel}: threshold={threshold} flags "
            f"{positive_fraction*100:.2f}% of voxels (undertrained-model miscalibration) "
            f"-- using adaptive threshold={adaptive_threshold:.4f} instead"
        )
        threshold = max(adaptive_threshold, threshold)
    elif positive_fraction == 0.0:
        adaptive_threshold = float(np.percentile(vol_np, 100 * (1 - max_positive_fraction)))
        logger.warning(
            f"t_idx={t_idx} ch={channel}: threshold={threshold} flags "
            f"0% of voxels (severe under-confidence) -- using adaptive "
            f"threshold={adaptive_threshold:.6f} instead"
        )
        threshold = adaptive_threshold
    return extract_peaks_from_volume(
        vol_np,
        threshold=threshold,
        voxel_size=DEFAULT_SCALE,
        nms_radius_um=hyperparams['nms_radius_um']
    )


def nodes_and_features_at_peaks(
    features: torch.Tensor, peaks: list, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared production top-level version of
    TrainingLoop._nodes_and_features_at_peaks() (P0-6, Part A5) -- builds
    (n, 3) node coords and (n, C) feature vectors at given peak locations.
    TrainingLoop._nodes_and_features_at_peaks() and
    src/submission_pipeline.py both delegate to this single implementation."""
    if len(peaks) == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, features.shape[1]), dtype=torch.float32, device=device),
        )
    nodes = torch.tensor(peaks, dtype=torch.float32, device=device)
    zc = torch.clamp(nodes[:, 0].long(), 0, features.shape[2] - 1)
    yc = torch.clamp(nodes[:, 1].long(), 0, features.shape[3] - 1)
    xc = torch.clamp(nodes[:, 2].long(), 0, features.shape[4] - 1)
    feats = features[0, :, zc, yc, xc].t()
    return nodes, feats


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
        deployed_sha: str = "unknown",
        progress_file: str | Path | None = None,
        split_identity: str = "unknown",
        strict_integrity_mode: bool = False,
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
            deployed_sha: git commit SHA of the code actually running -- written
                into the progress file (and logged by the caller) so "is this run
                using the code I think it is" is a 2-second check, not a
                post-mortem after a multi-hour run.
            progress_file: If given, path to overwrite (not append) a small JSON
                heartbeat after each epoch -- fetchable via `kaggle kernels
                output` mid-run without pulling the full raw log.
            split_identity: src/split_utils.py's compute_membership_sha256()
                identity of the split this run's train_loader/val_loader were
                built from -- embedded into every saved checkpoint
                ('split_membership_sha256') so evaluate_checkpoint.py can
                detect a checkpoint being scored against a different split
                than it was trained on (P0-2 fix, 2026-07-16). "unknown" (the
                default) is a valid value for callers that don't yet pass a
                real split identity -- see
                validate_checkpoint_split_compatibility()'s legacy-checkpoint
                handling, which treats an absent/unknown identity as a
                warning, not a hard failure.
            strict_integrity_mode: P0-7 (2026-07-19). If True, validate_epoch()'s
                technical sample-evaluation/GT-load failures increment their
                counter then immediately raise -- no ">50% tolerated" behavior.
                If False (default), retains the pre-P0-7 count-and-continue
                behavior including the existing >50% circuit-breaker. Independent
                of CompetitionDataset's strict_sample_coverage. Kaggle production
                training must pass True. Note: train_epoch()'s own fail-closed
                rules (technical GT-load failure, retained-pair empty GT, edge-
                target/edge-loss technical failure) are unconditional -- see
                _get_gt_nodes()/train_epoch() docstrings -- and do NOT depend on
                this flag, since a batch from self.train_loader is always a
                retained training pair by construction.
        """
        self.unet3d = unet3d
        self.transformer = transformer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = log_file
        self.deployed_sha = deployed_sha
        self.progress_file = Path(progress_file) if progress_file else None
        self.split_identity = split_identity
        self.strict_integrity_mode = strict_integrity_mode

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
            # Linear LR warmup over the first warmup_steps real training
            # batches, ramping from warmup_start_lr up to the configured
            # learning_rate, then holding at learning_rate for the rest of
            # training (ReduceLROnPlateau still applies per-epoch on top,
            # unchanged). Default 0 = no warmup, preserving existing
            # behavior for every caller that doesn't set this. Standard
            # fix for Adam-family early-training instability at higher lr
            # (not exotic -- e.g. the warmup schedules in Vaswani et al.
            # 2017 and BERT); added 2026-07-14 after lr=1e-2 was confirmed
            # to diverge (Loss: nan within ~100 real batches, v43) without
            # warmup.
            'warmup_steps': 0,
            'warmup_start_lr': 1e-4,
            # P0-4 (2026-07-17): max_validation_samples allows capping validation
            # to a deterministic subset of samples for faster iteration. When set,
            # selects the first N distinct sample IDs and evaluates every frame pair
            # for each selected sample (never truncates mid-sample). Default None =
            # full-fold validation (all samples).
            'max_validation_samples': None,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        # Tracks real batches trained across the whole run (not reset per
        # epoch) so warmup only ramps once, at the very start of training,
        # regardless of num_epochs.
        self._global_step = 0

        # Set random seed
        torch.manual_seed(self.hyperparams['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.hyperparams['seed'])
        np.random.seed(self.hyperparams['seed'])

        # Collect model parameters, split into decay/no-decay groups (2026-07-15):
        # AdamW previously applied weight_decay uniformly to ALL parameters,
        # including the detection head's deliberately negative RetinaNet-style
        # prior bias (src/model.py, prior_bias = log(1e-4/(1-1e-4)) ~= -9.21).
        # Standard practice excludes bias and norm-layer params from weight
        # decay -- decaying a 1D bias/norm param toward 0 fights whatever it was
        # deliberately initialized to represent, and provides no regularization
        # benefit those params don't have overfitting-prone weight matrices.
        named_params = list(unet3d.named_parameters()) + list(transformer.named_parameters())
        decay_params = [p for name, p in named_params if p.ndim > 1]
        no_decay_params = [p for name, p in named_params if p.ndim <= 1]

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            [
                {'params': decay_params, 'weight_decay': self.hyperparams['weight_decay']},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=self.hyperparams['learning_rate'],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Mixed precision (2026-07-14): v46 (GroupNorm added to UNet3D, see
        # commit 0e1e186) hit a real CUDA OOM on the T4 during backward()
        # ("Tried to allocate 2.00 GiB. ... 13.20 GiB memory in use" of the
        # T4's 14.56 GiB) -- v45 (same batch_size=1, no normalization) had
        # fit fine, so GroupNorm's added activation-memory overhead pushed
        # it over. autocast halves stored-activation memory for the
        # backward pass; GradScaler prevents fp16 gradient underflow.
        # enabled=False on CPU (self.device.type != 'cuda') so every local
        # CPU trace/test (scripts/local_smoke_train.py, tests/) is
        # unaffected -- torch.autocast(device_type='cpu', dtype=float16) is
        # not the intended use case and GradScaler is CUDA-only.
        self._amp_enabled = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp_enabled)

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
            # P0-7 (2026-07-19) new technical-failure counters: COUNTED_THEN_FATAL,
            # never tolerated. gt_node_load_failure covers a technical exception
            # from _get_gt_nodes() itself (missing GEFF, parse failure, malformed
            # attrs -- Rule A); retained_pair_zero_gt_nodes_failure covers a
            # DIFFERENT, successfully-parsed-but-empty-result case (Rule B) --
            # never conflated, see train_epoch()'s handling of each.
            'gt_node_load_failure': 0,
            'retained_pair_zero_gt_nodes_failure': 0,
        }
        # P0-7 (2026-07-19): biological-zero counter, kept separate from the
        # technical-failure counters above per ORD-2 -- a legitimate zero-positive-
        # edge batch is not a failure and must never be conflated with one.
        # GPU sanity gate Wave 2 (2026-07-19) adds two more: edge_supervised_
        # batches_total counts batches where the transformer actually received
        # real edge supervision (edge_targets not None, real edge_logits
        # produced, at least one positive edge present); ...with_nonzero_
        # transformer_grad is the subset of those where the captured gradient
        # snapshot (see self.last_transformer_gradient_snapshot below) was
        # finite and nonzero. Both are bookkeeping/observability counters, not
        # fail-closed gates themselves -- the gate's own pass/fail logic
        # (Wave 4) reads them, this class only measures and reports.
        self.epoch_biological_zero_counts = {
            'legitimate_zero_positive_edge_batches': 0,
            'edge_supervised_batches_total': 0,
            'edge_supervised_batches_with_nonzero_transformer_grad': 0,
        }
        # GPU sanity gate Wave 2 (2026-07-19): live gradient-snapshot capture,
        # updated in-place during train_epoch() with NO extra forward/backward
        # pass -- just a lightweight read of .grad right after the existing
        # backward() call already computes it. Two independently-tracked
        # targets per design §8.1: UNet captured on any batch where detection
        # loss succeeded (nearly every batch); Transformer captured only on
        # batches satisfying the edge-supervision conditions above. None until
        # at least one qualifying batch has occurred in the current run.
        self.last_unet_gradient_snapshot: float | None = None
        self.last_transformer_gradient_snapshot: float | None = None
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
                'predicted_nodes_total',
                'predicted_edges_total',
                'is_structural_zero',
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
                # Distinguishes "exactly zero" (structural failure -- a badly
                # trained but functioning model would produce small nonzero
                # scores, not exact zero) from "small but genuinely
                # undertrained", per DEFERRED_IMPROVEMENTS.md's monitoring plan.
                val_metrics.get("predicted_nodes_total", 0),
                val_metrics.get("predicted_edges_total", 0),
                val_metrics.get("is_structural_zero", True),
            ])

    def _write_progress_heartbeat(
        self, epoch: int, num_epochs: int, elapsed_seconds: float, train_loss: float,
        val_metrics: dict[str, float],
    ):
        """
        Overwrite (not append) a small JSON heartbeat after each epoch.

        Unlike the full raw log (only fetchable via `kaggle kernels logs`, which
        needs the whole run to be pulled and manually grepped), this is a real
        file under WORKING_DIR/self.progress_file's parent, fetchable mid-run via
        the already-working `kaggle kernels output` command -- no need to wait
        for completion or pull 40k+ log lines to check "is this run healthy".
        """
        if self.progress_file is None:
            return

        if val_metrics.get("is_structural_zero", False):
            health_status = "zero_detections"
        elif val_metrics.get("predicted_edges_total", 0) == 0:
            health_status = "zero_edges"
        elif val_metrics.get("score", 0.0) < 1e-6:
            health_status = "undertrained"
        else:
            health_status = "healthy"

        payload = {
            "deployed_sha": self.deployed_sha,
            "epoch": epoch,
            "num_epochs_budget": num_epochs,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "train_loss": train_loss,
            "val_score": val_metrics.get("score", 0.0),
            "predicted_nodes_total": val_metrics.get("predicted_nodes_total", 0),
            "predicted_edges_total": val_metrics.get("predicted_edges_total", 0),
            "health_status": health_status,
        }
        try:
            tmp_path = self.progress_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            tmp_path.replace(self.progress_file)  # atomic overwrite, not append
        except OSError as e:
            logger.warning(f"Failed to write progress heartbeat: {e}")
        _post_ntfy_heartbeat(payload)

    def _write_batch_heartbeat(self, batch_idx: int, effective_total: int, loss: float, max_sigmoid: float):
        """Mid-epoch heartbeat, same atomic-overwrite mechanism as
        _write_progress_heartbeat() but written every 5 batches during
        training, not just once per epoch. Every verification run this
        session used num_epochs=1, so the per-epoch heartbeat never fired
        until the run had already finished or errored -- zero real mid-run
        visibility despite the heartbeat mechanism existing. Deliberately a
        separate, simpler payload (no val_metrics -- those don't exist yet
        mid-epoch); validate_epoch()'s eventual full heartbeat overwrites
        this same file once available."""
        if self.progress_file is None:
            return
        payload = {
            "deployed_sha": self.deployed_sha,
            "phase": "training",
            "batch": batch_idx,
            "batch_total": effective_total,
            "train_loss": loss,
            "max_sigmoid": max_sigmoid,
        }
        try:
            tmp_path = self.progress_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            tmp_path.replace(self.progress_file)
        except OSError as e:
            logger.warning(f"Failed to write batch heartbeat: {e}")
        _post_ntfy_heartbeat(payload)

    def _get_gt_nodes(self, sample_id: str, t_idx: int) -> torch.Tensor:
        """
        Extract GT node coordinates from .geff at a specific timepoint.

        P0-7 (2026-07-19) Rule A: a TECHNICAL GT-load failure (missing .geff, GEFF
        parse exception, malformed node attributes) now RAISES instead of silently
        returning None -- this method's sole production call site (train_epoch())
        always operates on a RETAINED training pair (self.train_loader's
        CompetitionDataset is built with filter_unannotated_pairs=True), where a
        technical load failure is a real regression, never a benign data gap. A
        successfully-parsed .geff with genuinely zero GT nodes at a valid
        timepoint still returns an empty (0,3) tensor (Rule B) -- that alone is
        NOT a technical failure; train_epoch() separately treats an empty result
        from a retained pair as fatal (the retained-pair invariant), since this
        method alone cannot know whether its caller is operating on a retained
        pair.

        Args:
            sample_id: Dataset sample ID
            t_idx: Timepoint index

        Returns:
            (n_nodes, 3) tensor of [z, y, x] coordinates. n_nodes may be 0 for a
            valid timepoint with no GT nodes -- this never returns None.

        Raises:
            RuntimeError: on any technical GT-load failure (missing/unparseable
                .geff, malformed node attributes).
        """
        geff_path = self.data_dir / f"{sample_id}.geff"
        if not geff_path.exists():
            raise RuntimeError(
                f"Technical GT-load failure: expected .geff not found for sample "
                f"{sample_id} at {geff_path}."
            )

        try:
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
            raise RuntimeError(
                f"Technical GT-load failure: failed to parse .geff or read node "
                f"attributes for sample {sample_id} at t={t_idx} "
                f"(path={geff_path}): {e}"
            ) from e

    @staticmethod
    def _compute_warmup_lr(global_step: int, warmup_steps: int, start_lr: float, target_lr: float) -> float:
        """Linear ramp from start_lr to target_lr over warmup_steps calls.

        Uses (global_step+1)/warmup_steps, not global_step/warmup_steps: the
        latter never reaches fraction=1.0 (caught locally, 2026-07-14 -- a
        bare global_step/warmup_steps ramp topped out at 90% of target_lr
        for warmup_steps=10 and silently stayed there forever, since the
        caller stops invoking this once global_step==warmup_steps without
        fraction ever having reached 1.0).
        """
        fraction = (global_step + 1) / warmup_steps
        return start_lr + (target_lr - start_lr) * fraction

    def _generate_and_validate_heatmap_target(
        self, sample_id: str, t_idx: int, volume_shape: tuple[int, int, int, int],
        z: int, y: int, x: int,
    ) -> torch.Tensor:
        """
        Generate the (1, 2, Z, Y, X) GT heatmap target for a training pair and
        fail loudly if it's not what a retained pair should look like.

        P0-1 fix (2026-07-16): previously, ANY exception from generate_heatmap_targets
        was caught and silently replaced with an all-zero target (self.epoch_fallback_counts
        incremented but training continued). CompetitionDataset's pair index now
        guarantees every retained training pair has >=1 GT node at both t_idx and
        t_idx+1 (see CompetitionDataset._get_gt_counts_by_time), so a failure -- or an
        all-zero result -- reaching this point for a retained pair is a real
        regression, not an expected data gap, and must abort the run rather than
        train on a fabricated all-background target (see DetectionLoss's docstring:
        an all-zero target's weight_neg cancels out of loss.sum()/weights.sum()
        exactly, collapsing to plain BCE-against-zero -- the exact bug this dataset-
        level filtering exists to prevent).
        """
        geff_path = self.data_dir / f"{sample_id}.geff"
        try:
            heatmap_targets_dict, _ = generate_heatmap_targets(
                sample_id,
                str(geff_path),
                volume_shape,
                target_type='gaussian',
                target_ts=[t_idx, t_idx + 1],
                geff_cache=self._geff_cache,
            )
            # heatmaps[t] is (1, Z, Y, X) -- stack channel 0 (t_idx) and channel 1
            # (t_idx+1), then add batch dim to match logits' (B, 2, Z, Y, X) for
            # DetectionLoss/BCEWithLogitsLoss.
            zero_channel = torch.zeros((1, z, y, x), dtype=torch.float32)
            heatmap_ch0 = heatmap_targets_dict.get(t_idx, zero_channel)
            heatmap_ch1 = heatmap_targets_dict.get(t_idx + 1, zero_channel)
            if not isinstance(heatmap_ch0, torch.Tensor):
                heatmap_ch0 = torch.from_numpy(heatmap_ch0).float()
            if not isinstance(heatmap_ch1, torch.Tensor):
                heatmap_ch1 = torch.from_numpy(heatmap_ch1).float()
        except Exception as e:
            raise RuntimeError(
                f"Heatmap target generation failed for a retained training pair: "
                f"sample_id={sample_id} t_idx={t_idx} geff_path={geff_path}: {e}. "
                f"CompetitionDataset's annotation filtering should have excluded "
                f"this pair if either timepoint truly had zero GT nodes -- this "
                f"is a real regression (indexing, GEFF parsing, etc.), not an "
                f"expected data gap, so it must not be silently replaced with a "
                f"zero target."
            ) from e

        # Fail-loud invariant: a retained training pair is guaranteed (by
        # CompetitionDataset's GT-count filtering) to have >=1 GT node at both
        # t_idx and t_idx+1, so a correctly generated heatmap must have positive
        # target mass in both channels. A zero-sum channel here means the
        # filtering was bypassed or something broke downstream of it (wrong
        # coordinate/bounds handling, wrong time indexing) -- must abort, not
        # silently train on an effectively-empty target.
        ch0_sum = heatmap_ch0.sum().item()
        ch1_sum = heatmap_ch1.sum().item()
        if ch0_sum <= 0 or ch1_sum <= 0:
            raise RuntimeError(
                f"Retained training pair produced an all-zero heatmap target: "
                f"sample_id={sample_id} t_idx={t_idx} ch0_sum={ch0_sum} "
                f"ch1_sum={ch1_sum} geff_path={geff_path}. CompetitionDataset's "
                f"GT-count filtering should guarantee both timepoints have >=1 "
                f"GT node for a retained pair -- this indicates a real bug "
                f"(coordinate/bounds error, wrong time indexing, broken GEFF "
                f"parsing), not an expected data gap."
            )

        return torch.cat([heatmap_ch0, heatmap_ch1], dim=0).unsqueeze(0).to(self.device)

    @staticmethod
    def _compute_param_grad_norm(parameters) -> float | None:
        """GPU sanity gate Wave 2 (2026-07-19): L2 norm across every non-None
        .grad tensor in parameters, read immediately after backward() with no
        extra forward/backward pass. Returns None (not 0.0) if nothing in
        parameters had a gradient at all -- distinct from a real, legitimate
        gradient of exactly zero, which is itself a fail-closed gate signal
        (Wave 4) and must not be conflated with "never measured"."""
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if not grads:
            return None
        total_sq = sum(float((g.float() ** 2).sum().item()) for g in grads)
        return total_sq ** 0.5

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
        for key in self.epoch_biological_zero_counts:
            self.epoch_biological_zero_counts[key] = 0

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

            # Forward pass through UNet3D. autocast halves the stored
            # activation memory UNet3D's forward pass holds for backward()
            # -- by far the dominant memory consumer (full (128,64,256,256)
            # -scale feature maps) vs. the transformer's tiny point-feature
            # ops below, so wrapping just this call captures the large
            # majority of the saving needed. See self.scaler's __init__
            # comment for why this exists (real CUDA OOM on v46 after
            # GroupNorm was added).
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._amp_enabled):
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
            heatmap_target = self._generate_and_validate_heatmap_target(
                sample_id, int(t_idx), volume_shape, z, y, x
            )

            # detection_loss expects (B, 2, Z, Y, X) for both logits and targets.
            # logits.float(): logits left the autocast(dtype=float16) block above
            # as a float16 tensor: PyTorch's AMP docs explicitly recommend casting
            # tensors produced in an autocast region back to float32 once outside
            # it, since ops run there no longer get autocast's own dtype-safety
            # policy (BCEWithLogitsLoss itself is autocast-safe either way, but
            # this call executes in plain eager mode here, not inside autocast).
            detection_loss = self.detection_loss_fn(logits.float(), heatmap_target)

            # === EDGE LOSS (teacher forcing) ===
            # P0-7 (2026-07-19) Rule A / COUNTED_THEN_FATAL: _get_gt_nodes() now
            # RAISES on any technical GT-load failure (missing GEFF, parse
            # exception, malformed node attributes) instead of returning None --
            # this call site always operates on a RETAINED training pair
            # (self.train_loader is built with filter_unannotated_pairs=True),
            # so a technical failure here is a real regression, never a benign
            # data gap. Wrapped in a single try/except covering BOTH calls so a
            # failure loading either t or t+1 increments gt_node_load_failure
            # exactly once (not twice) for this batch, then immediately
            # propagates -- distinct from retained_pair_zero_gt_nodes_failure
            # below, which covers a DIFFERENT case (successful parse but an
            # empty node set), never a substitute for this counter.
            try:
                nodes_t = self._get_gt_nodes(sample_id, t_idx)
                nodes_t1 = self._get_gt_nodes(sample_id, t_idx + 1)
            except Exception as e:
                self.epoch_fallback_counts['gt_node_load_failure'] += 1
                raise RuntimeError(
                    f"Technical GT node load failure for retained training pair "
                    f"sample_id={sample_id} t_idx={t_idx}: {e}"
                ) from e

            # Rule B: a retained pair is guaranteed by CompetitionDataset's
            # GT-count filtering to have >=1 GT node at BOTH t_idx and t_idx+1.
            # An empty result here means that invariant was violated (a real bug
            # -- coordinate/indexing error, GT-count filtering bypassed, etc.),
            # NOT a legitimate biological zero -- must abort with a counted
            # failure, never silently fall through to the unconditional
            # edge_loss=0.0 default with no visibility (the exact bug this
            # closes).
            if nodes_t.shape[0] == 0 or nodes_t1.shape[0] == 0:
                self.epoch_fallback_counts['retained_pair_zero_gt_nodes_failure'] += 1
                raise RuntimeError(
                    f"Retained training pair produced zero GT nodes at one or "
                    f"both timepoints: sample_id={sample_id} t_idx={t_idx} "
                    f"n_nodes_t={nodes_t.shape[0]} n_nodes_t1={nodes_t1.shape[0]}. "
                    f"CompetitionDataset's GT-count filtering should guarantee "
                    f">=1 GT node at both timepoints for a retained pair -- this "
                    f"indicates a real bug, not an expected data gap."
                )

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

            # Generate real GT edge targets. Rule C: a technical failure here is
            # counted then immediately re-raised -- no edge_targets=None
            # continuation.
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
            except Exception as e:
                self.epoch_fallback_counts['edge_target_generation_failure'] += 1
                raise RuntimeError(
                    f"Edge target generation failed for retained training pair "
                    f"sample_id={sample_id} t_idx={t_idx}: {e}"
                ) from e

            edge_targets = edge_targets.to(self.device)
            division_mask = edge_metadata.get('division_mask', torch.zeros_like(edge_targets, dtype=torch.bool))
            division_mask = division_mask.to(self.device)

            # generate_edge_targets() (src/targets.py) returns a (n_t * n_t1,)
            # tensor and ONLY returns empty (0,) when n_t==0 or n_t1==0 --
            # verified directly against its source. Rule B above already
            # guarantees nodes_t/nodes_t1 are both non-empty by this point, so
            # edge_targets.numel() == 0 here is provably unreachable given that
            # invariant -- if it ever fires anyway, that invariant was silently
            # violated (a real bug, e.g. generate_edge_targets() re-deriving
            # n_t/n_t1 from something other than nodes_t/nodes_t1), not a
            # legitimate empty-candidate path. Treated as a counted technical
            # integrity failure, never as biological zero.
            if edge_targets.numel() == 0:
                self.epoch_fallback_counts['edge_target_generation_failure'] += 1
                raise RuntimeError(
                    f"generate_edge_targets() returned zero candidate edges for "
                    f"a retained pair with non-empty nodes_t/nodes_t1 "
                    f"(n_t={nodes_t.shape[0]} n_t1={nodes_t1.shape[0]}): "
                    f"sample_id={sample_id} t_idx={t_idx}. This contradicts the "
                    f"n_t * n_t1 candidate-edge contract -- a real bug, not a "
                    f"legitimate empty-candidate case."
                )

            # Rule D: generate_edge_targets() succeeded and produced a non-empty
            # but ALL-NEGATIVE target (no positive lineage/continuation edge) --
            # a legitimate biological zero, NOT a technical failure. This is
            # still real, valid training data (the transformer's false-edge
            # rejection signal) and must still run the transformer and
            # DivisionLoss over it, not be replaced with a disconnected
            # edge_loss=0.0 -- only the fallback COUNTER differs from the
            # ordinary positive-target path below; loss computation and Rule E's
            # fail-closed handling are identical either way.
            has_positive_edges = edge_targets.sum().item() > 0
            if not has_positive_edges:
                self.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] += 1

            # Rule E: a technical failure in the transformer call or DivisionLoss
            # is counted then immediately re-raised -- a caught technical
            # exception must never become edge_loss=0.0.
            try:
                edge_logits = self.transformer(nodes_t, nodes_t1, features_t, features_t1)
                edge_targets_float = edge_targets.float()
                edge_loss = self.division_loss_fn(
                    edge_logits.view(-1),
                    edge_targets_float,
                    division_mask
                )
            except Exception as e:
                self.epoch_fallback_counts['edge_loss_computation_failure'] += 1
                raise RuntimeError(
                    f"Edge loss computation failed for retained training pair "
                    f"sample_id={sample_id} t_idx={t_idx}: {e}"
                ) from e

            if not torch.isfinite(edge_loss):
                self.epoch_fallback_counts['edge_loss_computation_failure'] += 1
                raise RuntimeError(
                    f"edge_loss is non-finite (NaN/Inf) for retained training "
                    f"pair sample_id={sample_id} t_idx={t_idx}: "
                    f"{edge_loss.item()}"
                )

            # Total loss
            total_loss_item = (
                edge_loss +
                self.hyperparams['heatmap_loss_weight'] * detection_loss
            )

            # Backward pass. scaler.scale()/unscale_() are no-ops when
            # self._amp_enabled is False (GradScaler(enabled=False)), so
            # this is safe on CPU too.
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss_item).backward()

            # GPU sanity gate Wave 2 (2026-07-19): live gradient-snapshot
            # capture, read directly off backward()'s own output -- no extra
            # forward/backward pass. UNet: this point is only reached after
            # detection loss already succeeded (an earlier failure would have
            # raised before here), so every batch reaching here qualifies.
            # Transformer: only on batches with real positive-edge
            # supervision (has_positive_edges), matching design §8.1's
            # edge_supervised_batches_total condition -- edge_targets is
            # already guaranteed non-None/non-empty by the raises above, so
            # that condition reduces to this one check.
            unet_grad_norm = self._compute_param_grad_norm(self.unet3d.parameters())
            if unet_grad_norm is not None:
                self.last_unet_gradient_snapshot = unet_grad_norm

            if has_positive_edges:
                self.epoch_biological_zero_counts['edge_supervised_batches_total'] += 1
                transformer_grad_norm = self._compute_param_grad_norm(self.transformer.parameters())
                if transformer_grad_norm is not None:
                    self.last_transformer_gradient_snapshot = transformer_grad_norm
                    if transformer_grad_norm != 0.0 and math.isfinite(transformer_grad_norm):
                        self.epoch_biological_zero_counts[
                            'edge_supervised_batches_with_nonzero_transformer_grad'
                        ] += 1

            # Gradient clipping. unscale_() first so clip_grad_norm_ sees
            # real (not scaler-multiplied) gradient magnitudes.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.unet3d.parameters()) + list(self.transformer.parameters()),
                self.hyperparams['grad_clip']
            )

            # Linear LR warmup: ramp param_groups[0]['lr'] from warmup_start_lr
            # up to the configured learning_rate over the first warmup_steps
            # real batches, applied BEFORE optimizer.step() so this step
            # actually uses the ramped rate. No-op when warmup_steps=0 (the
            # default) -- self.optimizer already has the target lr from
            # __init__ in that case, so this branch never fires.
            warmup_steps = self.hyperparams['warmup_steps']
            if warmup_steps > 0 and self._global_step < warmup_steps:
                warmup_lr = self._compute_warmup_lr(
                    self._global_step, warmup_steps,
                    self.hyperparams['warmup_start_lr'], self.hyperparams['learning_rate'],
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._global_step += 1

            total_loss += total_loss_item.item()
            num_batches += 1

            # Every 5 batches (not 1/5th of the epoch, ~40 batches here) so
            # a genuinely slow epoch is visible within the first minute --
            # a real ~75min run was let continue for over 40min before
            # anyone had a rate/ETA signal to notice it was worth stopping.
            #
            # effective_total accounts for max_batches_per_epoch: without this,
            # a capped verification run (e.g. 1500 of a real 14,751-batch
            # epoch) logged both the batch-count denominator AND eta_remaining
            # against the full uncapped epoch size, making the ETA wildly
            # wrong (~6.5h shown for a run that actually finishes in ~40min) --
            # confirmed live against a real v40 run's log (2026-07-14).
            effective_total = min(len(self.train_loader), max_batches) if max_batches is not None else len(self.train_loader)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == effective_total:
                elapsed = time.time() - epoch_start_time
                rate = elapsed / (batch_idx + 1)
                eta_remaining = rate * (effective_total - (batch_idx + 1))
                # max_sigmoid: the only direct evidence of whether training is
                # actually moving the model's real output, as opposed to loss
                # (an indirect proxy the adaptive per-batch weighting makes
                # hard to compare across batches) or validation-time-only
                # sigmoid (previously the sole source, but only ever sampled
                # AFTER training stops -- this shows the live trend instead).
                # A single local step was directly measured to move this by
                # only ~9e-7 (2026-07-14) -- logging every training batch is
                # what actually answers whether that compounds over a real
                # run or plateaus, not a one-off before/after snapshot.
                with torch.no_grad():
                    max_sigmoid = torch.sigmoid(logits).max().item()
                logger.info(
                    f"Batch {batch_idx + 1}/{effective_total}, "
                    f"Loss: {total_loss_item.item():.6f}, "
                    f"max_sigmoid: {max_sigmoid:.8f}, "
                    f"elapsed={elapsed:.1f}s, {rate:.2f}s/batch, "
                    f"eta_remaining={eta_remaining:.0f}s"
                )
                self._write_batch_heartbeat(
                    batch_idx + 1, effective_total, total_loss_item.item(), max_sigmoid,
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

        P0-6 (Part A5): delegates to the shared top-level
        extract_inference_peaks() so validation and submission inference
        (src/submission_pipeline.py) can never silently diverge -- see that
        function's docstring for the adaptive-threshold rationale.
        """
        return extract_inference_peaks(
            detection_probs, channel=channel, t_idx=t_idx, hyperparams=self.hyperparams,
        )

    def _nodes_and_features_at_peaks(
        self, features: torch.Tensor, peaks: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (n, 3) node coords and (n, C) feature vectors at given peak
        locations. P0-6 (Part A5): delegates to the shared top-level
        nodes_and_features_at_peaks()."""
        return nodes_and_features_at_peaks(features, peaks, self.device)

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
        # P0-3 fix (2026-07-17): graph node identity for overlapping windows
        # is now owned by src/prediction_graph.py's PredictionGraphAssembler
        # (see its module docstring) instead of unconditionally add_node()'ing
        # both channels every batch -- that previously created two distinct
        # graph nodes for timepoint t_idx+1 (once as this batch's channel-1,
        # once as the next batch's own channel-0), inflating predicted node
        # count ~2x at non-boundary timepoints. This REQUIRES chronological
        # per-sample window order (self.val_loader must use shuffle=False --
        # see kaggle_kernel/train_kernel.py) -- the assembler raises
        # RuntimeError on any duplicate/backward/gapped t_idx per sample.
        assembler = PredictionGraphAssembler()
        max_batches = self.hyperparams.get('max_batches_per_epoch')
        max_validation_samples = self.hyperparams.get('max_validation_samples')

        # P0-4 (hardened): reject a max_validation_samples value that could
        # silently do the wrong thing -- bool is an int subclass in Python
        # (isinstance(True, int) is True), so it's checked explicitly first;
        # a non-positive cap has no sensible "select N samples" meaning.
        if max_validation_samples is not None:
            if isinstance(max_validation_samples, bool) or not isinstance(max_validation_samples, int):
                raise ValueError(
                    f"max_validation_samples must be an int or None, got "
                    f"{type(max_validation_samples).__name__}: {max_validation_samples!r}"
                )
            if max_validation_samples <= 0:
                raise ValueError(
                    f"max_validation_samples must be a positive int, got {max_validation_samples}"
                )

        # P0-7 (2026-07-19): validation accounting prefers CompetitionDataset's
        # own expected_sample_ids (the full split-file membership list) over
        # dataset.pairs -- deriving the "total" from .pairs let a sample that
        # silently failed to open (missing/unreadable Zarr) vanish from the
        # denominator too, undercounting validation_samples_total instead of
        # surfacing the shortfall in validation_samples_evaluated (bug F6).
        # Falls back to the pre-P0-7 .pairs-derived unique-sample-id list when
        # expected_sample_ids isn't available (e.g. test doubles predating
        # P0-7's dataset coverage contract that only mock .pairs) -- preserves
        # existing P0-4 cap-selection test coverage unrelated to bug F6.
        # getattr(..., None) (not a bare .dataset access) so a val_loader
        # without a .dataset attribute at all (a bare list, as used by several
        # existing unit tests) degrades to "expected ids unavailable" instead
        # of raising AttributeError.
        val_dataset = getattr(self.val_loader, 'dataset', None)
        has_expected_ids = (
            val_dataset is not None
            and hasattr(val_dataset, 'expected_sample_ids')
            and val_dataset.expected_sample_ids is not None
        )
        has_pairs = (
            val_dataset is not None
            and hasattr(val_dataset, 'pairs')
            and val_dataset.pairs is not None
        )

        allowed_sample_ids = None
        expected_fold_ids = None
        validation_samples_total = None
        validation_sample_cap = max_validation_samples

        if has_expected_ids:
            expected_fold_ids = list(val_dataset.expected_sample_ids)
        elif has_pairs:
            expected_fold_ids = []
            seen = set()
            for sample_id, _ in val_dataset.pairs:
                if sample_id not in seen:
                    expected_fold_ids.append(sample_id)
                    seen.add(sample_id)

        if expected_fold_ids is not None:
            validation_samples_total = len(expected_fold_ids)

            if max_validation_samples is not None:
                n_select = min(max_validation_samples, validation_samples_total)
                allowed_sample_ids = set(expected_fold_ids[:n_select])
                logger.info(
                    f"Validation sample selection: {len(allowed_sample_ids)} of "
                    f"{validation_samples_total} expected samples selected "
                    f"(cap={max_validation_samples}). Selected: {sorted(allowed_sample_ids)}"
                )
        elif max_validation_samples is not None:
            raise RuntimeError(
                f"max_validation_samples={max_validation_samples} requested but "
                f"val_loader.dataset does not expose 'expected_sample_ids' or "
                f"'pairs' for safe sample selection. Cannot guarantee complete "
                f"sample evaluation."
            )

        # validation_is_full_fold is computed a-posteriori, after the batch loop
        # and GT-loading below determine which expected sample IDs were
        # ACTUALLY, COMPLETELY evaluated -- see the assignment near
        # all_gt_graphs below. Default True here only for the legacy path
        # (expected_fold_ids unavailable, no cap requested), matching the
        # pre-P0-7 assumption for callers (e.g. bare-list unit tests) that
        # don't expose real coverage information at all.
        validation_is_full_fold = expected_fold_ids is None

        # max_batches_per_epoch is a training-only cap (P0-4); log this ONCE
        # before the loop when configured, not per-batch -- validate_epoch()
        # deliberately ignores it on every batch, and a per-batch warning is
        # pure log spam once past the threshold.
        if max_batches is not None:
            logger.info(
                f"max_batches_per_epoch={max_batches} is configured but is a "
                f"TRAINING-only cap; validate_epoch() ignores it and always "
                f"processes complete samples."
            )

        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.val_loader):
                sample_id = batch['sample_id'][0]

                # P0-4: Skip batches for samples not in the allowed set
                if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                    continue

                frame_t = batch['frame_t'].to(self.device)
                frame_t1 = batch['frame_t1'].to(self.device)
                t_idx = int(batch.get('t_idx', [0])[0])

                # Fail loud, immediately, on any out-of-order window for this
                # sample -- sequential canonical-node ownership is only valid
                # under strict chronological order (P0-3).
                assembler.validate_window_order(sample_id, t_idx)

                # Forward pass -- logits/detection_probs are (B, 2, Z, Y, X).
                # Same autocast rationale as train_epoch() above -- extra
                # memory headroom during eval too, no scaler needed here
                # (no backward pass, already under torch.no_grad()).
                x = torch.cat([frame_t, frame_t1], dim=1)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._amp_enabled):
                    logits, features = self.unet3d(x)
                # .float(): same rationale as train_epoch()'s detection_loss call --
                # logits left the autocast region as float16; detection_probs feeds
                # _peaks_for_channel()'s threshold comparison, which directly
                # determines predicted nodes and therefore val_score.
                detection_probs = torch.sigmoid(logits.float())

                peaks_t = self._peaks_for_channel(detection_probs, channel=0, t_idx=t_idx)
                peaks_t1 = self._peaks_for_channel(detection_probs, channel=1, t_idx=t_idx)

                # Canonical graph identity (P0-3): frame t_idx's source set is
                # peaks_t ONLY if this is the first window this sample has
                # ever produced -- otherwise it's the ALREADY-canonical nodes
                # from the prior window's channel-1 output, and peaks_t (this
                # window's channel-0 re-observation of the same real
                # timepoint) is ignored for node creation (counted
                # diagnostically only -- see PredictionGraphAssembler.
                # process_window()'s docstring for why: the real model is not
                # guaranteed to return identical overlapping predictions).
                # Frame t_idx+1 is always newly owned here from peaks_t1.
                source_ids, source_coords, target_ids, target_coords = assembler.process_window(
                    sample_id, t_idx, peaks_t, peaks_t1,
                )

                # Sample THIS window's features at the canonical coordinates
                # (not necessarily peaks_t/peaks_t1 -- source_coords may be
                # carried over from a prior window's forward pass, but this
                # window's feature tensor is still a valid array to index
                # into at those same (z,y,x) locations).
                nodes_t, features_t = self._nodes_and_features_at_peaks(features, source_coords)
                nodes_t1, features_t1 = self._nodes_and_features_at_peaks(features, target_coords)

                # Every 5 batches -- mirrors train_epoch()'s progress-logging cadence
                # (see comment there). Without this, a val_score=0.0 gives no way to
                # tell "the model detects nothing at all" (sigmoid stuck near 0) apart
                # from "it detects things but they don't match GT" from the CSV alone
                # -- exactly the diagnostic gap that made the real root cause of the
                # 2026-07-13 zero-score run (class-imbalance loss under-weighting)
                # invisible until the full log was pulled and manually inspected.
                if (_batch_idx + 1) % 5 == 0:
                    sig_min = detection_probs.min().item()
                    sig_max = detection_probs.max().item()
                    logger.info(
                        f"Val batch {_batch_idx + 1} | sample={sample_id} t_idx={t_idx} | "
                        f"sigmoid=[{sig_min:.4f}, {sig_max:.4f}] | "
                        f"raw peaks: {len(peaks_t)} (ch0), {len(peaks_t1)} (ch1) | "
                        f"canonical: {len(source_coords)} (frame {t_idx}), {len(target_coords)} (frame {t_idx + 1})"
                    )

                if len(source_coords) > 0 and len(target_coords) > 0:
                    edge_logits = self.transformer(nodes_t, nodes_t1, features_t, features_t1)
                    edge_probs = torch.sigmoid(edge_logits)
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

                # Coordinates cast to int to match the pl.Int64 schema
                # PredictionGraphAssembler registers (mirrors
                # run_pipeline.py:convert_nx_to_tracksdata()) -- passing float
                # values against an Int64 schema fails inside
                # evaluate_submission() with "unexpected value ... found
                # value of type Float64", silently caught by the
                # eval_failure fallback and masking the real validation
                # score with zeros.
                assembler.add_edges(sample_id, source_ids, target_ids, edges)

        # Circuit-breaker for a structurally-zero validation pass, evaluated
        # AFTER full coverage (P0-3 fix, 2026-07-17): the previous version
        # aborted early once the first 10 batches predicted zero nodes,
        # relying on shuffle=True so those 10 batches weren't just one
        # sample's real, legitimately-empty boundary frames. P0-3 requires
        # chronological (shuffle=False) order, under which the first N
        # batches ARE exactly one sample's early frames -- an early-boundary
        # abort would now misfire on real empty frames, not a structurally
        # dead model. So this check now runs once, over the complete
        # evaluated coverage's UNIQUE node total (post canonical-identity
        # dedup, not raw per-channel peak counts) -- still fails loud on a
        # genuinely zero-detection model, just never on an early false
        # positive from chronological ordering.
        diagnostics = assembler.diagnostics()
        total_predicted_nodes = diagnostics['predicted_nodes_total']
        total_predicted_edges = diagnostics['predicted_edges_total']
        if total_predicted_nodes == 0:
            raise RuntimeError(
                f"Validation aborted: the complete validation pass predicted ZERO "
                f"unique graph nodes (sigmoid never crossed "
                f"detection_threshold={self.hyperparams['detection_threshold']} anywhere, "
                f"across all {len(self.val_loader)} batches). validate_epoch() uses a "
                f"frozen model -- this cannot self-correct within the same validation "
                f"pass. Diagnose the detection head / loss weighting before retrying."
            )

        all_pred_graphs = assembler.pred_graphs()

        # Load each sample's GT graph once (not once per batch -- a sample
        # has many batches, all needing the same GT graph).
        for sample_id in all_pred_graphs:
            try:
                geff_path = self.data_dir / f"{sample_id}.geff"
                if not geff_path.exists():
                    # P0-7 COUNTED_THEN_FATAL: a missing GEFF is a technical
                    # GT-load failure; raise so the except block counts it and
                    # raises in strict mode rather than silently continuing.
                    raise FileNotFoundError(
                        f"expected .geff not found for sample {sample_id} "
                        f"at {geff_path}"
                    )
                gt_graph, gt_metadata = load_geff_ground_truth(str(geff_path))
                all_gt_graphs[sample_id] = gt_graph
                all_gt_metadata[sample_id] = gt_metadata
            except Exception as e:
                self.epoch_fallback_counts['evaluation_failure'] += 1
                # P0-7 STRICT VALIDATION INTEGRITY: no ">50% tolerated" behavior
                # in strict mode -- a technical GT-load failure is counted then
                # immediately raised, not tolerated up to the 50% threshold below.
                # getattr(..., False): a bare TrainingLoop.__new__() test double
                # (bypassing __init__) may not set self.strict_integrity_mode --
                # real production instances always go through __init__, which
                # defaults it to False anyway, so this is behaviorally identical
                # there (matches this class's existing getattr(self,
                # "deployed_sha", ...) pattern for the same reason).
                if getattr(self, 'strict_integrity_mode', False):
                    raise RuntimeError(
                        f"strict_integrity_mode=True: technical GT-load failure "
                        f"for sample {sample_id}: {e}"
                    ) from e
                logger.warning(f"Failed to load GT for {sample_id}: {e}")

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

        # P0-7 VALIDATION ACCOUNTING (a-posteriori): "evaluated" tracks the
        # PREDICTION side (all_pred_graphs) -- matching the pre-existing P0-4
        # convention -- computed here, after the batch loop, rather than
        # a-priori from the selection set, so a sample that was selected but
        # never produced a predicted window is correctly excluded. Deliberately
        # NOT intersected with all_gt_graphs: in strict_integrity_mode, any
        # per-sample GT-load failure already aborts validate_epoch() entirely
        # (see the loop above), so pred/gt coverage always coincide there; in
        # non-strict mode, treating a GT-load failure as "not evaluated" would
        # break existing pre-P0-7 test coverage that (correctly) counts a
        # sample as evaluated once its predictions are complete, independent of
        # a separately-tracked (epoch_fallback_counts['evaluation_failure'])
        # GT-load outcome.
        evaluated_sample_ids = set(all_pred_graphs)
        if expected_fold_ids is not None:
            validation_is_full_fold = (evaluated_sample_ids == set(expected_fold_ids))

        if getattr(self, 'strict_integrity_mode', False) and not all_gt_graphs:
            raise RuntimeError(
                "strict_integrity_mode=True: validation produced zero usable GT "
                "graphs -- val_score would be meaningless."
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

                # P0-4: Add provenance fields for validation coverage distinction
                val_metrics_clean['predicted_nodes_total'] = total_predicted_nodes
                val_metrics_clean['predicted_edges_total'] = total_predicted_edges
                val_metrics_clean['is_structural_zero'] = (total_predicted_nodes == 0)
                # P0-6 (adversarial finding 23): explicit evaluation-success
                # provenance, checked literally (is True) by
                # checkpoint_manifest.deployment_eligibility_errors() --
                # independent of any other metric field, including a
                # legitimate adjusted_edge_jaccard of exactly 0.0.
                val_metrics_clean['evaluation_completed_successfully'] = True
                val_metrics_clean['validation_is_full_fold'] = validation_is_full_fold
                val_metrics_clean['validation_samples_evaluated'] = (
                    len(evaluated_sample_ids) if expected_fold_ids is not None
                    else len(all_pred_graphs)
                )
                val_metrics_clean['validation_samples_total'] = (
                    validation_samples_total if validation_samples_total is not None
                    else len(all_pred_graphs)
                )
                val_metrics_clean['validation_sample_cap'] = validation_sample_cap

                # Log provenance
                if validation_is_full_fold:
                    logger.info(f"Validation used FULL-FOLD ({val_metrics_clean['validation_samples_evaluated']} samples)")
                else:
                    logger.info(
                        f"Validation CAPPED to {val_metrics_clean['validation_samples_evaluated']} "
                        f"of {val_metrics_clean['validation_samples_total']} samples"
                    )

                return val_metrics_clean
            except Exception as e:
                self.epoch_fallback_counts['evaluation_failure'] += 1
                if getattr(self, 'strict_integrity_mode', False):
                    raise RuntimeError(
                        f"strict_integrity_mode=True: evaluate_submission() "
                        f"failed: {e}"
                    ) from e
                logger.warning(f"Evaluation failed: {e}")
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

        # P0-4: Add provenance fields in all paths
        val_metrics['predicted_nodes_total'] = total_predicted_nodes
        val_metrics['predicted_edges_total'] = total_predicted_edges
        val_metrics['validation_is_full_fold'] = validation_is_full_fold
        val_metrics['validation_samples_evaluated'] = (
            len(evaluated_sample_ids) if expected_fold_ids is not None
            else len(all_pred_graphs)
        )
        val_metrics['validation_samples_total'] = (
            validation_samples_total if validation_samples_total is not None
            else len(all_pred_graphs)
        )
        val_metrics['validation_sample_cap'] = validation_sample_cap
        val_metrics['is_structural_zero'] = (total_predicted_nodes == 0)
        # P0-6 (adversarial finding 23): both fallback paths reaching here --
        # an evaluation exception, or no usable GT graphs at all -- are
        # EXPLICITLY marked as a non-successful evaluation, independent of
        # whatever score value they happen to report (both currently report
        # 0.0, but this field is what deployment_eligibility_errors() checks,
        # not the score itself, so this remains correct even if a future
        # change to the fallback value stops being exactly 0.0).
        val_metrics['evaluation_completed_successfully'] = False
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
        logger.info(f"Deployed code SHA: {self.deployed_sha}")
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

            # Unconditional checkpoint of the just-trained weights, BEFORE
            # validate_epoch() runs -- validate_epoch()'s circuit breaker can
            # raise and abort fit() entirely, and save_checkpoint() below
            # only fires on a val_score improvement, so every verification
            # run that hit the circuit breaker (v40-v45) lost its trained
            # weights completely, with no way to even inspect what the
            # model had learned. This survives that case: real weights are
            # on disk the moment training finishes, independent of whether
            # validation ever completes.
            self._save_last_checkpoint(epoch + 1, train_loss)

            # Validation
            val_metrics = self.validate_epoch()

            # Log epoch
            self._log_epoch(epoch + 1, train_loss, val_metrics)
            self._write_progress_heartbeat(
                epoch=epoch + 1, num_epochs=num_epochs,
                elapsed_seconds=time.monotonic() - fit_start_time,
                train_loss=train_loss, val_metrics=val_metrics,
            )

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

    def _save_last_checkpoint(self, epoch: int, train_loss: float):
        """Unconditional checkpoint of the current weights, independent of
        validation outcome -- see the call site in fit() for why this
        exists. Fixed filename (not val-score-keyed) so it always
        overwrites in place rather than accumulating; save_checkpoint()'s
        separate best-score-keyed files are unaffected."""
        checkpoint_path = self.checkpoint_dir / "last_checkpoint.pt"
        checkpoint = {
            'epoch': epoch,
            'unet3d_state_dict': self.unet3d.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'hyperparams': self.hyperparams,
            # P0-6 (Part B3): embedded for consistency with save_checkpoint(),
            # but last_checkpoint.pt can NEVER become deployment-eligible --
            # it has no 'val_metrics' key at all (see
            # checkpoint_manifest.deployment_eligibility_errors()), so this
            # never gets a manifest (Part B4/F6.7).
            'checkpoint_schema_version': checkpoint_manifest.CHECKPOINT_SCHEMA_VERSION,
            # getattr(): a bare TrainingLoop.__new__() test double (bypassing
            # __init__) may not set self.deployed_sha -- real production
            # instances always go through __init__, which defaults it to
            # "unknown" anyway, so this is behaviorally identical there.
            'training_code_sha': getattr(self, 'deployed_sha', 'unknown'),
            'model_contract': checkpoint_manifest.MODEL_CONTRACT,
        }
        # P0-2 fix (2026-07-16): same identity-embedding contract as
        # save_checkpoint() -- omit the key entirely (not a placeholder
        # value) when self.split_identity is "unknown".
        if self.split_identity != "unknown":
            checkpoint['split_membership_sha256'] = self.split_identity

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved unconditional last-epoch checkpoint: {checkpoint_path}")

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
            # P0-6 (Part B3): identity fields verified checkpoint deployment
            # (src/checkpoint_manifest.py) requires before a manifest may
            # reference this checkpoint.
            'checkpoint_schema_version': checkpoint_manifest.CHECKPOINT_SCHEMA_VERSION,
            'training_code_sha': getattr(self, 'deployed_sha', 'unknown'),
            'model_contract': checkpoint_manifest.MODEL_CONTRACT,
        }
        # P0-2 fix (2026-07-16): omit the key entirely (not a placeholder
        # value) when self.split_identity is "unknown", so
        # validate_checkpoint_split_compatibility()'s checkpoint.get(...) is
        # None legacy-checkpoint branch fires correctly instead of a
        # confusing hard mismatch against a literal "unknown" string.
        if self.split_identity != "unknown":
            checkpoint['split_membership_sha256'] = self.split_identity

        # P0-6 (collision-safety correction): determine the active manifest's
        # protected checkpoint, and this new checkpoint's own eligibility,
        # BEFORE writing a single byte to disk. Both checks are pure
        # in-memory operations (deployment_eligibility_errors() only reads
        # the `checkpoint` dict; it never touches the filesystem), so this
        # ordering costs nothing and closes a real gap: the derived filename
        # is deterministic from (epoch, rounded val_score) alone, so a
        # repeated/resumed epoch that lands on the SAME rounded score as the
        # currently-deployed checkpoint would otherwise have its bytes
        # silently destroyed by torch.save() before eligibility or the
        # active manifest were ever consulted.
        #
        # Fail-closed (adversarial finding 24): uses the shared production
        # manifest parser/validator
        # (checkpoint_manifest.read_active_manifest_checkpoint_path()), which
        # raises on ANY malformed JSON, duplicate/unknown/missing key, invalid
        # value, unsafe checkpoint filename, symlink, directory escape, or
        # missing referenced checkpoint file. This intentionally propagates --
        # never caught and downgraded to a warning -- because a malformed or
        # ambiguous active manifest means we cannot safely determine what to
        # protect, and proceeding anyway risks deleting or overwriting the
        # checkpoint a still-active (just unparseable-by-us) manifest
        # references. Nothing is written to disk if this raises.
        manifest_path = self.checkpoint_dir / checkpoint_manifest.MANIFEST_FILENAME
        protected_checkpoint_path = checkpoint_manifest.read_active_manifest_checkpoint_path(self.checkpoint_dir)
        eligibility_errors = checkpoint_manifest.deployment_eligibility_errors(checkpoint)

        collides_with_active = (
            protected_checkpoint_path is not None
            and checkpoint_path.resolve() == protected_checkpoint_path.resolve()
        )

        if collides_with_active and eligibility_errors:
            # An ineligible checkpoint must never be allowed to destroy the
            # active eligible checkpoint's bytes -- refuse outright, before
            # any write, rather than silently discarding it under a
            # different name (there is no legitimate reason to keep an
            # ineligible checkpoint that only exists because it happened to
            # collide with the deployed one).
            raise RuntimeError(
                f"Refusing to save checkpoint {checkpoint_path} -- its derived filename "
                f"collides with the checkpoint currently referenced by the active "
                f"deployment manifest ({manifest_path}), and this new checkpoint is NOT "
                f"deployment-eligible:\n" + "\n".join(f"  - {e}" for e in eligibility_errors) +
                "\nWriting to this path would destroy the active checkpoint's bytes with "
                "nothing eligible to replace it. No file was written."
            )

        if collides_with_active:
            # Eligible checkpoint that happens to share a filename with the
            # currently-deployed one (e.g. a resumed/repeated epoch landing
            # on the identical rounded val_score). Never overwrite the active
            # file in place -- save under a disambiguated filename instead,
            # so the active checkpoint's bytes remain completely untouched at
            # their original path for the entire duration of this call, and
            # only the NEW manifest (written further below, after this new
            # checkpoint is safely on disk) ever starts referencing the new
            # filename.
            disambiguator = 1
            while True:
                candidate_path = self.checkpoint_dir / f"epoch_{epoch}_val_score_{val_score:.4f}_r{disambiguator}.pt"
                if not candidate_path.exists():
                    break
                disambiguator += 1
            logger.warning(
                f"Derived checkpoint filename {checkpoint_path} collides with the actively "
                f"manifested checkpoint -- saving to {candidate_path} instead so the active "
                f"checkpoint is never overwritten."
            )
            checkpoint_path = candidate_path

        # P0-6 (fsync correction): save-then-fsync via the shared production
        # helper -- torch.save() alone only guarantees a Python-level
        # flush/close, not that the bytes are durably on disk, which matters
        # here because write_checkpoint_manifest() (below) immediately hashes
        # this same file.
        checkpoint_manifest.save_checkpoint_file(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        self.best_checkpoint_path = str(checkpoint_path)

        # P0-6 (Part B4/B5): only a checkpoint that passes every deployment-
        # eligibility check may get a manifest -- see
        # checkpoint_manifest.deployment_eligibility_errors() for the full
        # rule set (capped/partial/zero-node/zero-edge/structural-zero/
        # unknown-SHA/missing-split/malformed-hyperparameter/failed-evaluation
        # checkpoints are all rejected here, never manifested).
        if eligibility_errors:
            logger.info(
                f"Checkpoint {checkpoint_path} is NOT deployment-eligible -- no manifest "
                f"created/replaced. Reasons:\n" + "\n".join(f"  - {e}" for e in eligibility_errors)
            )
        else:
            checkpoint_manifest.write_checkpoint_manifest(checkpoint_path, checkpoint=checkpoint)
            protected_checkpoint_path = checkpoint_path
            logger.info(f"Deployment manifest written/replaced: {manifest_path}")

        # Clean up old checkpoints (keep last 3), but never delete the
        # checkpoint currently referenced by the active deployment manifest,
        # regardless of where it falls in the mtime ranking (Part B6).
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if protected_checkpoint_path is not None:
            resolved_protected = protected_checkpoint_path.resolve()
            checkpoints = [p for p in checkpoints if p.resolve() != resolved_protected]
        for old_checkpoint in checkpoints[3:]:
            old_checkpoint.unlink()
            logger.info(f"Deleted old checkpoint: {old_checkpoint}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        allow_split_mismatch: bool = False,
        allow_legacy_split: bool = False,
    ):
        """
        Load model checkpoint.

        P0-2 checkpoint/split-identity fix (2026-07-16): resuming TRAINING
        from a checkpoint trained under a different embryo-disjoint fold can
        directly contaminate the currently held-out embryo's weights -- a
        stricter, fail-by-default requirement than evaluate_checkpoint.py's
        warn-only legacy handling (see
        src/split_utils.py's validate_resume_checkpoint_split_identity()
        docstring for the full rationale). Unless this TrainingLoop's own
        split_identity is "unknown" (backward-compatibility case, logs a
        warning instead), a missing OR mismatched checkpoint identity raises
        RuntimeError by default -- pass allow_legacy_split=True /
        allow_split_mismatch=True only for a deliberate legacy warm start or
        cross-fold resume.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        validate_resume_checkpoint_split_identity(
            checkpoint, self.split_identity, checkpoint_path,
            allow_split_mismatch=allow_split_mismatch, allow_legacy_split=allow_legacy_split,
        )
        self.unet3d.load_state_dict(checkpoint['unet3d_state_dict'])
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint.get('val_metrics', {})
