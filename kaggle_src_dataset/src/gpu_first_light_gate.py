"""Bounded GPU infrastructure gate before any model-quality sanity run.

This gate deliberately does not call ``validate_epoch`` and does not claim
model quality.  It proves that reviewed source, real Kaggle data, CUDA, AMP,
both model branches, strict GT handling, optimizer stepping, reporting, and
non-deployable checkpoint serialization work together for a small fixed run.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src import checkpoint_manifest
from src.dataset import CompetitionDataset
from src.model import SimpleNodeTransformer, UNet3D
from src.split_utils import get_split_identity, load_and_validate_split
from src.train import TrainingLoop

REPORT_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_COUNT = 4
DEFAULT_MAX_BATCHES = 64
DEFAULT_TIME_BUDGET_SECONDS = 600.0


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def evaluate_first_light_report(report: dict[str, Any], *, require_checkpoint: bool = True) -> list[str]:
    """Return every fail-closed predicate violation; an empty list means PASS."""
    reasons: list[str] = []

    if type(report.get("schema_version")) is not int or report["schema_version"] != REPORT_SCHEMA_VERSION:
        reasons.append("report schema version is missing or unsupported")
    if report.get("gate_name") != "GPU-FIRST-LIGHT-01":
        reasons.append("gate name is missing or incorrect")
    if report.get("gate_scope") != "infrastructure_only_not_model_quality":
        reasons.append("gate scope does not explicitly prohibit model-quality claims")

    deployed_sha = report.get("deployed_sha")
    if not isinstance(deployed_sha, str) or len(deployed_sha) != 40 or any(
        char not in "0123456789abcdef" for char in deployed_sha
    ):
        reasons.append("deployed_sha is not a 40-character lowercase hexadecimal SHA")
    entrypoint_sha256 = report.get("gate_entrypoint_sha256")
    if not isinstance(entrypoint_sha256, str) or len(entrypoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in entrypoint_sha256
    ):
        reasons.append("gate entrypoint SHA-256 is missing or malformed")
    split_identity = report.get("split_membership_sha256")
    if not isinstance(split_identity, str) or len(split_identity) != 64 or any(
        char not in "0123456789abcdef" for char in split_identity
    ):
        reasons.append("split membership SHA-256 is missing or malformed")
    if report.get("import_origins_verified") is not True or not isinstance(report.get("import_origins"), dict) or not report["import_origins"]:
        reasons.append("production import origins were not verified")
    if not report.get("cuda_available") or report.get("device_type") != "cuda":
        reasons.append("CUDA device was not available and selected")
    if not report.get("cuda_arch_compatible"):
        reasons.append("CUDA compute capability was not verified compatible")
    if not report.get("gpu_name"):
        reasons.append("GPU name was not recorded")

    selected_sample_ids = report.get("selected_sample_ids")
    selected_sample_count = report.get("selected_sample_count")
    sample_count_requested = report.get("sample_count_requested")
    opened_sample_ids = report.get("successfully_opened_sample_ids")
    if (
        not isinstance(selected_sample_ids, list)
        or not selected_sample_ids
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in selected_sample_ids)
        or len(selected_sample_ids) != len(set(selected_sample_ids))
        or not _positive_int(selected_sample_count)
        or not _positive_int(sample_count_requested)
        or selected_sample_count != len(selected_sample_ids)
        or sample_count_requested != len(selected_sample_ids)
    ):
        reasons.append("selected sample identity/count is missing, empty, or inconsistent")
    elif (
        not isinstance(opened_sample_ids, list)
        or any(not isinstance(sample_id, str) for sample_id in opened_sample_ids)
        or set(opened_sample_ids) != set(selected_sample_ids)
    ):
        reasons.append("successfully opened samples do not exactly match the selected subset")

    requested_batches = report.get("requested_batches")
    completed_batches = report.get("completed_batches")
    valid_requested_batches = _positive_int(requested_batches)
    valid_completed_batches = _positive_int(completed_batches)
    if not valid_requested_batches or not valid_completed_batches or completed_batches != requested_batches:
        reasons.append("completed batch count does not equal the positive requested batch count")
    dataset_pair_count = report.get("dataset_pair_count")
    if (
        not _positive_int(dataset_pair_count)
        or not valid_requested_batches
        or dataset_pair_count < requested_batches
    ):
        reasons.append("selected subset does not expose enough dataset pairs for the requested run")
    if not _positive_finite(report.get("average_train_loss")):
        reasons.append("average training loss is not finite and positive")
    if not _positive_finite(report.get("last_unet_gradient_norm")):
        reasons.append("UNet gradient snapshot is absent, zero, or non-finite")
    if not _positive_finite(report.get("last_transformer_gradient_norm")):
        reasons.append("Transformer gradient snapshot is absent, zero, or non-finite")

    fallback_counts = report.get("fallback_counts")
    if not isinstance(fallback_counts, dict) or not fallback_counts:
        reasons.append("fallback counter map is missing or empty")
    elif any(not isinstance(value, int) or isinstance(value, bool) or value != 0 for value in fallback_counts.values()):
        reasons.append("at least one technical fallback counter is nonzero or invalid")

    biological_counts = report.get("biological_counts")
    if not isinstance(biological_counts, dict):
        reasons.append("biological counter map is missing")
    else:
        edge_supervised_total = biological_counts.get("edge_supervised_batches_total")
        transformer_nonzero_total = biological_counts.get("edge_supervised_batches_with_nonzero_transformer_grad")
        if not _positive_int(edge_supervised_total):
            reasons.append("no edge-supervised batch completed")
        if (
            not _positive_int(transformer_nonzero_total)
        ):
            reasons.append("no positive-edge batch produced a finite nonzero Transformer gradient")

    elapsed_seconds = report.get("elapsed_seconds")
    time_budget_seconds = report.get("time_budget_seconds")
    if not _positive_finite(elapsed_seconds) or not _positive_finite(time_budget_seconds):
        reasons.append("elapsed time or time budget is absent, non-finite, or non-positive")
    elif elapsed_seconds > time_budget_seconds:
        reasons.append("first-light run exceeded its wall-clock budget")

    if report.get("validation_performed") is not False:
        reasons.append("first-light gate must not perform validation")
    if report.get("deployment_manifest_generated") is not False:
        reasons.append("first-light gate must never generate a deployment manifest")
    allocated_bytes = report.get("peak_gpu_memory_allocated_bytes")
    reserved_bytes = report.get("peak_gpu_memory_reserved_bytes")
    if not _positive_int(allocated_bytes):
        reasons.append("peak allocated GPU memory was not recorded")
    if not _positive_int(reserved_bytes):
        reasons.append("peak reserved GPU memory was not recorded")

    if require_checkpoint:
        if not report.get("sanity_checkpoint_saved"):
            reasons.append("sanity checkpoint was not saved and round-trip verified")
        checkpoint_sha256 = report.get("sanity_checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in checkpoint_sha256
        ):
            reasons.append("sanity checkpoint SHA-256 is missing or malformed")

    return reasons


def write_report_atomic(report: dict[str, Any], report_path: str | Path) -> None:
    """Atomically write a deterministic, human-readable JSON report."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, report_path)


def _save_and_verify_sanity_checkpoint(
    loop: TrainingLoop,
    checkpoint_path: Path,
    *,
    deployed_sha: str,
    split_identity: str,
    selected_sample_ids: list[str],
) -> str:
    checkpoint = {
        "sanity_only": True,
        "deployment_eligible": False,
        "training_code_sha": deployed_sha,
        "split_membership_sha256": split_identity,
        "selected_sample_ids": selected_sample_ids,
        "completed_batches": loop.last_epoch_num_batches,
        "unet3d_state_dict": loop.unet3d.state_dict(),
        "transformer_state_dict": loop.transformer.state_dict(),
    }
    checkpoint_manifest.save_checkpoint_file(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if loaded.get("sanity_only") is not True or loaded.get("deployment_eligible") is not False:
        raise RuntimeError("sanity checkpoint round-trip lost its non-deployable identity markers")
    if loaded.get("training_code_sha") != deployed_sha or loaded.get("selected_sample_ids") != selected_sample_ids:
        raise RuntimeError("sanity checkpoint round-trip changed provenance or subset identity")
    return checkpoint_manifest.sha256_file(checkpoint_path)


def run_gpu_first_light_gate(
    *,
    data_dir: str | Path,
    split_file: str | Path,
    output_dir: str | Path,
    device: torch.device,
    deployed_sha: str,
    import_origins: dict[str, str],
    gate_entrypoint_sha256: str,
    cuda_arch_compatible: bool,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    max_batches: int = DEFAULT_MAX_BATCHES,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the bounded first-light gate and always emit ``gpu_first_light_report.json``."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0:
        raise ValueError("max_batches must be a positive integer")
    if not _positive_finite(time_budget_seconds):
        raise ValueError("time_budget_seconds must be finite and positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gpu_first_light_report.json"
    checkpoint_path = output_dir / "sanity_checkpoint.pt"
    manifest_path = output_dir / checkpoint_manifest.MANIFEST_FILENAME
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "gate_name": "GPU-FIRST-LIGHT-01",
        "gate_scope": "infrastructure_only_not_model_quality",
        "deployed_sha": deployed_sha,
        "gate_entrypoint_sha256": gate_entrypoint_sha256,
        "import_origins": import_origins,
        "import_origins_verified": bool(import_origins),
        "cuda_available": torch.cuda.is_available(),
        "cuda_arch_compatible": cuda_arch_compatible,
        "device_type": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "requested_batches": max_batches,
        "sample_count_requested": sample_count,
        "time_budget_seconds": float(time_budget_seconds),
        "validation_performed": False,
        "deployment_manifest_generated": False,
        "sanity_checkpoint_saved": False,
        "verdict": "FAIL",
        "failure_reasons": [],
    }

    try:
        if manifest_path.exists():
            raise RuntimeError(f"unexpected deployment manifest already exists at {manifest_path}")
        if device.type != "cuda" or not torch.cuda.is_available() or not cuda_arch_compatible:
            raise RuntimeError("GPU first-light gate requires a compatible CUDA device")

        split_file = Path(split_file)
        split_data = load_and_validate_split(split_file)
        train_ids = list(split_data["train"])
        if len(train_ids) < sample_count:
            raise RuntimeError(f"split has only {len(train_ids)} train samples, fewer than requested {sample_count}")
        selected_sample_ids = train_ids[:sample_count]
        split_identity = get_split_identity(split_file)

        dataset = CompetitionDataset(
            data_dir=data_dir,
            split_file=split_file,
            split_type="train",
            normalize=True,
            filter_unannotated_pairs=True,
            strict_sample_coverage=True,
            sample_id_allowlist=selected_sample_ids,
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)
        if len(loader) < max_batches:
            raise RuntimeError(f"selected subset has only {len(loader)} train batches, fewer than requested {max_batches}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        unet3d = UNet3D(in_channels=2, channels=(32, 64, 128)).to(device)
        transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4).to(device)
        torch.cuda.reset_peak_memory_stats(device)

        hyperparams = {
            "learning_rate": 3e-3,
            "warmup_steps": 300,
            "warmup_start_lr": 1e-4,
            "grad_clip": 1.0,
            "weight_decay": 1e-4,
            "heatmap_loss_weight": 1.0,
            "division_loss_weight": 2.5,
            "early_stopping_patience": 10,
            "edge_threshold": 0.5,
            "detection_threshold": 0.5,
            "nms_radius_um": 5.0,
            "seed": seed,
            "max_batches_per_epoch": max_batches,
            "max_validation_samples": None,
        }
        loop = TrainingLoop(
            unet3d=unet3d,
            transformer=transformer,
            train_loader=loader,
            val_loader=[],
            device=device,
            data_dir=data_dir,
            checkpoint_dir=str(output_dir),
            log_file=str(output_dir / "unused_training_log.csv"),
            hyperparams=hyperparams,
            deployed_sha=deployed_sha,
            progress_file=None,
            split_identity=split_identity,
            strict_integrity_mode=True,
        )
        average_loss = loop.train_epoch()

        report.update(
            {
                "split_membership_sha256": split_identity,
                "selected_sample_ids": selected_sample_ids,
                "selected_sample_count": len(selected_sample_ids),
                "dataset_pair_count": len(dataset),
                "successfully_opened_sample_ids": list(dataset.successfully_opened_sample_ids),
                "completed_batches": loop.last_epoch_num_batches,
                "average_train_loss": average_loss,
                "last_unet_gradient_norm": loop.last_unet_gradient_snapshot,
                "last_transformer_gradient_norm": loop.last_transformer_gradient_snapshot,
                "fallback_counts": dict(loop.epoch_fallback_counts),
                "biological_counts": dict(loop.epoch_biological_zero_counts),
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
        report["elapsed_seconds"] = time.perf_counter() - started
        preliminary_reasons = evaluate_first_light_report(report, require_checkpoint=False)
        if not preliminary_reasons:
            checkpoint_sha256 = _save_and_verify_sanity_checkpoint(
                loop,
                checkpoint_path,
                deployed_sha=deployed_sha,
                split_identity=split_identity,
                selected_sample_ids=selected_sample_ids,
            )
            report["sanity_checkpoint_saved"] = True
            report["sanity_checkpoint_sha256"] = checkpoint_sha256
        else:
            report["failure_reasons"] = preliminary_reasons
    except Exception as error:
        report["exception_type"] = type(error).__name__
        report["exception_message"] = str(error)
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["deployment_manifest_generated"] = manifest_path.exists()
        reasons = evaluate_first_light_report(report, require_checkpoint=True)
        if report.get("exception_message"):
            reasons.append(f"{report['exception_type']}: {report['exception_message']}")
        report["failure_reasons"] = list(dict.fromkeys(reasons))
        report["verdict"] = "PASS" if not report["failure_reasons"] else "FAIL"
        write_report_atomic(report, report_path)

    return report
