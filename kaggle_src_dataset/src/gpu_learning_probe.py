"""Bounded 512-batch GPU learning probe with two complete validation samples.

This stage follows the GPU first-light infrastructure gate.  It provides an
early model-learning signal without claiming full-fold quality or producing a
deployable checkpoint.
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
DEFAULT_TRAIN_BATCHES = 512
DEFAULT_VALIDATION_SAMPLES = 2
DEFAULT_TIME_BUDGET_SECONDS = 3600.0


def _positive_finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _nonnegative_finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def evaluate_learning_probe_report(report: dict[str, Any], *, require_checkpoint: bool = True) -> list[str]:
    """Return every technical-contract violation; empty means execution PASS."""
    reasons: list[str] = []

    if type(report.get("schema_version")) is not int or report["schema_version"] != REPORT_SCHEMA_VERSION:
        reasons.append("report schema version is missing or unsupported")
    if report.get("probe_name") != "GPU-LEARNING-PROBE-01":
        reasons.append("probe name is missing or incorrect")
    if report.get("probe_scope") != "bounded_learning_signal_not_model_quality":
        reasons.append("probe scope does not prohibit full model-quality claims")

    deployed_sha = report.get("deployed_sha")
    if not isinstance(deployed_sha, str) or len(deployed_sha) != 40 or any(
        char not in "0123456789abcdef" for char in deployed_sha
    ):
        reasons.append("deployed_sha is not a 40-character lowercase hexadecimal SHA")
    for field, label in (
        ("probe_entrypoint_sha256", "probe entrypoint SHA-256"),
        ("split_membership_sha256", "split membership SHA-256"),
    ):
        value = report.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            reasons.append(f"{label} is missing or malformed")

    if report.get("import_origins_verified") is not True or not isinstance(report.get("import_origins"), dict) or not report["import_origins"]:
        reasons.append("production import origins were not verified")
    if report.get("cuda_available") is not True or report.get("device_type") != "cuda":
        reasons.append("CUDA device was not available and selected")
    if report.get("cuda_arch_compatible") is not True:
        reasons.append("CUDA compute capability was not verified compatible")
    if not report.get("gpu_name"):
        reasons.append("GPU name was not recorded")

    requested_batches = report.get("requested_train_batches")
    completed_batches = report.get("completed_train_batches")
    if requested_batches != DEFAULT_TRAIN_BATCHES:
        reasons.append(f"requested training batches must equal {DEFAULT_TRAIN_BATCHES}")
    if (
        not _positive_int(requested_batches)
        or not _positive_int(completed_batches)
        or completed_batches != requested_batches
    ):
        reasons.append("completed training batches do not equal the positive requested count")
    if not _positive_int(report.get("train_dataset_pair_count")) or (
        _positive_int(requested_batches) and report["train_dataset_pair_count"] < requested_batches
    ):
        reasons.append("training split does not expose enough pairs for the requested run")
    if not _positive_finite(report.get("average_train_loss")):
        reasons.append("average training loss is not finite and positive")
    if not _positive_finite(report.get("last_unet_gradient_norm")):
        reasons.append("UNet gradient snapshot is absent, zero, or non-finite")
    if not _positive_finite(report.get("last_transformer_gradient_norm")):
        reasons.append("Transformer gradient snapshot is absent, zero, or non-finite")

    expected_train_ids = report.get("expected_train_sample_ids")
    opened_train_ids = report.get("successfully_opened_train_sample_ids")
    if (
        not isinstance(expected_train_ids, list)
        or not expected_train_ids
        or len(expected_train_ids) != len(set(expected_train_ids))
        or not isinstance(opened_train_ids, list)
        or set(opened_train_ids) != set(expected_train_ids)
    ):
        reasons.append("training sample coverage is missing or incomplete")

    requested_validation = report.get("requested_validation_samples")
    selected_validation = report.get("selected_validation_sample_ids")
    opened_validation = report.get("successfully_opened_validation_sample_ids")
    source_validation_total = report.get("source_validation_fold_sample_count")
    if (
        not _positive_int(requested_validation)
        or not isinstance(selected_validation, list)
        or len(selected_validation) != requested_validation
        or len(selected_validation) != len(set(selected_validation))
        or not isinstance(opened_validation, list)
        or selected_validation != opened_validation
    ):
        reasons.append("selected validation sample identity/coverage is incomplete")
    if requested_validation != DEFAULT_VALIDATION_SAMPLES:
        reasons.append(f"requested validation samples must equal {DEFAULT_VALIDATION_SAMPLES}")
    if not _positive_int(source_validation_total) or (
        _positive_int(requested_validation) and source_validation_total < requested_validation
    ):
        reasons.append("source validation fold count is missing or smaller than the requested subset")
    if report.get("full_fold_validation_performed") is not False:
        reasons.append("learning probe must explicitly report that full-fold validation was not performed")

    metrics = report.get("validation_metrics")
    if not isinstance(metrics, dict):
        reasons.append("validation metrics are missing")
    else:
        if metrics.get("evaluation_completed_successfully") is not True:
            reasons.append("validation evaluation did not complete successfully")
        if metrics.get("validation_samples_evaluated") != requested_validation:
            reasons.append("validation did not evaluate exactly the requested sample count")
        if metrics.get("validation_sample_cap") != requested_validation:
            reasons.append("validation sample cap does not match the requested sample count")
        if metrics.get("validation_samples_total") != requested_validation:
            reasons.append("validation loader does not contain exactly the requested sample subset")
        if metrics.get("validation_is_full_fold") is not True:
            reasons.append("selected validation subset was not processed completely")
        if not _positive_int(metrics.get("predicted_nodes_total")):
            reasons.append("validation produced no predicted nodes")
        if not _nonnegative_int(metrics.get("predicted_edges_total")):
            reasons.append("validation predicted edge count is missing or invalid")
        if metrics.get("is_structural_zero") is not False:
            reasons.append("validation is structurally zero or missing its structural-zero flag")
        for field in ("edge_jaccard", "adjusted_edge_jaccard", "division_jaccard", "score"):
            if not _nonnegative_finite(metrics.get(field)):
                reasons.append(f"validation metric {field} is absent, negative, or non-finite")

    if not isinstance(report.get("learning_signal_observed"), bool):
        reasons.append("learning signal observation must be reported as a boolean")

    for field in ("train_fallback_counts", "post_validation_fallback_counts"):
        counters = report.get(field)
        if not isinstance(counters, dict) or not counters:
            reasons.append(f"{field} is missing or empty")
        elif any(not isinstance(value, int) or isinstance(value, bool) or value != 0 for value in counters.values()):
            reasons.append(f"{field} contains a nonzero or invalid technical fallback")

    biological_counts = report.get("train_biological_counts")
    if not isinstance(biological_counts, dict):
        reasons.append("training biological counter map is missing")
    else:
        if not _positive_int(biological_counts.get("edge_supervised_batches_total")):
            reasons.append("no edge-supervised training batch completed")
        if not _positive_int(biological_counts.get("edge_supervised_batches_with_nonzero_transformer_grad")):
            reasons.append("no edge-supervised batch produced a nonzero Transformer gradient")

    elapsed = report.get("elapsed_seconds")
    budget = report.get("time_budget_seconds")
    if budget != DEFAULT_TIME_BUDGET_SECONDS:
        reasons.append(f"time budget must equal {DEFAULT_TIME_BUDGET_SECONDS} seconds")
    if not _positive_finite(elapsed) or not _positive_finite(budget):
        reasons.append("elapsed time or time budget is absent, non-finite, or non-positive")
    elif elapsed > budget:
        reasons.append("learning probe exceeded its wall-clock budget")
    if not _positive_int(report.get("peak_gpu_memory_allocated_bytes")):
        reasons.append("peak allocated GPU memory was not recorded")
    if not _positive_int(report.get("peak_gpu_memory_reserved_bytes")):
        reasons.append("peak reserved GPU memory was not recorded")
    training_elapsed = report.get("training_elapsed_seconds")
    if not _positive_finite(training_elapsed) or (
        _positive_finite(elapsed) and training_elapsed > elapsed
    ):
        reasons.append("training elapsed time is absent or inconsistent with total elapsed time")

    if report.get("deployment_manifest_generated") is not False:
        reasons.append("learning probe must never generate a deployment manifest")
    if require_checkpoint:
        if report.get("probe_checkpoint_saved") is not True:
            reasons.append("probe checkpoint was not saved and round-trip verified")
        checkpoint_sha = report.get("probe_checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(
            char not in "0123456789abcdef" for char in checkpoint_sha
        ):
            reasons.append("probe checkpoint SHA-256 is missing or malformed")

    return reasons


def write_report_atomic(report: dict[str, Any], report_path: str | Path) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, report_path)


def _save_and_verify_probe_checkpoint(
    loop: TrainingLoop,
    checkpoint_path: Path,
    *,
    deployed_sha: str,
    split_identity: str,
    validation_metrics: dict[str, Any],
) -> str:
    checkpoint = {
        "probe_only": True,
        "sanity_only": True,
        "deployment_eligible": False,
        "training_code_sha": deployed_sha,
        "split_membership_sha256": split_identity,
        "completed_train_batches": loop.last_epoch_num_batches,
        "validation_metrics": validation_metrics,
        "unet3d_state_dict": loop.unet3d.state_dict(),
        "transformer_state_dict": loop.transformer.state_dict(),
    }
    checkpoint_manifest.save_checkpoint_file(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        loaded.get("probe_only") is not True
        or loaded.get("sanity_only") is not True
        or loaded.get("deployment_eligible") is not False
    ):
        raise RuntimeError("probe checkpoint round-trip lost its non-deployable identity markers")
    if loaded.get("training_code_sha") != deployed_sha or loaded.get("split_membership_sha256") != split_identity:
        raise RuntimeError("probe checkpoint round-trip changed provenance")
    return checkpoint_manifest.sha256_file(checkpoint_path)


def run_gpu_learning_probe(
    *,
    data_dir: str | Path,
    split_file: str | Path,
    output_dir: str | Path,
    device: torch.device,
    deployed_sha: str,
    import_origins: dict[str, str],
    probe_entrypoint_sha256: str,
    cuda_arch_compatible: bool,
    train_batches: int = DEFAULT_TRAIN_BATCHES,
    validation_samples: int = DEFAULT_VALIDATION_SAMPLES,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the bounded learning probe and always emit its JSON report."""
    for name, value in (("train_batches", train_batches), ("validation_samples", validation_samples)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not _positive_finite(time_budget_seconds):
        raise ValueError("time_budget_seconds must be finite and positive")
    if train_batches != DEFAULT_TRAIN_BATCHES:
        raise ValueError(f"train_batches must equal {DEFAULT_TRAIN_BATCHES}")
    if validation_samples != DEFAULT_VALIDATION_SAMPLES:
        raise ValueError(f"validation_samples must equal {DEFAULT_VALIDATION_SAMPLES}")
    if time_budget_seconds != DEFAULT_TIME_BUDGET_SECONDS:
        raise ValueError(f"time_budget_seconds must equal {DEFAULT_TIME_BUDGET_SECONDS}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gpu_learning_probe_report.json"
    checkpoint_path = output_dir / "learning_probe_checkpoint.pt"
    manifest_path = output_dir / checkpoint_manifest.MANIFEST_FILENAME
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "probe_name": "GPU-LEARNING-PROBE-01",
        "probe_scope": "bounded_learning_signal_not_model_quality",
        "deployed_sha": deployed_sha,
        "probe_entrypoint_sha256": probe_entrypoint_sha256,
        "import_origins": import_origins,
        "import_origins_verified": bool(import_origins),
        "cuda_available": torch.cuda.is_available(),
        "cuda_arch_compatible": cuda_arch_compatible,
        "device_type": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "requested_train_batches": train_batches,
        "requested_validation_samples": validation_samples,
        "full_fold_validation_performed": False,
        "time_budget_seconds": float(time_budget_seconds),
        "deployment_manifest_generated": False,
        "probe_checkpoint_saved": False,
        "learning_signal_observed": False,
        "verdict": "FAIL",
        "failure_reasons": [],
    }

    try:
        if manifest_path.exists():
            raise RuntimeError(f"unexpected deployment manifest already exists at {manifest_path}")
        if device.type != "cuda" or not torch.cuda.is_available() or not cuda_arch_compatible:
            raise RuntimeError("GPU learning probe requires a compatible CUDA device")

        split_file = Path(split_file)
        split_data = load_and_validate_split(split_file)
        train_ids = list(split_data["train"])
        validation_ids = list(split_data["validation"])
        if len(validation_ids) < validation_samples:
            raise RuntimeError(
                f"split has only {len(validation_ids)} validation samples, fewer than requested {validation_samples}"
            )
        selected_validation_ids = validation_ids[:validation_samples]
        split_identity = get_split_identity(split_file)

        train_dataset = CompetitionDataset(
            data_dir=data_dir,
            split_file=split_file,
            split_type="train",
            normalize=True,
            filter_unannotated_pairs=True,
            strict_sample_coverage=True,
        )
        validation_dataset = CompetitionDataset(
            data_dir=data_dir,
            split_file=split_file,
            split_type="validation",
            normalize=True,
            filter_unannotated_pairs=False,
            strict_sample_coverage=True,
            sample_id_allowlist=selected_validation_ids,
        )
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
        if len(train_loader) < train_batches:
            raise RuntimeError(
                f"training split has only {len(train_loader)} batches, fewer than requested {train_batches}"
            )

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
            "max_batches_per_epoch": train_batches,
            "max_validation_samples": validation_samples,
        }
        loop = TrainingLoop(
            unet3d=unet3d,
            transformer=transformer,
            train_loader=train_loader,
            val_loader=validation_loader,
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
        train_fallback_counts = dict(loop.epoch_fallback_counts)
        train_biological_counts = dict(loop.epoch_biological_zero_counts)
        training_elapsed = time.perf_counter() - started
        validation_metrics = {key: _json_scalar(value) for key, value in loop.validate_epoch().items()}

        report.update(
            {
                "split_membership_sha256": split_identity,
                "expected_train_sample_ids": train_ids,
                "successfully_opened_train_sample_ids": list(train_dataset.successfully_opened_sample_ids),
                "train_dataset_pair_count": len(train_dataset),
                "completed_train_batches": loop.last_epoch_num_batches,
                "average_train_loss": average_loss,
                "last_unet_gradient_norm": loop.last_unet_gradient_snapshot,
                "last_transformer_gradient_norm": loop.last_transformer_gradient_snapshot,
                "train_fallback_counts": train_fallback_counts,
                "train_biological_counts": train_biological_counts,
                "selected_validation_sample_ids": selected_validation_ids,
                "source_validation_fold_sample_count": len(validation_ids),
                "successfully_opened_validation_sample_ids": list(validation_dataset.successfully_opened_sample_ids),
                "validation_metrics": validation_metrics,
                "post_validation_fallback_counts": dict(loop.epoch_fallback_counts),
                "training_elapsed_seconds": training_elapsed,
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
        report["learning_signal_observed"] = bool(
            _positive_finite(validation_metrics.get("score"))
            and _positive_finite(validation_metrics.get("adjusted_edge_jaccard"))
            and _positive_int(validation_metrics.get("predicted_edges_total"))
        )
        report["elapsed_seconds"] = time.perf_counter() - started
        preliminary_reasons = evaluate_learning_probe_report(report, require_checkpoint=False)
        if not preliminary_reasons:
            checkpoint_sha = _save_and_verify_probe_checkpoint(
                loop,
                checkpoint_path,
                deployed_sha=deployed_sha,
                split_identity=split_identity,
                validation_metrics=validation_metrics,
            )
            report["probe_checkpoint_saved"] = True
            report["probe_checkpoint_sha256"] = checkpoint_sha
        else:
            report["failure_reasons"] = preliminary_reasons
    except Exception as error:
        report["exception_type"] = type(error).__name__
        report["exception_message"] = str(error)
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["deployment_manifest_generated"] = manifest_path.exists()
        reasons = evaluate_learning_probe_report(report, require_checkpoint=True)
        if report.get("exception_message"):
            reasons.append(f"{report['exception_type']}: {report['exception_message']}")
        report["failure_reasons"] = list(dict.fromkeys(reasons))
        report["verdict"] = "PASS" if not report["failure_reasons"] else "FAIL"
        write_report_atomic(report, report_path)

    return report
