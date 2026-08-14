"""Bounded two-sample, multi-frame calibration diagnostic.

This diagnostic reuses the immutable GPU learning-probe checkpoint.  It runs
exactly four deterministic annotated frames from each of the probe's two
validation samples, freezes every float32 probability volume, and evaluates a
common threshold/NMS grid without the adaptive inference fallback.  Its result
is evidence for a later experiment, never a production configuration or a
checkpoint-promotion decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataset import CompetitionDataset
from src.evaluation import DEFAULT_MAX_DISTANCE, DEFAULT_SCALE, load_geff_ground_truth
from src.gpu_coordinate_diagnostic import (
    NMS_RADII_UM,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_PROBE_SHA,
    SOURCE_SPLIT_SHA256,
    THRESHOLD_RULES,
    build_sweep,
    sha256_file,
)
from src.model import UNet3D
from src.split_utils import get_split_identity

REPORT_SCHEMA_VERSION = 1
DIAGNOSTIC_NAME = "GPU-MULTIFRAME-CALIBRATION-01"
DIAGNOSTIC_SCOPE = "bounded_two_sample_calibration_not_model_quality_or_production_selection"
SAMPLE_IDS = ("44b6_0113de3b", "44b6_0b24845f")
FRAMES_PER_SAMPLE = 4
EXPECTED_FRAME_COUNT = len(SAMPLE_IDS) * FRAMES_PER_SAMPLE
TIME_BUDGET_SECONDS = 1200.0
PROBABILITY_FILENAME = "multiframe_probability_f32.npz"
REPORT_FILENAME = "gpu_multiframe_calibration_report.json"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_report_atomic(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_annotated_frames(data_dir: Path, dataset: CompetitionDataset) -> list[dict[str, Any]]:
    """Select the earliest four reachable annotated frames per sample."""
    pair_to_index = {pair: index for index, pair in enumerate(dataset.pairs)}
    selections: list[dict[str, Any]] = []
    for sample_id in SAMPLE_IDS:
        graph, _metadata = load_geff_ground_truth(str(data_dir / f"{sample_id}.geff"))
        attrs = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        if attrs.height == 0:
            raise RuntimeError(f"selected sample {sample_id} has no ground-truth nodes")
        selected_for_sample = 0
        for frame_timepoint in sorted({int(value) for value in attrs["t"].to_list()}):
            if (sample_id, frame_timepoint) in pair_to_index:
                window_t_idx, channel = frame_timepoint, 0
            elif (sample_id, frame_timepoint - 1) in pair_to_index:
                window_t_idx, channel = frame_timepoint - 1, 1
            else:
                continue
            rows = (
                attrs.filter(attrs["t"] == frame_timepoint)
                .select(["node_id", "z", "y", "x"])
                .to_dicts()
            )
            nodes = [
                {
                    "node_id": int(row["node_id"]),
                    "zyx": [float(row["z"]), float(row["y"]), float(row["x"])],
                }
                for row in rows
            ]
            artifact_key = f"sample_{SAMPLE_IDS.index(sample_id)}_t_{frame_timepoint}_c_{channel}"
            selections.append(
                {
                    "artifact_key": artifact_key,
                    "sample_id": sample_id,
                    "dataset_index": pair_to_index[(sample_id, window_t_idx)],
                    "window_t_idx": window_t_idx,
                    "model_output_channel": channel,
                    "frame_timepoint": frame_timepoint,
                    "frame_role": "frame_t" if channel == 0 else "frame_t1",
                    "ground_truth_nodes": nodes,
                }
            )
            selected_for_sample += 1
            if selected_for_sample == FRAMES_PER_SAMPLE:
                break
        if selected_for_sample != FRAMES_PER_SAMPLE:
            raise RuntimeError(
                f"sample {sample_id} exposes {selected_for_sample} reachable annotated frames; "
                f"exactly {FRAMES_PER_SAMPLE} are required"
            )
    if len(selections) != EXPECTED_FRAME_COUNT:
        raise RuntimeError("selected frame count does not match the bounded contract")
    return selections


def aggregate_sweeps(frame_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Micro-aggregate a common grid, retaining per-sample evidence."""
    expected_pairs = [(rule, radius) for rule in THRESHOLD_RULES for radius in NMS_RADII_UM]
    rows: list[dict[str, Any]] = []
    for rule, radius in expected_pairs:
        matching_rows = [
            row
            for frame in frame_results
            for row in frame["grid"]
            if row["threshold_rule"] == rule and row["nms_radius_um"] == radius
        ]
        if len(matching_rows) != EXPECTED_FRAME_COUNT:
            raise RuntimeError("incomplete frame grid during aggregation")
        tp = sum(row["tp"] for row in matching_rows)
        fp = sum(row["fp"] for row in matching_rows)
        fn = sum(row["fn"] for row in matching_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_sample = []
        for sample_id in SAMPLE_IDS:
            sample_frames = [frame for frame in frame_results if frame["sample_id"] == sample_id]
            sample_rows = [
                row
                for frame in sample_frames
                for row in frame["grid"]
                if row["threshold_rule"] == rule and row["nms_radius_um"] == radius
            ]
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "frames": len(sample_frames),
                    "tp": sum(row["tp"] for row in sample_rows),
                    "fp": sum(row["fp"] for row in sample_rows),
                    "fn": sum(row["fn"] for row in sample_rows),
                    "frames_with_match": sum(row["tp"] > 0 for row in sample_rows),
                }
            )
        rows.append(
            {
                "threshold_rule": rule,
                "nms_radius_um": radius,
                "frames": EXPECTED_FRAME_COUNT,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "predicted_count": tp + fp,
                "ground_truth_count": tp + fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "per_sample": per_sample,
            }
        )
    return rows


def rank_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose one descriptive candidate without authorizing production use."""
    rule_order = {rule: index for index, rule in enumerate(THRESHOLD_RULES)}

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        samples_with_match = sum(item["tp"] > 0 for item in row["per_sample"])
        return (
            -samples_with_match,
            -row["f1"],
            -row["recall"],
            row["predicted_count"],
            abs(row["nms_radius_um"] - 5.0),
            rule_order[row["threshold_rule"]],
        )

    candidate = dict(min(rows, key=rank))
    candidate["recommendation_only"] = True
    candidate["production_configuration_selected"] = False
    return candidate


def classify_scientific_result(aggregate_rows: list[dict[str, Any]]) -> str:
    if any(all(sample["tp"] > 0 for sample in row["per_sample"]) for row in aggregate_rows):
        return "GATED_MATCHES_IN_BOTH_SAMPLES"
    if any(row["tp"] > 0 for row in aggregate_rows):
        return "GATED_MATCHES_IN_ONE_SAMPLE_ONLY"
    if any(row["predicted_count"] > 0 for row in aggregate_rows):
        return "PREDICTIONS_WITHOUT_GATED_MATCHES"
    return "NO_EXTRACTABLE_PEAKS"


def _artifact_entries(probability_path: Path) -> dict[str, np.ndarray]:
    with np.load(probability_path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def evaluate_multiframe_calibration_report(
    report: dict[str, Any], probability_path: Path | None = None
) -> list[str]:
    """Return fail-closed technical violations; scientific weakness may PASS."""
    reasons: list[str] = []
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1:
        reasons.append("report schema version is missing or unsupported")
    if (
        report.get("diagnostic_name") != DIAGNOSTIC_NAME
        or report.get("diagnostic_scope") != DIAGNOSTIC_SCOPE
    ):
        reasons.append("diagnostic identity or scope is incorrect")
    if not _is_lower_hex(report.get("deployed_sha"), 40):
        reasons.append("deployed SHA is malformed")
    if not _is_lower_hex(report.get("diagnostic_entrypoint_sha256"), 64):
        reasons.append("diagnostic entrypoint SHA-256 is malformed")
    if report.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        reasons.append("source checkpoint SHA-256 is incorrect")
    if report.get("source_checkpoint_training_code_sha") != SOURCE_PROBE_SHA:
        reasons.append("source checkpoint training-code SHA is incorrect")
    if report.get("split_membership_sha256") != SOURCE_SPLIT_SHA256:
        reasons.append("split identity is incorrect")
    if report.get("checkpoint_loaded_weights_only") is not True:
        reasons.append("source checkpoint was not loaded with weights_only=True")
    if report.get("import_origins_verified") is not True or not report.get("import_origins"):
        reasons.append("production import origins were not verified")
    if report.get("device_type") != "cuda" or report.get("cuda_available") is not True:
        reasons.append("CUDA was not selected")
    if report.get("cuda_arch_compatible") is not True:
        reasons.append("CUDA architecture is incompatible")
    if (
        report.get("forward_pass_count") != EXPECTED_FRAME_COUNT
        or report.get("eval_mode") is not True
        or report.get("no_grad") is not True
        or report.get("inference_autocast_dtype") != "float16"
    ):
        reasons.append("inference forward-pass contract is incorrect")
    for field in ("optimizer_steps", "backward_calls", "adaptive_fallback_calls"):
        if report.get(field) != 0:
            reasons.append(f"{field} must be exactly zero")
    for field in ("deployment_manifest_generated", "new_checkpoint_generated"):
        if report.get(field) is not False:
            reasons.append(f"{field} must remain false")
    if report.get("model_state_sha256_before") != report.get("model_state_sha256_after"):
        reasons.append("model state changed")
    if report.get("selected_sample_ids") != list(SAMPLE_IDS):
        reasons.append("selected sample identities are incorrect")
    if report.get("successfully_opened_sample_ids") != list(SAMPLE_IDS):
        reasons.append("selected sample coverage is incomplete")
    if report.get("frames_per_sample") != FRAMES_PER_SAMPLE:
        reasons.append("frames-per-sample cap is incorrect")
    peak_memory = report.get("peak_gpu_memory_bytes")
    if not isinstance(peak_memory, int) or isinstance(peak_memory, bool) or peak_memory <= 0:
        reasons.append("peak GPU memory is missing or nonpositive")
    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        reasons.append("frame evidence is missing or incomplete")
        frames = []
    elif any(
        sum(frame.get("sample_id") == sample_id for frame in frames) != FRAMES_PER_SAMPLE
        for sample_id in SAMPLE_IDS
    ):
        reasons.append("per-sample frame coverage is incomplete")
    else:
        identities = [(frame.get("sample_id"), frame.get("frame_timepoint")) for frame in frames]
        if len(identities) != len(set(identities)):
            reasons.append("selected frame identities are duplicated")
        for frame in frames:
            nodes = frame.get("ground_truth_nodes")
            if not isinstance(nodes, list) or not nodes:
                reasons.append("selected frame has empty ground truth")
                break
            try:
                gt = np.asarray([node["zyx"] for node in nodes], dtype=np.float64).reshape(-1, 3)
            except (KeyError, TypeError, ValueError):
                reasons.append("selected frame ground-truth coordinates are malformed")
                break
            shape = np.asarray((64, 256, 256), dtype=np.float64)
            if not np.isfinite(gt).all() or not ((gt >= 0) & (gt < shape)).all():
                reasons.append("selected frame ground-truth coordinates are invalid")
                break
            if (
                frame.get("coordinate_order") != ["z", "y", "x"]
                or frame.get("voxel_size_um") != list(DEFAULT_SCALE)
                or frame.get("match_gate_um") != DEFAULT_MAX_DISTANCE
            ):
                reasons.append("frame coordinate convention is incorrect")
                break
            grid = frame.get("grid")
            expected_pairs = {(rule, radius) for rule in THRESHOLD_RULES for radius in NMS_RADII_UM}
            if (
                not isinstance(grid, list)
                or len(grid) != len(expected_pairs)
                or {(row.get("threshold_rule"), row.get("nms_radius_um")) for row in grid}
                != expected_pairs
            ):
                reasons.append("selected frame threshold/NMS grid is incomplete")
                break
    aggregate = report.get("aggregate_grid")
    if not isinstance(aggregate, list) or len(aggregate) != len(THRESHOLD_RULES) * len(
        NMS_RADII_UM
    ):
        reasons.append("aggregate grid is missing or incomplete")
    if report.get("production_configuration_selected") is not False:
        reasons.append("diagnostic must not select a production configuration")
    candidate = report.get("calibration_candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("recommendation_only") is not True
        or candidate.get("production_configuration_selected") is not False
    ):
        reasons.append("calibration candidate is missing or claims production authority")
    if report.get("scientific_result") not in {
        "GATED_MATCHES_IN_BOTH_SAMPLES",
        "GATED_MATCHES_IN_ONE_SAMPLE_ONLY",
        "PREDICTIONS_WITHOUT_GATED_MATCHES",
        "NO_EXTRACTABLE_PEAKS",
    }:
        reasons.append("scientific result is missing or unsupported")
    if report.get("checkpoint_promotion_performed") is not False:
        reasons.append("diagnostic must not promote a checkpoint")
    if report.get("time_budget_seconds") != TIME_BUDGET_SECONDS:
        reasons.append("declared time budget is incorrect")
    elapsed = report.get("elapsed_seconds")
    if (
        not isinstance(elapsed, int | float)
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed < 0
        or elapsed > TIME_BUDGET_SECONDS
    ):
        reasons.append("elapsed time is invalid or exceeds the bounded budget")

    if probability_path is not None and frames:
        artifact = report.get("probability_artifact", {})
        try:
            if sha256_file(probability_path) != artifact.get("sha256"):
                reasons.append("probability artifact byte SHA-256 does not match")
            volumes = _artifact_entries(probability_path)
            expected_keys = {frame["artifact_key"] for frame in frames}
            if set(volumes) != expected_keys:
                reasons.append("probability artifact keys do not match selected frames")
            else:
                recomputed_frames = []
                for frame in frames:
                    volume = volumes[frame["artifact_key"]]
                    if volume.dtype != np.float32 or volume.shape != (64, 256, 256):
                        reasons.append("probability volume dtype or shape is incorrect")
                        continue
                    if not np.isfinite(volume).all() or volume.min() < 0 or volume.max() > 1:
                        reasons.append("probability volume is nonfinite or outside [0,1]")
                        continue
                    gt = np.asarray(
                        [node["zyx"] for node in frame["ground_truth_nodes"]], dtype=np.float64
                    )
                    grid, best = build_sweep(volume, gt)
                    recomputed = dict(frame)
                    recomputed["probability_sha256"] = hashlib.sha256(volume.tobytes()).hexdigest()
                    recomputed["grid"] = grid
                    recomputed["best_point"] = best
                    recomputed_frames.append(recomputed)
                if len(recomputed_frames) == EXPECTED_FRAME_COUNT:
                    if _sha256_json(frames) != _sha256_json(recomputed_frames):
                        reasons.append("frame grids do not recompute from frozen probabilities")
                    recomputed_aggregate = aggregate_sweeps(recomputed_frames)
                    if _sha256_json(aggregate) != _sha256_json(recomputed_aggregate):
                        reasons.append("aggregate grid does not recompute")
                    if _sha256_json(report.get("calibration_candidate")) != _sha256_json(
                        rank_aggregate(recomputed_aggregate)
                    ):
                        reasons.append("calibration candidate does not recompute")
                    if report.get("scientific_result") != classify_scientific_result(
                        recomputed_aggregate
                    ):
                        reasons.append("scientific result does not recompute")
        except Exception as error:
            reasons.append(f"probability artifact could not be safely validated: {error}")
    return reasons


def run_gpu_multiframe_calibration(
    *,
    data_dir: Path,
    split_file: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    deployed_sha: str,
    import_origins: dict[str, str],
    diagnostic_entrypoint_sha256: str,
) -> dict[str, Any]:
    """Run the bounded calibration diagnostic and always write a report."""
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    probability_path = output_dir / PROBABILITY_FILENAME
    manifest_path = output_dir / "checkpoint_manifest.json"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnostic_name": DIAGNOSTIC_NAME,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "deployed_sha": deployed_sha,
        "diagnostic_entrypoint_sha256": diagnostic_entrypoint_sha256,
        "source_checkpoint_sha256": None,
        "source_checkpoint_training_code_sha": None,
        "checkpoint_loaded_weights_only": False,
        "split_membership_sha256": None,
        "import_origins": import_origins,
        "import_origins_verified": bool(import_origins),
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_arch_compatible": False,
        "eval_mode": False,
        "no_grad": False,
        "forward_pass_count": 0,
        "inference_autocast_dtype": None,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "adaptive_fallback_calls": 0,
        "selected_sample_ids": list(SAMPLE_IDS),
        "successfully_opened_sample_ids": [],
        "frames_per_sample": FRAMES_PER_SAMPLE,
        "peak_gpu_memory_bytes": None,
        "production_configuration_selected": False,
        "checkpoint_promotion_performed": False,
        "deployment_manifest_generated": False,
        "new_checkpoint_generated": False,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "elapsed_seconds": None,
        "execution_verdict": "FAIL",
        "failure_reasons": [],
    }
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("multiframe calibration requires CUDA")
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError(f"unexpected source checkpoint SHA-256: {checkpoint_sha}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        report["checkpoint_loaded_weights_only"] = True
        if checkpoint.get("training_code_sha") != SOURCE_PROBE_SHA:
            raise RuntimeError("checkpoint training-code SHA does not match the immutable probe")
        if not (checkpoint.get("probe_only") is True and checkpoint.get("sanity_only") is True):
            raise RuntimeError("source checkpoint lost its probe-only identity")
        if checkpoint.get("deployment_eligible") is not False:
            raise RuntimeError("source checkpoint unexpectedly claims deployment eligibility")
        report["source_checkpoint_sha256"] = checkpoint_sha
        report["source_checkpoint_training_code_sha"] = checkpoint["training_code_sha"]
        split_identity = get_split_identity(split_file)
        if split_identity != SOURCE_SPLIT_SHA256:
            raise RuntimeError("active split identity does not match the immutable probe split")
        report["split_membership_sha256"] = split_identity
        major, _minor = torch.cuda.get_device_capability(device)
        report["cuda_arch_compatible"] = major >= 7
        if not report["cuda_arch_compatible"]:
            raise RuntimeError("CUDA device compute capability is below 7.0")

        dataset = CompetitionDataset(
            data_dir=data_dir,
            split_file=split_file,
            split_type="validation",
            normalize=True,
            filter_unannotated_pairs=False,
            strict_sample_coverage=True,
            sample_id_allowlist=list(SAMPLE_IDS),
        )
        if dataset.successfully_opened_sample_ids != list(SAMPLE_IDS):
            raise RuntimeError("selected validation sample coverage is incomplete")
        report["successfully_opened_sample_ids"] = list(dataset.successfully_opened_sample_ids)
        selections = select_annotated_frames(data_dir, dataset)
        model = UNet3D(in_channels=2, channels=(32, 64, 128)).to(device)
        model.load_state_dict(checkpoint["unet3d_state_dict"], strict=True)
        report["model_state_sha256_before"] = _state_dict_sha256(model.state_dict())
        model.eval()
        report["eval_mode"] = not model.training
        torch.cuda.reset_peak_memory_stats(device)
        probabilities: dict[str, np.ndarray] = {}
        frame_results = []
        with torch.no_grad():
            report["no_grad"] = not torch.is_grad_enabled()
            for selection in selections:
                item = dataset[selection["dataset_index"]]
                if (
                    item.get("sample_id") != selection["sample_id"]
                    or int(item.get("t_idx", -1)) != selection["window_t_idx"]
                ):
                    raise RuntimeError("dataset returned a different selected frame")
                frame_t = item["frame_t"].unsqueeze(0).to(device)
                frame_t1 = item["frame_t1"].unsqueeze(0).to(device)
                if tuple(frame_t.shape) != (1, 1, 64, 256, 256) or tuple(frame_t1.shape) != (
                    1,
                    1,
                    64,
                    256,
                    256,
                ):
                    raise RuntimeError("selected frame pair has an unexpected tensor shape")
                input_pair = torch.cat([frame_t, frame_t1], dim=1)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    report["inference_autocast_dtype"] = "float16"
                    logits, features = model(input_pair)
                    report["forward_pass_count"] += 1
                    probability = (
                        torch.sigmoid(logits.float())[0, selection["model_output_channel"]]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                del features, logits, input_pair, frame_t, frame_t1
                probabilities[selection["artifact_key"]] = probability
                gt = np.asarray(
                    [node["zyx"] for node in selection["ground_truth_nodes"]], dtype=np.float64
                )
                grid, best = build_sweep(probability, gt)
                frame_result = dict(selection)
                frame_result["coordinate_order"] = ["z", "y", "x"]
                frame_result["voxel_size_um"] = list(DEFAULT_SCALE)
                frame_result["match_gate_um"] = DEFAULT_MAX_DISTANCE
                frame_result["probability_shape"] = list(probability.shape)
                frame_result["probability_sha256"] = hashlib.sha256(
                    probability.tobytes()
                ).hexdigest()
                frame_result["grid"] = grid
                frame_result["best_point"] = best
                frame_results.append(frame_result)
                if time.monotonic() - started > TIME_BUDGET_SECONDS:
                    raise RuntimeError("multiframe calibration exceeded its time budget")

        report["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))

        np.savez(probability_path, **probabilities)
        # All report evidence is derived again from the real round-tripped arrays.
        frozen = _artifact_entries(probability_path)
        for frame in frame_results:
            volume = frozen[frame["artifact_key"]]
            if hashlib.sha256(volume.tobytes()).hexdigest() != frame["probability_sha256"]:
                raise RuntimeError("round-tripped probability volume changed")
        aggregate = aggregate_sweeps(frame_results)
        report.update(
            {
                "model_state_sha256_after": _state_dict_sha256(model.state_dict()),
                "frames": frame_results,
                "aggregate_grid": aggregate,
                "calibration_candidate": rank_aggregate(aggregate),
                "scientific_result": classify_scientific_result(aggregate),
                "probability_artifact": {
                    "file": PROBABILITY_FILENAME,
                    "sha256": sha256_file(probability_path),
                    "array_count": len(frozen),
                    "keys": sorted(frozen),
                    "dtype": "float32",
                    "shape_per_array": [64, 256, 256],
                },
            }
        )
        report["elapsed_seconds"] = time.monotonic() - started
        report["deployment_manifest_generated"] = manifest_path.exists()
        report["new_checkpoint_generated"] = any(output_dir.glob("*.pt"))
        report["failure_reasons"] = evaluate_multiframe_calibration_report(report, probability_path)
    except Exception as error:
        report["exception_type"] = type(error).__name__
        report["exception_message"] = str(error)
        report["failure_reasons"] = [f"{type(error).__name__}: {error}"]
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["deployment_manifest_generated"] = manifest_path.exists()
        report["new_checkpoint_generated"] = any(output_dir.glob("*.pt"))
        for field in ("deployment_manifest_generated", "new_checkpoint_generated"):
            reason = f"{field} must remain false"
            if report[field] is True and reason not in report["failure_reasons"]:
                report["failure_reasons"].append(reason)
        if report["elapsed_seconds"] > TIME_BUDGET_SECONDS:
            reason = "elapsed time is invalid or exceeds the bounded budget"
            if reason not in report["failure_reasons"]:
                report["failure_reasons"].append(reason)
        report["execution_verdict"] = "PASS" if not report["failure_reasons"] else "FAIL"
        write_report_atomic(report, report_path)
    return report
