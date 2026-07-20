"""One-frame, one-forward diagnostic for detection calibration and alignment.

This module is deliberately separate from training and submission.  It loads
the immutable bounded-probe checkpoint, freezes one probability volume, then
measures explicit threshold/NMS settings against ground-truth coordinates.
Scientific outcomes never promote a checkpoint; only broken execution or
evidence integrity makes the diagnostic fail.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.dataset import CompetitionDataset
from src.evaluation import DEFAULT_MAX_DISTANCE, DEFAULT_SCALE, load_geff_ground_truth
from src.model import UNet3D
from src.split_utils import get_split_identity
from src.train import extract_peaks_from_volume, pool_kernel_from_um

REPORT_SCHEMA_VERSION = 1
DIAGNOSTIC_NAME = "GPU-COORDINATE-DIAGNOSTIC-01"
DIAGNOSTIC_SCOPE = "one_frozen_frame_threshold_nms_not_model_quality"
SOURCE_PROBE_SHA = "893edcf5099ed1ac2531e25f0d07f8d1be9fb015"
SOURCE_CHECKPOINT_SHA256 = "53ece6b89e6e8735ffec94600d346763695b5386e07d9bc45e6fc068b1cd8c87"
SOURCE_SPLIT_SHA256 = "8277f93b10a3747b8f42e06a2dfbed7e96cbc8646920b1cc9ea56aea7d7b5732"
SAMPLE_ID = "44b6_0113de3b"
THRESHOLD_RULES = (
    "fixed_0.5",
    "fixed_0.4",
    "fixed_0.3",
    "fixed_0.1",
    "fixed_0.01",
    "q99",
    "q99_5",
    "q99_9",
)
NMS_RADII_UM = (2.5, 5.0, 7.5)
PROBABILITY_FILENAME = "frame_probability_f32.npy"
REPORT_FILENAME = "gpu_coordinate_diagnostic_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _probability_summary(probability: np.ndarray) -> dict[str, Any]:
    return {
        "min": float(probability.min()),
        "max": float(probability.max()),
        "mean": float(probability.mean()),
        "std": float(probability.std()),
        "nonfinite_count": int((~np.isfinite(probability)).sum()),
        "voxels_total": probability.size,
        "quantiles": {
            name: float(np.percentile(probability, percentile))
            for name, percentile in (
                ("p50", 50),
                ("p90", 90),
                ("p99", 99),
                ("p99_5", 99.5),
                ("p99_9", 99.9),
                ("p99_99", 99.99),
            )
        },
    }


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


def threshold_values(volume: np.ndarray) -> dict[str, float]:
    """Return the declared thresholds from one immutable float32 volume."""
    return {
        "fixed_0.5": 0.5,
        "fixed_0.4": 0.4,
        "fixed_0.3": 0.3,
        "fixed_0.1": 0.1,
        "fixed_0.01": 0.01,
        "q99": float(np.percentile(volume, 99.0)),
        "q99_5": float(np.percentile(volume, 99.5)),
        "q99_9": float(np.percentile(volume, 99.9)),
    }


def _physical_distances(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
) -> np.ndarray:
    if len(predicted) == 0 or len(ground_truth) == 0:
        return np.empty((len(predicted), len(ground_truth)), dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    delta = (predicted[:, None, :] - ground_truth[None, :, :]) * scale_array
    return np.linalg.norm(delta, axis=2)


def gated_optimal_matches(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    gate_um: float = DEFAULT_MAX_DISTANCE,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, then minimum-distance, one-to-one matching."""
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 3)
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 3)
    if len(predicted) == 0 or len(ground_truth) == 0:
        return []

    distances = _physical_distances(predicted, ground_truth, scale)
    n_pred, n_gt = distances.shape
    # One row per GT and one column per prediction, plus a private unmatched
    # column per GT.  This is O(n_gt * n_pred), not O((n_gt+n_pred)^2): low
    # threshold sweep points may legitimately expose tens of thousands of
    # local maxima, while the annotated frame remains sparse.
    unmatched_cost = (n_gt + 1) * gate_um + 1.0
    forbidden_cost = unmatched_cost * 4.0
    cost = np.full((n_gt, n_pred + n_gt), forbidden_cost, dtype=np.float64)
    cost[:, :n_pred] = np.where(distances.T <= gate_um, distances.T, forbidden_cost)
    for gt_index in range(n_gt):
        cost[gt_index, n_pred + gt_index] = unmatched_cost

    rows, columns = linear_sum_assignment(cost)
    matches = []
    for gt_index, column in zip(rows, columns, strict=True):
        if column < n_pred and distances[column, gt_index] <= gate_um:
            matches.append((int(column), int(gt_index), float(distances[column, gt_index])))
    return sorted(matches)


def _distance_summary(distances: list[float]) -> dict[str, float | None]:
    if not distances:
        return {"min": None, "median": None, "p95": None, "max": None}
    values = np.asarray(distances, dtype=np.float64)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def build_sweep(
    volume: np.ndarray, gt_coords: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = threshold_values(volume)
    rows: list[dict[str, Any]] = []
    all_peaks: dict[tuple[str, float], list[list[float]]] = {}

    for rule in THRESHOLD_RULES:
        threshold = thresholds[rule]
        for radius in NMS_RADII_UM:
            peaks = extract_peaks_from_volume(
                volume,
                threshold=threshold,
                voxel_size=DEFAULT_SCALE,
                nms_radius_um=radius,
            )
            predicted = np.asarray(peaks, dtype=np.float64).reshape(-1, 3)
            matches = gated_optimal_matches(predicted, gt_coords)
            tp = len(matches)
            fp = len(predicted) - tp
            fn = len(gt_coords) - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            peak_list = predicted.tolist()
            all_peaks[(rule, radius)] = peak_list
            rows.append(
                {
                    "threshold_rule": rule,
                    "threshold_value": threshold,
                    "nms_radius_um": radius,
                    "pool_kernel_zyx": list(pool_kernel_from_um(radius, DEFAULT_SCALE)),
                    "voxels_above_threshold": int((volume > threshold).sum()),
                    "positive_fraction": float((volume > threshold).mean()),
                    "predicted_count": len(predicted),
                    "predicted_coords_sha256": _sha256_json(peak_list),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "matched_distance_um": _distance_summary([item[2] for item in matches]),
                }
            )

    rule_order = {rule: index for index, rule in enumerate(THRESHOLD_RULES)}

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        median = row["matched_distance_um"]["median"]
        return (
            -row["f1"],
            -row["tp"],
            math.inf if median is None else median,
            abs(row["predicted_count"] - len(gt_coords)),
            abs(row["nms_radius_um"] - 5.0),
            rule_order[row["threshold_rule"]],
        )

    best_row = min(rows, key=rank)
    best_peaks = np.asarray(
        all_peaks[(best_row["threshold_rule"], best_row["nms_radius_um"])], dtype=np.float64
    ).reshape(-1, 3)
    best_matches = gated_optimal_matches(best_peaks, gt_coords)
    best = dict(best_row)
    best["predicted_coords_zyx"] = best_peaks.tolist()
    best["matches"] = [
        {
            "pred_index": pred_index,
            "gt_index": gt_index,
            "pred_zyx": best_peaks[pred_index].tolist(),
            "gt_zyx": gt_coords[gt_index].tolist(),
            "delta_zyx": (best_peaks[pred_index] - gt_coords[gt_index]).tolist(),
            "distance_um": distance,
        }
        for pred_index, gt_index, distance in best_matches
    ]
    return rows, best


def build_alignment_hypotheses(
    predicted_zyx: np.ndarray, ground_truth_zyx: np.ndarray, volume_shape: tuple[int, int, int]
) -> dict[str, Any]:
    """Conditionally compare transforms for one identity-ranked peak set.

    This is a sensitivity table, not an alignment adjudication: the threshold/NMS
    point was selected under the production coordinate convention.  The report
    states that limitation explicitly so a favorable alternative transform cannot
    be mistaken for an independently optimized result.
    """
    predicted = np.asarray(predicted_zyx, dtype=np.float64).reshape(-1, 3)
    ground_truth = np.asarray(ground_truth_zyx, dtype=np.float64).reshape(-1, 3)
    axis_names = ("z", "y", "x")
    permutations = []
    shape = np.asarray(volume_shape, dtype=np.float64)
    for order in itertools.permutations(range(3)):
        transformed = predicted[:, order] if len(predicted) else predicted.copy()
        matches = gated_optimal_matches(transformed, ground_truth)
        in_bounds = (
            int(((transformed >= 0) & (transformed < shape)).all(axis=1).sum())
            if len(transformed)
            else 0
        )
        permutations.append(
            {
                "predicted_axis_order": "".join(axis_names[index] for index in order),
                "matches_within_gate": len(matches),
                "in_bounds_count": in_bounds,
                "distance_um": _distance_summary([match[2] for match in matches]),
            }
        )

    scale_rows = []
    for name, scale in (
        ("physical_microns", DEFAULT_SCALE),
        ("anisotropy_ratio", (4.0, 1.0, 1.0)),
        ("unit_voxels", (1.0, 1.0, 1.0)),
    ):
        matches = gated_optimal_matches(predicted, ground_truth, scale=scale)
        scale_rows.append(
            {
                "scale_name": name,
                "scale_zyx": list(scale),
                "matches_within_gate": len(matches),
                "distance": _distance_summary([match[2] for match in matches]),
            }
        )

    nearest = []
    distances = _physical_distances(predicted, ground_truth)
    if len(predicted):
        for gt_index in range(len(ground_truth)):
            pred_index = int(np.argmin(distances[:, gt_index]))
            nearest.append(
                {
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "pred_zyx": predicted[pred_index].tolist(),
                    "gt_zyx": ground_truth[gt_index].tolist(),
                    "delta_zyx": (predicted[pred_index] - ground_truth[gt_index]).tolist(),
                    "distance_um": float(distances[pred_index, gt_index]),
                }
            )
    return {
        "interpretation": "conditional_sensitivity_at_identity_ranked_best_point",
        "can_adjudicate_alignment": False,
        "axis_permutations": permutations,
        "scale_variants": scale_rows,
        "ungated_nearest_prediction_per_gt": nearest,
    }


def classify_outcome(volume: np.ndarray, grid: list[dict[str, Any]]) -> str:
    production_rows = [row for row in grid if row["threshold_rule"] == "fixed_0.5"]
    if any(row["tp"] > 0 for row in production_rows):
        return "PRODUCTION_THRESHOLD_HAS_SPATIAL_SIGNAL"
    nonproduction = [row for row in grid if row["threshold_rule"] != "fixed_0.5"]
    underconfident = int((volume > 0.5).sum()) == 0
    if underconfident and any(row["tp"] > 0 for row in nonproduction):
        return "UNDERCONFIDENT_WITH_ANY_GATED_MATCH"
    if underconfident and any(row["predicted_count"] > 0 for row in nonproduction):
        return "UNDERCONFIDENT_WITHOUT_GATED_MATCH"
    if all(row["predicted_count"] == 0 for row in grid):
        return "NO_EXTRACTABLE_PEAKS"
    return "INCONCLUSIVE"


def evaluate_coordinate_diagnostic_report(
    report: dict[str, Any],
    probability_volume: np.ndarray | None = None,
    probability_path: Path | None = None,
) -> list[str]:
    """Return fail-closed technical violations; scientific zero signal is valid."""
    reasons: list[str] = []
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1:
        reasons.append("report schema version is missing or unsupported")
    if (
        report.get("diagnostic_name") != DIAGNOSTIC_NAME
        or report.get("diagnostic_scope") != DIAGNOSTIC_SCOPE
    ):
        reasons.append("diagnostic identity or scope is incorrect")
    if not _is_lower_hex(report.get("deployed_sha"), 40):
        reasons.append("deployed diagnostic-code SHA is malformed")
    if not _is_lower_hex(report.get("diagnostic_entrypoint_sha256"), 64):
        reasons.append("diagnostic entrypoint SHA-256 is malformed")
    if report.get("split_membership_sha256") != SOURCE_SPLIT_SHA256:
        reasons.append("active split identity does not match the immutable probe split")
    if report.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        reasons.append("source checkpoint SHA-256 does not match the immutable probe artifact")
    if report.get("source_checkpoint_training_code_sha") != SOURCE_PROBE_SHA:
        reasons.append("source checkpoint training-code SHA is incorrect")
    if report.get("checkpoint_loaded_weights_only") is not True:
        reasons.append("source checkpoint was not loaded with weights_only=True")
    if report.get("import_origins_verified") is not True or not report.get("import_origins"):
        reasons.append("production import origins were not verified")
    if report.get("device_type") != "cuda" or report.get("cuda_available") is not True:
        reasons.append("CUDA was not selected")
    if report.get("cuda_arch_compatible") is not True:
        reasons.append("CUDA architecture is not compatible")
    if (
        report.get("forward_pass_count") != 1
        or report.get("eval_mode") is not True
        or report.get("no_grad") is not True
    ):
        reasons.append("diagnostic did not use exactly one eval/no-grad forward pass")
    if report.get("inference_autocast_dtype") != "float16":
        reasons.append("diagnostic did not reproduce production float16 autocast")
    for field in ("optimizer_steps", "backward_calls", "adaptive_fallback_calls"):
        if report.get(field) != 0:
            reasons.append(f"{field} must be exactly zero")
    for field in ("deployment_manifest_generated", "new_checkpoint_generated"):
        if report.get(field) is not False:
            reasons.append(f"{field} must be false")
    if report.get("model_state_sha256_before") != report.get("model_state_sha256_after"):
        reasons.append("model state changed during the diagnostic")
    if not _is_lower_hex(report.get("model_state_sha256_before"), 64):
        reasons.append("model state SHA-256 is malformed")

    frame = report.get("frame", {})
    if frame.get("sample_id") != SAMPLE_ID:
        reasons.append("diagnostic sample identity is incorrect")
    if frame.get("coordinate_order") != ["z", "y", "x"] or frame.get("coordinate_unit") != "voxel":
        reasons.append("coordinate convention is incorrect")
    if (
        frame.get("voxel_size_um") != list(DEFAULT_SCALE)
        or frame.get("match_gate_um") != DEFAULT_MAX_DISTANCE
    ):
        reasons.append("physical scale or match gate is incorrect")
    if frame.get("model_output_channel") not in (0, 1):
        reasons.append("model output channel is invalid")
    if frame.get("frame_timepoint") != frame.get("window_t_idx", -99) + frame.get(
        "model_output_channel", -99
    ):
        reasons.append("frame timepoint does not match the selected window/channel")
    if frame.get("selection_rule") != "first selected validation sample, earliest annotated frame":
        reasons.append("frame selection rule is incorrect")
    expected_role = "frame_t" if frame.get("model_output_channel") == 0 else "frame_t1"
    if frame.get("frame_role") != expected_role:
        reasons.append("frame role does not match the selected channel")

    gt = report.get("ground_truth", {})
    nodes = gt.get("nodes")
    gt_valid = False
    if not isinstance(nodes, list) or not nodes:
        reasons.append("ground-truth coordinate list is empty or missing")
    else:
        gt_array = np.asarray([node.get("zyx") for node in nodes], dtype=np.float64)
        shape = np.asarray(frame.get("probability_shape", []), dtype=np.int64)
        if gt_array.shape != (len(nodes), 3) or not np.isfinite(gt_array).all():
            reasons.append("ground-truth coordinates are malformed or nonfinite")
        elif shape.shape != (3,) or ((gt_array < 0) | (gt_array >= shape)).any():
            reasons.append("ground-truth coordinates are outside the probability volume")
        else:
            gt_valid = True
        if gt.get("count") != len(nodes):
            reasons.append("ground-truth count does not match the coordinate list")
        if gt.get("finite_count") != len(nodes) or gt.get("in_bounds_count") != len(nodes):
            reasons.append("ground-truth finite/in-bounds counters are incorrect")
        if gt.get("out_of_bounds_count") != 0:
            reasons.append("ground-truth out-of-bounds count is nonzero")
        if gt_valid and gt.get("coords_sha256") != _sha256_json(
            [node.get("zyx") for node in nodes]
        ):
            reasons.append("ground-truth coordinate hash is incorrect")
        node_ids = [node.get("node_id") for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            reasons.append("ground-truth node IDs are duplicated")

    grid = report.get("grid")
    expected_pairs = {(rule, radius) for rule in THRESHOLD_RULES for radius in NMS_RADII_UM}
    actual_pairs = (
        {(row.get("threshold_rule"), row.get("nms_radius_um")) for row in grid}
        if isinstance(grid, list)
        else set()
    )
    if (
        not isinstance(grid, list)
        or len(grid) != len(expected_pairs)
        or actual_pairs != expected_pairs
    ):
        reasons.append("threshold/NMS grid is incomplete or duplicated")
    expected_sweep = {
        "threshold_operator": ">",
        "percentile_method": "numpy_default_linear",
        "threshold_rules": list(THRESHOLD_RULES),
        "nms_radii_um": list(NMS_RADII_UM),
        "expected_grid_points": len(expected_pairs),
    }
    if report.get("sweep_definition") != expected_sweep:
        reasons.append("threshold/NMS sweep definition is incorrect")

    if probability_volume is not None:
        volume = np.asarray(probability_volume)
        if volume.dtype != np.float32 or list(volume.shape) != frame.get("probability_shape"):
            reasons.append("frozen probability volume dtype or shape is incorrect")
        elif not np.isfinite(volume).all() or volume.min() < 0.0 or volume.max() > 1.0:
            reasons.append("frozen probability volume is nonfinite or outside [0,1]")
        elif gt_valid and isinstance(grid, list):
            artifact = report.get("probability_artifact", {})
            if artifact.get("file") != PROBABILITY_FILENAME:
                reasons.append("probability artifact filename is incorrect")
            if artifact.get("dtype") != "float32" or artifact.get("shape") != list(volume.shape):
                reasons.append("probability artifact metadata is incorrect")
            if artifact.get("nbytes") != volume.nbytes or not _is_lower_hex(
                artifact.get("sha256"), 64
            ):
                reasons.append("probability artifact size or SHA-256 is incorrect")
            if _sha256_json(report.get("probability")) != _sha256_json(
                _probability_summary(volume)
            ):
                reasons.append("probability statistics do not recompute from the frozen volume")
            gt_array = np.asarray([node["zyx"] for node in nodes], dtype=np.float64)
            recomputed_grid, recomputed_best = build_sweep(volume, gt_array)
            if _sha256_json(grid) != _sha256_json(recomputed_grid):
                reasons.append("threshold/NMS grid does not recompute from the frozen volume")
            if _sha256_json(report.get("best_point")) != _sha256_json(recomputed_best):
                reasons.append("best point does not recompute from the frozen volume")
            recomputed_alignment = build_alignment_hypotheses(
                np.asarray(recomputed_best["predicted_coords_zyx"], dtype=np.float64),
                gt_array,
                tuple(volume.shape),
            )
            if _sha256_json(report.get("alignment_hypotheses")) != _sha256_json(
                recomputed_alignment
            ):
                reasons.append("alignment hypotheses do not recompute from the frozen volume")
            if report.get("diagnostic_outcome") != classify_outcome(volume, recomputed_grid):
                reasons.append("diagnostic outcome does not recompute from the frozen volume")
            if report.get("production_threshold_underconfident") is not (
                int((volume > 0.5).sum()) == 0
            ):
                reasons.append("production under-confidence flag is incorrect")
    if probability_path is not None:
        artifact = report.get("probability_artifact", {})
        try:
            if sha256_file(probability_path) != artifact.get("sha256"):
                reasons.append("probability artifact byte SHA-256 does not match the report")
            loaded = np.load(probability_path, allow_pickle=False)
            if probability_volume is not None and not np.array_equal(loaded, probability_volume):
                reasons.append("probability artifact bytes do not match the validated volume")
        except Exception as error:
            reasons.append(f"probability artifact could not be loaded safely: {error}")
    return reasons


def _select_earliest_annotated_frame(
    data_dir: Path, dataset: CompetitionDataset
) -> tuple[int, int, int, list[dict[str, Any]]]:
    graph, _metadata = load_geff_ground_truth(str(data_dir / f"{SAMPLE_ID}.geff"))
    attrs = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    if attrs.height == 0:
        raise RuntimeError("selected sample has no ground-truth nodes")
    earliest = int(attrs["t"].min())
    pair_to_index = {pair: index for index, pair in enumerate(dataset.pairs)}
    if (SAMPLE_ID, earliest) in pair_to_index:
        window_t_idx, channel = earliest, 0
    elif (SAMPLE_ID, earliest - 1) in pair_to_index:
        window_t_idx, channel = earliest - 1, 1
    else:
        raise RuntimeError(f"no validation window contains earliest annotated frame t={earliest}")
    frame_rows = attrs.filter(attrs["t"] == earliest).select(["node_id", "z", "y", "x"]).to_dicts()
    nodes = [
        {
            "node_id": int(row["node_id"]),
            "zyx": [float(row["z"]), float(row["y"]), float(row["x"])],
        }
        for row in frame_rows
    ]
    return pair_to_index[(SAMPLE_ID, window_t_idx)], window_t_idx, channel, nodes


def run_gpu_coordinate_diagnostic(
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
    """Run the one-frame diagnostic and always write a technical verdict."""
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
        "deployment_manifest_generated": False,
        "new_checkpoint_generated": False,
        "execution_verdict": "FAIL",
        "failure_reasons": [],
    }
    contract_evaluated = False
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("coordinate diagnostic requires CUDA")
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
        active_split_identity = get_split_identity(split_file)
        if active_split_identity != SOURCE_SPLIT_SHA256:
            raise RuntimeError("active split identity does not match the immutable probe split")
        if checkpoint.get("split_membership_sha256") != active_split_identity:
            raise RuntimeError("checkpoint and active split identities do not match")
        report["split_membership_sha256"] = active_split_identity
        major, minor = torch.cuda.get_device_capability(device)
        report["cuda_arch_compatible"] = f"sm_{major}{minor}" in torch.cuda.get_arch_list()
        if not report["cuda_arch_compatible"]:
            raise RuntimeError("CUDA architecture is incompatible with this PyTorch build")

        dataset = CompetitionDataset(
            data_dir=data_dir,
            split_file=split_file,
            split_type="validation",
            normalize=True,
            filter_unannotated_pairs=False,
            strict_sample_coverage=True,
            sample_id_allowlist=[SAMPLE_ID],
        )
        dataset_index, window_t_idx, channel, gt_nodes = _select_earliest_annotated_frame(
            data_dir, dataset
        )
        item = dataset[dataset_index]
        if item.get("sample_id") != SAMPLE_ID or int(item.get("t_idx", -1)) != window_t_idx:
            raise RuntimeError(
                "dataset returned a different sample/window than the selected diagnostic frame"
            )
        if tuple(item["frame_t"].shape) != (1, 64, 256, 256) or tuple(item["frame_t1"].shape) != (
            1,
            64,
            256,
            256,
        ):
            raise RuntimeError("diagnostic frame pair has an unexpected tensor shape")
        frame_t = item["frame_t"].unsqueeze(0).to(device)
        frame_t1 = item["frame_t1"].unsqueeze(0).to(device)

        model = UNet3D(in_channels=2, channels=(32, 64, 128)).to(device)
        model.load_state_dict(checkpoint["unet3d_state_dict"], strict=True)
        state_before = _state_dict_sha256(model.state_dict())
        model.eval()
        report["eval_mode"] = not model.training
        input_pair = torch.cat([frame_t, frame_t1], dim=1)
        with torch.no_grad():
            report["no_grad"] = not torch.is_grad_enabled()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                report["inference_autocast_dtype"] = "float16"
                logits, features = model(input_pair)
                report["forward_pass_count"] += 1
                probability = (
                    torch.sigmoid(logits.float())[0, channel].cpu().numpy().astype(np.float32)
                )
        input_shape = list(input_pair.shape)
        del features, logits, input_pair, frame_t, frame_t1
        state_after = _state_dict_sha256(model.state_dict())
        np.save(probability_path, probability, allow_pickle=False)
        probability = np.load(probability_path, allow_pickle=False)

        gt_coords = np.asarray([node["zyx"] for node in gt_nodes], dtype=np.float64)
        grid, best = build_sweep(probability, gt_coords)
        alignment_hypotheses = build_alignment_hypotheses(
            np.asarray(best["predicted_coords_zyx"], dtype=np.float64),
            gt_coords,
            tuple(probability.shape),
        )
        shape = list(probability.shape)
        in_bounds = ((gt_coords >= 0) & (gt_coords < np.asarray(shape))).all(axis=1)
        report.update(
            {
                "model_state_sha256_before": state_before,
                "model_state_sha256_after": state_after,
                "frame": {
                    "selection_rule": "first selected validation sample, earliest annotated frame",
                    "sample_id": SAMPLE_ID,
                    "dataset_index": dataset_index,
                    "window_t_idx": window_t_idx,
                    "model_output_channel": channel,
                    "frame_timepoint": window_t_idx + channel,
                    "frame_role": "frame_t" if channel == 0 else "frame_t1",
                    "input_shape": input_shape,
                    "probability_shape": shape,
                    "coordinate_order": ["z", "y", "x"],
                    "coordinate_unit": "voxel",
                    "voxel_size_um": list(DEFAULT_SCALE),
                    "match_gate_um": DEFAULT_MAX_DISTANCE,
                },
                "probability_artifact": {
                    "file": PROBABILITY_FILENAME,
                    "sha256": sha256_file(probability_path),
                    "dtype": str(probability.dtype),
                    "shape": shape,
                    "nbytes": probability.nbytes,
                },
                "probability": _probability_summary(probability),
                "ground_truth": {
                    "count": len(gt_nodes),
                    "nodes": gt_nodes,
                    "coords_sha256": _sha256_json([node["zyx"] for node in gt_nodes]),
                    "finite_count": int(np.isfinite(gt_coords).all(axis=1).sum()),
                    "in_bounds_count": int(in_bounds.sum()),
                    "out_of_bounds_count": int((~in_bounds).sum()),
                },
                "sweep_definition": {
                    "threshold_operator": ">",
                    "percentile_method": "numpy_default_linear",
                    "threshold_rules": list(THRESHOLD_RULES),
                    "nms_radii_um": list(NMS_RADII_UM),
                    "expected_grid_points": len(THRESHOLD_RULES) * len(NMS_RADII_UM),
                },
                "grid": grid,
                "best_point": best,
                "alignment_hypotheses": alignment_hypotheses,
                "production_threshold_underconfident": int((probability > 0.5).sum()) == 0,
                "diagnostic_outcome": classify_outcome(probability, grid),
            }
        )
        report["deployment_manifest_generated"] = manifest_path.exists()
        report["new_checkpoint_generated"] = any(output_dir.glob("*.pt"))
        reasons = evaluate_coordinate_diagnostic_report(report, probability, probability_path)
        contract_evaluated = True
        report["failure_reasons"] = reasons
    except Exception as error:
        report["exception_type"] = type(error).__name__
        report["exception_message"] = str(error)
        report["failure_reasons"] = [f"{type(error).__name__}: {error}"]
    finally:
        report["deployment_manifest_generated"] = manifest_path.exists()
        report["new_checkpoint_generated"] = any(output_dir.glob("*.pt"))
        for field in ("deployment_manifest_generated", "new_checkpoint_generated"):
            reason = f"{field} must remain false"
            if report[field] is True and reason not in report["failure_reasons"]:
                report["failure_reasons"].append(reason)
        if not contract_evaluated and not report["failure_reasons"]:
            report["failure_reasons"] = evaluate_coordinate_diagnostic_report(
                report,
                probability if "probability" in locals() else None,
                probability_path if probability_path.exists() else None,
            )
        report["execution_verdict"] = "PASS" if not report["failure_reasons"] else "FAIL"
        write_report_atomic(report, report_path)
    return report
