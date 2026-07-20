"""Contract tests for the one-frame GPU coordinate diagnostic."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation import DEFAULT_MAX_DISTANCE, DEFAULT_SCALE
from src.gpu_coordinate_diagnostic import (
    DIAGNOSTIC_NAME,
    DIAGNOSTIC_SCOPE,
    NMS_RADII_UM,
    REPORT_SCHEMA_VERSION,
    SAMPLE_ID,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_PROBE_SHA,
    SOURCE_SPLIT_SHA256,
    THRESHOLD_RULES,
    build_alignment_hypotheses,
    build_sweep,
    classify_outcome,
    evaluate_coordinate_diagnostic_report,
    gated_optimal_matches,
    run_gpu_coordinate_diagnostic,
    sha256_file,
    threshold_values,
    write_report_atomic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    REPO_ROOT / "kaggle_kernel_coordinate_diagnostic" / "gpu_coordinate_diagnostic_kernel.py"
)
METADATA_PATH = REPO_ROOT / "kaggle_kernel_coordinate_diagnostic" / "kernel-metadata.json"


def _volume_and_gt():
    volume = np.zeros((5, 8, 8), dtype=np.float32)
    volume[1, 2, 3] = 0.4
    volume[3, 6, 6] = 0.2
    gt = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    return volume, gt


def _passing_report(volume, gt):
    grid, best = build_sweep(volume, gt)
    alignment = build_alignment_hypotheses(
        np.asarray(best["predicted_coords_zyx"]), gt, tuple(volume.shape)
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnostic_name": DIAGNOSTIC_NAME,
        "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "deployed_sha": "a" * 40,
        "diagnostic_entrypoint_sha256": "b" * 64,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_checkpoint_training_code_sha": SOURCE_PROBE_SHA,
        "checkpoint_loaded_weights_only": True,
        "split_membership_sha256": SOURCE_SPLIT_SHA256,
        "import_origins": {"src.train": "/kaggle/input/st-act-src/src/train.py"},
        "import_origins_verified": True,
        "cuda_available": True,
        "device_type": "cuda",
        "gpu_name": "Tesla T4",
        "cuda_arch_compatible": True,
        "eval_mode": True,
        "no_grad": True,
        "forward_pass_count": 1,
        "inference_autocast_dtype": "float16",
        "optimizer_steps": 0,
        "backward_calls": 0,
        "adaptive_fallback_calls": 0,
        "deployment_manifest_generated": False,
        "new_checkpoint_generated": False,
        "model_state_sha256_before": "d" * 64,
        "model_state_sha256_after": "d" * 64,
        "frame": {
            "selection_rule": "first selected validation sample, earliest annotated frame",
            "sample_id": SAMPLE_ID,
            "dataset_index": 0,
            "window_t_idx": 0,
            "model_output_channel": 0,
            "frame_timepoint": 0,
            "frame_role": "frame_t",
            "input_shape": [1, 2, *volume.shape],
            "probability_shape": list(volume.shape),
            "coordinate_order": ["z", "y", "x"],
            "coordinate_unit": "voxel",
            "voxel_size_um": list(DEFAULT_SCALE),
            "match_gate_um": DEFAULT_MAX_DISTANCE,
        },
        "ground_truth": {
            "count": len(gt),
            "nodes": [{"node_id": 7, "zyx": row.tolist()} for row in gt],
            "finite_count": len(gt),
            "in_bounds_count": len(gt),
            "out_of_bounds_count": 0,
            "coords_sha256": hashlib.sha256(
                json.dumps(
                    [row.tolist() for row in gt],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
        "probability_artifact": {
            "file": "frame_probability_f32.npy",
            "sha256": "f" * 64,
            "dtype": "float32",
            "shape": list(volume.shape),
            "nbytes": volume.nbytes,
        },
        "probability": {
            "min": float(volume.min()),
            "max": float(volume.max()),
            "mean": float(volume.mean()),
            "std": float(volume.std()),
            "nonfinite_count": 0,
            "voxels_total": volume.size,
            "quantiles": {
                name: float(np.percentile(volume, percentile))
                for name, percentile in (
                    ("p50", 50),
                    ("p90", 90),
                    ("p99", 99),
                    ("p99_5", 99.5),
                    ("p99_9", 99.9),
                    ("p99_99", 99.99),
                )
            },
        },
        "grid": grid,
        "best_point": best,
        "alignment_hypotheses": alignment,
        "sweep_definition": {
            "threshold_operator": ">",
            "percentile_method": "numpy_default_linear",
            "threshold_rules": list(THRESHOLD_RULES),
            "nms_radii_um": list(NMS_RADII_UM),
            "expected_grid_points": 24,
        },
        "production_threshold_underconfident": int((volume > 0.5).sum()) == 0,
        "diagnostic_outcome": classify_outcome(volume, grid),
        "execution_verdict": "PASS",
        "failure_reasons": [],
    }


def test_complete_report_recomputes_from_frozen_volume():
    volume, gt = _volume_and_gt()
    assert evaluate_coordinate_diagnostic_report(_passing_report(volume, gt), volume) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", 1.0, "schema version"),
        ("source_checkpoint_sha256", "0" * 64, "checkpoint SHA-256"),
        ("source_checkpoint_training_code_sha", "0" * 40, "training-code SHA"),
        ("split_membership_sha256", "0" * 64, "active split identity"),
        ("checkpoint_loaded_weights_only", False, "weights_only"),
        ("import_origins_verified", False, "import origins"),
        ("device_type", "cpu", "CUDA"),
        ("cuda_arch_compatible", False, "architecture"),
        ("forward_pass_count", 2, "exactly one"),
        ("inference_autocast_dtype", "float32", "float16 autocast"),
        ("optimizer_steps", 1, "optimizer_steps"),
        ("adaptive_fallback_calls", 1, "adaptive_fallback_calls"),
        ("new_checkpoint_generated", True, "new_checkpoint_generated"),
        ("model_state_sha256_after", "e" * 64, "model state changed"),
    ],
)
def test_each_technical_violation_fails_closed(field, value, expected):
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    report[field] = value
    assert any(
        expected in reason for reason in evaluate_coordinate_diagnostic_report(report, volume)
    )


def test_coordinate_order_scale_and_channel_timepoint_fail_closed():
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    report["frame"]["coordinate_order"] = ["x", "y", "z"]
    report["frame"]["voxel_size_um"] = [4.0, 1.0, 1.0]
    report["frame"]["frame_timepoint"] = 1
    reasons = evaluate_coordinate_diagnostic_report(report, volume)
    assert any("coordinate convention" in reason for reason in reasons)
    assert any("physical scale" in reason for reason in reasons)
    assert any("timepoint" in reason for reason in reasons)


@pytest.mark.parametrize("coords", [[[np.nan, 1, 1]], [[9, 1, 1]]])
def test_nonfinite_or_out_of_bounds_gt_fails(coords):
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    report["ground_truth"]["nodes"] = [{"node_id": 7, "zyx": coords[0]}]
    assert any(
        "ground-truth coordinates" in reason
        for reason in evaluate_coordinate_diagnostic_report(report, volume)
    )


def test_empty_gt_fails_but_zero_scientific_signal_does_not():
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    report["ground_truth"]["nodes"] = []
    assert any(
        "empty" in reason for reason in evaluate_coordinate_diagnostic_report(report, volume)
    )

    flat = np.zeros_like(volume)
    grid, _best = build_sweep(flat, gt)
    assert classify_outcome(flat, grid) == "NO_EXTRACTABLE_PEAKS"


def test_thresholds_are_exactly_derived_from_frozen_volume():
    volume, _gt = _volume_and_gt()
    thresholds = threshold_values(volume)
    assert thresholds["fixed_0.5"] == 0.5
    assert thresholds["fixed_0.3"] == 0.3
    assert thresholds["q99"] == pytest.approx(np.percentile(volume, 99.0))
    assert thresholds["q99_5"] == pytest.approx(np.percentile(volume, 99.5))
    assert list(thresholds) == list(THRESHOLD_RULES)


def test_grid_is_exact_cartesian_product_and_tampering_fails():
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    assert len(report["grid"]) == len(THRESHOLD_RULES) * len(NMS_RADII_UM) == 24
    assert {(r["threshold_rule"], r["nms_radius_um"]) for r in report["grid"]} == {
        (rule, radius) for rule in THRESHOLD_RULES for radius in NMS_RADII_UM
    }

    report["grid"][0]["tp"] += 1
    assert any(
        "does not recompute" in reason
        for reason in evaluate_coordinate_diagnostic_report(report, volume)
    )


def test_missing_or_duplicate_grid_point_fails():
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    report["grid"][-1] = dict(report["grid"][0])
    assert any(
        "grid is incomplete" in reason
        for reason in evaluate_coordinate_diagnostic_report(report, volume)
    )


def test_anisotropic_physical_distance_and_exact_gate_boundary():
    predicted = np.asarray([[0.0, 0.0, 0.0]])
    gt_z4 = np.asarray([[4.0, 0.0, 0.0]])
    match = gated_optimal_matches(predicted, gt_z4)
    assert match[0][2] == pytest.approx(4 * 1.625)

    exact_gate = np.asarray([[DEFAULT_MAX_DISTANCE / DEFAULT_SCALE[0], 0.0, 0.0]])
    assert gated_optimal_matches(predicted, exact_gate)[0][2] == pytest.approx(DEFAULT_MAX_DISTANCE)


def test_matcher_maximizes_cardinality_before_total_distance():
    # P0 is 6um from both GTs. P1 can only reach GT0. A nearest-first greedy
    # choice could consume GT0 with P0 and return one match; the optimum is two.
    x_scale = DEFAULT_SCALE[2]
    predicted = np.asarray([[0.0, 0.0, 6.0 / x_scale], [0.0, 0.0, -1.0 / x_scale]])
    ground_truth = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 12.0 / x_scale]])
    matches = gated_optimal_matches(predicted, ground_truth)
    assert len(matches) == 2
    assert {(pred, gt) for pred, gt, _distance in matches} == {(0, 1), (1, 0)}


def test_alignment_hypotheses_expose_axis_permutation_signal():
    # Stored prediction is xyz=(6,0,0), while GT is zyx=(0,0,6). Reversing
    # the stored tuple must recover the exact match; the official order does not.
    predicted = np.asarray([[6.0, 0.0, 0.0]])
    gt = np.asarray([[0.0, 0.0, 6.0]])
    result = build_alignment_hypotheses(predicted, gt, (8, 8, 8))
    by_order = {row["predicted_axis_order"]: row for row in result["axis_permutations"]}
    assert by_order["zyx"]["matches_within_gate"] == 0
    assert by_order["xyz"]["matches_within_gate"] == 1
    assert by_order["xyz"]["distance_um"]["max"] == pytest.approx(0.0)


def test_underconfident_volume_with_aligned_lower_threshold_peak_is_signal():
    volume, gt = _volume_and_gt()
    grid, _best = build_sweep(volume, gt)
    assert int((volume > 0.5).sum()) == 0
    assert classify_outcome(volume, grid) == "UNDERCONFIDENT_WITH_ANY_GATED_MATCH"


def test_underconfident_volume_with_wrong_peak_has_no_spatial_signal():
    volume = np.zeros((5, 8, 8), dtype=np.float32)
    volume[4, 7, 7] = 0.4
    gt = np.asarray([[0.0, 0.0, 0.0]])
    grid, _best = build_sweep(volume, gt)
    assert classify_outcome(volume, grid) == "UNDERCONFIDENT_WITHOUT_GATED_MATCH"


def test_atomic_writer_rejects_nan_and_leaves_no_report(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_report_atomic({"bad": float("nan")}, path)
    assert not path.exists()


def test_probability_sidecar_bytes_are_verified(tmp_path):
    volume, gt = _volume_and_gt()
    report = _passing_report(volume, gt)
    path = tmp_path / "frame_probability_f32.npy"
    np.save(path, volume, allow_pickle=False)
    report["probability_artifact"]["sha256"] = sha256_file(path)
    assert evaluate_coordinate_diagnostic_report(report, volume, path) == []

    path.write_bytes(path.read_bytes()[:-8] + b"corrupt!")
    reasons = evaluate_coordinate_diagnostic_report(report, volume, path)
    assert any("byte SHA-256" in reason for reason in reasons)


def test_cpu_invocation_fails_before_data_or_checkpoint_access_and_writes_report(tmp_path):
    report = run_gpu_coordinate_diagnostic(
        data_dir=tmp_path / "missing-data",
        split_file=tmp_path / "missing-split.json",
        checkpoint_path=tmp_path / "missing-checkpoint.pt",
        output_dir=tmp_path,
        device=torch.device("cpu"),
        deployed_sha="a" * 40,
        import_origins={"src.train": "/verified/src/train.py"},
        diagnostic_entrypoint_sha256="b" * 64,
    )
    assert report["execution_verdict"] == "FAIL"
    assert report["exception_type"] == "RuntimeError"
    assert (tmp_path / "gpu_coordinate_diagnostic_report.json").exists()
    assert not (tmp_path / "frame_probability_f32.npy").exists()
    assert not list(tmp_path.glob("*.pt"))


def test_diagnostic_source_has_no_training_adaptive_or_checkpoint_write_calls():
    source = (REPO_ROOT / "src" / "gpu_coordinate_diagnostic.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "extract_inference_peaks" not in called_names
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert "save_checkpoint" not in called_attributes
    assert "write_checkpoint_manifest" not in called_attributes
    assert "train_epoch" not in called_attributes


def test_root_and_kaggle_diagnostic_modules_are_byte_identical():
    root = (REPO_ROOT / "src" / "gpu_coordinate_diagnostic.py").read_bytes()
    mirror = (
        REPO_ROOT / "kaggle_src_dataset" / "src" / "gpu_coordinate_diagnostic.py"
    ).read_bytes()
    assert hashlib.sha256(root).digest() == hashlib.sha256(mirror).digest()


def test_kernel_metadata_attaches_exact_probe_output_and_has_no_push_commands():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["kernel_sources"] == ["drbhatiasanjay/st-act-gpu-learning-probe"]
    assert metadata["dataset_sources"] == ["drbhatiasanjay/st-act-src"]
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    kernel = KERNEL_PATH.read_text(encoding="utf-8")
    assert "learning_probe_checkpoint.pt" in kernel
    assert "kaggle kernels push" not in kernel
    assert "kaggle datasets version" not in kernel


def test_kernel_structurally_verifies_nonempty_import_list_and_strict_sha():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    verify_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "verify_import_origins"
    ]
    sha_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "validate_git_sha_file"
    ]
    assert len(verify_calls) == 1
    assert isinstance(verify_calls[0].args[1], ast.Name)
    assert verify_calls[0].args[1].id == "modules_to_verify"
    assert len(sha_calls) == 1
    assert len(sha_calls[0].args) == 1
