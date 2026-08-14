"""Contracts for the bounded two-sample multiframe calibration diagnostic."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.evaluation import DEFAULT_MAX_DISTANCE, DEFAULT_SCALE
from src.gpu_coordinate_diagnostic import (
    NMS_RADII_UM,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_PROBE_SHA,
    SOURCE_SPLIT_SHA256,
    THRESHOLD_RULES,
)
from src.gpu_multiframe_calibration import (
    DIAGNOSTIC_NAME,
    DIAGNOSTIC_SCOPE,
    EXPECTED_FRAME_COUNT,
    FRAMES_PER_SAMPLE,
    SAMPLE_IDS,
    TIME_BUDGET_SECONDS,
    aggregate_sweeps,
    classify_scientific_result,
    evaluate_multiframe_calibration_report,
    rank_aggregate,
    run_gpu_multiframe_calibration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = REPO_ROOT / "kaggle_kernel_multiframe_calibration"
KERNEL_PATH = KERNEL_DIR / "gpu_multiframe_calibration_kernel.py"
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"


def _grid(tp=0, fp=0, fn=1):
    return [
        {
            "threshold_rule": rule,
            "nms_radius_um": radius,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        for rule in THRESHOLD_RULES
        for radius in NMS_RADII_UM
    ]


def _frames(*, matched_samples=()):
    frames = []
    for sample_id in SAMPLE_IDS:
        for frame_index in range(FRAMES_PER_SAMPLE):
            matched = sample_id in matched_samples
            frames.append(
                {
                    "artifact_key": f"{sample_id}_{frame_index}",
                    "sample_id": sample_id,
                    "dataset_index": frame_index,
                    "window_t_idx": frame_index,
                    "model_output_channel": 0,
                    "frame_timepoint": frame_index,
                    "frame_role": "frame_t",
                    "ground_truth_nodes": [{"node_id": frame_index, "zyx": [1.0, 2.0, 3.0]}],
                    "coordinate_order": ["z", "y", "x"],
                    "voxel_size_um": list(DEFAULT_SCALE),
                    "match_gate_um": DEFAULT_MAX_DISTANCE,
                    "probability_shape": [64, 256, 256],
                    "probability_sha256": "a" * 64,
                    "grid": _grid(tp=int(matched), fp=2 if matched else 0, fn=0 if matched else 1),
                    "best_point": {},
                }
            )
    return frames


def _passing_report():
    frames = _frames(matched_samples=SAMPLE_IDS)
    aggregate = aggregate_sweeps(frames)
    return {
        "schema_version": 1,
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
        "forward_pass_count": EXPECTED_FRAME_COUNT,
        "inference_autocast_dtype": "float16",
        "optimizer_steps": 0,
        "backward_calls": 0,
        "adaptive_fallback_calls": 0,
        "selected_sample_ids": list(SAMPLE_IDS),
        "successfully_opened_sample_ids": list(SAMPLE_IDS),
        "frames_per_sample": FRAMES_PER_SAMPLE,
        "peak_gpu_memory_bytes": 1,
        "model_state_sha256_before": "c" * 64,
        "model_state_sha256_after": "c" * 64,
        "frames": frames,
        "aggregate_grid": aggregate,
        "calibration_candidate": rank_aggregate(aggregate),
        "scientific_result": classify_scientific_result(aggregate),
        "production_configuration_selected": False,
        "checkpoint_promotion_performed": False,
        "deployment_manifest_generated": False,
        "new_checkpoint_generated": False,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "elapsed_seconds": 10.0,
        "execution_verdict": "PASS",
        "failure_reasons": [],
    }


def test_scope_is_exactly_two_samples_four_frames_and_24_points():
    assert len(SAMPLE_IDS) == 2
    assert FRAMES_PER_SAMPLE == 4
    assert EXPECTED_FRAME_COUNT == 8
    assert len(THRESHOLD_RULES) * len(NMS_RADII_UM) == 24
    assert TIME_BUDGET_SECONDS == 1200.0


def test_aggregate_micro_counts_and_per_sample_evidence():
    aggregate = aggregate_sweeps(_frames(matched_samples=SAMPLE_IDS))
    assert len(aggregate) == 24
    row = aggregate[0]
    assert (row["tp"], row["fp"], row["fn"]) == (8, 16, 0)
    assert row["precision"] == pytest.approx(1 / 3)
    assert row["recall"] == pytest.approx(1.0)
    assert [item["sample_id"] for item in row["per_sample"]] == list(SAMPLE_IDS)
    assert all(item["frames_with_match"] == FRAMES_PER_SAMPLE for item in row["per_sample"])


def test_aggregate_refuses_missing_frame_grid():
    frames = _frames()
    frames[0]["grid"].pop()
    with pytest.raises(RuntimeError, match="incomplete frame grid"):
        aggregate_sweeps(frames)


def test_rank_prefers_evidence_in_both_samples_before_single_sample_f1():
    both = {
        "threshold_rule": "fixed_0.4",
        "nms_radius_um": 5.0,
        "tp": 2,
        "fp": 98,
        "fn": 6,
        "predicted_count": 100,
        "ground_truth_count": 8,
        "precision": 0.02,
        "recall": 0.25,
        "f1": 0.037,
        "per_sample": [{"tp": 1}, {"tp": 1}],
    }
    one = dict(both)
    one.update(
        {
            "threshold_rule": "q99_9",
            "tp": 4,
            "fp": 0,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 0.667,
            "per_sample": [{"tp": 4}, {"tp": 0}],
        }
    )
    selected = rank_aggregate([one, both])
    assert selected["threshold_rule"] == "fixed_0.4"
    assert selected["recommendation_only"] is True
    assert selected["production_configuration_selected"] is False


@pytest.mark.parametrize(
    ("matched_samples", "expected"),
    [
        (SAMPLE_IDS, "GATED_MATCHES_IN_BOTH_SAMPLES"),
        ((SAMPLE_IDS[0],), "GATED_MATCHES_IN_ONE_SAMPLE_ONLY"),
    ],
)
def test_scientific_result_distinguishes_cross_sample_signal(matched_samples, expected):
    assert (
        classify_scientific_result(aggregate_sweeps(_frames(matched_samples=matched_samples)))
        == expected
    )


def test_structurally_complete_report_passes():
    assert evaluate_multiframe_calibration_report(_passing_report()) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", 1.0, "schema version"),
        ("source_checkpoint_sha256", "0" * 64, "checkpoint SHA-256"),
        ("source_checkpoint_training_code_sha", "0" * 40, "training-code SHA"),
        ("split_membership_sha256", "0" * 64, "split identity"),
        ("checkpoint_loaded_weights_only", False, "weights_only"),
        ("import_origins_verified", False, "import origins"),
        ("device_type", "cpu", "CUDA"),
        ("cuda_arch_compatible", False, "architecture"),
        ("forward_pass_count", 7, "forward-pass"),
        ("inference_autocast_dtype", "float32", "forward-pass"),
        ("optimizer_steps", 1, "optimizer_steps"),
        ("adaptive_fallback_calls", 1, "adaptive_fallback_calls"),
        ("successfully_opened_sample_ids", [SAMPLE_IDS[0]], "sample coverage"),
        ("peak_gpu_memory_bytes", 0, "GPU memory"),
        ("production_configuration_selected", True, "production configuration"),
        ("checkpoint_promotion_performed", True, "promote"),
        ("deployment_manifest_generated", True, "deployment_manifest_generated"),
        ("new_checkpoint_generated", True, "new_checkpoint_generated"),
        ("elapsed_seconds", TIME_BUDGET_SECONDS + 1, "elapsed time"),
    ],
)
def test_each_technical_violation_fails_closed(field, value, expected):
    report = _passing_report()
    report[field] = value
    assert any(expected in reason for reason in evaluate_multiframe_calibration_report(report))


def test_duplicate_frame_identity_and_wrong_coordinate_convention_fail():
    report = _passing_report()
    report["frames"][1]["frame_timepoint"] = report["frames"][0]["frame_timepoint"]
    report["frames"][1]["sample_id"] = report["frames"][0]["sample_id"]
    report["frames"][2]["coordinate_order"] = ["x", "y", "z"]
    reasons = evaluate_multiframe_calibration_report(report)
    assert any("duplicated" in reason for reason in reasons)
    assert any("coordinate convention" in reason for reason in reasons)


def test_candidate_cannot_claim_production_authority():
    report = _passing_report()
    report["calibration_candidate"]["production_configuration_selected"] = True
    assert any(
        "production authority" in reason
        for reason in evaluate_multiframe_calibration_report(report)
    )


def test_cpu_invocation_fails_before_data_access_and_writes_report(tmp_path):
    report = run_gpu_multiframe_calibration(
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
    assert (tmp_path / "gpu_multiframe_calibration_report.json").exists()
    assert not (tmp_path / "multiframe_probability_f32.npz").exists()


def test_preexisting_forbidden_artifacts_are_never_hidden(tmp_path):
    (tmp_path / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "forbidden.pt").write_bytes(b"not a checkpoint")
    report = run_gpu_multiframe_calibration(
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
    assert report["deployment_manifest_generated"] is True
    assert report["new_checkpoint_generated"] is True
    assert any("deployment_manifest_generated" in reason for reason in report["failure_reasons"])
    assert any("new_checkpoint_generated" in reason for reason in report["failure_reasons"])


def test_source_has_exactly_one_model_call_site_and_no_training_calls():
    source = (REPO_ROOT / "src" / "gpu_multiframe_calibration.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    model_calls = [
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "model"
    ]
    attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert len(model_calls) == 1
    assert "backward" not in attributes
    assert "step" not in attributes
    assert "save_checkpoint" not in attributes
    assert "write_checkpoint_manifest" not in attributes
    assert "train_epoch" not in attributes


def test_root_and_kaggle_modules_are_byte_identical():
    root = (REPO_ROOT / "src" / "gpu_multiframe_calibration.py").read_bytes()
    mirror = (
        REPO_ROOT / "kaggle_src_dataset" / "src" / "gpu_multiframe_calibration.py"
    ).read_bytes()
    assert hashlib.sha256(root).digest() == hashlib.sha256(mirror).digest()


def test_kernel_metadata_and_entrypoint_are_execution_only():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["kernel_sources"] == ["drbhatiasanjay/st-act-gpu-learning-probe"]
    assert metadata["dataset_sources"] == ["drbhatiasanjay/st-act-src"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    kernel = KERNEL_PATH.read_text(encoding="utf-8")
    assert "learning_probe_checkpoint.pt" in kernel
    assert "verify_import_origins" in kernel
    assert "validate_git_sha_file" in kernel
    assert "kaggle kernels push" not in kernel
    assert "kaggle datasets version" not in kernel


def test_exact_changed_scope_contract_names_are_present():
    assert DIAGNOSTIC_SCOPE.endswith("not_model_quality_or_production_selection")
    source = (REPO_ROOT / "src" / "gpu_multiframe_calibration.py").read_text(encoding="utf-8")
    for marker in (
        "production_configuration_selected",
        "checkpoint_promotion_performed",
        "adaptive_fallback_calls",
        "recommendation_only",
    ):
        assert marker in source
