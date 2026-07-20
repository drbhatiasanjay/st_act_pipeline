"""Fail-closed contract tests for the bounded GPU learning probe."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.gpu_learning_probe import (
    evaluate_learning_probe_report,
    run_gpu_learning_probe,
    write_report_atomic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "kaggle_kernel_learning_probe" / "gpu_learning_probe_kernel.py"
METADATA_PATH = REPO_ROOT / "kaggle_kernel_learning_probe" / "kernel-metadata.json"


def _passing_report():
    return {
        "schema_version": 1,
        "probe_name": "GPU-LEARNING-PROBE-01",
        "probe_scope": "bounded_learning_signal_not_model_quality",
        "deployed_sha": "a" * 40,
        "probe_entrypoint_sha256": "b" * 64,
        "split_membership_sha256": "c" * 64,
        "import_origins": {"src.train": "/kaggle/input/st-act-src/src/train.py"},
        "import_origins_verified": True,
        "cuda_available": True,
        "cuda_arch_compatible": True,
        "device_type": "cuda",
        "gpu_name": "Tesla T4",
        "requested_train_batches": 512,
        "completed_train_batches": 512,
        "train_dataset_pair_count": 2000,
        "average_train_loss": 1.25,
        "last_unet_gradient_norm": 0.5,
        "last_transformer_gradient_norm": 0.25,
        "expected_train_sample_ids": ["train-a", "train-b"],
        "successfully_opened_train_sample_ids": ["train-a", "train-b"],
        "requested_validation_samples": 2,
        "selected_validation_sample_ids": ["val-a", "val-b"],
        "successfully_opened_validation_sample_ids": ["val-a", "val-b"],
        "source_validation_fold_sample_count": 3,
        "full_fold_validation_performed": False,
        "validation_metrics": {
            "evaluation_completed_successfully": True,
            "validation_samples_evaluated": 2,
            "validation_samples_total": 2,
            "validation_sample_cap": 2,
            "validation_is_full_fold": True,
            "predicted_nodes_total": 10,
            "predicted_edges_total": 4,
            "is_structural_zero": False,
            "edge_jaccard": 0.1,
            "adjusted_edge_jaccard": 0.08,
            "division_jaccard": 0.0,
            "score": 0.08,
        },
        "train_fallback_counts": {"edge_target_generation_failure": 0},
        "post_validation_fallback_counts": {
            "edge_target_generation_failure": 0,
            "evaluation_failure": 0,
        },
        "train_biological_counts": {
            "edge_supervised_batches_total": 500,
            "edge_supervised_batches_with_nonzero_transformer_grad": 450,
        },
        "elapsed_seconds": 900.0,
        "training_elapsed_seconds": 700.0,
        "time_budget_seconds": 3600.0,
        "peak_gpu_memory_allocated_bytes": 100,
        "peak_gpu_memory_reserved_bytes": 200,
        "deployment_manifest_generated": False,
        "probe_checkpoint_saved": True,
        "probe_checkpoint_sha256": "d" * 64,
        "learning_signal_observed": True,
    }


def test_complete_report_passes():
    assert evaluate_learning_probe_report(_passing_report()) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", 1.0, "schema version"),
        ("deployed_sha", "unknown", "deployed_sha"),
        ("probe_entrypoint_sha256", "bad", "entrypoint SHA-256"),
        ("split_membership_sha256", "bad", "split membership"),
        ("import_origins_verified", False, "import origins"),
        ("cuda_available", False, "CUDA device"),
        ("cuda_arch_compatible", False, "compute capability"),
        ("completed_train_batches", 511, "completed training batches"),
        ("average_train_loss", float("nan"), "training loss"),
        ("last_unet_gradient_norm", 0.0, "UNet gradient"),
        ("last_transformer_gradient_norm", float("inf"), "Transformer gradient"),
        ("elapsed_seconds", 3601.0, "wall-clock budget"),
        ("deployment_manifest_generated", True, "deployment manifest"),
        ("peak_gpu_memory_allocated_bytes", 0, "allocated GPU memory"),
        ("probe_checkpoint_saved", False, "checkpoint was not saved"),
    ],
)
def test_each_primary_failure_is_fail_closed(field, value, expected):
    report = _passing_report()
    report[field] = value
    assert any(expected in reason for reason in evaluate_learning_probe_report(report))


def test_incomplete_train_sample_coverage_fails():
    report = _passing_report()
    report["successfully_opened_train_sample_ids"] = ["train-a"]
    assert any("training sample coverage" in reason for reason in evaluate_learning_probe_report(report))


def test_incomplete_selected_validation_coverage_fails():
    report = _passing_report()
    report["successfully_opened_validation_sample_ids"] = ["val-a"]
    assert any("validation sample identity" in reason for reason in evaluate_learning_probe_report(report))


def test_opening_any_unselected_validation_sample_fails_boundedness_contract():
    report = _passing_report()
    report["successfully_opened_validation_sample_ids"].append("val-c")
    assert any("validation sample identity" in reason for reason in evaluate_learning_probe_report(report))


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("evaluation_completed_successfully", False, "did not complete"),
        ("validation_samples_evaluated", 1, "exactly the requested"),
        ("validation_sample_cap", None, "sample cap"),
        ("validation_samples_total", 1, "requested sample subset"),
        ("predicted_nodes_total", 0, "no predicted nodes"),
        ("validation_is_full_fold", False, "processed completely"),
        ("is_structural_zero", True, "structurally zero"),
        ("score", float("nan"), "metric score"),
    ],
)
def test_validation_contract_failures_are_fail_closed(field, value, expected):
    report = _passing_report()
    report["validation_metrics"][field] = value
    assert any(expected in reason for reason in evaluate_learning_probe_report(report))


@pytest.mark.parametrize("counter_field", ["train_fallback_counts", "post_validation_fallback_counts"])
def test_any_nonzero_technical_fallback_fails(counter_field):
    report = _passing_report()
    report[counter_field]["technical_failure"] = 1
    assert any(counter_field in reason for reason in evaluate_learning_probe_report(report))


@pytest.mark.parametrize(
    "counter",
    ["edge_supervised_batches_total", "edge_supervised_batches_with_nonzero_transformer_grad"],
)
def test_missing_edge_training_signal_fails(counter):
    report = _passing_report()
    report["train_biological_counts"][counter] = 0
    assert evaluate_learning_probe_report(report)


def test_zero_quality_score_is_reportable_without_becoming_a_technical_failure():
    report = _passing_report()
    report["validation_metrics"]["edge_jaccard"] = 0.0
    report["validation_metrics"]["adjusted_edge_jaccard"] = 0.0
    report["validation_metrics"]["score"] = 0.0
    report["learning_signal_observed"] = False
    assert evaluate_learning_probe_report(report) == []


def test_pre_checkpoint_evaluation_checks_execution_predicates_only():
    report = _passing_report()
    report["probe_checkpoint_saved"] = False
    report.pop("probe_checkpoint_sha256")
    assert evaluate_learning_probe_report(report, require_checkpoint=False) == []


def test_atomic_report_is_deterministic_json(tmp_path):
    path = tmp_path / "report.json"
    report = {"z": 1, "a": [2, 3]}
    write_report_atomic(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')
    assert not path.with_suffix(".json.tmp").exists()


def test_cpu_invocation_fails_closed_but_still_writes_report(tmp_path):
    report = run_gpu_learning_probe(
        data_dir=tmp_path / "unused",
        split_file=tmp_path / "unused.json",
        output_dir=tmp_path,
        device=torch.device("cpu"),
        deployed_sha="a" * 40,
        import_origins={"src.train": "/verified/src/train.py"},
        probe_entrypoint_sha256="b" * 64,
        cuda_arch_compatible=False,
    )
    assert report["verdict"] == "FAIL"
    assert report["exception_type"] == "RuntimeError"
    assert (tmp_path / "gpu_learning_probe_report.json").exists()
    assert not (tmp_path / "learning_probe_checkpoint.pt").exists()
    assert not (tmp_path / "checkpoint_manifest.json").exists()


def test_root_and_kaggle_probe_modules_are_byte_identical():
    root = (REPO_ROOT / "src" / "gpu_learning_probe.py").read_bytes()
    mirror = (REPO_ROOT / "kaggle_src_dataset" / "src" / "gpu_learning_probe.py").read_bytes()
    assert hashlib.sha256(root).digest() == hashlib.sha256(mirror).digest()


def test_kernel_metadata_is_private_t4_gpu_and_uses_expected_inputs():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["code_file"] == "gpu_learning_probe_kernel.py"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["drbhatiasanjay/st-act-src"]
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]


def test_kernel_structurally_verifies_nonempty_import_list_and_strict_sha():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    verify_calls = [
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "verify_import_origins"
    ]
    sha_calls = [
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "validate_git_sha_file"
    ]
    assert len(verify_calls) == 1
    assert isinstance(verify_calls[0].args[1], ast.Name)
    assert verify_calls[0].args[1].id == "modules_to_verify"
    assert len(sha_calls) == 1
    assert len(sha_calls[0].args) == 1


def test_probe_calls_validation_but_never_generates_manifest_or_pushes_kaggle():
    probe_tree = ast.parse((REPO_ROOT / "src" / "gpu_learning_probe.py").read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(probe_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "validate_epoch" in called_attributes
    assert "write_checkpoint_manifest" not in called_attributes
    kernel_source = KERNEL_PATH.read_text(encoding="utf-8")
    assert "kaggle kernels push" not in kernel_source
    assert "kaggle datasets version" not in kernel_source


def test_validation_dataset_is_bounded_before_dataloader_iteration():
    source = (REPO_ROOT / "src" / "gpu_learning_probe.py").read_text(encoding="utf-8")
    assert "sample_id_allowlist=selected_validation_ids" in source


def test_default_probe_is_512_batches_two_validation_samples_one_hour():
    from src import gpu_learning_probe

    assert gpu_learning_probe.DEFAULT_TRAIN_BATCHES == 512
    assert gpu_learning_probe.DEFAULT_VALIDATION_SAMPLES == 2
    assert gpu_learning_probe.DEFAULT_TIME_BUDGET_SECONDS == 3600.0
