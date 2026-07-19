"""Fail-closed contract tests for the bounded GPU first-light gate."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.gpu_first_light_gate import (
    evaluate_first_light_report,
    run_gpu_first_light_gate,
    write_report_atomic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "kaggle_kernel_gate" / "gpu_first_light_kernel.py"
METADATA_PATH = REPO_ROOT / "kaggle_kernel_gate" / "kernel-metadata.json"


def _passing_report():
    return {
        "schema_version": 1,
        "gate_name": "GPU-FIRST-LIGHT-01",
        "gate_scope": "infrastructure_only_not_model_quality",
        "deployed_sha": "a" * 40,
        "gate_entrypoint_sha256": "c" * 64,
        "split_membership_sha256": "d" * 64,
        "import_origins": {"src.train": "/kaggle/input/st-act-src/src/train.py"},
        "import_origins_verified": True,
        "cuda_available": True,
        "cuda_arch_compatible": True,
        "device_type": "cuda",
        "gpu_name": "Tesla T4",
        "selected_sample_ids": ["s1", "s2", "s3", "s4"],
        "selected_sample_count": 4,
        "sample_count_requested": 4,
        "successfully_opened_sample_ids": ["s1", "s2", "s3", "s4"],
        "dataset_pair_count": 200,
        "requested_batches": 64,
        "completed_batches": 64,
        "average_train_loss": 1.25,
        "last_unet_gradient_norm": 0.5,
        "last_transformer_gradient_norm": 0.25,
        "fallback_counts": {"edge_target_generation_failure": 0, "edge_loss_computation_failure": 0},
        "biological_counts": {
            "edge_supervised_batches_total": 64,
            "edge_supervised_batches_with_nonzero_transformer_grad": 3,
        },
        "elapsed_seconds": 90.0,
        "time_budget_seconds": 600.0,
        "validation_performed": False,
        "deployment_manifest_generated": False,
        "peak_gpu_memory_allocated_bytes": 100,
        "peak_gpu_memory_reserved_bytes": 200,
        "sanity_checkpoint_saved": True,
        "sanity_checkpoint_sha256": "b" * 64,
    }


def test_complete_report_passes():
    assert evaluate_first_light_report(_passing_report()) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", 1.0, "schema version"),
        ("deployed_sha", "unknown", "deployed_sha"),
        ("gate_entrypoint_sha256", "bad", "entrypoint SHA-256"),
        ("split_membership_sha256", "bad", "split membership"),
        ("import_origins_verified", False, "import origins"),
        ("cuda_available", False, "CUDA device"),
        ("cuda_arch_compatible", False, "compute capability"),
        ("completed_batches", 63, "completed batch count"),
        ("requested_batches", True, "completed batch count"),
        ("selected_sample_count", 4.0, "sample identity/count"),
        ("average_train_loss", float("nan"), "training loss"),
        ("last_unet_gradient_norm", 0.0, "UNet gradient"),
        ("last_transformer_gradient_norm", float("inf"), "Transformer gradient"),
        ("elapsed_seconds", 601.0, "wall-clock budget"),
        ("validation_performed", True, "must not perform validation"),
        ("deployment_manifest_generated", True, "deployment manifest"),
        ("peak_gpu_memory_allocated_bytes", 0, "allocated GPU memory"),
        ("sanity_checkpoint_saved", False, "checkpoint was not saved"),
    ],
)
def test_each_primary_failure_is_fail_closed(field, value, expected):
    report = _passing_report()
    report[field] = value
    assert any(expected in reason for reason in evaluate_first_light_report(report))


def test_any_nonzero_technical_fallback_fails():
    report = _passing_report()
    report["fallback_counts"]["edge_target_generation_failure"] = 1
    assert any("fallback" in reason for reason in evaluate_first_light_report(report))


@pytest.mark.parametrize(
    "counter",
    ["edge_supervised_batches_total", "edge_supervised_batches_with_nonzero_transformer_grad"],
)
def test_missing_edge_learning_signal_fails(counter):
    report = _passing_report()
    report["biological_counts"][counter] = 0
    assert evaluate_first_light_report(report)


def test_pre_checkpoint_evaluation_can_validate_training_predicates_only():
    report = _passing_report()
    report["sanity_checkpoint_saved"] = False
    report.pop("sanity_checkpoint_sha256")
    assert evaluate_first_light_report(report, require_checkpoint=False) == []


def test_atomic_report_is_deterministic_json(tmp_path):
    path = tmp_path / "report.json"
    report = {"z": 1, "a": [2, 3]}
    write_report_atomic(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')
    assert not path.with_suffix(".json.tmp").exists()


def test_cpu_invocation_fails_closed_but_still_writes_report(tmp_path):
    report = run_gpu_first_light_gate(
        data_dir=tmp_path / "unused",
        split_file=tmp_path / "unused.json",
        output_dir=tmp_path,
        device=torch.device("cpu"),
        deployed_sha="a" * 40,
        import_origins={"src.train": "/verified/src/train.py"},
        gate_entrypoint_sha256="b" * 64,
        cuda_arch_compatible=False,
    )
    assert report["verdict"] == "FAIL"
    assert report["exception_type"] == "RuntimeError"
    assert (tmp_path / "gpu_first_light_report.json").exists()
    assert not (tmp_path / "sanity_checkpoint.pt").exists()
    assert not (tmp_path / "checkpoint_manifest.json").exists()


def test_root_and_kaggle_gate_modules_are_byte_identical():
    root = (REPO_ROOT / "src" / "gpu_first_light_gate.py").read_bytes()
    mirror = (REPO_ROOT / "kaggle_src_dataset" / "src" / "gpu_first_light_gate.py").read_bytes()
    assert hashlib.sha256(root).digest() == hashlib.sha256(mirror).digest()


def test_kernel_metadata_is_private_t4_gpu_and_uses_expected_inputs():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["code_file"] == "gpu_first_light_kernel.py"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["drbhatiasanjay/st-act-src"]
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]


def test_kernel_structurally_verifies_nonempty_import_list_and_strict_sha():
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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


def test_gate_never_calls_validation_deployment_manifest_or_kaggle_push():
    paths = [KERNEL_PATH, REPO_ROOT / "src" / "gpu_first_light_gate.py"]
    forbidden_calls = {"validate_epoch", "write_checkpoint_manifest"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert forbidden_calls.isdisjoint(called_attributes | called_names)
    kernel_source = KERNEL_PATH.read_text(encoding="utf-8")
    assert "kaggle kernels push" not in kernel_source
    assert "kaggle datasets version" not in kernel_source


def test_default_run_is_four_samples_sixty_four_batches_ten_minutes():
    from src import gpu_first_light_gate

    assert gpu_first_light_gate.DEFAULT_SAMPLE_COUNT == 4
    assert gpu_first_light_gate.DEFAULT_MAX_BATCHES == 64
    assert gpu_first_light_gate.DEFAULT_TIME_BUDGET_SECONDS == 600.0
