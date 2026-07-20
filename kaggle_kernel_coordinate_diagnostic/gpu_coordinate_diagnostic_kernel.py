"""Kaggle T4 entrypoint for the one-frame coordinate diagnostic."""

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

MAX_SEARCH_DEPTH = 5


def find_all_kaggle_input_dirs(marker_relpath: str, max_depth: int = MAX_SEARCH_DEPTH) -> list[str]:
    if not os.path.exists("/kaggle/input"):
        return []
    matches: list[str] = []
    for dirpath, dirnames, _filenames in os.walk("/kaggle/input"):
        depth = dirpath[len("/kaggle/input") :].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        if os.path.isfile(os.path.join(dirpath, marker_relpath)):
            matches.append(dirpath)
    return matches


def find_exactly_one_kaggle_input_dir(marker_relpath: str) -> str:
    matches = find_all_kaggle_input_dirs(marker_relpath)
    if not matches:
        raise RuntimeError(f"No directory beneath /kaggle/input contains {marker_relpath!r}.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple directories beneath /kaggle/input contain {marker_relpath!r}, exactly "
            f"one is required: {sorted(matches)}"
        )
    return matches[0]


logging.basicConfig(
    level=logging.INFO,
    format="[GPU Coordinate Diagnostic] %(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger("GPUCoordinateDiagnostic")

if not os.path.exists("/kaggle/input"):
    raise RuntimeError("GPU coordinate diagnostic must run inside Kaggle")

source_root = Path(find_exactly_one_kaggle_input_dir(os.path.join("src", "dataset.py")))
checkpoint_root = Path(find_exactly_one_kaggle_input_dir("learning_probe_checkpoint.pt"))
checkpoint_path = checkpoint_root / "learning_probe_checkpoint.pt"
sys.path.insert(0, str(source_root))

from src.deployment_provenance import validate_git_sha_file  # noqa: E402

deployed_sha = validate_git_sha_file(source_root / "GIT_SHA.txt")
logger.info("Diagnostic code SHA: %s", deployed_sha)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "tracksdata==0.1.0rc6",
        "zarr>=3.0.0",
        "numcodecs>=0.11.0",
        "geff>=1.0.0",
        "geff-spec",
        "polars>=1.36.0",
        "dask",
        "bidict",
        "rustworkx",
        "psygnal",
        "donfig",
        "google-crc32c",
        "typer",
        "ilpy",
        "pyscipopt",
        "blosc2>=2.0.0",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "--force-reinstall",
        "polars>=1.36.0",
        "polars-runtime-32",
    ],
    check=True,
)

import polars as _polars_check  # noqa: E402
from polars._plr import PySeries as _PySeriesCheck  # noqa: E402, F401

logger.info("polars %s extension verified OK", _polars_check.__version__)

import src.data_loader  # noqa: E402
import src.dataset  # noqa: E402
import src.evaluation  # noqa: E402
import src.gpu_coordinate_diagnostic  # noqa: E402
import src.model  # noqa: E402
import src.train  # noqa: E402
from src.deployment_provenance import verify_import_origins  # noqa: E402
from src.gpu_coordinate_diagnostic import run_gpu_coordinate_diagnostic  # noqa: E402
from src.split_utils import resolve_split_file_path  # noqa: E402

modules_to_verify = [
    src.data_loader,
    src.dataset,
    src.evaluation,
    src.gpu_coordinate_diagnostic,
    src.model,
    src.train,
]
verify_import_origins(source_root, modules_to_verify)
import_origins = {
    module.__name__: str(Path(module.__file__).resolve()) for module in modules_to_verify
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
entrypoint_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
competition_train_dir = Path(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
)
split_file = resolve_split_file_path(kaggle_src_dataset_dir=source_root)
output_dir = Path("/kaggle/working/gpu_coordinate_diagnostic")

report = run_gpu_coordinate_diagnostic(
    data_dir=competition_train_dir,
    split_file=split_file,
    checkpoint_path=checkpoint_path,
    output_dir=output_dir,
    device=device,
    deployed_sha=deployed_sha,
    import_origins=import_origins,
    diagnostic_entrypoint_sha256=entrypoint_sha256,
)
logger.info("Coordinate diagnostic execution verdict: %s", report["execution_verdict"])
logger.info("Scientific diagnostic outcome: %s", report.get("diagnostic_outcome"))
logger.info("Report: %s", output_dir / "gpu_coordinate_diagnostic_report.json")
if report["execution_verdict"] != "PASS":
    raise RuntimeError(f"GPU coordinate diagnostic failed: {report['failure_reasons']}")
