"""
Kaggle inference kernel for ST-ACT -- Code Competition compliant submission.

This is a SEPARATE kernel from train_kernel.py, built after discovering (from
the real /rules page, not guessed) that biohub-cell-tracking-during-development
is a Code Competition: the graded submission must be a committed Notebook with
"Internet access disabled", GPU/CPU runtime <= 12 hours, writing submission.csv.
train_kernel.py's live `pip install` from PyPI cannot run here -- every
dependency must already be present. This script installs from a pre-downloaded
wheels Dataset instead, via `pip install --no-index --find-links=...`.

P0-6: submission-path parity, fail-closed submission validation, verified
checkpoint deployment. Checkpoint selection is now exclusively via a verified
checkpoint_manifest.json (never filename/mtime guessing -- see
src/checkpoint_manifest.py), and graph construction is exclusively via the
shared production pipeline (src/submission_pipeline.py) so this kernel can
never silently diverge from generate_submission.py's local smoke-test path.
"""

import logging
import os
import sys
from pathlib import Path

import torch

# Exact-one discovery (Part C1): never silently select the first directory,
# directory order, or modification time -- zero matches raises, multiple
# matches raises and lists every candidate.
MAX_SEARCH_DEPTH = 5


def find_all_kaggle_input_dirs(marker_relpath: str, max_depth: int = MAX_SEARCH_DEPTH) -> list[str]:
    if not os.path.exists("/kaggle/input"):
        return []
    matches: list[str] = []
    for dirpath, dirnames, _filenames in os.walk("/kaggle/input"):
        depth = dirpath[len("/kaggle/input"):].count(os.sep)
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


KAGGLE_MODE = os.path.exists("/kaggle/input")

if KAGGLE_MODE:
    # Kaggle mode: never fall back to repository-local source code.
    KAGGLE_SRC_DATASET_DIR = find_exactly_one_kaggle_input_dir(os.path.join("src", "dataset.py"))
    sys.path.insert(0, KAGGLE_SRC_DATASET_DIR)
else:
    # Local non-Kaggle execution (e.g. import-structure smoke tests) may
    # retain a repository fallback.
    KAGGLE_SRC_DATASET_DIR = None
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='[Kaggle Inference] %(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger("KaggleInference")

if KAGGLE_MODE:
    try:
        logger.info(f"/kaggle/input contents: {os.listdir('/kaggle/input')}")
    except Exception as e:
        logger.warning(f"Could not list /kaggle/input: {e}")
    logger.info(f"Selected source dataset (exact-one discovery): {KAGGLE_SRC_DATASET_DIR}")

# === ENVIRONMENT SETUP ===
# Deployed code identity (Part C3): read the SHA embedded at push time by
# scripts/sync_kaggle_src.py. In Kaggle mode, missing/unreadable/empty/
# malformed values raise -- "unknown" is never a valid fallback here, unlike
# the pre-P0-6 kernel.
if KAGGLE_MODE:
    _sha_file = Path(KAGGLE_SRC_DATASET_DIR) / "GIT_SHA.txt"
    if not _sha_file.exists():
        raise RuntimeError(
            f"GIT_SHA.txt not found in selected source dataset {KAGGLE_SRC_DATASET_DIR} -- "
            f"was this pushed via scripts/sync_kaggle_src.py?"
        )
    _raw_sha = _sha_file.read_text(encoding="utf-8").strip()
    if not _raw_sha:
        raise RuntimeError(f"GIT_SHA.txt at {_sha_file} is empty or whitespace-only.")
    if len(_raw_sha) != 40 or _raw_sha != _raw_sha.lower() or not all(c in "0123456789abcdef" for c in _raw_sha):
        raise RuntimeError(
            f"GIT_SHA.txt at {_sha_file} does not contain a 40-character lowercase hex git "
            f"SHA: {_raw_sha!r}"
        )
    DEPLOYED_SHA = _raw_sha
else:
    DEPLOYED_SHA = "unknown (local execution)"
logger.info(f"Deployed code SHA: {DEPLOYED_SHA}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    # Same fail-fast GPU compute-capability check as train_kernel.py -- fail
    # in milliseconds if an incompatible GPU (e.g. P100) was allocated,
    # instead of after loading test data.
    major, minor = torch.cuda.get_device_capability(device)
    cc_string = f"sm_{major}{minor}"
    supported_archs = torch.cuda.get_arch_list()
    if cc_string not in supported_archs:
        raise RuntimeError(
            f"CUDA capability mismatch: {torch.cuda.get_device_name(device)} is "
            f"{cc_string}, but this PyTorch build only supports: "
            f"{', '.join(supported_archs)}. Reselect a supported GPU (T4)."
        )
    logger.info(f"GPU compute capability {cc_string} verified compatible.")

# === OFFLINE DEPENDENCY INSTALL ===
# Same 17 packages as train_kernel.py's pip install list, but from a local
# wheels Dataset (drbhatiasanjay/st-act-wheels, pre-downloaded for
# manylinux2014_x86_64 cp312 -- matches Kaggle's real base image) instead of
# live PyPI -- required because Internet access is disabled for the real
# graded run. --no-deps stays load-bearing (see train_kernel.py's documented
# numpy-corruption lesson): the wheels list already includes every real
# transitive dependency, checked via the same audit method used earlier.
if KAGGLE_MODE:
    import subprocess

    _wheels_matches = find_all_kaggle_input_dirs("tracksdata-0.1.0rc6-py3-none-any.whl")
    if not _wheels_matches:
        raise RuntimeError(
            "st-act-wheels dataset not found under /kaggle/input -- attach it "
            "in the notebook's Input panel before running."
        )
    if len(_wheels_matches) > 1:
        raise RuntimeError(
            f"Multiple datasets under /kaggle/input contain the wheels marker file, "
            f"exactly one is required: {sorted(_wheels_matches)}"
        )
    WHEELS_DIR = _wheels_matches[0]
    logger.info(f"Installing dependencies offline from {WHEELS_DIR}")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
            f"--find-links={WHEELS_DIR}",
            "tracksdata==0.1.0rc6", "zarr>=3.0.0", "numcodecs>=0.11.0",
            "geff>=1.0.0", "geff-spec",
            "dask", "bidict", "rustworkx", "psygnal", "donfig", "google-crc32c",
            "typer", "ilpy", "pyscipopt", "blosc2>=2.0.0",
        ],
        check=True,
    )

    # polars needs --force-reinstall, separately from the rest: a real run
    # (v22-27 of the training kernel) proved plain `pip install polars` --
    # offline OR from PyPI, doesn't matter -- leaves Kaggle's pre-existing
    # base-image polars untouched if it already satisfies the version bound,
    # even though that pre-existing install may be missing its compiled
    # extension entirely. This exact fix already exists in train_kernel.py;
    # omitting it here (a real bug, not a hypothetical) reproduced the
    # identical "Polars binary is missing!" failure on the very first real
    # run of this script.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
            "--force-reinstall", f"--find-links={WHEELS_DIR}",
            "polars>=1.36.0", "polars-runtime-32",
        ],
        check=True,
    )
    logger.info("Offline dependency installation complete.")

    import polars as _pl_check
    try:
        from polars._plr import PySeries as _PySeriesCheck  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"polars compiled extension (_plr) is broken after offline install: {e}"
        ) from e
    logger.info(f"polars {_pl_check.__version__} extension verified OK.")

import src.checkpoint_manifest  # noqa: E402
import src.dataset  # noqa: E402
import src.inference  # noqa: E402
import src.model  # noqa: E402
import src.prediction_graph  # noqa: E402
import src.submission_exporter  # noqa: E402
import src.submission_pipeline  # noqa: E402
import src.train  # noqa: E402
from src.checkpoint_manifest import find_single_manifest, load_verified_checkpoint  # noqa: E402
from src.model import SimpleNodeTransformer, UNet3D  # noqa: E402
from src.submission_exporter import export_submission, validate_submission  # noqa: E402
from src.submission_pipeline import run_submission_inference  # noqa: E402


def verify_import_origins(expected_root: str | Path) -> None:
    """Part C2: after importing every production module this kernel depends
    on, verify each one's real __file__ resolves underneath the exact-one
    selected source dataset -- never underneath a stray repository checkout,
    a different attached dataset, or anywhere else. Raises loudly on any
    module resolving outside expected_root."""
    expected_root_resolved = Path(expected_root).resolve()
    modules_to_check = [
        src.dataset,
        src.model,
        src.train,
        src.submission_pipeline,
        src.checkpoint_manifest,
        src.submission_exporter,
        src.prediction_graph,
        src.inference,
    ]
    for module in modules_to_check:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"Module {module.__name__} has no __file__ -- cannot verify import origin.")
        resolved = Path(module_file).resolve()
        try:
            resolved.relative_to(expected_root_resolved)
        except ValueError as e:
            raise RuntimeError(
                f"Module {module.__name__} resolved to {resolved}, which is NOT beneath the "
                f"selected source dataset {expected_root_resolved} -- refusing to run "
                f"inference against code from an unverified location."
            ) from e
        logger.info(f"Verified import origin: {module.__name__} -> {resolved}")


if KAGGLE_MODE:
    verify_import_origins(KAGGLE_SRC_DATASET_DIR)


def main() -> None:
    # === VERIFIED CHECKPOINT SELECTION (Part C4) ===
    # No recursive epoch_*.pt / newest-mtime / filename-guessing selection --
    # exactly one manifest beneath /kaggle/input, hash-verified before
    # torch.load() ever runs.
    manifest_path = find_single_manifest("/kaggle/input")
    checkpoint, manifest, checkpoint_path = load_verified_checkpoint(
        manifest_path, expected_source_sha=DEPLOYED_SHA, map_location=device,
    )
    logger.info(f"Verified manifest: {manifest_path}")
    logger.info(f"Verified checkpoint: {checkpoint_path}")
    logger.info(
        f"Manifest identity: training_code_sha={manifest['training_code_sha']} "
        f"mounted_source_sha={DEPLOYED_SHA} split_membership_sha256={manifest['split_membership_sha256']} "
        f"model_contract={manifest['model_contract']} epoch={manifest['epoch']} "
        f"coverage={manifest['validation_samples_evaluated']}/{manifest['validation_samples_total']} "
        f"datasets={manifest['num_datasets']} predicted_nodes_total={manifest['predicted_nodes_total']} "
        f"predicted_edges_total={manifest['predicted_edges_total']} "
        f"adjusted_edge_jaccard={manifest['adjusted_edge_jaccard']}"
    )

    hyperparams = checkpoint["hyperparams"]

    unet3d = UNet3D(in_channels=2, channels=(32, 64, 128)).to(device)
    transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4).to(device)
    # Strict state-dict loading (Part B8) -- never downgraded to a warning.
    unet3d.load_state_dict(checkpoint["unet3d_state_dict"], strict=True)
    transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=True)
    unet3d.eval()
    transformer.eval()

    # === REAL TEST DATA (swapped in at grading time, per rules) ===
    test_dir = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test")
    test_zarrs = sorted(test_dir.glob("*.zarr"))
    logger.info(f"Found {len(test_zarrs)} real test sample(s): {[z.stem for z in test_zarrs]}")

    # === SHARED PRODUCTION SUBMISSION PIPELINE (Part C5) ===
    # No direct IndexedRXGraph construction, node-attribute schema setup, or
    # add_node/add_edge loops in this kernel -- run_submission_inference()
    # owns all graph assembly (via PredictionGraphAssembler), identically to
    # generate_submission.py's local path. Raises RuntimeError (Part C6) if
    # test_zarrs is empty rather than producing a header-only "successful"
    # dry run.
    pred_graphs, diagnostics = run_submission_inference(
        test_dir=test_dir,
        test_zarrs=test_zarrs,
        unet3d=unet3d,
        transformer=transformer,
        device=device,
        hyperparams=hyperparams,
    )

    required_dataset_ids = diagnostics["required_dataset_ids"]
    logger.info(
        f"Inference complete in {diagnostics['total_elapsed_seconds']:.1f}s: "
        f"{diagnostics['total_unique_nodes']} nodes, {diagnostics['total_unique_edges']} edges, "
        f"{diagnostics['total_accepted_edges']}/{diagnostics['total_candidate_edges']} edges accepted."
    )
    for sample_id in required_dataset_ids:
        sample_diag = diagnostics["per_sample"][sample_id]
        logger.info(
            f"  {sample_id}: pairs={sample_diag['processed_pair_count']}/"
            f"{sample_diag['expected_pair_count']} nodes={sample_diag['unique_node_count']} "
            f"edges={sample_diag['unique_edge_count']} elapsed={sample_diag['elapsed_seconds']:.1f}s"
        )
    if "cuda_max_memory_allocated_bytes" in diagnostics:
        logger.info(f"CUDA peak memory allocated: {diagnostics['cuda_max_memory_allocated_bytes']} bytes")

    out_path = "/kaggle/working/submission.csv"
    logger.info(f"Exporting submission to {out_path}")
    csv_path = export_submission(pred_graphs, out_path, required_dataset_ids=required_dataset_ids)
    is_valid = validate_submission(csv_path, required_dataset_ids=required_dataset_ids)
    logger.info(f"validate_submission() result: {is_valid}")
    if not is_valid:
        raise RuntimeError("Generated submission.csv failed validate_submission() -- do not submit.")


if __name__ == "__main__":
    main()
