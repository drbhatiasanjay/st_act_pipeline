"""
Kaggle training kernel for ST-ACT (Spatio-Temporal Anisotropic Cell Tracker).

Full training: wall-clock-budgeted run (up to 11h of the 12h Kaggle GPU cap)
on the full 199-sample train/val split.
- Trains until the wall-clock budget or num_epochs upper bound is hit
- Saves checkpoint (best val score, guaranteed at least after epoch 1) and
  training log for local evaluation
- Superseded the earlier 200-batch-capped sanity check once that validated
  the pipeline end-to-end (real data, real GPU, real checkpoint save)
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Kaggle "script"-type kernel pushes only upload the single code_file --
# sibling files/folders in the local push directory (e.g. the bundled
# kaggle_kernel/src/) are NOT included, confirmed by pulling the kernel back
# down and finding only this one file present. The correct mechanism is a
# Kaggle Dataset (drbhatiasanjay/st-act-src, containing src/ + data_split.json)
# attached via dataset_sources in kernel-metadata.json.
#
# The assumed mount path (/kaggle/input/st-act-src) 404'd on import TWICE
# despite the dataset showing as attached in the website editor's Input
# panel -- rather than guess a third exact path, search every directory
# under /kaggle/input for one that actually contains src/dataset.py, and use
# whatever that real path turns out to be.
#
# A one-level os.listdir() scan is NOT enough: Kaggle's current layout
# nests attached datasets under /kaggle/input/datasets/<owner>/<slug>/ (and
# competition data under /kaggle/input/competitions/<slug>/) rather than
# flat /kaggle/input/<slug>/. Confirmed by a real failed run (Version #17
# and #18 both) whose own diagnostic logged
# "/kaggle/input contents: ['competitions', 'datasets']" -- neither of
# those top-level names itself contains src/dataset.py, so the old loop
# always left KAGGLE_SRC_DATASET_DIR as None and every run died on
# `from src.dataset import CompetitionDataset` before training ever
# started. Walk recursively instead (capped at a shallow depth so this
# can't run away scanning a huge competition input tree), and take the
# first directory that actually contains src/dataset.py.
KAGGLE_SRC_DATASET_DIR = None
if os.path.exists("/kaggle/input"):
    MAX_SEARCH_DEPTH = 5
    for dirpath, dirnames, _filenames in os.walk("/kaggle/input"):
        depth = dirpath[len("/kaggle/input"):].count(os.sep)
        if depth >= MAX_SEARCH_DEPTH:
            dirnames[:] = []
            continue
        if "src" in dirnames and os.path.isfile(os.path.join(dirpath, "src", "dataset.py")):
            KAGGLE_SRC_DATASET_DIR = dirpath
            break

if KAGGLE_SRC_DATASET_DIR:
    sys.path.insert(0, KAGGLE_SRC_DATASET_DIR)
else:
    # Local run (or dataset not found/attached): src/ is a sibling of this
    # file's parent directory locally; on Kaggle this leaves imports to fail
    # loudly below rather than silently resolving to the wrong "src".
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Kaggle Training] %(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger("KaggleTraining")

# Detect environment
KAGGLE_MODE = os.path.exists("/kaggle/input")
if KAGGLE_MODE:
    INPUT_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
    WORKING_DIR = Path("/kaggle/working")
    OUTPUT_DIR = Path("/kaggle/output")
    # Diagnostic: the st-act-src dataset was visibly attached in the website
    # editor's Input panel yet the assumed mount path
    # (/kaggle/input/st-act-src) still 404'd on import -- log the REAL
    # contents of /kaggle/input instead of guessing the path a third time.
    try:
        logger.info(f"/kaggle/input contents: {os.listdir('/kaggle/input')}")
    except Exception as e:
        logger.warning(f"Could not list /kaggle/input: {e}")
else:
    INPUT_DIR = Path("data/staging")
    WORKING_DIR = Path(".")
    OUTPUT_DIR = Path(".")

WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"{'='*80}")
logger.info("ST-ACT KAGGLE FULL TRAINING")
logger.info(f"{'='*80}")
logger.info(f"Kaggle Mode: {KAGGLE_MODE}")
logger.info(f"Input Dir: {INPUT_DIR}")
logger.info(f"Working Dir: {WORKING_DIR}")

# Environment setup
logger.info("\n" + f"{'='*80}")
logger.info("ENVIRONMENT SETUP")
logger.info(f"{'='*80}")

# Deployed code identity: read the SHA embedded at push time by
# scripts/sync_kaggle_src.py -- turns "is this run using the code I think I
# committed" into a 2-second check of the first few log lines instead of a
# post-mortem after a multi-hour run (see DEFERRED_IMPROVEMENTS.md).
DEPLOYED_SHA = "unknown (GIT_SHA.txt not found -- was this pushed via scripts/sync_kaggle_src.py?)"
if KAGGLE_SRC_DATASET_DIR:
    sha_file = Path(KAGGLE_SRC_DATASET_DIR) / "GIT_SHA.txt"
    if sha_file.exists():
        DEPLOYED_SHA = sha_file.read_text().strip()
logger.info(f"Deployed code SHA: {DEPLOYED_SHA}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

    # Fail loud in milliseconds, not after Zarr data loading has already
    # started: a real run got a P100 (compute capability sm_60) allocated
    # via CLI push (Kaggle's kernels-push API can't request T4, confirmed
    # earlier this session), and PyTorch's installed build only ships
    # compiled kernels for sm_70+ -- "no kernel image is available for
    # execution on the device" only surfaced ~1s into UNet3D's first
    # conv3d call, after Zarr loading had already spent real time. Compare
    # the actual hardware's compute capability against what this specific
    # PyTorch build was compiled for, instead of discovering the mismatch
    # mid-forward-pass.
    major, minor = torch.cuda.get_device_capability(device)
    cc_string = f"sm_{major}{minor}"
    supported_archs = torch.cuda.get_arch_list()
    if cc_string not in supported_archs:
        raise RuntimeError(
            f"CUDA capability mismatch: {torch.cuda.get_device_name(device)} is "
            f"{cc_string}, but this PyTorch build only supports: "
            f"{', '.join(supported_archs)}. Reselect a supported GPU (T4) in the "
            f"website's Save & Run All dialog -- kernels push via CLI/API cannot "
            f"request a specific accelerator and may allocate an incompatible one."
        )
    logger.info(f"GPU compute capability {cc_string} verified compatible with this PyTorch build.")

# Set random seed
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
logger.info(f"Random seed set: {SEED}")

# Hyperparameters
HYPERPARAMS = {
    # Bumped 1e-3 -> 1e-2 (2026-07-14): the 1e-3 decision (see git history)
    # was based ENTIRELY on a trivial single-example memorization test
    # (crosses sigmoid=0.5 by step 50 on that test) -- that test's own
    # comment flagged "which this trivial synthetic test can't fully
    # represent." Two independent REAL-data experiments since then both
    # contradict its extrapolation: a local CPU run (35 real diverse-data
    # steps, max_sigmoid crawled 0.000082->0.000169, nowhere near 0.5) and a
    # real Kaggle GPU run (v42, 5000 real steps, shuffle=True confirmed
    # active via diverse validation sample IDs) where validation's max
    # sigmoid still rounded to <0.00005 -- i.e. no real movement across 5035
    # real steps combined. The BCE gradient at the init prior is verified
    # near-maximal (~-0.9999 per positive voxel, computed directly, not
    # assumed) so the loss function itself isn't obviously starved of signal
    # at the output layer -- what's unverified is whether that gradient
    # meaningfully moves real weights through the full multi-layer UNet3D at
    # 1e-3, which the trivial test's much simpler effective landscape didn't
    # stress-test. 1e-2 was previously rejected specifically because it
    # "converged in ~2 steps" on that SAME trivial test and was assumed
    # likely to destabilize on real data -- that assumption was never tested
    # on real data either. Testing it for real now that 1e-3 has failed
    # across two real-data experiments; the circuit breaker + heartbeat
    # infrastructure will catch instability quickly and cheaply if it occurs.
    # 1e-2 confirmed for real (v43, 2026-07-14): Loss: nan starting
    # somewhere between batch 5 and 100, never recovered through the
    # remaining ~900 batches -- genuine numeric divergence, not just a
    # theoretical risk. Both tested endpoints are now real, verified
    # failures: 1e-3 (no movement across 5035 real steps) and 1e-2 (NaN
    # within ~100 real steps). Testing the geometric mean of the two,
    # sqrt(1e-3*1e-2) ~= 3e-3, as the principled next point in this
    # log-scale search rather than guessing -- neither endpoint's extreme.
    'learning_rate': 3e-3,
    # warmup_steps=300 (2026-07-15): v48 (GroupNorm + gradient checkpointing,
    # same lr=3e-3, warmup_steps=0 default) showed max_sigmoid collapsing
    # fast within the FIRST ~30 real batches (8e-5 -> ~1.6e-5), then sitting
    # roughly flat near that floor for the remaining ~4970 batches -- a fast
    # early crash into an apparent saturation trap, not a slow continuous
    # drift (confirmed via fine-grained, every-5-batch log inspection, not
    # the coarser every-200-batch view that looked like gradual decay).
    # TrainingLoop's linear warmup (src/train.py, built 2026-07-14 in direct
    # response to v43's lr=1e-2 divergence) was already implemented and
    # tested but never actually enabled on any real run through v48 --
    # this is the first run to turn it on. 300 steps is ~10x past where the
    # collapse originates, giving the model room to find a stable operating
    # point before the full 3e-3 rate applies. warmup_start_lr uses
    # TrainingLoop's existing default (1e-4).
    'warmup_steps': 300,
    'grad_clip': 1.0,
    'weight_decay': 1e-4,
    'heatmap_loss_weight': 1.0,
    'division_loss_weight': 2.5,
    'early_stopping_patience': 10,
    'edge_threshold': 0.5,
    'detection_threshold': 0.5,
    'nms_radius_um': 5.0,
    'seed': SEED,
    'batch_size': 1,  # Memory-constrained
    # TEMPORARY for this run only: an intermediate ~34min-train verification,
    # sized up from the earlier 200-batch (~5min) check, which showed the
    # v40 (2026-07-14): 1500 batches with shuffle=True + all 4 fixes (adaptive
    # loss, bias init, lr=1e-3, shuffle=True) trained cleanly -- healthy,
    # mildly-decreasing real loss throughout (~1.61 -> ~1.42 by batch 1000),
    # no collapse -- but validation's circuit breaker still fired (first 10
    # frozen-model batches predicted zero nodes above detection_threshold=0.5).
    # Bumping to 5000 (real GPU rate confirmed 1.54s/batch -> ~2.1h train) to
    # test whether more real steps clear the fixed 0.5 threshold before
    # committing to a full ~6.3h uncapped epoch. REMOVE this line before the
    # next real full training run (see git history / DEFERRED_IMPROVEMENTS.md
    # for the full-run config this temporarily overrides).
    # Back to 5000 (2026-07-14): lr=3e-3 already confirmed stable at 1000
    # batches (v44, no NaN) but validation still showed zero detections --
    # inconclusive at that budget since v42's comparable 1e-3 result used
    # 5x more steps. Scaling to the same 5000-batch budget for a fair
    # comparison, now with per-batch max_sigmoid logging (see train.py) so
    # this run shows the real trend directly instead of another isolated
    # before/after snapshot.
    'max_batches_per_epoch': 5000,
    # Full run: no max_batches_per_epoch cap (that was only for the earlier
    # sanity check -- real epoch size is ~14,751 batches: 149 train samples x
    # ~99 consecutive-frame pairs each). Real per-batch rate at sanity-check
    # time (v26, before the Zarr per-item loader-caching fix in f5fd65c) was
    # ~1.37s/batch, implying ~5.6h/epoch train-only -- but that number
    # predates the loader-cache fix and may now be pessimistic. Rather than
    # guess a fixed epoch count from a possibly-stale rate, num_epochs below
    # is a generous upper bound and max_wall_clock_seconds does the real
    # gating, sized from the ACTUAL measured epoch time during this run
    # (see TrainingLoop.fit()).
    'num_epochs': 1,
    # 11h of Kaggle's 12h GPU session cap, leaving a ~1h buffer for
    # dependency install, notebook-to-HTML conversion, and validate_epoch
    # cost that may be higher than the sanity run's (that run's val was
    # near-instant because the undertrained model predicted zero detections
    # everywhere, making NMS/assignment near-no-ops -- not representative of
    # a model that's actually learning).
    'max_wall_clock_seconds': 11 * 3600,
}

logger.info("\nHyperparameters:")
for key, val in HYPERPARAMS.items():
    logger.info(f"  {key}: {val}")

# Kaggle's base image does not have this project's non-standard pinned
# dependencies (confirmed by a real failed run: ModuleNotFoundError on
# tracksdata after the src/ import path itself was fixed). Install them from
# requirements.txt's exact pins before importing any project code.
# tracksdata==0.1.0rc6 is pre-1.0 and version-sensitive (see CLAUDE.md).
#
# --no-deps is deliberate and load-bearing, not an optimization: two real
# runs showed pip's normal dependency resolution -- even WITH an explicit
# numpy<2.4 constraint added -- silently reinstalls numpy in a way that
# leaves it internally inconsistent (numpy's own strings.py importing names
# from numpy._core.umath that the installed umath binary doesn't have).
# Kaggle's base image already ships numpy/scipy/pandas current enough for
# these packages' actual runtime needs; --no-deps stops pip from touching
# them via transitive resolution at all. If a genuinely-missing transitive
# dependency surfaces (not numpy/scipy), it needs to be added to this list
# explicitly rather than removing --no-deps.
if KAGGLE_MODE:
    import subprocess
    logger.info("Installing non-standard dependencies...")
    # --no-deps means transitive deps must be listed explicitly. Real full
    # dependency tree checked locally (importlib.metadata.requires) for
    # tracksdata/geff/zarr; geff-spec confirmed missing by a real Kaggle run
    # (ModuleNotFoundError: No module named 'geff_spec'). bidict/rustworkx/
    # psygnal/donfig/google-crc32c/typer are niche enough to be unlikely on
    # Kaggle's base image; ilpy/imagecodecs deliberately omitted for now
    # (heavier, solver-binding-adjacent -- add only if actually needed, since
    # tracksdata's own ILP tracker isn't used here, only its graph/geff I/O
    # and metric functions per this project's established usage).
    #
    # polars pinned to >=1.36.0 (tracksdata's real declared requirement, not
    # a guess): a bare "polars" with no version constraint let pip treat an
    # already-installed older polars as "satisfied" and skip reinstalling it
    # (--no-deps means pip won't upgrade an unconstrained already-present
    # package), and that old polars lacks the Float16 dtype tracksdata's own
    # internals reference -- confirmed by a real run's AttributeError.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "tracksdata==0.1.0rc6", "zarr>=3.0.0", "numcodecs>=0.11.0",
            "geff>=1.0.0", "geff-spec", "polars>=1.36.0", "dask",
            "bidict", "rustworkx", "psygnal", "donfig", "google-crc32c", "typer",
            # tracksdata/__init__.py unconditionally imports tracksdata.solvers
            # (-> ilpy) at package init, even though this project's code path
            # never calls tracksdata's own ILP solver (only its graph/geff I/O
            # and metric functions) -- confirmed required by a real run's
            # ModuleNotFoundError, so it can't be skipped as originally assumed.
            # ilpy itself requires pyscipopt (checked via
            # importlib.metadata.requires('ilpy') locally, not a guess).
            "ilpy", "pyscipopt",
            # blosc2 is in requirements.txt for real (Zarr v3 compression
            # codec) but was missed from earlier install rounds -- a full
            # recursive scan of tracksdata's package source for every
            # third-party top-level import (not just following one
            # ModuleNotFoundError at a time) confirmed it's referenced.
            "blosc2>=2.0.0",
        ],
        check=True,
    )
    logger.info("Dependency installation complete.")

    # v20-v22 traced a real bug through three rounds: v20's plain install
    # left `polars._plr` (the compiled extension) broken, silently swallowed
    # by polars/series/series.py's `with contextlib.suppress(ImportError):
    # from polars._plr import PyDataFrame, PySeries` -- every later
    # DataFrame/Series call then raised NameError: name 'PySeries' is not
    # defined, caught by this project's own try/except fallbacks and
    # silently replaced with all-zero heatmap targets and empty GT node
    # sets for EVERY timepoint, no crash (confirmed via v20's real log).
    # v21 added a fail-loud check (kept below) and tried --force-reinstall,
    # which made the symptom clearer but not fixed: "Polars binary is
    # missing!" / "could not find Polars' Rust module". v22's `pip show -f
    # polars` gave the real answer: `Requires: polars-runtime-32` -- modern
    # polars ships as a thin Python package plus a SEPARATE compiled
    # extension package (polars-runtime-32) that provides polars._plr.
    # This project's --no-deps flag (deliberately there to stop pip from
    # touching numpy/scipy, see the block above) was also blocking polars'
    # own required runtime companion from ever installing. Fix: install it
    # explicitly by name, same trick already used for ilpy/pyscipopt.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "--force-reinstall", "polars>=1.36.0", "polars-runtime-32"],
        check=True,
    )

    # Fail loud, not silent: verify polars' compiled extension actually
    # loaded before training starts, instead of letting the same bug
    # resurface as a quietly-swallowed per-timepoint warning again.
    import polars as _pl_check
    try:
        from polars._plr import PySeries as _PySeriesCheck  # noqa: F401
    except ImportError as e:
        logger.error(f"polars._plr failed to import after force-reinstall: {e}")
        logger.error(f"polars version: {_pl_check.__version__}")
        raise RuntimeError(
            "polars compiled extension (_plr) is broken -- GT node/heatmap "
            "loading would silently degrade to all-zero targets. Aborting "
            "instead of training on garbage data."
        ) from e
    logger.info(f"polars {_pl_check.__version__} extension verified OK.")

# For local development: import from src/
# For Kaggle: dependencies come from the attached st-act-src Dataset
try:
    from src.dataset import CompetitionDataset
    from src.model import SimpleNodeTransformer, UNet3D
    from src.train import TrainingLoop
    LOCAL_IMPORTS = True
except ImportError:
    LOCAL_IMPORTS = False
    logger.error("Could not import from src/. This kernel requires the src/ directory to be present.")
    logger.error(f"KAGGLE_SRC_DATASET_DIR resolved to: {KAGGLE_SRC_DATASET_DIR}")
    logger.error(f"sys.path: {sys.path}")
    raise

logger.info(f"Local imports: {LOCAL_IMPORTS}")

# === DATA LOADING ===
logger.info("\n" + f"{'='*80}")
logger.info("DATA LOADING")
logger.info(f"{'='*80}")

if KAGGLE_SRC_DATASET_DIR:
    data_split_file = Path(KAGGLE_SRC_DATASET_DIR) / "data_split.json"
else:
    data_split_file = Path("data_split.json")
if not data_split_file.exists():
    logger.error(f"data_split.json not found at {data_split_file}")
    raise FileNotFoundError("Missing data_split.json")

logger.info("Creating datasets...")
# NOTE: KAGGLE_MODE's exact input subdirectory structure under INPUT_DIR
# (e.g. INPUT_DIR/"train" vs INPUT_DIR itself) has NOT been verified against
# the real Kaggle mount in this session -- smoketest.py's own docstring notes
# a prior guessed path was wrong once already. Verify this path exists on
# Kaggle (e.g. via a quick `os.listdir(INPUT_DIR)` in the actual kernel logs)
# before trusting a full sanity-check run to use it correctly.
train_data_dir = (INPUT_DIR / "train") if KAGGLE_MODE else Path("data/staging/train")
logger.info(f"Using data_dir: {train_data_dir}")
try:
    train_dataset = CompetitionDataset(
        data_dir=train_data_dir,
        split_file=data_split_file,
        split_type='train',
        normalize=True
    )
    val_dataset = CompetitionDataset(
        data_dir=train_data_dir,
        split_file=data_split_file,
        split_type='validation',
        normalize=True
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
except Exception as e:
    logger.error(f"Failed to create datasets: {e}")
    raise

# Create data loaders. train_loader MUST shuffle: real .geff ground truth exists
# only at sparse per-sample timepoint ranges (see CLAUDE.md), so with
# shuffle=False the model walks long, unbroken all-background stretches in the
# same fixed order every epoch -- confirmed live via a local diagnostic
# (2026-07-14): loss collapsed to ~0.0001 and max_sigmoid went flat at the
# init floor for 19+ consecutive steps once a sample's sparse GT ran out.
# val_loader MUST ALSO shuffle -- the original "order doesn't matter, and the
# circuit-breaker checks the first N batches specifically" reasoning was
# wrong: verified directly (2026-07-14) that the first validation sample
# (44b6_0b24845f) has ZERO real GT nodes for t=0 through t=10 (first real
# node at t=11), so with shuffle=False the circuit-breaker's first 10
# batches are structurally guaranteed to have nothing to detect, regardless
# of model quality -- this is exactly what v40 (1500 batches) and v41 (5000
# batches) both hit, unchanged, proving the check wasn't measuring the model
# at all. Same root-cause class as the train_loader fix above, just on the
# validation side.
train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True)

logger.info(f"Train loader batches: {len(train_loader)}")
logger.info(f"Val loader batches: {len(val_loader)}")

# === MODEL INITIALIZATION ===
logger.info("\n" + f"{'='*80}")
logger.info("MODEL INITIALIZATION")
logger.info(f"{'='*80}")

logger.info("Creating models...")
unet3d = UNet3D(in_channels=2, channels=(32, 64, 128))
transformer = SimpleNodeTransformer(hidden_dim=128, num_heads=4, num_blocks=4)

unet3d.to(device)
transformer.to(device)

unet3d_params = sum(p.numel() for p in unet3d.parameters())
transformer_params = sum(p.numel() for p in transformer.parameters())
total_params = unet3d_params + transformer_params

logger.info(f"UNet3D parameters: {unet3d_params:,}")
logger.info(f"Transformer parameters: {transformer_params:,}")
logger.info(f"Total parameters: {total_params:,}")

# === TRAINING LOOP SETUP ===
logger.info("\n" + f"{'='*80}")
logger.info("TRAINING LOOP SETUP")
logger.info(f"{'='*80}")

checkpoint_dir = WORKING_DIR / "checkpoints"
log_file = WORKING_DIR / "training_log.csv"
progress_file = WORKING_DIR / "training_progress.json"

logger.info(f"Checkpoint dir: {checkpoint_dir}")
logger.info(f"Log file: {log_file}")
logger.info(f"Progress heartbeat file: {progress_file}")

training_loop = TrainingLoop(
    unet3d=unet3d,
    transformer=transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    data_dir=train_data_dir,
    checkpoint_dir=str(checkpoint_dir),
    log_file=str(log_file),
    hyperparams=HYPERPARAMS,
    deployed_sha=DEPLOYED_SHA,
    progress_file=progress_file,
)

logger.info("Training loop initialized")

# === FULL TRAINING (WALL-CLOCK BUDGETED) ===
logger.info("\n" + f"{'='*80}")
logger.info("FULL TRAINING (WALL-CLOCK BUDGETED)")
logger.info(f"{'='*80}")

num_epochs = HYPERPARAMS['num_epochs']
max_wall_clock_seconds = HYPERPARAMS['max_wall_clock_seconds']
logger.info(f"Training for up to {num_epochs} epochs, budget={max_wall_clock_seconds:.0f}s")

try:
    training_loop.fit(num_epochs=num_epochs, max_wall_clock_seconds=max_wall_clock_seconds)
except Exception as e:
    logger.error(f"Training failed: {e}")
    logger.error("Saving partial checkpoint before exit...")
    try:
        partial_checkpoint = {
            'unet3d_state_dict': unet3d.state_dict(),
            'transformer_state_dict': transformer.state_dict(),
            'error': str(e),
        }
        torch.save(partial_checkpoint, WORKING_DIR / "partial_checkpoint.pt")
        logger.info("Partial checkpoint saved")
    except Exception as save_error:
        logger.error(f"Failed to save partial checkpoint: {save_error}")
    raise

# === OUTPUT & VERIFICATION ===
logger.info("\n" + f"{'='*80}")
logger.info("OUTPUT & VERIFICATION")
logger.info(f"{'='*80}")

# Save model summary
logger.info("Saving model summary...")
# num_epochs above is the upper bound passed to fit(); the wall-clock budget
# may have stopped training earlier -- count actual completed epochs from
# the log file (one data row per completed epoch) rather than assume the cap.
epochs_completed = 0
if Path(log_file).exists():
    with open(log_file) as _f:
        epochs_completed = max(0, sum(1 for _ in _f) - 1)
summary_path = WORKING_DIR / "model_summary.txt"
with open(summary_path, 'w') as f:
    f.write("ST-ACT KAGGLE FULL TRAINING\n")
    f.write(f"{'='*80}\n\n")
    f.write("MODEL SUMMARY\n")
    f.write(f"UNet3D parameters: {unet3d_params:,}\n")
    f.write(f"Transformer parameters: {transformer_params:,}\n")
    f.write(f"Total parameters: {total_params:,}\n\n")
    f.write("TRAINING CONFIGURATION\n")
    for key, val in HYPERPARAMS.items():
        f.write(f"{key}: {val}\n")
    f.write(f"\nTRAINING EPOCHS COMPLETED: {epochs_completed} (budget upper bound: {num_epochs})\n")
    f.write(f"Training dataset size: {len(train_dataset)}\n")
    f.write(f"Validation dataset size: {len(val_dataset)}\n\n")
    f.write("CHECKPOINTS\n")
    if training_loop.best_checkpoint_path:
        f.write(f"Best checkpoint: {training_loop.best_checkpoint_path}\n")
        f.write(f"Best validation score: {training_loop.best_val_score:.6f}\n")

logger.info(f"Model summary saved to {summary_path}")

# Verify training log exists and has data
logger.info("\nVerifying training log...")
log_path = Path(log_file)
if log_path.exists():
    with open(log_path) as f:
        lines = f.readlines()
    logger.info(f"Training log created: {log_path}")
    logger.info(f"Log has {len(lines)} lines (header + {len(lines)-1} epochs)")

    # Print first few lines for verification
    logger.info("First few log lines:")
    for line in lines[:min(3, len(lines))]:
        logger.info(f"  {line.strip()}")
else:
    logger.warning(f"Training log not found at {log_path}")

# List output files
logger.info("\nOutput files in working directory:")
for f in sorted(WORKING_DIR.glob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / (1024*1024)
        logger.info(f"  {f.name}: {size_mb:.2f} MB")

logger.info("\n" + f"{'='*80}")
logger.info("FULL TRAINING COMPLETE")
logger.info(f"{'='*80}")
logger.info("\nNext steps:")
logger.info("1. Download outputs from /kaggle/working/")
logger.info("2. Check training_log.csv for metrics and loss curves")
logger.info("3. Verify validation score is non-zero and metrics are reasonable")
logger.info("4. Push the best checkpoint to st-act-checkpoint dataset for inference")
logger.info("5. If issues found, diagnose from logs and fix src/train.py")
