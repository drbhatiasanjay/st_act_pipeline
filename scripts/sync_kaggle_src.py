"""
Sync src/ into kaggle_src_dataset/src/, embed a git SHA marker, and verify the
sync actually took before (optionally) pushing a new Kaggle Dataset version.

Built after a real incident this session: an adjacent investigation session
concluded a 5.6 GPU-hour training run had executed on a stale, pre-fix Kaggle
dataset snapshot (later independently re-checked and found NOT to be true --
see DEFERRED_IMPROVEMENTS.md -- but the underlying gap it correctly identified
is real regardless: there was no automated way to confirm "the code Kaggle
mounts actually matches what's committed" before triggering a run, only manual
`cp` + eyeballing). This script removes the manual-copy failure mode entirely
and gives every future run a 2-second way to confirm its own code identity
(the embedded SHA, logged as the first line of ENVIRONMENT SETUP).

P0-2 fix (2026-07-16): also syncs the embryo-disjoint split files
(data_split.json, data_splits/*.json). Before this fix, only src/**/*.py was
synced -- a real training run reading data_split.json from
kaggle_src_dataset/ (see kaggle_kernel/train_kernel.py's
resolve_split_file_path(kaggle_src_dataset_dir=...) call) would have silently
gotten whatever stale split snapshot happened to be sitting in
kaggle_src_dataset/, independent of what was actually committed to the repo
root -- the exact same "code Kaggle mounts doesn't match what's committed"
failure mode this script was built to close for src/**/*.py, just for split
files instead.

Usage:
    python scripts/sync_kaggle_src.py                 # sync + verify only
    python scripts/sync_kaggle_src.py --push -m "..."  # sync, verify, then push
"""

import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATASET_DIR = REPO_ROOT / "kaggle_src_dataset"
DATASET_SRC_DIR = DATASET_DIR / "src"
SHA_FILE = DATASET_DIR / "GIT_SHA.txt"

# Relative (POSIX-style) paths of every split file that must be mounted
# alongside src/ on Kaggle -- kept in sync with src/split_utils.py's
# DEFAULT_SPLIT_FILE and scripts/build_train_val_split.py's write_folds().
SPLIT_FILES = [
    "data_split.json",
    "data_splits/embryo_44b6_validation.json",
    "data_splits/embryo_6bba_validation.json",
]


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def working_tree_is_clean(repo_root: Path = REPO_ROOT) -> bool:
    """True iff `git status --porcelain` reports no staged, unstaged, or
    untracked changes at all. Used to guard both the GIT_SHA.txt marker
    write and --push -- P0-2 fix (2026-07-16): writing 'current git SHA' into
    a tracked file while OTHER tracked files have uncommitted changes is
    misleading (it implies the synced content == that exact commit, when it
    may actually include uncommitted diffs beyond it), and pushing a Kaggle
    Dataset version from a dirty tree uploads code that doesn't match any
    real, reviewable commit at all."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip() == ""


def sync_git_sha(repo_root: Path = REPO_ROOT, sha_file: Path = SHA_FILE) -> str | None:
    """Write the current git HEAD SHA to sha_file, but ONLY when the working
    tree is clean (see working_tree_is_clean() docstring for why). Returns
    the SHA actually written, or None if the write was skipped because the
    tree was dirty -- in which case sha_file's existing content (if any) is
    left untouched rather than overwritten with a misleading marker."""
    if not working_tree_is_clean(repo_root):
        print(
            f"WARNING: working tree has uncommitted changes -- leaving "
            f"{sha_file.relative_to(repo_root)} untouched rather than writing "
            f"a misleading 'clean HEAD' marker. Commit before deploying to "
            f"get an accurate GIT_SHA.txt.",
            file=sys.stderr,
        )
        return None
    sha = git_sha(repo_root)
    sha_file.write_text(sha + "\n")
    return sha


def sync_src(src_dir: Path = SRC_DIR, dataset_src_dir: Path = DATASET_SRC_DIR) -> None:
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(src_dir)
        dest = dataset_src_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, dest)


def sync_split_files(repo_root: Path = REPO_ROOT, dataset_dir: Path = DATASET_DIR) -> None:
    """Copy every file in SPLIT_FILES from repo_root to dataset_dir, preserving
    the same relative path. Raises FileNotFoundError (fail loud, not a silent
    skip) if a required split file is missing at the source -- a Kaggle bundle
    missing a split file is exactly the kind of gap this script exists to
    prevent, not something to quietly work around."""
    for rel in SPLIT_FILES:
        src = repo_root / rel
        if not src.exists():
            raise FileNotFoundError(
                f"Required split file not found at {src} -- cannot sync a "
                f"Kaggle bundle without every file in SPLIT_FILES present."
            )
        dest = dataset_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def verify_sync(
    src_dir: Path = SRC_DIR,
    dataset_src_dir: Path = DATASET_SRC_DIR,
    repo_root: Path = REPO_ROOT,
    dataset_dir: Path = DATASET_DIR,
) -> list[str]:
    """Return a list of mismatched/missing relative paths across BOTH the
    Python source tree (src/**/*.py) and the split files (SPLIT_FILES) --
    empty means fully synced. Each entry is either a src/-relative .py path
    (matching the original pre-P0-2 format) or a SPLIT_FILES entry."""
    mismatches: list[str] = []

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(src_dir)
        dest = dataset_src_dir / rel
        if not dest.exists() or not filecmp.cmp(py_file, dest, shallow=False):
            mismatches.append(str(rel))

    for rel in SPLIT_FILES:
        src = repo_root / rel
        dest = dataset_dir / rel
        if not src.exists():
            # Source itself missing -- can't confirm sync either way, but this
            # is still a real problem worth surfacing, not a silent skip.
            mismatches.append(rel)
            continue
        if not dest.exists() or not filecmp.cmp(src, dest, shallow=False):
            mismatches.append(rel)

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push a new Kaggle Dataset version after syncing")
    parser.add_argument("-m", "--message", default=None, help="Dataset version message (required with --push)")
    args = parser.parse_args()

    if args.push and not args.message:
        parser.error("--push requires -m/--message")

    # P0-2 fix (2026-07-16): --push uploads a Kaggle Dataset version claiming
    # to be "this code" -- from a dirty tree, that claim is false (the
    # uploaded content wouldn't match any real, reviewable commit). Checked
    # BEFORE doing any sync work, so a doomed --push fails fast.
    if args.push and not working_tree_is_clean():
        print(
            "ERROR: working tree has uncommitted changes -- refusing to push "
            "a Kaggle Dataset version that would not match any real commit. "
            "Commit (or stash) your changes first.",
            file=sys.stderr,
        )
        sys.exit(1)

    sha = git_sha()
    print(f"Current git SHA: {sha}")

    sync_src()
    sync_split_files()
    written_sha = sync_git_sha()
    print("Synced src/ -> kaggle_src_dataset/src/")
    print(f"Synced {len(SPLIT_FILES)} split file(s) -> kaggle_src_dataset/")
    if written_sha:
        print(f"Wrote {SHA_FILE.relative_to(REPO_ROOT)} = {written_sha}")
    else:
        print(f"Skipped updating {SHA_FILE.relative_to(REPO_ROOT)} (working tree not clean)")

    mismatches = verify_sync()
    if mismatches:
        print(f"ERROR: {len(mismatches)} file(s) still mismatched after sync: {mismatches}", file=sys.stderr)
        sys.exit(1)
    print("Verified: kaggle_src_dataset/ matches src/ and the active split files exactly.")

    if args.push:
        print(f"Pushing new dataset version: {args.message}")
        subprocess.run(
            ["py", "-m", "kaggle", "datasets", "version", "-p", ".", "-m", args.message, "--dir-mode", "zip"],
            cwd=REPO_ROOT / "kaggle_src_dataset", check=True,
        )


if __name__ == "__main__":
    main()
