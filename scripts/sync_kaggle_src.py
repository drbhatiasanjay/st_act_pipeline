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
DATASET_SRC_DIR = REPO_ROOT / "kaggle_src_dataset" / "src"
SHA_FILE = REPO_ROOT / "kaggle_src_dataset" / "GIT_SHA.txt"


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def sync_src() -> None:
    for py_file in SRC_DIR.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(SRC_DIR)
        dest = DATASET_SRC_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, dest)


def verify_sync() -> list[str]:
    """Return a list of mismatched relative paths (empty = fully synced)."""
    mismatches = []
    for py_file in SRC_DIR.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(SRC_DIR)
        dest = DATASET_SRC_DIR / rel
        if not dest.exists() or not filecmp.cmp(py_file, dest, shallow=False):
            mismatches.append(str(rel))
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push a new Kaggle Dataset version after syncing")
    parser.add_argument("-m", "--message", default=None, help="Dataset version message (required with --push)")
    args = parser.parse_args()

    if args.push and not args.message:
        parser.error("--push requires -m/--message")

    sha = git_sha()
    print(f"Current git SHA: {sha}")

    sync_src()
    SHA_FILE.write_text(sha + "\n")
    print(f"Synced src/ -> kaggle_src_dataset/src/, wrote {SHA_FILE.relative_to(REPO_ROOT)}")

    mismatches = verify_sync()
    if mismatches:
        print(f"ERROR: {len(mismatches)} file(s) still mismatched after sync: {mismatches}", file=sys.stderr)
        sys.exit(1)
    print("Verified: kaggle_src_dataset/src/ matches src/ exactly.")

    if args.push:
        print(f"Pushing new dataset version: {args.message}")
        subprocess.run(
            ["py", "-m", "kaggle", "datasets", "version", "-p", ".", "-m", args.message, "--dir-mode", "zip"],
            cwd=REPO_ROOT / "kaggle_src_dataset", check=True,
        )


if __name__ == "__main__":
    main()
