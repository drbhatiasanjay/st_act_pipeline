"""
Unit tests for scripts/sync_kaggle_src.py's sync_src / sync_split_files /
verify_sync -- the P0-2 fix (2026-07-16) that extends this script to also
sync data_split.json + data_splits/*.json, not just src/**/*.py. Before this
fix, a Kaggle run reading data_split.json from kaggle_src_dataset/ could
silently use a stale split snapshot never verified against what was actually
committed -- the same class of gap this script was originally built to close
for src/**/*.py.

Fully synthetic -- every test builds its own tmp_path repo/dataset directory
pair and never touches the real repo's src/ or kaggle_src_dataset/.

Run: py -m pytest tests/test_sync_kaggle_src.py -v
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.sync_kaggle_src import (
    SPLIT_FILES,
    sync_git_sha,
    sync_split_files,
    sync_src,
    verify_sync,
    working_tree_is_clean,
)


def make_git_repo(tmp_path):
    """Init a real, throwaway git repo under tmp_path with one committed
    file. working_tree_is_clean()/sync_git_sha() shell out to real `git`, so
    a real (if tiny) repo is the only faithful way to test them -- the -c
    user.name/user.email flags scope identity to this one invocation only,
    never touching global git config."""
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    git_env_args = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]

    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    (repo_root / "committed.txt").write_text("committed content\n")
    subprocess.run(["git", "add", "committed.txt"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", *git_env_args, "commit", "-q", "-m", "initial commit"], cwd=repo_root, check=True,
    )
    return repo_root


def make_repo(tmp_path, py_files: dict[str, str], split_files: dict[str, str] | None = None):
    """Build a synthetic {repo}/src/ tree with the given {relpath: content}
    Python files, and optionally {repo}/{relpath} split files. Returns
    (repo_root, src_dir, dataset_dir, dataset_src_dir)."""
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    dataset_dir = repo_root / "kaggle_src_dataset"
    dataset_src_dir = dataset_dir / "src"

    for rel, content in py_files.items():
        path = src_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    for rel, content in (split_files or {}).items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    return repo_root, src_dir, dataset_dir, dataset_src_dir


class TestSyncSrc:
    def test_copies_python_files_to_dataset_src_dir(self, tmp_path):
        _repo_root, src_dir, _dataset_dir, dataset_src_dir = make_repo(
            tmp_path, {"dataset.py": "# real content", "nested/helper.py": "# nested content"}
        )
        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)

        assert (dataset_src_dir / "dataset.py").read_text() == "# real content"
        assert (dataset_src_dir / "nested" / "helper.py").read_text() == "# nested content"

    def test_overwrites_stale_content(self, tmp_path):
        _repo_root, src_dir, _dataset_dir, dataset_src_dir = make_repo(
            tmp_path, {"dataset.py": "# new content"}
        )
        dataset_src_dir.mkdir(parents=True)
        (dataset_src_dir / "dataset.py").write_text("# stale old content")

        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)

        assert (dataset_src_dir / "dataset.py").read_text() == "# new content"

    def test_skips_pycache(self, tmp_path):
        _repo_root, src_dir, _dataset_dir, dataset_src_dir = make_repo(tmp_path, {"dataset.py": "# content"})
        pycache_file = src_dir / "__pycache__" / "dataset.cpython-312.py"
        pycache_file.parent.mkdir(parents=True)
        pycache_file.write_text("# should not be synced")

        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)

        assert not (dataset_src_dir / "__pycache__").exists()


class TestSyncSplitFiles:
    def test_copies_split_files_to_dataset_dir(self, tmp_path):
        split_content = {rel: f"content-of-{rel}" for rel in SPLIT_FILES}
        repo_root, _src_dir, dataset_dir, _dataset_src_dir = make_repo(
            tmp_path, {}, split_files=split_content
        )

        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)

        for rel in SPLIT_FILES:
            assert (dataset_dir / rel).read_text() == f"content-of-{rel}"

    def test_overwrites_stale_split_file_content(self, tmp_path):
        split_content = {rel: f"fresh-{rel}" for rel in SPLIT_FILES}
        repo_root, _src_dir, dataset_dir, _dataset_src_dir = make_repo(
            tmp_path, {}, split_files=split_content
        )
        for rel in SPLIT_FILES:
            stale_path = dataset_dir / rel
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            stale_path.write_text(f"stale-{rel}")

        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)

        for rel in SPLIT_FILES:
            assert (dataset_dir / rel).read_text() == f"fresh-{rel}"

    def test_raises_file_not_found_when_source_split_file_missing(self, tmp_path):
        # Only provide the first SPLIT_FILES entry -- the rest are missing.
        split_content = {SPLIT_FILES[0]: "content"}
        repo_root, _src_dir, dataset_dir, _dataset_src_dir = make_repo(
            tmp_path, {}, split_files=split_content
        )

        expected_basename = SPLIT_FILES[1].rsplit("/", 1)[-1]
        with pytest.raises(FileNotFoundError, match=expected_basename):
            sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)


class TestVerifySync:
    def _full_repo(self, tmp_path):
        split_content = {rel: f"content-of-{rel}" for rel in SPLIT_FILES}
        return make_repo(
            tmp_path,
            {"dataset.py": "# real content", "nested/helper.py": "# nested content"},
            split_files=split_content,
        )

    def test_fully_synced_repo_reports_no_mismatches(self, tmp_path):
        repo_root, src_dir, dataset_dir, dataset_src_dir = self._full_repo(tmp_path)
        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)
        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)

        mismatches = verify_sync(
            src_dir=src_dir, dataset_src_dir=dataset_src_dir, repo_root=repo_root, dataset_dir=dataset_dir,
        )
        assert mismatches == []

    def test_detects_missing_python_file(self, tmp_path):
        repo_root, src_dir, dataset_dir, dataset_src_dir = self._full_repo(tmp_path)
        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)
        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)
        (dataset_src_dir / "nested" / "helper.py").unlink()

        mismatches = verify_sync(
            src_dir=src_dir, dataset_src_dir=dataset_src_dir, repo_root=repo_root, dataset_dir=dataset_dir,
        )
        assert os.path.join("nested", "helper.py") in mismatches

    def test_detects_python_file_content_mismatch(self, tmp_path):
        repo_root, src_dir, dataset_dir, dataset_src_dir = self._full_repo(tmp_path)
        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)
        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)
        (dataset_src_dir / "dataset.py").write_text("# tampered content")

        mismatches = verify_sync(
            src_dir=src_dir, dataset_src_dir=dataset_src_dir, repo_root=repo_root, dataset_dir=dataset_dir,
        )
        assert "dataset.py" in mismatches

    def test_detects_missing_split_file(self, tmp_path):
        repo_root, src_dir, dataset_dir, dataset_src_dir = self._full_repo(tmp_path)
        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)
        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)
        (dataset_dir / SPLIT_FILES[0]).unlink()

        mismatches = verify_sync(
            src_dir=src_dir, dataset_src_dir=dataset_src_dir, repo_root=repo_root, dataset_dir=dataset_dir,
        )
        assert SPLIT_FILES[0] in mismatches

    def test_detects_split_file_content_mismatch(self, tmp_path):
        repo_root, src_dir, dataset_dir, dataset_src_dir = self._full_repo(tmp_path)
        sync_src(src_dir=src_dir, dataset_src_dir=dataset_src_dir)
        sync_split_files(repo_root=repo_root, dataset_dir=dataset_dir)
        (dataset_dir / SPLIT_FILES[1]).write_text("# tampered split content")

        mismatches = verify_sync(
            src_dir=src_dir, dataset_src_dir=dataset_src_dir, repo_root=repo_root, dataset_dir=dataset_dir,
        )
        assert SPLIT_FILES[1] in mismatches
        assert SPLIT_FILES[0] not in mismatches

    def test_before_any_sync_everything_is_reported_mismatched(self, tmp_path):
        repo_root, src_dir, dataset_dir, dataset_src_dir = self._full_repo(tmp_path)
        # No sync_src()/sync_split_files() call -- dataset_dir is empty.

        mismatches = verify_sync(
            src_dir=src_dir, dataset_src_dir=dataset_src_dir, repo_root=repo_root, dataset_dir=dataset_dir,
        )
        assert "dataset.py" in mismatches
        assert os.path.join("nested", "helper.py") in mismatches
        for rel in SPLIT_FILES:
            assert rel in mismatches


class TestWorkingTreeIsClean:
    """P0-2 fix (2026-07-16): guards both the GIT_SHA.txt marker write and
    --push -- see working_tree_is_clean()'s docstring."""

    def test_clean_repo_returns_true(self, tmp_path):
        repo_root = make_git_repo(tmp_path)
        assert working_tree_is_clean(repo_root) is True

    def test_modified_tracked_file_returns_false(self, tmp_path):
        repo_root = make_git_repo(tmp_path)
        (repo_root / "committed.txt").write_text("modified content\n")
        assert working_tree_is_clean(repo_root) is False

    def test_untracked_new_file_returns_false(self, tmp_path):
        repo_root = make_git_repo(tmp_path)
        (repo_root / "new_untracked_file.txt").write_text("new\n")
        assert working_tree_is_clean(repo_root) is False


class TestSyncGitSha:
    def test_clean_tree_writes_current_head_sha(self, tmp_path):
        repo_root = make_git_repo(tmp_path)
        sha_file = repo_root / "GIT_SHA.txt"
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()

        written = sync_git_sha(repo_root=repo_root, sha_file=sha_file)

        assert written == head_sha
        assert sha_file.read_text().strip() == head_sha

    def test_dirty_tree_skips_write_and_leaves_existing_content_untouched(self, tmp_path):
        repo_root = make_git_repo(tmp_path)
        sha_file = repo_root / "GIT_SHA.txt"
        sha_file.write_text("stale-sha-marker\n")
        (repo_root / "committed.txt").write_text("uncommitted modification\n")  # dirties the tree

        written = sync_git_sha(repo_root=repo_root, sha_file=sha_file)

        assert written is None
        assert sha_file.read_text() == "stale-sha-marker\n"

    def test_dirty_tree_does_not_create_sha_file_if_absent(self, tmp_path):
        repo_root = make_git_repo(tmp_path)
        sha_file = repo_root / "GIT_SHA.txt"
        (repo_root / "committed.txt").write_text("uncommitted modification\n")  # dirties the tree

        written = sync_git_sha(repo_root=repo_root, sha_file=sha_file)

        assert written is None
        assert not sha_file.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
