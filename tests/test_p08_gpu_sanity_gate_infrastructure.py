"""
Tests for GPU sanity gate infrastructure (GPU_SANITY_GATE_DESIGN_2026-07-18_v4.md).

Wave 1 (this section, Part A): CompetitionDataset's sample_id_allowlist constructor
param, and every function in src/deployment_provenance.py (find_all_kaggle_input_dirs,
find_exactly_one_kaggle_input_dir, validate_git_sha_file, verify_import_origins).

TestExactOneDiscoveryOnSharedModule DOES exercise this module's own copy of the
exact-one discovery functions (not a re-test of the kernel scripts) -- this is
necessary, not redundant: src/deployment_provenance.py's copy is a separate,
independently-editable piece of code, and only test_p07_training_integrity.py's
TestExactOneSourceDiscovery covers the kernel scripts' own literal copies (via
AST extraction of kaggle_kernel/train_kernel.py). Without this module's own
tests, a future edit to src/deployment_provenance.py's copy alone could regress
silently. See the module's own docstring for why the kernel scripts can't
import this module's copy at the point they need discovery (a bootstrap
ordering constraint), which is why both copies exist and both need coverage.

Run: py -m pytest tests/test_p08_gpu_sanity_gate_infrastructure.py -v
"""
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CompetitionDataset
from src.deployment_provenance import (
    find_all_kaggle_input_dirs,
    find_exactly_one_kaggle_input_dir,
    validate_git_sha_file,
    verify_import_origins,
)

# ---------------------------------------------------------------------------
# Section A1 -- CompetitionDataset(sample_id_allowlist=...)
# ---------------------------------------------------------------------------

def _write_split_file(tmp_path: Path, train_ids: list[str], validation_ids: list[str] | None = None) -> Path:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({
        "train": train_ids,
        "validation": validation_ids or [],
    }))
    return split_path


class TestSampleIdAllowlist:
    def test_none_allowlist_does_not_filter(self, tmp_path):
        split_path = _write_split_file(tmp_path, ["a", "b", "c"])
        dataset = CompetitionDataset(
            data_dir=tmp_path, split_file=split_path, split_type="train",
            sample_id_allowlist=None,
        )
        assert dataset.sample_ids == ["a", "b", "c"]

    def test_allowlist_subset_filters_sample_ids(self, tmp_path):
        split_path = _write_split_file(tmp_path, ["a", "b", "c", "d"])
        dataset = CompetitionDataset(
            data_dir=tmp_path, split_file=split_path, split_type="train",
            sample_id_allowlist=["b", "d"],
        )
        assert sorted(dataset.sample_ids) == ["b", "d"]

    def test_allowlist_narrows_expected_sample_ids(self, tmp_path):
        """expected_sample_ids (set by _build_pair_index) must reflect only the
        allowlisted subset -- the filtering must happen BEFORE the pair-index
        build loop, not after (design §3.0)."""
        split_path = _write_split_file(tmp_path, ["a", "b", "c", "d"])
        dataset = CompetitionDataset(
            data_dir=tmp_path, split_file=split_path, split_type="train",
            sample_id_allowlist=["a", "c"],
        )
        assert sorted(dataset.expected_sample_ids) == ["a", "c"]

    def test_allowlist_id_not_in_split_raises(self, tmp_path):
        split_path = _write_split_file(tmp_path, ["a", "b"])
        with pytest.raises(ValueError, match="not present in split"):
            CompetitionDataset(
                data_dir=tmp_path, split_file=split_path, split_type="train",
                sample_id_allowlist=["a", "does_not_exist"],
            )

    def test_allowlist_with_strict_sample_coverage_raises_on_missing_zarr(self, tmp_path):
        """Both flags must compose: allowlist filters the input set first, then
        strict_sample_coverage still fails loud on a real coverage gap within
        that filtered set (neither flag silently absorbs the other's job)."""
        split_path = _write_split_file(tmp_path, ["a", "b"])
        with pytest.raises(RuntimeError, match="strict_sample_coverage=True"):
            CompetitionDataset(
                data_dir=tmp_path, split_file=split_path, split_type="train",
                sample_id_allowlist=["a"], strict_sample_coverage=True,
            )

    def test_explicitly_empty_allowlist_raises(self, tmp_path):
        """An explicitly empty allowlist ([]) must raise, not be treated as a
        valid zero-sample dataset -- distinct from sample_id_allowlist=None,
        which means 'no restriction at all', not 'restrict to nothing'. A
        caller passing [] almost certainly has an unpopulated upstream value
        (e.g. an empty K-expansion result), which must fail loud, not
        silently produce a dataset with zero samples that would then pass
        every other check vacuously."""
        split_path = _write_split_file(tmp_path, ["a", "b"])
        with pytest.raises(ValueError, match="empty"):
            CompetitionDataset(
                data_dir=tmp_path, split_file=split_path, split_type="train",
                sample_id_allowlist=[],
            )

    def test_duplicate_ids_in_allowlist_raise_listing_duplicates(self, tmp_path):
        """A duplicated ID in sample_id_allowlist must raise, not be silently
        collapsed via set() -- a caller passing duplicates has a real bug
        (e.g. double-counting in a candidate-expansion loop) that a silent
        dedup would hide rather than surface."""
        split_path = _write_split_file(tmp_path, ["a", "b", "c"])
        with pytest.raises(ValueError, match=r"duplicate sample IDs.*\['a'\]"):
            CompetitionDataset(
                data_dir=tmp_path, split_file=split_path, split_type="train",
                sample_id_allowlist=["a", "a", "b", "a"],
            )

    def test_filtered_sample_ids_preserve_split_file_order_not_allowlist_order(self, tmp_path):
        """Deterministic ordering: the filtered self.sample_ids must follow the
        ORIGINAL split-file order, not the order IDs happen to appear in the
        allowlist argument -- filtering is `[s for s in self.sample_ids if s in
        allowlist_set]`, a single deterministic pass over the split's own
        order, never re-sorted by the (possibly differently-ordered, and in
        Wave 4's K-expansion case, insertion-ordered) allowlist list."""
        split_path = _write_split_file(tmp_path, ["c", "a", "b", "d"])
        dataset = CompetitionDataset(
            data_dir=tmp_path, split_file=split_path, split_type="train",
            sample_id_allowlist=["b", "a"],  # deliberately reversed vs split order
        )
        assert dataset.sample_ids == ["a", "b"]  # split order (c,a,b,d) filtered, not allowlist order (b,a)

    def test_excluded_samples_never_reach_pair_index_construction(self, tmp_path, monkeypatch):
        """A sample_id filtered out by sample_id_allowlist must never be visited
        by _build_pair_index()'s per-sample loop at all -- not merely excluded
        from the final pairs list after being processed. Proven by giving the
        excluded sample a real, openable .zarr directory (so it WOULD produce
        real pairs if incorrectly visited) and recording every sample_id
        _get_loader is actually called with."""
        split_path = _write_split_file(tmp_path, ["a", "b"])
        (tmp_path / "a.zarr").mkdir()
        (tmp_path / "b.zarr").mkdir()  # exists -- would yield real pairs if visited

        visited_sample_ids = []

        class _FakeLoader:
            def get_shape(self):
                return (3, 4, 4, 4)

        def fake_get_loader(self, sample_id):
            visited_sample_ids.append(sample_id)
            return _FakeLoader()

        monkeypatch.setattr(CompetitionDataset, "_get_loader", fake_get_loader)

        dataset = CompetitionDataset(
            data_dir=tmp_path, split_file=split_path, split_type="train",
            sample_id_allowlist=["a"],
        )

        assert visited_sample_ids == ["a"], (
            f"excluded sample 'b' must never be visited by _build_pair_index's "
            f"per-sample loop, got {visited_sample_ids}"
        )
        assert all(sample_id == "a" for sample_id, _ in dataset.pairs)
        assert dataset.pairs != []

    def test_post_build_invariant_catches_mutation_during_build_pair_index(self, tmp_path, monkeypatch):
        """Codex review, PR #4, 2026-07-19: proves the invariant runs AFTER
        _build_pair_index() and is actually EFFECTIVE at catching mutation,
        not merely unreachable-by-construction. Monkeypatches
        _build_pair_index to call the real implementation and then corrupt
        self.sample_ids afterward -- simulating a bug introduced during index
        construction that a check positioned only right after the pre-build
        filter step could never observe (it would already have run and
        returned by the time such a mutation happened)."""
        split_path = _write_split_file(tmp_path, ["a", "b"])
        real_build_pair_index = CompetitionDataset._build_pair_index

        def corrupting_build_pair_index(self):
            real_build_pair_index(self)
            self.sample_ids.append("mutated_during_build")

        monkeypatch.setattr(CompetitionDataset, "_build_pair_index", corrupting_build_pair_index)

        with pytest.raises(RuntimeError, match="invariant violated"):
            CompetitionDataset(
                data_dir=tmp_path, split_file=split_path, split_type="train",
                sample_id_allowlist=["a"],
            )


# ---------------------------------------------------------------------------
# Section A2 -- src/deployment_provenance.py
# ---------------------------------------------------------------------------

def _patch_fake_kaggle_input(monkeypatch, real_root: Path):
    """Redirect the literal '/kaggle/input' path to a real temp directory tree,
    for the duration of one test. Mirrors
    tests/test_p07_training_integrity.py's identical helper (duplicated here,
    not imported, to keep this new test file self-contained and not create a
    cross-test-file dependency)."""
    real_exists = os.path.exists
    real_walk = os.walk
    real_isfile = os.path.isfile
    real_root_str = str(real_root)

    def fake_exists(path):
        if str(path) == "/kaggle/input":
            return real_root.exists()
        return real_exists(path)

    def fake_walk(path, *a, **kw):
        if str(path) == "/kaggle/input":
            for dirpath, dirnames, filenames in real_walk(real_root_str, *a, **kw):
                yield "/kaggle/input" + dirpath[len(real_root_str):], dirnames, filenames
        else:
            yield from real_walk(path, *a, **kw)

    def fake_isfile(path):
        spath = str(path)
        if spath.startswith("/kaggle/input"):
            rel = spath[len("/kaggle/input"):].lstrip("/\\")
            return real_isfile(str(real_root / rel)) if rel else real_isfile(real_root_str)
        return real_isfile(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os, "walk", fake_walk)
    monkeypatch.setattr(os.path, "isfile", fake_isfile)


class TestExactOneDiscoveryOnSharedModule:
    """Sanity-checks src/deployment_provenance.py's copy behaves identically to
    the kernel scripts' own copies (already fully covered by
    test_p07_training_integrity.py::TestExactOneSourceDiscovery) -- this is a
    consistency check on the shared module, not a re-test of the kernels."""

    def test_zero_matches_raises(self, monkeypatch, tmp_path):
        _patch_fake_kaggle_input(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match="No directory"):
            find_exactly_one_kaggle_input_dir(os.path.join("src", "dataset.py"))

    def test_exactly_one_match_returns_it(self, monkeypatch, tmp_path):
        d = tmp_path / "st-act-src" / "src"
        d.mkdir(parents=True)
        (d / "dataset.py").touch()
        _patch_fake_kaggle_input(monkeypatch, tmp_path)
        found = find_exactly_one_kaggle_input_dir(os.path.join("src", "dataset.py"))
        assert found.replace("\\", "/").endswith("st-act-src")

    def test_multiple_matches_raises_listing_candidates(self, monkeypatch, tmp_path):
        for name in ("dataset-a", "dataset-b"):
            d = tmp_path / name / "src"
            d.mkdir(parents=True)
            (d / "dataset.py").touch()
        _patch_fake_kaggle_input(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError, match="Multiple directories"):
            find_exactly_one_kaggle_input_dir(os.path.join("src", "dataset.py"))

    def test_find_all_returns_empty_list_when_no_kaggle_input(self, monkeypatch, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        _patch_fake_kaggle_input(monkeypatch, nonexistent)
        assert find_all_kaggle_input_dirs(os.path.join("src", "dataset.py")) == []


class TestValidateGitShaFile:
    VALID_SHA = "a" * 40

    def test_missing_file_always_raises(self, tmp_path):
        """No allow_unknown / allow-missing escape hatch exists on this
        function at all -- GPU-SANITY-GATE-01's canonical provenance path
        never accepts "unknown" under any circumstance, unlike
        train_kernel.py's own separate, untouched local-execution fallback."""
        with pytest.raises(RuntimeError, match="GIT_SHA.txt not found"):
            validate_git_sha_file(tmp_path / "GIT_SHA.txt")

    def test_no_allow_unknown_parameter_exists(self):
        """Structural guard: fails loud (TypeError) if a future edit
        reintroduces an allow_unknown/allow_missing-style parameter on this
        function -- the strict gate provenance path must never regain one."""
        import inspect
        params = inspect.signature(validate_git_sha_file).parameters
        assert list(params) == ["sha_file_path"], (
            f"validate_git_sha_file must have exactly one parameter "
            f"(sha_file_path), got {list(params)} -- no allow_unknown or "
            f"similar escape hatch is permitted on the gate's strict "
            f"provenance path."
        )

    def test_empty_file_raises(self, tmp_path):
        sha_file = tmp_path / "GIT_SHA.txt"
        sha_file.write_text("")
        with pytest.raises(RuntimeError, match="empty or whitespace"):
            validate_git_sha_file(sha_file)

    def test_whitespace_only_file_raises(self, tmp_path):
        sha_file = tmp_path / "GIT_SHA.txt"
        sha_file.write_text("   \n")
        with pytest.raises(RuntimeError, match="empty or whitespace"):
            validate_git_sha_file(sha_file)

    def test_too_short_sha_raises(self, tmp_path):
        sha_file = tmp_path / "GIT_SHA.txt"
        sha_file.write_text("abc123")
        with pytest.raises(RuntimeError, match="40-character"):
            validate_git_sha_file(sha_file)

    def test_uppercase_sha_raises(self, tmp_path):
        sha_file = tmp_path / "GIT_SHA.txt"
        sha_file.write_text("A" * 40)
        with pytest.raises(RuntimeError, match="40-character"):
            validate_git_sha_file(sha_file)

    def test_non_hex_chars_raise(self, tmp_path):
        sha_file = tmp_path / "GIT_SHA.txt"
        sha_file.write_text("g" * 40)
        with pytest.raises(RuntimeError, match="40-character"):
            validate_git_sha_file(sha_file)

    def test_valid_sha_returned_stripped(self, tmp_path):
        sha_file = tmp_path / "GIT_SHA.txt"
        sha_file.write_text(f"{self.VALID_SHA}\n")
        assert validate_git_sha_file(sha_file) == self.VALID_SHA


class TestVerifyImportOrigins:
    def _fake_module(self, name: str, file_path: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__file__ = file_path
        return m

    def test_passes_when_all_modules_beneath_expected_root(self, tmp_path):
        root = tmp_path / "st-act-src"
        (root / "src").mkdir(parents=True)
        mod_file = root / "src" / "dataset.py"
        mod_file.touch()
        fake_mod = self._fake_module("src.dataset", str(mod_file))
        verify_import_origins(root, [fake_mod])  # must not raise

    def test_raises_when_a_module_resolves_outside_expected_root(self, tmp_path):
        root = tmp_path / "st-act-src"
        root.mkdir()
        other = tmp_path / "some_other_checkout" / "src" / "dataset.py"
        other.parent.mkdir(parents=True)
        other.touch()
        fake_mod = self._fake_module("src.dataset", str(other))
        with pytest.raises(RuntimeError, match="NOT beneath"):
            verify_import_origins(root, [fake_mod])

    def test_raises_when_module_has_no_file_attribute(self, tmp_path):
        root = tmp_path / "st-act-src"
        root.mkdir()
        fake_mod = types.ModuleType("src.builtin_like")
        # deliberately no __file__ set
        with pytest.raises(RuntimeError, match="no __file__"):
            verify_import_origins(root, [fake_mod])

    def test_empty_module_list_raises_instead_of_vacuously_succeeding(self, tmp_path):
        """Codex review, PR #4, 2026-07-19: an empty modules list must not
        silently 'pass' -- that would defeat the entire point of this
        provenance gate for a caller that (by bug) forgot to supply its
        module list."""
        root = tmp_path / "st-act-src"
        root.mkdir()
        with pytest.raises(RuntimeError, match="empty module list"):
            verify_import_origins(root, [])
