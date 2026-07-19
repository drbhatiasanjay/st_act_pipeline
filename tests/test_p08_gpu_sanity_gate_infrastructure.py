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

Wave 2 (Sections B-C): src/targets.py's generate_edge_targets() GT-topology-only
hard/easy negative edge split (no dependence on model logits anywhere), and
src/train.py's TrainingLoop gradient-snapshot/edge-supervision-counter
instrumentation in train_epoch().

Run: py -m pytest tests/test_p08_gpu_sanity_gate_infrastructure.py -v
"""
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.targets as targets_module
import src.train as train_module
from src.dataset import CompetitionDataset
from src.deployment_provenance import (
    find_all_kaggle_input_dirs,
    find_exactly_one_kaggle_input_dir,
    validate_git_sha_file,
    verify_import_origins,
)
from src.targets import generate_edge_targets
from src.train import TrainingLoop

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


# ---------------------------------------------------------------------------
# Section B -- src/targets.py: generate_edge_targets() hard/easy negative
# split (Wave 2, GT-topology only, no model logits anywhere)
# ---------------------------------------------------------------------------

class _FakeEdgeGraph:
    """Minimal stand-in for the loaded .geff ground-truth graph --
    implements only what generate_edge_targets() actually calls:
    node_attrs(attr_keys=...), dividing_nodes(), has_edge(src, tgt)."""

    def __init__(self, gt_rows, edges=(), dividing=()):
        self._gt_rows = gt_rows
        self._edges = set(edges)
        self._dividing = set(dividing)

    def node_attrs(self, attr_keys=None):
        import polars as pl
        return pl.DataFrame(self._gt_rows)

    def dividing_nodes(self):
        return self._dividing

    def has_edge(self, src, tgt):
        return (src, tgt) in self._edges


def _patch_geff_load(monkeypatch, graph):
    monkeypatch.setattr(targets_module, "load_geff_cached", lambda path, cache: (graph, object()))


class TestEdgeHardEasyNegativeSplit:
    def _three_way_fixture(self):
        """2x2 candidate grid engineered to produce all three categories at
        once: (A,C) positive, (B,C) hard negative (both matched, no GT edge),
        (A,D) and (B,D) easy negative (D unmatched at t+1)."""
        gt_rows = [
            {"t": 0, "node_id": "gA", "z": 0.0, "y": 0.0, "x": 0.0},
            {"t": 0, "node_id": "gB", "z": 0.0, "y": 10.0, "x": 10.0},
            {"t": 1, "node_id": "gC", "z": 0.0, "y": 0.0, "x": 0.0},
        ]
        graph = _FakeEdgeGraph(gt_rows, edges={("gA", "gC")})
        nodes_t = torch.tensor([[0.0, 0.0, 0.0], [0.0, 10.0, 10.0]])       # A, B
        nodes_t1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 500.0, 500.0]])    # C, D (D unmatched)
        return graph, nodes_t, nodes_t1

    def test_three_way_partition_matches_expected_composition(self, monkeypatch):
        graph, nodes_t, nodes_t1 = self._three_way_fixture()
        _patch_geff_load(monkeypatch, graph)

        edge_labels, metadata = generate_edge_targets(
            sample_id="fake", geff_path="unused.geff",
            nodes_t=nodes_t, nodes_t1=nodes_t1, t=0,
        )

        # row-major (i,j): (A,C) (A,D) (B,C) (B,D)
        assert edge_labels.tolist() == [1, 0, 0, 0]
        assert metadata["hard_negative_mask"].tolist() == [False, False, True, False]
        assert metadata["easy_negative_mask"].tolist() == [False, True, False, True]
        assert metadata["num_positive_edges"] == 1
        assert metadata["num_hard_negative_edges"] == 1
        assert metadata["num_easy_negative_edges"] == 2
        assert metadata["num_negative_edges"] == 3

    def test_masks_aligned_no_overlap_union_equals_negatives_never_include_positive(self, monkeypatch):
        graph, nodes_t, nodes_t1 = self._three_way_fixture()
        _patch_geff_load(monkeypatch, graph)

        edge_labels, metadata = generate_edge_targets(
            sample_id="fake", geff_path="unused.geff",
            nodes_t=nodes_t, nodes_t1=nodes_t1, t=0,
        )
        hard = metadata["hard_negative_mask"]
        easy = metadata["easy_negative_mask"]

        assert hard.shape == edge_labels.shape
        assert easy.shape == edge_labels.shape
        assert not (hard & easy).any(), "hard/easy negative masks must never overlap"
        assert torch.equal(hard | easy, edge_labels == 0), "union of both masks must equal exactly the negatives"
        assert not (hard & (edge_labels == 1)).any(), "hard_negative_mask must never include a positive edge"
        assert not (easy & (edge_labels == 1)).any(), "easy_negative_mask must never include a positive edge"
        assert metadata["num_hard_negative_edges"] == int(hard.sum().item())
        assert metadata["num_easy_negative_edges"] == int(easy.sum().item())

    def test_zero_hard_negatives_is_legitimate_not_raised(self, monkeypatch):
        """A scenario with no both-matched-but-unconnected pairs at all --
        num_hard_negative_edges must be exactly 0, never raised, never
        substituted for."""
        gt_rows = [
            {"t": 0, "node_id": "gA", "z": 0.0, "y": 0.0, "x": 0.0},
            {"t": 1, "node_id": "gC", "z": 0.0, "y": 0.0, "x": 0.0},
        ]
        graph = _FakeEdgeGraph(gt_rows, edges={("gA", "gC")})
        _patch_geff_load(monkeypatch, graph)
        nodes_t = torch.tensor([[0.0, 0.0, 0.0]])          # matches gA
        nodes_t1 = torch.tensor([[0.0, 500.0, 500.0]])      # unmatched (far from gC)

        edge_labels, metadata = generate_edge_targets(
            sample_id="fake", geff_path="unused.geff",
            nodes_t=nodes_t, nodes_t1=nodes_t1, t=0,
        )

        assert metadata["num_hard_negative_edges"] == 0
        assert metadata["hard_negative_mask"].tolist() == [False]
        assert metadata["num_easy_negative_edges"] == 1

    def test_empty_node_set_returns_empty_masks_and_zero_counts(self, monkeypatch):
        graph = _FakeEdgeGraph(gt_rows=[])
        _patch_geff_load(monkeypatch, graph)

        edge_labels, metadata = generate_edge_targets(
            sample_id="fake", geff_path="unused.geff",
            nodes_t=torch.zeros((0, 3)), nodes_t1=torch.zeros((3, 3)), t=0,
        )

        assert metadata["hard_negative_mask"].shape == (0,)
        assert metadata["easy_negative_mask"].shape == (0,)
        assert metadata["num_hard_negative_edges"] == 0
        assert metadata["num_easy_negative_edges"] == 0


# ---------------------------------------------------------------------------
# Section C -- src/train.py: TrainingLoop gradient-snapshot capture and
# edge-supervision counters (Wave 2)
# ---------------------------------------------------------------------------

class TestComputeParamGradNorm:
    def test_returns_none_when_no_gradients_present(self):
        p = nn.Parameter(torch.ones(3))  # never used in a backward pass
        assert TrainingLoop._compute_param_grad_norm([p]) is None

    def test_computes_l2_norm_across_params(self):
        p1 = nn.Parameter(torch.zeros(2))
        p2 = nn.Parameter(torch.zeros(2))
        p1.grad = torch.tensor([3.0, 0.0])
        p2.grad = torch.tensor([4.0, 0.0])
        # L2 norm of concatenated grads [3, 0, 4, 0] == 5.0
        assert TrainingLoop._compute_param_grad_norm([p1, p2]) == pytest.approx(5.0)

    def test_ignores_params_with_no_grad(self):
        p1 = nn.Parameter(torch.zeros(1))
        p2 = nn.Parameter(torch.zeros(1))
        p1.grad = torch.tensor([3.0])
        # p2.grad left None -- must be ignored, not treated as zero-contribution
        # in a way that changes None-vs-real-value semantics for the caller.
        assert TrainingLoop._compute_param_grad_norm([p1, p2]) == pytest.approx(3.0)


def _make_grad_harness(monkeypatch, *, generate_edge_targets_return):
    """Minimal TrainingLoop (via __new__ bypass) whose train_epoch() runs ONE
    real backward() pass through a tiny real nn.Module graph -- unlike
    test_p07_training_integrity.py's harness (which deliberately multiplies
    fake output by 0.0*self.p so gradients are always exactly zero by
    construction, fine for THAT file's purposes), this harness needs REAL,
    generally-nonzero gradients on both unet3d and transformer parameters to
    exercise the gradient-snapshot capture meaningfully."""
    device = torch.device("cpu")
    z, y, x = 4, 4, 4

    class _RealGradUNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1))

        def forward(self, x_in):
            return torch.zeros(1, 2, z, y, x) + self.p, torch.zeros(1, 8, z, y, x)

    class _RealGradTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1))

        def forward(self, nodes_t, nodes_t1, features_t, features_t1):
            n = nodes_t.shape[0] * nodes_t1.shape[0]
            return self.p.expand(n).clone()

    unet3d = _RealGradUNet3D()
    transformer = _RealGradTransformer()

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.unet3d = unet3d
    loop.transformer = transformer
    loop.device = device
    loop.data_dir = Path("unused")
    loop.hyperparams = {
        'heatmap_loss_weight': 1.0, 'grad_clip': 1.0, 'warmup_steps': 0,
        'max_batches_per_epoch': None,
    }
    loop.optimizer = torch.optim.AdamW(
        list(unet3d.parameters()) + list(transformer.parameters()), lr=1e-4,
    )
    loop._amp_enabled = False
    loop.scaler = torch.amp.GradScaler('cpu', enabled=False)
    # Real (not disconnected-constant) losses so backward() actually populates
    # .grad on both fake modules' parameters via a real computation graph.
    loop.detection_loss_fn = lambda logits, target: (logits ** 2).mean()
    loop.division_loss_fn = lambda logits, targets, mask: (logits ** 2).mean()
    loop.epoch_fallback_counts = {
        'heatmap_generation_failure': 0,
        'edge_target_generation_failure': 0,
        'edge_loss_computation_failure': 0,
        'evaluation_failure': 0,
        'gt_node_load_failure': 0,
        'retained_pair_zero_gt_nodes_failure': 0,
    }
    loop.epoch_biological_zero_counts = {
        'legitimate_zero_positive_edge_batches': 0,
        'edge_supervised_batches_total': 0,
        'edge_supervised_batches_with_nonzero_transformer_grad': 0,
    }
    loop.last_unet_gradient_snapshot = None
    loop.last_transformer_gradient_snapshot = None
    loop._global_step = 0
    loop._geff_cache = {}
    loop.last_epoch_wall_clock_seconds = 0.0
    loop.last_epoch_num_batches = 0
    loop.progress_file = None

    batch = {
        "frame_t": torch.zeros(1, 1, z, y, x),
        "frame_t1": torch.zeros(1, 1, z, y, x),
        "sample_id": ["fake_sample"],
        "t_idx": torch.tensor([0]),
    }
    loop.train_loader = [batch]

    monkeypatch.setattr(
        loop, "_generate_and_validate_heatmap_target",
        lambda sample_id, t_idx, volume_shape, z_, y_, x_: torch.zeros(1, 2, z_, y_, x_),
    )
    monkeypatch.setattr(
        loop, "_get_gt_nodes",
        lambda sample_id, t_idx: torch.tensor([[1.0, 1.0, 1.0]]),
    )
    monkeypatch.setattr(train_module, "generate_edge_targets", lambda *a, **kw: generate_edge_targets_return)

    return loop


class TestGradientSnapshotAndEdgeSupervisionCounters:
    def test_unet_snapshot_captured_and_transformer_counters_fire_on_positive_edges(self, monkeypatch):
        edge_labels = torch.tensor([1, 0])  # at least one positive edge
        metadata = {'division_mask': torch.zeros(2, dtype=torch.bool)}
        loop = _make_grad_harness(monkeypatch, generate_edge_targets_return=(edge_labels, metadata))

        loop.train_epoch()

        assert loop.last_unet_gradient_snapshot is not None
        assert loop.last_unet_gradient_snapshot > 0.0
        assert loop.last_transformer_gradient_snapshot is not None
        assert loop.last_transformer_gradient_snapshot > 0.0
        assert loop.epoch_biological_zero_counts['edge_supervised_batches_total'] == 1
        assert loop.epoch_biological_zero_counts['edge_supervised_batches_with_nonzero_transformer_grad'] == 1
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 0

    def test_all_negative_batch_counts_supervision_without_transformer_snapshot(self, monkeypatch):
        """Rule D legitimate-zero-positive-edge batch: UNet gradient must
        still be captured (detection loss is independent of edge positivity),
        but the transformer gradient snapshot/nonzero counter must NOT be
        touched. edge_supervised_batches_total still increments because the
        valid all-negative targets train false-edge rejection."""
        edge_labels = torch.tensor([0, 0])  # all-negative -- no positive edges at all
        metadata = {'division_mask': torch.zeros(2, dtype=torch.bool)}
        loop = _make_grad_harness(monkeypatch, generate_edge_targets_return=(edge_labels, metadata))

        loop.train_epoch()

        assert loop.last_unet_gradient_snapshot is not None
        assert loop.last_unet_gradient_snapshot > 0.0
        assert loop.last_transformer_gradient_snapshot is None
        assert loop.epoch_biological_zero_counts['edge_supervised_batches_total'] == 1
        assert loop.epoch_biological_zero_counts['edge_supervised_batches_with_nonzero_transformer_grad'] == 0
        assert loop.epoch_biological_zero_counts['legitimate_zero_positive_edge_batches'] == 1

    def test_snapshots_are_unscaled_before_capture(self, monkeypatch):
        class _ScaleByEight:
            def scale(self, loss):
                return loss * 8.0

            def unscale_(self, optimizer):
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is not None:
                            parameter.grad.div_(8.0)

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        edge_labels = torch.tensor([1, 0])
        metadata = {'division_mask': torch.zeros(2, dtype=torch.bool)}
        loop = _make_grad_harness(monkeypatch, generate_edge_targets_return=(edge_labels, metadata))
        loop.scaler = _ScaleByEight()

        loop.train_epoch()

        # Each tiny module contributes d(p**2)/dp == 2 at p == 1. Capturing
        # before unscale_ would incorrectly report 16 instead.
        assert loop.last_unet_gradient_snapshot == pytest.approx(2.0)
        assert loop.last_transformer_gradient_snapshot == pytest.approx(2.0)

    def test_transformer_snapshot_resets_when_next_epoch_has_no_positive_edges(self, monkeypatch):
        positive = (
            torch.tensor([1, 0]),
            {'division_mask': torch.zeros(2, dtype=torch.bool)},
        )
        loop = _make_grad_harness(monkeypatch, generate_edge_targets_return=positive)
        loop.train_epoch()
        assert loop.last_transformer_gradient_snapshot is not None

        all_negative = (
            torch.tensor([0, 0]),
            {'division_mask': torch.zeros(2, dtype=torch.bool)},
        )
        monkeypatch.setattr(
            train_module,
            "generate_edge_targets",
            lambda *args, **kwargs: all_negative,
        )
        loop.train_epoch()

        assert loop.last_unet_gradient_snapshot is not None
        assert loop.last_transformer_gradient_snapshot is None
        assert loop.epoch_biological_zero_counts['edge_supervised_batches_total'] == 1
        assert loop.epoch_biological_zero_counts['edge_supervised_batches_with_nonzero_transformer_grad'] == 0
