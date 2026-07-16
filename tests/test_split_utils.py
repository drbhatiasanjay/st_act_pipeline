"""
Unit tests for the P0-2 fix: leave-one-embryo-out split generation
(scripts/build_train_val_split.py) and split-file resolution/validation
(src/split_utils.py).

Kaggle's own competition documentation (fetched live via `kaggle competitions
pages -c biohub-cell-tracking-during-development --page-name data-description
--content`) states folder names follow `{embryo_id}_{field_of_view}`, the
first underscore-delimited segment is the embryo ID, and "multiple samples may
share the same embryo." The ORIGINAL data_split.json (stratified by that
prefix, not disjoint by it) had real embryo-level leakage between train and
validation -- see the P0-2 audit and scripts/build_train_val_split.py's module
docstring. data_split.json has since been REPLACED with an exact copy of
data_splits/embryo_44b6_validation.json's lists (see
TestRootDataSplitJsonIsValidCompatibilityAlias below) so any legacy caller
that still hardcodes that filename gets a genuinely embryo-disjoint split.

All tests here are fully synthetic -- no dependency on the real competition
zip or large Kaggle data. The one exception (explicitly using the small,
already-checked-in data_split.json / data_splits/*.json to prove the
compatibility alias is correct, and to prove realistic-shape 71/128 counts)
reads small local JSON files already in the repo, not large Kaggle data.

Run: py -m pytest tests/test_split_utils.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_train_val_split import (
    build_leave_one_embryo_out_folds,
)
from scripts.build_train_val_split import (
    extract_embryo_id as generator_extract_embryo_id,
)
from src.split_utils import (
    DEFAULT_SPLIT_FILE,
    compute_membership_sha256,
    extract_embryo_id,
    get_split_identity,
    load_and_validate_split,
    resolve_split_file_path,
    validate_checkpoint_split_compatibility,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_DATA_SPLIT_JSON = os.path.join(REPO_ROOT, "data_split.json")


class TestExtractEmbryoId:
    """Test 1: embryo_id extraction."""

    def test_generator_extract_embryo_id(self):
        assert generator_extract_embryo_id("44b6_0113de3b") == "44b6"
        assert generator_extract_embryo_id("6bba_05b6850b") == "6bba"

    def test_split_utils_extract_embryo_id(self):
        assert extract_embryo_id("44b6_0113de3b") == "44b6"
        assert extract_embryo_id("6bba_05b6850b") == "6bba"


def make_synthetic_inventory(counts: dict[str, int]) -> list[str]:
    """Build a synthetic {embryo_id}_{hash} inventory with the given per-embryo
    sample counts, without touching any real data."""
    samples = []
    for embryo_id, count in counts.items():
        for i in range(count):
            samples.append(f"{embryo_id}_{i:08x}")
    return samples


class TestBuildLeaveOneEmbryoOutFolds:
    def test_deterministic_across_repeated_calls(self):
        """Test 2: generated folds are deterministic -- no shuffling, no seed."""
        samples = make_synthetic_inventory({"aaaa": 5, "bbbb": 3, "cccc": 4})
        folds_1 = build_leave_one_embryo_out_folds(samples)
        folds_2 = build_leave_one_embryo_out_folds(samples)

        assert folds_1 == folds_2

    def test_each_fold_has_no_embryo_overlap(self):
        """Test 3."""
        samples = make_synthetic_inventory({"aaaa": 5, "bbbb": 3, "cccc": 4})
        folds = build_leave_one_embryo_out_folds(samples)

        for fold_name, split_data in folds.items():
            train_embryos = {extract_embryo_id(s) for s in split_data["train"]}
            val_embryos = {extract_embryo_id(s) for s in split_data["validation"]}
            assert train_embryos & val_embryos == set(), (
                f"{fold_name}: embryo overlap {train_embryos & val_embryos}"
            )

    def test_every_input_sample_appears_exactly_once_per_fold(self):
        """Test 4."""
        samples = make_synthetic_inventory({"aaaa": 5, "bbbb": 3, "cccc": 4})
        folds = build_leave_one_embryo_out_folds(samples)

        for fold_name, split_data in folds.items():
            combined = split_data["train"] + split_data["validation"]
            assert sorted(combined) == sorted(samples), (
                f"{fold_name}: train+validation does not equal the input inventory exactly once"
            )
            assert len(combined) == len(set(combined)), f"{fold_name}: a sample appears more than once"

    def test_no_sample_appears_in_both_train_and_validation(self):
        """Test 5."""
        samples = make_synthetic_inventory({"aaaa": 5, "bbbb": 3, "cccc": 4})
        folds = build_leave_one_embryo_out_folds(samples)

        for fold_name, split_data in folds.items():
            overlap = set(split_data["train"]) & set(split_data["validation"])
            assert overlap == set(), f"{fold_name}: sample-ID overlap {overlap}"

    def test_realistic_shape_produces_expected_counts(self):
        """Test 6: current 199-sample inventory (71 x 44b6, 128 x 6bba) produces
        128/71 for 44b6-held-out and 71/128 for 6bba-held-out. Uses the real,
        small, already-checked-in data_split.json (not large Kaggle data) as
        the source of the real 199 sample IDs, so this test is tied to the
        actual audited inventory rather than a hand-picked synthetic shape."""
        if not os.path.exists(REAL_DATA_SPLIT_JSON):
            pytest.skip("data_split.json not found in this environment")

        with open(REAL_DATA_SPLIT_JSON) as f:
            old_split = json.load(f)
        all_199_samples = old_split["train"] + old_split["validation"]
        assert len(all_199_samples) == 199, "expected the real 199-sample inventory"

        folds = build_leave_one_embryo_out_folds(all_199_samples)

        assert folds["embryo_44b6_validation"]["metadata"]["train_count"] == 128
        assert folds["embryo_44b6_validation"]["metadata"]["validation_count"] == 71
        assert folds["embryo_6bba_validation"]["metadata"]["train_count"] == 71
        assert folds["embryo_6bba_validation"]["metadata"]["validation_count"] == 128

    def test_fold_json_is_serializable_and_sorted(self):
        """Sanity check on the exact contract build_train_val_split.py writes:
        lists must be sorted (no shuffling), and the whole structure must
        round-trip through json."""
        samples = make_synthetic_inventory({"bbbb": 3, "aaaa": 5})
        folds = build_leave_one_embryo_out_folds(samples)

        for split_data in folds.values():
            assert split_data["train"] == sorted(split_data["train"])
            assert split_data["validation"] == sorted(split_data["validation"])
            json.dumps(split_data)  # must not raise


class TestRootDataSplitJsonIsValidCompatibilityAlias:
    """Test 7 (revised): the root data_split.json was the OLD stratified split
    (real embryo-level leakage, both embryos present in both train and
    validation -- see the P0-2 audit). It has since been REPLACED with an exact
    copy of data_splits/embryo_44b6_validation.json's train/validation lists
    (plus a compatibility_alias_for metadata field), so any legacy caller that
    still hardcodes "data_split.json" gets a genuinely embryo-disjoint split
    rather than a leaking one. These tests prove the replacement is correct,
    not that the old (now-removed) contamination is rejected."""

    def require_real_data_split_json(self):
        if not os.path.exists(REAL_DATA_SPLIT_JSON):
            pytest.skip("data_split.json not found in this environment")

    def test_load_and_validate_split_succeeds(self):
        self.require_real_data_split_json()
        result = load_and_validate_split(Path(REAL_DATA_SPLIT_JSON))
        assert result["train"]
        assert result["validation"]

    def test_zero_embryo_overlap(self):
        self.require_real_data_split_json()
        result = load_and_validate_split(Path(REAL_DATA_SPLIT_JSON))
        train_embryos = {extract_embryo_id(s) for s in result["train"]}
        val_embryos = {extract_embryo_id(s) for s in result["validation"]}
        assert train_embryos & val_embryos == set()

    def test_train_and_validation_lists_exactly_match_primary_fold(self):
        self.require_real_data_split_json()
        primary_fold_path = Path(REPO_ROOT) / "data_splits" / "embryo_44b6_validation.json"
        if not primary_fold_path.exists():
            pytest.skip("data_splits/embryo_44b6_validation.json not found in this environment")

        root_split = load_and_validate_split(Path(REAL_DATA_SPLIT_JSON))
        primary_fold = load_and_validate_split(primary_fold_path)

        assert root_split["train"] == primary_fold["train"]
        assert root_split["validation"] == primary_fold["validation"]

    def test_counts_are_128_train_71_validation(self):
        self.require_real_data_split_json()
        result = load_and_validate_split(Path(REAL_DATA_SPLIT_JSON))
        assert len(result["train"]) == 128
        assert len(result["validation"]) == 71
        assert result["metadata"]["train_embryos"] == ["6bba"]
        assert result["metadata"]["validation_embryos"] == ["44b6"]

    def test_metadata_declares_compatibility_alias(self):
        self.require_real_data_split_json()
        with open(REAL_DATA_SPLIT_JSON) as f:
            data = json.load(f)
        assert data["metadata"]["method"] == "leave_one_embryo_out"
        assert data["metadata"]["compatibility_alias_for"] == "data_splits/embryo_44b6_validation.json"


class TestLoadAndValidateSplitGuard:
    def _write_split(self, tmp_path, train, validation, metadata=None):
        path = tmp_path / "split.json"
        data = {"train": train, "validation": validation, "metadata": metadata or {}}
        path.write_text(json.dumps(data))
        return path

    def test_empty_train_is_rejected(self, tmp_path):
        """Test 8a."""
        path = self._write_split(tmp_path, [], ["aaaa_1"])
        with pytest.raises(RuntimeError, match="train list is empty"):
            load_and_validate_split(path)

    def test_empty_validation_is_rejected(self, tmp_path):
        """Test 8b."""
        path = self._write_split(tmp_path, ["bbbb_1"], [])
        with pytest.raises(RuntimeError, match="validation list is empty"):
            load_and_validate_split(path)

    def test_duplicate_ids_in_train_are_rejected(self, tmp_path):
        """Test 9a."""
        path = self._write_split(tmp_path, ["aaaa_1", "aaaa_1"], ["bbbb_1"])
        with pytest.raises(RuntimeError, match="duplicate sample IDs in train"):
            load_and_validate_split(path)

    def test_duplicate_ids_in_validation_are_rejected(self, tmp_path):
        """Test 9b."""
        path = self._write_split(tmp_path, ["aaaa_1"], ["bbbb_1", "bbbb_1"])
        with pytest.raises(RuntimeError, match="duplicate sample IDs in validation"):
            load_and_validate_split(path)

    def test_incorrect_train_count_metadata_is_rejected(self, tmp_path):
        """Test 10a."""
        path = self._write_split(
            tmp_path, ["aaaa_1", "aaaa_2"], ["bbbb_1"],
            metadata={"train_count": 99, "validation_count": 1},
        )
        with pytest.raises(RuntimeError, match="train_count=99"):
            load_and_validate_split(path)

    def test_incorrect_validation_count_metadata_is_rejected(self, tmp_path):
        """Test 10b."""
        path = self._write_split(
            tmp_path, ["aaaa_1"], ["bbbb_1", "bbbb_2"],
            metadata={"train_count": 1, "validation_count": 99},
        )
        with pytest.raises(RuntimeError, match="validation_count=99"):
            load_and_validate_split(path)

    def test_incorrect_total_samples_metadata_is_rejected(self, tmp_path):
        """Test 10c."""
        path = self._write_split(
            tmp_path, ["aaaa_1"], ["bbbb_1"],
            metadata={"total_samples": 99},
        )
        with pytest.raises(RuntimeError, match="total_samples=99"):
            load_and_validate_split(path)

    def test_valid_disjoint_split_passes(self, tmp_path):
        """Positive case: a genuinely embryo-disjoint split must NOT raise."""
        path = self._write_split(
            tmp_path, ["aaaa_1", "aaaa_2"], ["bbbb_1"],
            metadata={"train_count": 2, "validation_count": 1, "total_samples": 3, "method": "leave_one_embryo_out"},
        )
        result = load_and_validate_split(path)
        assert result["train"] == ["aaaa_1", "aaaa_2"]
        assert result["validation"] == ["bbbb_1"]

    def test_full_inventory_completeness_check(self, tmp_path):
        path = self._write_split(tmp_path, ["aaaa_1"], ["bbbb_1"])
        # Missing "cccc_1" from the expected full inventory -> must raise.
        with pytest.raises(RuntimeError, match="does not cover the full expected inventory"):
            load_and_validate_split(path, full_inventory=["aaaa_1", "bbbb_1", "cccc_1"])

        # Exact match -> must NOT raise.
        load_and_validate_split(path, full_inventory=["aaaa_1", "bbbb_1"])

    def test_missing_split_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_validate_split(tmp_path / "does_not_exist.json")

    def test_embryo_appearing_in_both_train_and_validation_is_rejected(self, tmp_path):
        """P0-2 direct regression test (2026-07-16): exercises the
        embryo-disjointness guard directly against a hand-built synthetic
        split where embryo 44b6 has samples on both sides, rather than only
        checking that the generated leave-one-embryo-out folds happen to be
        disjoint. This is the exact leakage shape the original (pre-P0-2)
        data_split.json had."""
        path = self._write_split(
            tmp_path,
            train=["44b6_train_fov", "6bba_train_fov"],
            validation=["44b6_validation_fov"],
        )
        with pytest.raises(RuntimeError, match="44b6"):
            load_and_validate_split(path)


class TestLoadAndValidateSplitMetadataConsistency:
    """P0-2 fix (2026-07-16): load_and_validate_split() now cross-checks
    metadata['group_key'], ['train_embryos'], ['validation_embryos'], and
    ['per_embryo_counts'] against what's actually derivable from the train/
    validation sample-ID lists -- so a hand-edited or stale metadata block can
    never silently disagree with the real split contents. Each field is only
    checked when present (matches the existing train_count/validation_count/
    total_samples pattern)."""

    VALID_TRAIN = ["aaaa_1", "aaaa_2"]
    VALID_VALIDATION = ["bbbb_1"]

    def _valid_metadata(self) -> dict:
        return {
            "group_key": "sample_id.split('_', 1)[0]",
            "train_embryos": ["aaaa"],
            "validation_embryos": ["bbbb"],
            "per_embryo_counts": {
                "aaaa": {"total": 2, "in_train": 2, "in_validation": 0},
                "bbbb": {"total": 1, "in_train": 0, "in_validation": 1},
            },
        }

    def _write(self, tmp_path, metadata):
        path = tmp_path / "split.json"
        data = {"train": self.VALID_TRAIN, "validation": self.VALID_VALIDATION, "metadata": metadata}
        path.write_text(json.dumps(data))
        return path

    def test_fully_consistent_metadata_passes(self, tmp_path):
        path = self._write(tmp_path, self._valid_metadata())
        result = load_and_validate_split(path)
        assert result["train"] == self.VALID_TRAIN

    def test_incorrect_group_key_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["group_key"] = "sample_id[:4]"
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match="group_key"):
            load_and_validate_split(path)

    def test_incorrect_train_embryos_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["train_embryos"] = ["wrong"]
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match="train_embryos"):
            load_and_validate_split(path)

    def test_incorrect_validation_embryos_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["validation_embryos"] = ["wrong"]
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match="validation_embryos"):
            load_and_validate_split(path)

    def test_incorrect_per_embryo_total_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["per_embryo_counts"]["aaaa"]["total"] = 99
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match=r"\['total'\]=99"):
            load_and_validate_split(path)

    def test_incorrect_in_train_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["per_embryo_counts"]["aaaa"]["in_train"] = 99
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match=r"\['in_train'\]=99"):
            load_and_validate_split(path)

    def test_incorrect_in_validation_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["per_embryo_counts"]["bbbb"]["in_validation"] = 99
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match=r"\['in_validation'\]=99"):
            load_and_validate_split(path)

    def test_missing_embryo_entry_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        del metadata["per_embryo_counts"]["bbbb"]
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match="missing entries"):
            load_and_validate_split(path)

    def test_unexpected_embryo_entry_is_rejected(self, tmp_path):
        metadata = self._valid_metadata()
        metadata["per_embryo_counts"]["cccc"] = {"total": 1, "in_train": 1, "in_validation": 0}
        path = self._write(tmp_path, metadata)
        with pytest.raises(RuntimeError, match="do not appear in either"):
            load_and_validate_split(path)


class TestSplitFileResolution:
    def test_env_var_selects_requested_fold(self, monkeypatch, tmp_path):
        """Test 11."""
        monkeypatch.setenv("ST_ACT_SPLIT_FILE", "data_splits/embryo_6bba_validation.json")
        resolved = resolve_split_file_path(repo_root=tmp_path)
        assert resolved == tmp_path / "data_splits" / "embryo_6bba_validation.json"

    def test_default_selection_uses_embryo_44b6_validation(self, monkeypatch, tmp_path):
        """Test 12."""
        monkeypatch.delenv("ST_ACT_SPLIT_FILE", raising=False)
        assert DEFAULT_SPLIT_FILE == "data_splits/embryo_44b6_validation.json"
        resolved = resolve_split_file_path(repo_root=tmp_path)
        assert resolved == tmp_path / "data_splits" / "embryo_44b6_validation.json"

    def test_absolute_path_env_var_used_as_is(self, monkeypatch, tmp_path):
        abs_path = tmp_path / "somewhere" / "custom_split.json"
        monkeypatch.setenv("ST_ACT_SPLIT_FILE", str(abs_path))
        resolved = resolve_split_file_path(repo_root=tmp_path / "unrelated_root")
        assert resolved == abs_path

    def test_kaggle_src_dataset_dir_takes_precedence_over_repo_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ST_ACT_SPLIT_FILE", raising=False)
        kaggle_dir = tmp_path / "kaggle_input" / "st-act-src"
        resolved = resolve_split_file_path(repo_root=tmp_path / "local_root", kaggle_src_dataset_dir=kaggle_dir)
        assert resolved == kaggle_dir / "data_splits" / "embryo_44b6_validation.json"


class TestComputeMembershipSha256:
    """P0-2 checkpoint/split-identity fix (2026-07-16): compute_membership_sha256()
    is the canonical fingerprint checkpoints embed and evaluation-time code
    compares against (see validate_checkpoint_split_compatibility below)."""

    def test_deterministic_for_same_membership(self):
        h1 = compute_membership_sha256(["a_1", "a_2"], ["b_1"])
        h2 = compute_membership_sha256(["a_1", "a_2"], ["b_1"])
        assert h1 == h2

    def test_independent_of_input_list_order(self):
        h1 = compute_membership_sha256(["a_2", "a_1"], ["b_1"])
        h2 = compute_membership_sha256(["a_1", "a_2"], ["b_1"])
        assert h1 == h2

    def test_differs_when_a_sample_moves_between_train_and_validation(self):
        h_before = compute_membership_sha256(["a_1", "a_2"], ["b_1"])
        h_after = compute_membership_sha256(["a_1"], ["a_2", "b_1"])
        assert h_before != h_after

    def test_differs_when_a_sample_is_added(self):
        h_before = compute_membership_sha256(["a_1"], ["b_1"])
        h_after = compute_membership_sha256(["a_1", "a_2"], ["b_1"])
        assert h_before != h_after

    def test_is_a_valid_sha256_hex_digest(self):
        h = compute_membership_sha256(["a_1"], ["b_1"])
        assert len(h) == 64
        int(h, 16)  # must not raise -- valid hex


class TestGetSplitIdentity:
    def _write_split(self, tmp_path, train, validation):
        path = tmp_path / "split.json"
        path.write_text(json.dumps({"train": train, "validation": validation, "metadata": {}}))
        return path

    def test_matches_compute_membership_sha256_directly(self, tmp_path):
        path = self._write_split(tmp_path, ["aaaa_1", "aaaa_2"], ["bbbb_1"])
        identity = get_split_identity(path)
        assert identity == compute_membership_sha256(["aaaa_1", "aaaa_2"], ["bbbb_1"])

    def test_real_primary_fold_identity_matches_its_own_metadata(self):
        primary_fold_path = Path(REPO_ROOT) / "data_splits" / "embryo_44b6_validation.json"
        if not primary_fold_path.exists():
            pytest.skip("data_splits/embryo_44b6_validation.json not found in this environment")
        with open(primary_fold_path) as f:
            data = json.load(f)
        assert get_split_identity(primary_fold_path) == data["metadata"]["membership_sha256"]


class TestLoadAndValidateSplitMembershipSha256:
    def _write(self, tmp_path, train, validation, membership_sha256):
        path = tmp_path / "split.json"
        data = {
            "train": train, "validation": validation,
            "metadata": {"membership_sha256": membership_sha256},
        }
        path.write_text(json.dumps(data))
        return path

    def test_correct_membership_sha256_passes(self, tmp_path):
        train, validation = ["aaaa_1"], ["bbbb_1"]
        correct_hash = compute_membership_sha256(train, validation)
        path = self._write(tmp_path, train, validation, correct_hash)
        result = load_and_validate_split(path)
        assert result["train"] == train

    def test_incorrect_membership_sha256_is_rejected(self, tmp_path):
        path = self._write(tmp_path, ["aaaa_1"], ["bbbb_1"], "0" * 64)
        with pytest.raises(RuntimeError, match="membership_sha256"):
            load_and_validate_split(path)

    def test_real_data_split_json_membership_sha256_is_internally_consistent(self):
        if not os.path.exists(REAL_DATA_SPLIT_JSON):
            pytest.skip("data_split.json not found in this environment")
        # load_and_validate_split() would already raise if this were wrong --
        # this test asserts it explicitly rather than only relying on the
        # absence of an exception.
        with open(REAL_DATA_SPLIT_JSON) as f:
            data = json.load(f)
        expected = compute_membership_sha256(data["train"], data["validation"])
        assert data["metadata"]["membership_sha256"] == expected
        load_and_validate_split(Path(REAL_DATA_SPLIT_JSON))  # must not raise


class TestValidateCheckpointSplitCompatibility:
    """P0-2 checkpoint/split-identity fix (2026-07-16): a checkpoint's
    val_score was selected/early-stopped against one specific held-out
    embryo -- evaluating it against a DIFFERENT split silently produces a
    meaningless score. This guard catches that before it happens."""

    ACTIVE_IDENTITY = "a" * 64
    ACTIVE_PATH = Path("data_splits/embryo_44b6_validation.json")

    def test_matching_identity_does_not_raise(self):
        checkpoint = {"split_membership_sha256": self.ACTIVE_IDENTITY}
        validate_checkpoint_split_compatibility(checkpoint, self.ACTIVE_IDENTITY, self.ACTIVE_PATH)

    def test_missing_identity_warns_but_does_not_raise(self, caplog):
        checkpoint = {}  # legacy checkpoint, predates this fix
        with caplog.at_level("WARNING"):
            validate_checkpoint_split_compatibility(checkpoint, self.ACTIVE_IDENTITY, self.ACTIVE_PATH)
        assert any("no saved split_membership_sha256" in r.message for r in caplog.records)

    def test_mismatched_identity_raises_by_default(self):
        checkpoint = {"split_membership_sha256": "b" * 64}
        with pytest.raises(RuntimeError, match="different embryo-disjoint folds"):
            validate_checkpoint_split_compatibility(checkpoint, self.ACTIVE_IDENTITY, self.ACTIVE_PATH)

    def test_mismatched_identity_with_allow_mismatch_warns_instead_of_raising(self, caplog):
        checkpoint = {"split_membership_sha256": "b" * 64}
        with caplog.at_level("WARNING"):
            validate_checkpoint_split_compatibility(
                checkpoint, self.ACTIVE_IDENTITY, self.ACTIVE_PATH, allow_mismatch=True,
            )
        assert any("does NOT match" in r.message for r in caplog.records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
