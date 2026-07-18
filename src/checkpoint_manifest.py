"""
Verified checkpoint manifest: schema, discovery, deployment eligibility,
transactional writing, and fail-closed loading (P0-6).

A checkpoint alone (epoch_N_val_score_X.XXXX.pt) carries no verifiable proof
of what produced it -- production inference (generate_submission.py,
kaggle_kernel_inference/inference_kernel.py) previously selected a checkpoint
by filename pattern or newest-mtime, with no way to confirm the mounted
source code actually matches what trained it, that validation was a genuine
full-fold pass (not a capped or structural-zero run), or that the checkpoint
bytes on disk match what was actually saved. This module makes checkpoint
deployment fail-closed: a manifest is only ever created for a checkpoint that
passed every deployment-eligibility check (see deployment_eligibility_errors),
and loading a checkpoint for production inference (load_verified_checkpoint)
independently re-verifies every one of those conditions plus a byte-exact
hash match before torch.load() ever runs.
"""

import errno
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

MODEL_CONTRACT = "edge_logits_v1"
CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "checkpoint_manifest.json"

_REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "checkpoint_file",
    "checkpoint_sha256",
    "training_code_sha",
    "split_membership_sha256",
    "model_contract",
    "epoch",
    "validation_is_full_fold",
    "validation_samples_evaluated",
    "validation_samples_total",
    "num_datasets",
    "predicted_nodes_total",
    "predicted_edges_total",
    "is_structural_zero",
    "adjusted_edge_jaccard",
}

_REQUIRED_CHECKPOINT_KEYS = {
    "unet3d_state_dict",
    "transformer_state_dict",
    "hyperparams",
    "epoch",
    "checkpoint_schema_version",
    "training_code_sha",
    "model_contract",
    "split_membership_sha256",
    "val_metrics",
}


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 hex digest (lowercase) of a file's exact bytes."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint_file(checkpoint: dict, checkpoint_path: str | Path) -> None:
    """
    Save checkpoint (a torch-serializable dict) to checkpoint_path, then
    flush and fsync the underlying file descriptor before returning.

    Required before any caller computes this file's SHA-256 or otherwise
    treats it as durably on disk (Part B5's "flush/fsync where supported"
    step) -- torch.save() only guarantees the bytes reached the OS's page
    cache via a Python-level close/flush, not that they are durably written
    to the physical storage device. Hashing (or manifesting) a checkpoint
    immediately after a bare torch.save() with no fsync risks the manifest
    referencing a hash that a subsequent crash could leave inconsistent with
    what actually landed on disk.

    fsync failures are re-raised UNLESS the OS reports the operation is
    genuinely unsupported on this filesystem/platform (errno.ENOTSUP, or the
    Windows-specific case where fsync is reported invalid for the underlying
    handle) -- that narrow, documented case is logged and treated as
    non-fatal; every other fsync failure (disk full, I/O error, etc.) must
    propagate, never be silently swallowed.
    """
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError as e:
            if e.errno in (errno.ENOTSUP, errno.EINVAL):
                logger.warning(
                    f"fsync is not supported for {checkpoint_path} on this platform/"
                    f"filesystem ({e}) -- continuing without it."
                )
            else:
                raise


def _is_lowercase_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(c in "0123456789abcdef" for c in value)


def _is_exact_schema_version(value: Any, expected: int) -> bool:
    """True only if value is a genuine int equal to expected -- Python's `==`
    treats bool as a subtype of int (True == 1) and freely compares int/float
    (1.0 == 1), so a bare `value == expected` equality check would wrongly
    accept True, 1.0, or other bool/float lookalikes as a valid schema
    version. Both schema_version fields are frozen fail-closed contract
    values, so the type itself must be checked, not just numeric equality."""
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"Duplicate JSON key in manifest: {k!r}")
        d[k] = v
    return d


def _parse_manifest_bytes(raw_bytes: bytes) -> dict:
    """Strict manifest parse: rejects invalid UTF-8/JSON, duplicate keys,
    missing fields, and unexpected fields. Does not validate field values --
    see _validate_manifest_semantics for that."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Manifest is not valid UTF-8: {e}") from e

    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise ValueError(f"Manifest is not valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"Manifest top level must be a JSON object, got {type(parsed).__name__}")

    actual_keys = set(parsed.keys())
    missing = _REQUIRED_MANIFEST_FIELDS - actual_keys
    unexpected = actual_keys - _REQUIRED_MANIFEST_FIELDS
    if missing:
        raise ValueError(f"Manifest missing required field(s): {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Manifest has unexpected field(s): {sorted(unexpected)}")

    return parsed


def _validate_manifest_semantics(manifest: dict) -> None:
    """Full schema/semantic validation of an already key-set-verified manifest dict."""
    if not _is_exact_schema_version(manifest.get("schema_version"), MANIFEST_SCHEMA_VERSION):
        raise ValueError(
            f"manifest['schema_version'] must be exactly the int {MANIFEST_SCHEMA_VERSION} "
            f"(not bool, not float, not str), got {manifest.get('schema_version')!r}"
        )

    checkpoint_file = manifest.get("checkpoint_file")
    if not isinstance(checkpoint_file, str) or checkpoint_file == "":
        raise ValueError(f"manifest['checkpoint_file'] must be a non-empty string, got {checkpoint_file!r}")
    if os.path.isabs(checkpoint_file):
        raise ValueError(f"manifest['checkpoint_file'] must not be an absolute path: {checkpoint_file!r}")
    if "/" in checkpoint_file or "\\" in checkpoint_file or ".." in checkpoint_file:
        raise ValueError(
            f"manifest['checkpoint_file'] must be a bare filename with no path "
            f"separators or '..': {checkpoint_file!r}"
        )
    if not checkpoint_file.endswith(".pt"):
        raise ValueError(f"manifest['checkpoint_file'] must end with .pt: {checkpoint_file!r}")

    if not _is_lowercase_hex(manifest.get("checkpoint_sha256"), 64):
        raise ValueError(
            f"manifest['checkpoint_sha256'] must be 64-character lowercase hex, "
            f"got {manifest.get('checkpoint_sha256')!r}"
        )
    if not _is_lowercase_hex(manifest.get("training_code_sha"), 40):
        raise ValueError(
            f"manifest['training_code_sha'] must be 40-character lowercase hex, "
            f"got {manifest.get('training_code_sha')!r}"
        )
    if not _is_lowercase_hex(manifest.get("split_membership_sha256"), 64):
        raise ValueError(
            f"manifest['split_membership_sha256'] must be 64-character lowercase hex, "
            f"got {manifest.get('split_membership_sha256')!r}"
        )

    if manifest.get("model_contract") != MODEL_CONTRACT:
        raise ValueError(
            f"manifest['model_contract'] must be {MODEL_CONTRACT!r}, got "
            f"{manifest.get('model_contract')!r}"
        )

    epoch = manifest.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError(f"manifest['epoch'] must be a positive non-bool int, got {epoch!r}")

    if manifest.get("validation_is_full_fold") is not True:
        raise ValueError(
            f"manifest['validation_is_full_fold'] must be True, got "
            f"{manifest.get('validation_is_full_fold')!r}"
        )

    def _positive_int(key: str) -> int:
        val = manifest.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ValueError(f"manifest['{key}'] must be a positive non-bool int, got {val!r}")
        return val

    evaluated = _positive_int("validation_samples_evaluated")
    total = _positive_int("validation_samples_total")
    if evaluated != total:
        raise ValueError(
            f"manifest['validation_samples_evaluated']={evaluated} must equal "
            f"manifest['validation_samples_total']={total}"
        )

    num_datasets = _positive_int("num_datasets")
    if num_datasets != evaluated:
        raise ValueError(
            f"manifest['num_datasets']={num_datasets} must equal "
            f"manifest['validation_samples_evaluated']={evaluated}"
        )

    _positive_int("predicted_nodes_total")
    _positive_int("predicted_edges_total")

    if manifest.get("is_structural_zero") is not False:
        raise ValueError(
            f"manifest['is_structural_zero'] must be False, got "
            f"{manifest.get('is_structural_zero')!r}"
        )

    adj = manifest.get("adjusted_edge_jaccard")
    if isinstance(adj, bool) or not isinstance(adj, int | float):
        raise ValueError(
            f"manifest['adjusted_edge_jaccard'] must be a finite non-negative number, "
            f"got {adj!r}"
        )
    if not math.isfinite(float(adj)) or float(adj) < 0:
        raise ValueError(
            f"manifest['adjusted_edge_jaccard'] must be a finite non-negative number, "
            f"got {adj!r}"
        )


def find_single_manifest(root: str | Path, max_depth: int = 5) -> Path:
    """Search beneath root (bounded by max_depth) for exactly one file named
    checkpoint_manifest.json. Raises when zero or more than one is found --
    never selects by first result, directory order, mtime, or checkpoint
    filename."""
    root = Path(root)
    root_str = str(root)
    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_str):
        depth = dirpath[len(root_str):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        if MANIFEST_FILENAME in filenames:
            candidates.append(Path(dirpath) / MANIFEST_FILENAME)

    if not candidates:
        raise RuntimeError(f"No {MANIFEST_FILENAME} found beneath {root} (max_depth={max_depth}).")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple {MANIFEST_FILENAME} files found beneath {root}, exactly one is "
            f"required: {sorted(str(c) for c in candidates)}"
        )
    return candidates[0]


def read_active_manifest_checkpoint_path(manifest_dir: str | Path) -> Path | None:
    """
    Fail-closed lookup of the checkpoint currently protected by the active
    manifest in manifest_dir. Returns None only when no manifest file exists
    yet (a legitimate first-checkpoint state) -- otherwise returns the
    resolved Path of the checkpoint the active manifest references.

    Raises (ValueError / FileNotFoundError) on ANY malformed, invalid, or
    ambiguous manifest state: malformed JSON, duplicate JSON keys, unknown or
    missing fields, invalid field values, an unsafe checkpoint filename
    (absolute path, path separators, '..', wrong suffix), a symlinked
    checkpoint path, a directory-escaping checkpoint path, or a referenced
    checkpoint file that does not exist.

    This is the single shared helper production code must use to determine
    "what checkpoint must survive cleanup" -- callers must never catch these
    exceptions and substitute a warning-and-proceed fallback (doing so risks
    deleting the checkpoint an ambiguous/corrupted manifest was still
    protecting). It deliberately does NOT verify the checkpoint file's own
    SHA-256 against the manifest -- that full trust verification belongs to
    load_verified_checkpoint() for the (expensive, whole-file-read) production
    inference-loading path; this helper only needs to safely identify which
    file to protect from deletion, not certify it as loadable.
    """
    manifest_dir = Path(manifest_dir)
    manifest_path = manifest_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None

    raw_bytes = manifest_path.read_bytes()
    manifest = _parse_manifest_bytes(raw_bytes)
    _validate_manifest_semantics(manifest)

    checkpoint_file = manifest["checkpoint_file"]
    checkpoint_path = manifest_dir / checkpoint_file
    if checkpoint_path.is_symlink():
        raise ValueError(
            f"Active manifest's checkpoint path must not be a symlink: {checkpoint_path}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Active manifest {manifest_path} references a checkpoint that does not exist: "
            f"{checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise ValueError(
            f"Active manifest's checkpoint path is not a regular file: {checkpoint_path}"
        )
    if checkpoint_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError(
            f"Active manifest's checkpoint path {checkpoint_path} does not physically "
            f"resolve beside the manifest {manifest_path} (possible symlink escape)."
        )

    return checkpoint_path


def validate_inference_hyperparams(hyperparams: Any) -> dict:
    """Fail-closed validation of the hyperparameters production inference
    needs (detection_threshold, edge_threshold, nms_radius_um, and the
    optional max_positive_voxel_fraction). Returns the same dict unchanged
    on success -- never substitutes defaults for a missing or invalid field."""
    if not isinstance(hyperparams, dict):
        raise ValueError(f"hyperparams must be a dict, got {type(hyperparams).__name__}")

    def _bounded(key: str, low: float, high: float, low_incl: bool = True, high_incl: bool = True) -> None:
        if key not in hyperparams:
            raise ValueError(f"hyperparams missing required field {key!r}")
        val = hyperparams[key]
        if isinstance(val, bool) or not isinstance(val, int | float):
            raise ValueError(
                f"hyperparams[{key!r}] must be int or float, not bool, got "
                f"{type(val).__name__}: {val!r}"
            )
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"hyperparams[{key!r}] must be finite, got {val!r}")
        ok_low = fval >= low if low_incl else fval > low
        ok_high = fval <= high if high_incl else fval < high
        if not (ok_low and ok_high):
            raise ValueError(f"hyperparams[{key!r}]={val!r} is out of the required range")

    _bounded("detection_threshold", 0, 1)
    _bounded("edge_threshold", 0, 1)
    _bounded("nms_radius_um", 0, math.inf, low_incl=False, high_incl=False)
    if "max_positive_voxel_fraction" in hyperparams:
        _bounded("max_positive_voxel_fraction", 0, 1, low_incl=False, high_incl=True)

    return hyperparams


def deployment_eligibility_errors(checkpoint: Any) -> list[str]:
    """Return every reason (human-readable) checkpoint is NOT eligible for
    manifest creation/replacement -- empty list means fully eligible. Never
    raises -- callers decide whether to skip manifest creation (expected,
    logged) or treat a non-empty list as a hard failure."""
    errors: list[str] = []

    if not isinstance(checkpoint, dict):
        errors.append(f"checkpoint must be a dict, got {type(checkpoint).__name__}")
        return errors

    # An obviously incomplete checkpoint (missing the actual model weights)
    # must never become the active manifested checkpoint, even though the
    # load path would separately reject it later -- fail-closed means this
    # is caught before write_checkpoint_manifest() ever runs, not after.
    if "unet3d_state_dict" not in checkpoint:
        errors.append("checkpoint missing required key 'unet3d_state_dict'")
    if "transformer_state_dict" not in checkpoint:
        errors.append("checkpoint missing required key 'transformer_state_dict'")

    schema_version = checkpoint.get("checkpoint_schema_version")
    if not _is_exact_schema_version(schema_version, CHECKPOINT_SCHEMA_VERSION):
        errors.append(
            f"checkpoint_schema_version must be exactly the int {CHECKPOINT_SCHEMA_VERSION} "
            f"(not bool, not float, not str), got {schema_version!r}"
        )

    training_code_sha = checkpoint.get("training_code_sha")
    if not _is_lowercase_hex(training_code_sha, 40):
        errors.append(
            f"training_code_sha must be a 40-character lowercase hex git SHA, got "
            f"{training_code_sha!r}"
        )

    split_sha = checkpoint.get("split_membership_sha256")
    if not _is_lowercase_hex(split_sha, 64):
        errors.append(
            f"split_membership_sha256 must be a 64-character lowercase hex hash, got "
            f"{split_sha!r}"
        )

    model_contract = checkpoint.get("model_contract")
    if model_contract != MODEL_CONTRACT:
        errors.append(f"model_contract must be {MODEL_CONTRACT!r}, got {model_contract!r}")

    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        errors.append(f"epoch must be a positive non-bool int, got {epoch!r}")

    hyperparams = checkpoint.get("hyperparams")
    try:
        validate_inference_hyperparams(hyperparams)
    except ValueError as e:
        errors.append(f"invalid inference hyperparameters: {e}")

    val_metrics = checkpoint.get("val_metrics")
    if not isinstance(val_metrics, dict):
        errors.append(
            f"val_metrics must be a dict, got "
            f"{'None' if val_metrics is None else type(val_metrics).__name__}"
        )
        return errors

    def _positive_int(key: str) -> int | None:
        val = val_metrics.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            errors.append(f"val_metrics[{key!r}] must be a positive non-bool int, got {val!r}")
            return None
        return val

    if val_metrics.get("evaluation_completed_successfully") is not True:
        errors.append(
            f"val_metrics['evaluation_completed_successfully'] must be literal True -- "
            f"got {val_metrics.get('evaluation_completed_successfully')!r}. This is a "
            f"fallback/failed-evaluation checkpoint (an exception during evaluation, or "
            f"no usable ground truth) and can never be deployment-eligible regardless of "
            f"what its other metric fields report."
        )

    if val_metrics.get("validation_is_full_fold") is not True:
        errors.append(
            f"val_metrics['validation_is_full_fold'] must be True, got "
            f"{val_metrics.get('validation_is_full_fold')!r}"
        )

    evaluated = _positive_int("validation_samples_evaluated")
    total = _positive_int("validation_samples_total")
    if evaluated is not None and total is not None and evaluated != total:
        errors.append(
            f"val_metrics['validation_samples_evaluated']={evaluated} must equal "
            f"val_metrics['validation_samples_total']={total}"
        )

    num_datasets = _positive_int("num_datasets")
    if num_datasets is not None and evaluated is not None and num_datasets != evaluated:
        errors.append(
            f"val_metrics['num_datasets']={num_datasets} must equal "
            f"val_metrics['validation_samples_evaluated']={evaluated}"
        )

    _positive_int("predicted_nodes_total")
    _positive_int("predicted_edges_total")

    if val_metrics.get("is_structural_zero") is not False:
        errors.append(
            f"val_metrics['is_structural_zero'] must be False, got "
            f"{val_metrics.get('is_structural_zero')!r}"
        )

    adj = val_metrics.get("adjusted_edge_jaccard")
    if isinstance(adj, bool) or not isinstance(adj, int | float) or not math.isfinite(float(adj)) or float(adj) < 0:
        errors.append(
            f"val_metrics['adjusted_edge_jaccard'] must be a finite non-negative number, "
            f"got {adj!r}"
        )

    return errors


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    checkpoint: dict | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Transactionally create/replace checkpoint_manifest.json beside
    checkpoint_path (B5): write to a temp file in the same directory, fsync,
    strictly parse+validate the temp manifest, verify it self-consistently
    references this checkpoint's real file hash, only then os.replace() it
    into place, then re-read and re-verify the now-active manifest. On any
    failure before or after replacement, the old manifest (if any) is left
    untouched or atomically restored -- never left partially written.

    Raises ValueError if checkpoint is not deployment-eligible (see
    deployment_eligibility_errors) -- callers that want to skip ineligible
    checkpoints silently (the expected case) must check eligibility
    themselves before calling this.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if checkpoint_path.is_symlink():
        raise ValueError(f"Checkpoint path must not be a symlink: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path must be a regular file: {checkpoint_path}")
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Checkpoint path must have a .pt suffix, got: {checkpoint_path}")

    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    errors = deployment_eligibility_errors(checkpoint)
    if errors:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not eligible for manifest creation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    if output_path is None:
        output_path = checkpoint_path.parent / MANIFEST_FILENAME
    else:
        output_path = Path(output_path)
        # Direct writes must obey the same canonical-filename contract as
        # discovery (find_single_manifest) and loading (load_verified_
        # checkpoint) -- otherwise the direct path could create a
        # differently-named file that no discovery/load path would ever
        # trust, silently orphaning it, or worse, be mistaken for the real
        # deployment manifest by a caller that doesn't re-check the name.
        if output_path.name != MANIFEST_FILENAME:
            raise ValueError(
                f"Manifest output_path must be named {MANIFEST_FILENAME!r}, got "
                f"{output_path.name!r}: {output_path}"
            )

    resolved_checkpoint_parent = checkpoint_path.resolve().parent
    resolved_manifest_parent = output_path.resolve().parent
    if resolved_checkpoint_parent != resolved_manifest_parent:
        raise ValueError(
            f"Manifest must be written beside its checkpoint: checkpoint parent "
            f"{resolved_checkpoint_parent} != manifest parent {resolved_manifest_parent}"
        )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    val_metrics = checkpoint["val_metrics"]

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "training_code_sha": checkpoint["training_code_sha"],
        "split_membership_sha256": checkpoint["split_membership_sha256"],
        "model_contract": checkpoint["model_contract"],
        "epoch": checkpoint["epoch"],
        "validation_is_full_fold": val_metrics["validation_is_full_fold"],
        "validation_samples_evaluated": val_metrics["validation_samples_evaluated"],
        "validation_samples_total": val_metrics["validation_samples_total"],
        "num_datasets": val_metrics["num_datasets"],
        "predicted_nodes_total": val_metrics["predicted_nodes_total"],
        "predicted_edges_total": val_metrics["predicted_edges_total"],
        "is_structural_zero": val_metrics["is_structural_zero"],
        "adjusted_edge_jaccard": float(val_metrics["adjusted_edge_jaccard"]),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    old_manifest_bytes: bytes | None = None
    if output_path.exists():
        old_manifest_bytes = output_path.read_bytes()

    fd, tmp_name = tempfile.mkstemp(
        dir=str(output_path.parent), prefix=".checkpoint_manifest.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(manifest_bytes)
            f.flush()
            os.fsync(f.fileno())

        temp_parsed = _parse_manifest_bytes(tmp_path.read_bytes())
        _validate_manifest_semantics(temp_parsed)
        if temp_parsed["checkpoint_file"] != checkpoint_path.name or temp_parsed["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError(
                "Temporary manifest failed self-consistency check before replacement -- "
                "refusing to replace the active manifest."
            )

        os.replace(str(tmp_path), str(output_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    try:
        active_bytes = output_path.read_bytes()
        active_parsed = _parse_manifest_bytes(active_bytes)
        _validate_manifest_semantics(active_parsed)
        if active_bytes != manifest_bytes:
            raise RuntimeError("Active manifest bytes do not match what write_checkpoint_manifest() wrote.")
    except Exception:
        logger.error(
            f"Post-replacement manifest verification FAILED for {output_path} -- "
            f"restoring previous manifest state."
        )
        if old_manifest_bytes is not None:
            fd2, tmp2_name = tempfile.mkstemp(
                dir=str(output_path.parent), prefix=".checkpoint_manifest.restore.", suffix=".tmp"
            )
            tmp2_path = Path(tmp2_name)
            with os.fdopen(fd2, "wb") as f2:
                f2.write(old_manifest_bytes)
                f2.flush()
                os.fsync(f2.fileno())
            os.replace(str(tmp2_path), str(output_path))
        else:
            output_path.unlink(missing_ok=True)
        raise

    logger.info(f"Checkpoint manifest written: {output_path} (checkpoint_sha256={checkpoint_sha256})")
    return output_path


def load_verified_checkpoint(
    manifest_path: str | Path,
    expected_source_sha: str,
    map_location: Any,
) -> tuple[dict, dict, Path]:
    """
    Fail-closed checkpoint load (B7): every verification step below must
    pass, IN ORDER, before torch.load() is ever called on the referenced
    checkpoint file. Returns (checkpoint, manifest, checkpoint_path).
    """
    manifest_path = Path(manifest_path)

    # 1. manifest exists.
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Manifest path is not a regular file: {manifest_path}")

    # 1b. manifest must have the exact canonical filename -- production
    # deployment identity has exactly one manifest name; a differently-named
    # but otherwise-valid manifest must never be treated as authoritative.
    if manifest_path.name != MANIFEST_FILENAME:
        raise ValueError(
            f"Manifest file must be named {MANIFEST_FILENAME!r}, got "
            f"{manifest_path.name!r}: {manifest_path}"
        )

    raw_bytes = manifest_path.read_bytes()

    # 2/3/4: strict JSON parse, duplicate-key rejection, exact key set.
    # 5/6: types + full schema/semantic validation.
    manifest = _parse_manifest_bytes(raw_bytes)
    _validate_manifest_semantics(manifest)

    # 7. expected source SHA itself must be well-formed.
    if not _is_lowercase_hex(expected_source_sha, 40):
        raise ValueError(
            f"expected_source_sha must be a 40-character lowercase hex git SHA, got "
            f"{expected_source_sha!r}"
        )

    # 8. manifest training SHA must equal expected (mounted) source SHA.
    if manifest["training_code_sha"] != expected_source_sha:
        raise ValueError(
            f"Manifest training_code_sha={manifest['training_code_sha']!r} does not "
            f"match expected_source_sha={expected_source_sha!r} -- the mounted source "
            f"code does not match the code that trained this checkpoint."
        )

    # 9. safe, non-symlink .pt checkpoint path physically beside the manifest.
    checkpoint_file = manifest["checkpoint_file"]
    checkpoint_path = manifest_path.parent / checkpoint_file
    if checkpoint_path.is_symlink():
        raise ValueError(f"Checkpoint path must not be a symlink: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file referenced by manifest not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a regular file: {checkpoint_path}")
    if checkpoint_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError(
            f"Checkpoint path {checkpoint_path} does not physically resolve beside "
            f"manifest {manifest_path} (possible symlink escape)."
        )

    # 10. actual checkpoint hash computed and matched against the manifest.
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != manifest["checkpoint_sha256"]:
        raise ValueError(
            f"Checkpoint file hash mismatch: manifest declares "
            f"{manifest['checkpoint_sha256']!r}, actual file hash is {actual_sha256!r}. "
            f"Refusing to load a checkpoint whose bytes do not match its verified manifest."
        )

    # 11. only now may torch.load() run.
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 12. required checkpoint keys present.
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Loaded checkpoint must be a dict, got {type(checkpoint).__name__}")
    missing_keys = _REQUIRED_CHECKPOINT_KEYS - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required key(s): {sorted(missing_keys)}")

    # 13. checkpoint schema/hyperparams/source/split/contract/epoch/val-metrics/eligibility.
    errors = deployment_eligibility_errors(checkpoint)
    if errors:
        raise ValueError(
            f"Checkpoint {checkpoint_path} referenced by verified manifest {manifest_path} "
            f"is not deployment-eligible:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # 14. manifest and checkpoint values must match exactly.
    val_metrics = checkpoint["val_metrics"]
    mismatches: list[str] = []

    def _check(field: str, checkpoint_value: Any, manifest_value: Any) -> None:
        if checkpoint_value != manifest_value:
            mismatches.append(f"{field}: checkpoint={checkpoint_value!r} manifest={manifest_value!r}")

    _check("training_code_sha", checkpoint["training_code_sha"], manifest["training_code_sha"])
    _check("split_membership_sha256", checkpoint["split_membership_sha256"], manifest["split_membership_sha256"])
    _check("model_contract", checkpoint["model_contract"], manifest["model_contract"])
    _check("epoch", checkpoint["epoch"], manifest["epoch"])
    _check("validation_is_full_fold", val_metrics["validation_is_full_fold"], manifest["validation_is_full_fold"])
    _check("validation_samples_evaluated", val_metrics["validation_samples_evaluated"], manifest["validation_samples_evaluated"])
    _check("validation_samples_total", val_metrics["validation_samples_total"], manifest["validation_samples_total"])
    _check("num_datasets", val_metrics["num_datasets"], manifest["num_datasets"])
    _check("predicted_nodes_total", val_metrics["predicted_nodes_total"], manifest["predicted_nodes_total"])
    _check("predicted_edges_total", val_metrics["predicted_edges_total"], manifest["predicted_edges_total"])
    _check("is_structural_zero", val_metrics["is_structural_zero"], manifest["is_structural_zero"])
    if float(val_metrics["adjusted_edge_jaccard"]) != float(manifest["adjusted_edge_jaccard"]):
        mismatches.append(
            f"adjusted_edge_jaccard: checkpoint={val_metrics['adjusted_edge_jaccard']!r} "
            f"manifest={manifest['adjusted_edge_jaccard']!r}"
        )

    if mismatches:
        raise ValueError(
            f"Checkpoint {checkpoint_path} values do not match verified manifest "
            f"{manifest_path}:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )

    logger.info(
        f"Verified checkpoint loaded: {checkpoint_path} (sha256={actual_sha256}) | "
        f"manifest={manifest_path} | training_code_sha={manifest['training_code_sha']} | "
        f"model_contract={manifest['model_contract']} | epoch={manifest['epoch']} | "
        f"split={manifest['split_membership_sha256']} | "
        f"coverage={manifest['validation_samples_evaluated']}/{manifest['validation_samples_total']} | "
        f"datasets={manifest['num_datasets']} | nodes={manifest['predicted_nodes_total']} | "
        f"edges={manifest['predicted_edges_total']} | "
        f"adjusted_edge_jaccard={manifest['adjusted_edge_jaccard']}"
    )

    return checkpoint, manifest, checkpoint_path
