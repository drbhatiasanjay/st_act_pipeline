"""
Shared source-provenance verification for Kaggle kernels and the GPU sanity gate.

GPU sanity gate design (2026-07-19, Correction 2 in GPU_SANITY_GATE_DESIGN_2026-07-18_v4.md)
proposed extracting find_exactly_one_kaggle_input_dir(), verify_import_origins(), and strict
GIT_SHA.txt validation -- which already existed, independently, in kaggle_kernel_inference/
inference_kernel.py (P0-6, "Part C1"/"Part C2"/"Part C3") and kaggle_kernel/train_kernel.py
(P0-7) -- into one shared module all three call sites import identically.

IMPORTANT BOOTSTRAP CONSTRAINT, discovered during implementation (2026-07-19): the discovery
functions (find_all_kaggle_input_dirs / find_exactly_one_kaggle_input_dir) run BEFORE sys.path
includes any src/ directory at all -- their entire job is finding which directory to add to
sys.path in the first place. A Kaggle-kernel entry point therefore CANNOT `from
src.deployment_provenance import find_exactly_one_kaggle_input_dir` to do that discovery: `src`
isn't importable yet. This means the discovery functions must remain literally duplicated,
verbatim, at the top of every Kaggle-kernel entry point's own bootstrap section (train_kernel.py,
inference_kernel.py, and any future kernel entry point such as a gate-runner script) -- this
module is their canonical reference to copy from and diff against, not something they can import
at that point. validate_git_sha_file() and verify_import_origins() do NOT have this constraint
(both existing kernels already insert the discovered directory into sys.path before reaching
their SHA-validation and import-verification steps), so any NEW kernel entry point written after
this module exists should import those two directly rather than re-embedding them. The two
existing kernels (train_kernel.py, inference_kernel.py) were deliberately left unmodified during
the GPU-sanity-gate Wave 1 pass that created this module: retrofitting them would also require
touching tests/test_p07_training_integrity.py's AST-extraction tests (which locate and exec the
literal inline if-block by source-text pattern match), which is outside this pass's authorized
file scope for no functional gain -- both kernels' existing inline implementations were
independently verified byte-for-byte equivalent to this module's logic before this module was
written.
"""

import os
from pathlib import Path
from types import ModuleType

MAX_SEARCH_DEPTH = 5


def find_all_kaggle_input_dirs(marker_relpath: str, max_depth: int = MAX_SEARCH_DEPTH) -> list[str]:
    """Return every directory beneath /kaggle/input containing marker_relpath."""
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
    """Exact-one discovery: never silently select the first directory, directory
    order, or modification time. Zero matches raises; multiple matches raises and
    lists every candidate."""
    matches = find_all_kaggle_input_dirs(marker_relpath)
    if not matches:
        raise RuntimeError(f"No directory beneath /kaggle/input contains {marker_relpath!r}.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple directories beneath /kaggle/input contain {marker_relpath!r}, exactly "
            f"one is required: {sorted(matches)}"
        )
    return matches[0]


def validate_git_sha_file(sha_file_path: str | Path) -> str:
    """Strict, unconditional GIT_SHA.txt validation for GPU-SANITY-GATE-01 and any
    other canonical gate provenance path -- missing, empty/whitespace-only, or
    malformed (not exactly 40 lowercase hex characters) all raise, always, with
    no "unknown"/allow-missing escape hatch of any kind. Deliberately stricter
    than train_kernel.py's own local-execution fallback (which stays untouched,
    per the GPU sanity gate scope decision -- see this module's docstring): the
    gate's whole purpose is verifying real deployed-code identity before an
    expensive run, so there is no legitimate "local execution, skip the check"
    case for it the way there is for local development of the training kernel."""
    sha_file_path = Path(sha_file_path)
    if not sha_file_path.exists():
        raise RuntimeError(
            f"GIT_SHA.txt not found at {sha_file_path} -- was this pushed via "
            f"scripts/sync_kaggle_src.py?"
        )
    raw_sha = sha_file_path.read_text(encoding="utf-8").strip()
    if not raw_sha:
        raise RuntimeError(f"GIT_SHA.txt at {sha_file_path} is empty or whitespace-only.")
    if len(raw_sha) != 40 or raw_sha != raw_sha.lower() or not all(
        c in "0123456789abcdef" for c in raw_sha
    ):
        raise RuntimeError(
            f"GIT_SHA.txt at {sha_file_path} does not contain a 40-character lowercase "
            f"hex git SHA: {raw_sha!r}"
        )
    return raw_sha


def verify_import_origins(expected_root: str | Path, modules: list[ModuleType]) -> None:
    """Part C2: after importing every production module the caller depends on,
    verify each one's real __file__ resolves underneath the exact-one selected
    source dataset -- never underneath a stray repository checkout, a different
    attached dataset, or anywhere else. Raises loudly on any module resolving
    outside expected_root. Caller supplies the exact module list since different
    callers (train_kernel.py, inference_kernel.py, the GPU sanity gate) depend on
    different production modules."""
    expected_root_resolved = Path(expected_root).resolve()
    for module in modules:
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
                f"against code from an unverified location."
            ) from e
