"""
Run execution tracking: fine-grained stage timing, live progress/ETA,
detection-result caching, and per-dataset checkpointing for cheap reruns.

Why this exists: the placeholder detector's threshold miscalibration caused a
2.5+ hour stuck run with zero visibility into what was happening or how close
it was to done (2026-07-03, see STATE.md). Every future run should instead:
  - Stream structured, per-stage timing to a JSONL log AS IT HAPPENS (not just
    at the end), so a killed/crashed run still leaves a useful partial record.
  - Print live progress and an ETA for remaining work.
  - Cache detection output (the volumes are static; thresholds/tracker costs
    are what actually change between runs, especially during Phase 4 tuning)
    so unrelated reruns don't repeat expensive detection.
  - Checkpoint completed datasets so a rerun after a partial failure (e.g. 1
    of 5 datasets erroring) only reprocesses what didn't finish, not
    everything.
"""

import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(".cache")
DETECTION_CACHE_DIR = CACHE_DIR / "detections"
DATASET_CACHE_DIR = CACHE_DIR / "datasets"
LOG_DIR = Path("logs/runs")


def _stable_hash(obj: Any) -> str:
    """Deterministic short hash of a JSON-serializable config dict."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes:.0f}m{secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f}h{minutes:.0f}m"


def _git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


class RunTracker:
    """
    Tracks one pipeline invocation across multiple units (e.g. datasets).
    Writes an append-only, line-flushed JSONL event log to
    logs/runs/<run_id>.jsonl -- partial progress survives a crash or kill,
    unlike the earlier stuck run which left nothing to diagnose.
    """

    def __init__(self, run_label: str, total_units: int, log_dir: Path = LOG_DIR):
        self.run_id = time.strftime("%Y%m%dT%H%M%S") + f"_{run_label}"
        self.run_label = run_label
        self.total_units = total_units
        self.completed_units = 0
        self.unit_durations = []
        self.start_time = time.time()

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{self.run_id}.jsonl"
        self._log_file = open(self.log_path, "a", buffering=1)

        self._emit({
            "event": "run_start",
            "run_id": self.run_id,
            "run_label": run_label,
            "total_units": total_units,
            "git_commit": _git_commit_hash(),
        })

    def _emit(self, record: dict):
        record.setdefault("timestamp", time.time())
        self._log_file.write(json.dumps(record, default=str) + "\n")
        self._log_file.flush()
        try:
            os.fsync(self._log_file.fileno())
        except OSError:
            pass  # not fatal if the platform/filesystem doesn't support fsync here

    def start_unit(self, unit_id: str) -> "UnitTimer":
        return UnitTimer(self, unit_id)

    def unit_complete(self, unit_id: str, duration: float, stage_timings: dict, extra: dict = None, cached: bool = False):
        self.completed_units += 1
        self.unit_durations.append(duration)
        record = {
            "event": "unit_complete",
            "unit_id": unit_id,
            "duration_s": round(duration, 3),
            "stage_timings_s": {k: round(v, 3) for k, v in stage_timings.items()},
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "cached": cached,
        }
        if extra:
            record["extra"] = extra
        self._emit(record)
        self._print_progress()

    def unit_failed(self, unit_id: str, error: str, stage_timings: dict = None):
        record = {
            "event": "unit_failed",
            "unit_id": unit_id,
            "error": error,
            "stage_timings_s": {k: round(v, 3) for k, v in (stage_timings or {}).items()},
        }
        self._emit(record)
        # Failures still count toward "processed" for progress/ETA purposes.
        self.completed_units += 1
        self._print_progress()

    def _print_progress(self):
        elapsed = time.time() - self.start_time
        remaining_units = max(0, self.total_units - self.completed_units)
        if self.unit_durations:
            avg = sum(self.unit_durations) / len(self.unit_durations)
            eta_str = _format_duration(avg * remaining_units)
        else:
            eta_str = "unknown"
        print(
            f"[{self.run_label}] {self.completed_units}/{self.total_units} done "
            f"| elapsed={_format_duration(elapsed)} | eta_remaining={eta_str}",
            flush=True,
        )

    def run_end(self, summary: dict = None):
        elapsed = time.time() - self.start_time
        record = {
            "event": "run_end",
            "run_id": self.run_id,
            "elapsed_s": round(elapsed, 3),
            "completed_units": self.completed_units,
            "total_units": self.total_units,
        }
        if summary:
            record["summary"] = summary
        self._emit(record)
        self._log_file.close()
        print(f"[{self.run_label}] RUN COMPLETE: {self.completed_units}/{self.total_units} "
              f"in {_format_duration(elapsed)}. Log: {self.log_path}")


class UnitTimer:
    """Times named stages within one unit (e.g. one dataset's load/detect/track/...)."""

    def __init__(self, tracker: RunTracker, unit_id: str):
        self.tracker = tracker
        self.unit_id = unit_id
        self.stage_timings = {}
        self._stage_start = None
        self._current_stage = None
        self._unit_start = time.time()

    def stage(self, name: str) -> "UnitTimer":
        self._close_current_stage()
        self._current_stage = name
        self._stage_start = time.time()
        return self

    def _close_current_stage(self):
        if self._current_stage is not None:
            self.stage_timings[self._current_stage] = time.time() - self._stage_start
            self._current_stage = None

    def done(self, extra: dict = None, cached: bool = False) -> float:
        self._close_current_stage()
        duration = time.time() - self._unit_start
        self.tracker.unit_complete(self.unit_id, duration, self.stage_timings, extra, cached=cached)
        return duration

    def failed(self, error: str):
        self._close_current_stage()
        self.tracker.unit_failed(self.unit_id, str(error), self.stage_timings)


# ---------------------------------------------------------------------------
# Detection caching: the zarr volumes are static; detection thresholds are
# what change between runs (rarely) while tracker costs change often (Phase 4
# tuning). Cache detection output per (dataset, detection-relevant-config) so
# runs that only touch tracker costs skip detection entirely.
# ---------------------------------------------------------------------------

def detection_cache_key(zarr_path: str, cnn_threshold: float, unet_threshold: float,
                         cnn_offset: float, unet_offset: float, max_candidates: int) -> str:
    path = Path(zarr_path)
    stat = path.stat() if path.exists() else None
    config = {
        "zarr_path": str(path.resolve()),
        "mtime": stat.st_mtime if stat else None,
        "cnn_threshold": cnn_threshold,
        "unet_threshold": unet_threshold,
        "cnn_offset": cnn_offset,
        "unet_offset": unet_offset,
        "max_candidates": max_candidates,
    }
    return _stable_hash(config)


def load_detection_cache(cache_key: str) -> Optional[dict]:
    cache_path = DETECTION_CACHE_DIR / f"{cache_key}.pkl"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None  # corrupt/incompatible cache -- treat as a miss, not an error


def save_detection_cache(cache_key: str, centroids_by_t: dict, motion_vectors_by_t: dict):
    DETECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DETECTION_CACHE_DIR / f"{cache_key}.pkl"
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"centroids_by_t": centroids_by_t, "motion_vectors_by_t": motion_vectors_by_t}, f)
    tmp_path.replace(cache_path)  # atomic replace on both POSIX and Windows (same volume)


# ---------------------------------------------------------------------------
# Per-dataset checkpointing: skip datasets whose full detect+track pipeline
# already completed successfully under the SAME config, so a rerun after
# e.g. dataset 3-of-5 failing only reprocesses dataset 3.
# ---------------------------------------------------------------------------

def dataset_checkpoint_key(dataset_id: str, zarr_path: str, full_config: dict) -> str:
    path = Path(zarr_path)
    stat = path.stat() if path.exists() else None
    config = {"dataset_id": dataset_id, "zarr_path": str(path.resolve()),
              "mtime": stat.st_mtime if stat else None, **full_config}
    return _stable_hash(config)


def load_dataset_checkpoint(checkpoint_key: str) -> Optional[dict]:
    cache_path = DATASET_CACHE_DIR / f"{checkpoint_key}.pkl"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_dataset_checkpoint(checkpoint_key: str, lineage_graph, timings: dict):
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATASET_CACHE_DIR / f"{checkpoint_key}.pkl"
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"lineage_graph": lineage_graph, "timings": timings}, f)
    tmp_path.replace(cache_path)
