"""
Kaggle run registry: a local SQLite database logging functional, technical,
and performance parameters for every Kaggle kernel run (probes, training,
diagnostics), so runs can be compared systematically instead of re-deriving
"what changed since last time" from memory or scattered JSON files each time.

Why this exists: by 2026-08-14 this project had run 3+ real Kaggle GPU
kernels (learning probe v2, v3, a real training run) with no structured way
to compare them -- each comparison meant manually re-reading raw report JSONs
side by side. This gives one queryable table covering config (deployed SHA,
hyperparams, GPU), functional results (val scores, structural-zero flag,
predicted node/edge counts), and performance (wall-clock, batches/sec, peak
GPU memory) for every run.

Complementary to src/run_tracker.py, not a replacement: that module tracks
fine-grained progress *within* one local pipeline run (per-dataset caching,
live ETA). This tracks *across* Kaggle kernel runs, one row per run,
populated after the fact from each run's real report JSON / training_log.csv
/ kaggle_check_run.py output.

Usage:
    python scripts/run_registry.py init
    python scripts/run_registry.py ingest-probe-report <report.json> --run-id <id>
    python scripts/run_registry.py ingest-training-run --run-id <id> --deployed-sha <sha> ...
    python scripts/run_registry.py list
    python scripts/run_registry.py show <run-id>
    python scripts/run_registry.py compare <run-id-1> <run-id-2> [...]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path("kaggle_runs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    kernel_slug TEXT,
    kernel_version INTEGER,
    run_type TEXT,                      -- probe | training | diagnostic | calibration
    logged_at TEXT,                     -- when this row was written (UTC ISO)

    -- Technical / config
    deployed_sha TEXT,
    gpu_name TEXT,
    device_type TEXT,
    cuda_available INTEGER,
    config_json TEXT,                   -- full hyperparams dict as JSON

    -- Scope
    requested_train_batches INTEGER,
    completed_train_batches INTEGER,
    num_epochs INTEGER,
    train_dataset_pair_count INTEGER,
    validation_samples_total INTEGER,
    validation_samples_evaluated INTEGER,
    full_fold_validation_performed INTEGER,

    -- Functional results
    verdict TEXT,                       -- PASS | FAIL | COMPLETE | ERROR | RUNNING
    average_train_loss REAL,
    last_train_loss REAL,
    max_sigmoid_final REAL,
    max_sigmoid_min REAL,
    max_sigmoid_max REAL,
    last_unet_gradient_norm REAL,
    last_transformer_gradient_norm REAL,
    val_edge_jaccard REAL,
    val_adjusted_edge_jaccard REAL,
    val_division_jaccard REAL,
    val_score REAL,
    predicted_nodes_total INTEGER,
    predicted_edges_total INTEGER,
    gt_nodes_total INTEGER,
    is_structural_zero INTEGER,
    learning_signal_observed INTEGER,

    -- Performance
    elapsed_seconds REAL,
    training_elapsed_seconds REAL,
    seconds_per_batch REAL,
    peak_gpu_memory_allocated_bytes INTEGER,
    peak_gpu_memory_reserved_bytes INTEGER,

    -- Integrity
    train_fallback_counts_json TEXT,
    post_validation_fallback_counts_json TEXT,

    -- Free text / provenance
    notes TEXT,
    raw_report_path TEXT
);
"""


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Initialized {db_path.resolve()}")


def _upsert(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = list(row.keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "run_id")
    sql = f"""
        INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})
        ON CONFLICT(run_id) DO UPDATE SET {updates}
    """
    conn.execute(sql, row)
    conn.commit()


def ingest_probe_report(
    report_path: Path, run_id: str, kernel_slug: str, kernel_version: int | None = None,
    run_type: str = "probe", notes: str = "", db_path: Path = DB_PATH,
) -> None:
    """Ingest a gpu_learning_probe_report.json (or same-shaped report) into the registry."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    vm = report.get("validation_metrics", {})

    row = {
        "run_id": run_id,
        "kernel_slug": kernel_slug,
        "kernel_version": kernel_version,
        "run_type": run_type,
        "logged_at": datetime.now(UTC).isoformat(),
        "deployed_sha": report.get("deployed_sha"),
        "gpu_name": report.get("gpu_name"),
        "device_type": report.get("device_type"),
        "cuda_available": int(bool(report.get("cuda_available"))),
        "config_json": json.dumps({
            "requested_validation_samples": report.get("requested_validation_samples"),
            "time_budget_seconds": report.get("time_budget_seconds"),
        }),
        "requested_train_batches": report.get("requested_train_batches"),
        "completed_train_batches": report.get("completed_train_batches"),
        "num_epochs": None,
        "train_dataset_pair_count": report.get("train_dataset_pair_count"),
        "validation_samples_total": vm.get("validation_samples_total"),
        "validation_samples_evaluated": vm.get("validation_samples_evaluated"),
        "full_fold_validation_performed": int(bool(report.get("full_fold_validation_performed"))),
        "verdict": report.get("verdict"),
        "average_train_loss": report.get("average_train_loss"),
        "last_train_loss": None,
        "max_sigmoid_final": None,
        "max_sigmoid_min": None,
        "max_sigmoid_max": None,
        "last_unet_gradient_norm": report.get("last_unet_gradient_norm"),
        "last_transformer_gradient_norm": report.get("last_transformer_gradient_norm"),
        "val_edge_jaccard": vm.get("edge_jaccard"),
        "val_adjusted_edge_jaccard": vm.get("adjusted_edge_jaccard"),
        "val_division_jaccard": vm.get("division_jaccard"),
        "val_score": vm.get("score"),
        "predicted_nodes_total": vm.get("predicted_nodes_total"),
        "predicted_edges_total": vm.get("predicted_edges_total"),
        "gt_nodes_total": vm.get("num_gt_nodes_total"),
        "is_structural_zero": int(bool(vm.get("is_structural_zero"))),
        "learning_signal_observed": int(bool(report.get("learning_signal_observed"))),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "training_elapsed_seconds": report.get("training_elapsed_seconds"),
        "seconds_per_batch": (
            report.get("training_elapsed_seconds") / report.get("completed_train_batches")
            if report.get("training_elapsed_seconds") and report.get("completed_train_batches")
            else None
        ),
        "peak_gpu_memory_allocated_bytes": report.get("peak_gpu_memory_allocated_bytes"),
        "peak_gpu_memory_reserved_bytes": report.get("peak_gpu_memory_reserved_bytes"),
        "train_fallback_counts_json": json.dumps(report.get("train_fallback_counts", {})),
        "post_validation_fallback_counts_json": json.dumps(report.get("post_validation_fallback_counts", {})),
        "notes": notes,
        "raw_report_path": str(report_path.resolve()),
    }
    conn = get_conn(db_path)
    _upsert(conn, row)
    conn.close()
    print(f"Ingested {run_id} from {report_path}")


def ingest_training_run(
    run_id: str, kernel_slug: str, training_log_csv: Path, progress_json: Path,
    manifest_json: Path | None = None, kernel_version: int | None = None,
    run_type: str = "training", notes: str = "", db_path: Path = DB_PATH,
) -> None:
    """Ingest a real train_kernel.py run from its training_log.csv (last row),
    training_progress.json (real total wall-clock, includes validation -- the
    CSV's epoch_wall_clock_seconds is train-phase only, do not confuse the
    two), and optionally checkpoint_manifest.json for split/provenance detail."""
    import csv as csv_module

    with open(training_log_csv, newline="", encoding="utf-8") as f:
        csv_rows = list(csv_module.DictReader(f))
    if not csv_rows:
        raise ValueError(f"{training_log_csv} has no data rows")
    last = csv_rows[-1]

    progress = json.loads(progress_json.read_text(encoding="utf-8"))

    manifest = {}
    if manifest_json and manifest_json.exists():
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    def _f(v):
        return float(v) if v not in (None, "") else None

    def _i(v):
        return int(float(v)) if v not in (None, "") else None

    row = {
        "run_id": run_id,
        "kernel_slug": kernel_slug,
        "kernel_version": kernel_version,
        "run_type": run_type,
        "logged_at": datetime.now(UTC).isoformat(),
        "deployed_sha": progress.get("deployed_sha") or manifest.get("training_code_sha"),
        "gpu_name": progress.get("gpu_name"),
        "device_type": "cuda" if progress.get("cuda_available") else ("cpu" if progress.get("cuda_available") is not None else None),
        "cuda_available": int(bool(progress.get("cuda_available"))) if progress.get("cuda_available") is not None else None,
        "config_json": json.dumps({"learning_rate": last.get("learning_rate")}),
        "requested_train_batches": _i(last.get("num_batches")),
        "completed_train_batches": _i(last.get("num_batches")),
        "num_epochs": _i(progress.get("num_epochs_budget")) or _i(last.get("epoch")),
        "train_dataset_pair_count": None,
        "validation_samples_total": manifest.get("validation_samples_total"),
        "validation_samples_evaluated": manifest.get("validation_samples_evaluated"),
        "full_fold_validation_performed": int(bool(manifest.get("validation_is_full_fold"))),
        "verdict": "COMPLETE",
        "average_train_loss": _f(last.get("train_loss")),
        "last_train_loss": _f(progress.get("train_loss")),
        "max_sigmoid_final": _f(progress.get("max_sigmoid_final")),
        "max_sigmoid_min": _f(progress.get("max_sigmoid_min")),
        "max_sigmoid_max": _f(progress.get("max_sigmoid_max")),
        "last_unet_gradient_norm": None,
        "last_transformer_gradient_norm": None,
        "val_edge_jaccard": _f(last.get("val_edge_jaccard")),
        "val_adjusted_edge_jaccard": _f(last.get("val_adjusted_edge_jaccard")),
        "val_division_jaccard": _f(last.get("val_division_jaccard")),
        "val_score": _f(last.get("val_score")),
        "predicted_nodes_total": _i(last.get("predicted_nodes_total")),
        "predicted_edges_total": _i(last.get("predicted_edges_total")),
        "gt_nodes_total": None,
        "is_structural_zero": int(str(last.get("is_structural_zero")).lower() == "true"),
        "learning_signal_observed": None,
        # elapsed_seconds is the REAL total (train + full validation) from the
        # progress heartbeat; training_log.csv's epoch_wall_clock_seconds is
        # train-phase-only and would understate total cost if used here.
        "elapsed_seconds": _f(progress.get("elapsed_seconds")),
        "training_elapsed_seconds": _f(last.get("epoch_wall_clock_seconds")),
        "seconds_per_batch": (
            _f(last.get("epoch_wall_clock_seconds")) / _i(last.get("num_batches"))
            if last.get("epoch_wall_clock_seconds") and last.get("num_batches")
            else None
        ),
        "peak_gpu_memory_allocated_bytes": _i(progress.get("peak_gpu_memory_allocated_bytes")),
        "peak_gpu_memory_reserved_bytes": _i(progress.get("peak_gpu_memory_reserved_bytes")),
        "train_fallback_counts_json": json.dumps({
            "heatmap_failures": last.get("heatmap_failures"),
            "edge_target_failures": last.get("edge_target_failures"),
            "edge_loss_failures": last.get("edge_loss_failures"),
            "eval_failures": last.get("eval_failures"),
        }),
        "post_validation_fallback_counts_json": "{}",
        "notes": notes,
        "raw_report_path": str(training_log_csv.resolve()),
    }
    conn = get_conn(db_path)
    _upsert(conn, row)
    conn.close()
    print(f"Ingested {run_id} from {training_log_csv} + {progress_json}")


def list_runs(db_path: Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT run_id, run_type, deployed_sha, verdict, completed_train_batches, "
        "max_sigmoid_final, val_score, is_structural_zero, elapsed_seconds "
        "FROM runs ORDER BY logged_at"
    ).fetchall()
    conn.close()
    if not rows:
        print("No runs logged yet.")
        return
    print(f"{'run_id':<30} {'type':<10} {'sha':<10} {'verdict':<8} {'batches':>8} {'val_score':>10} {'struct0':>8} {'elapsed_s':>10}")
    for r in rows:
        sha = (r["deployed_sha"] or "")[:8]
        print(f"{r['run_id']:<30} {r['run_type'] or '':<10} {sha:<10} {r['verdict'] or '':<8} "
              f"{r['completed_train_batches'] if r['completed_train_batches'] is not None else '':>8} "
              f"{r['val_score'] if r['val_score'] is not None else '':>10} "
              f"{r['is_structural_zero'] if r['is_structural_zero'] is not None else '':>8} "
              f"{r['elapsed_seconds'] if r['elapsed_seconds'] is not None else '':>10}")


def compare_runs(run_ids: list[str], db_path: Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    fields = [
        "run_id", "run_type", "deployed_sha", "verdict", "completed_train_batches",
        "num_epochs", "average_train_loss", "max_sigmoid_final",
        "val_edge_jaccard", "val_adjusted_edge_jaccard", "val_score",
        "predicted_nodes_total", "gt_nodes_total", "is_structural_zero",
        "elapsed_seconds", "training_elapsed_seconds", "seconds_per_batch",
    ]
    rows = {}
    for rid in run_ids:
        row = conn.execute(f"SELECT {', '.join(fields)} FROM runs WHERE run_id = ?", (rid,)).fetchone()
        if row:
            rows[rid] = row
        else:
            print(f"Warning: run_id '{rid}' not found")
    conn.close()
    if not rows:
        return
    label_w = max(len(f) for f in fields) + 2
    col_w = 25
    print(f"{'field':<{label_w}}" + "".join(f"{rid:>{col_w}}" for rid in rows))
    for f in fields:
        line = f"{f:<{label_w}}"
        for rid in rows:
            val = rows[rid][f]
            val_str = str(val) if val is not None else "-"
            if f == "deployed_sha" and val_str != "-":
                val_str = val_str[:10]
            line += f"{val_str:>{col_w}}"
        print(line)


def show_run(run_id: str, db_path: Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    if not row:
        print(f"No run found: {run_id}")
        return
    for k in row.keys():
        print(f"  {k}: {row[k]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    p_ingest = sub.add_parser("ingest-probe-report")
    p_ingest.add_argument("report_path", type=Path)
    p_ingest.add_argument("--run-id", required=True)
    p_ingest.add_argument("--kernel-slug", required=True)
    p_ingest.add_argument("--kernel-version", type=int, default=None)
    p_ingest.add_argument("--run-type", default="probe")
    p_ingest.add_argument("--notes", default="")

    p_train = sub.add_parser("ingest-training-run")
    p_train.add_argument("--run-id", required=True)
    p_train.add_argument("--kernel-slug", required=True)
    p_train.add_argument("--training-log-csv", required=True, type=Path)
    p_train.add_argument("--progress-json", required=True, type=Path)
    p_train.add_argument("--manifest-json", type=Path, default=None)
    p_train.add_argument("--kernel-version", type=int, default=None)
    p_train.add_argument("--run-type", default="training")
    p_train.add_argument("--notes", default="")

    sub.add_parser("list")

    p_show = sub.add_parser("show")
    p_show.add_argument("run_id")

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("run_ids", nargs="+")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
    elif args.command == "ingest-probe-report":
        ingest_probe_report(
            args.report_path, args.run_id, args.kernel_slug, args.kernel_version,
            args.run_type, args.notes,
        )
    elif args.command == "ingest-training-run":
        ingest_training_run(
            args.run_id, args.kernel_slug, args.training_log_csv, args.progress_json,
            args.manifest_json, args.kernel_version, args.run_type, args.notes,
        )
    elif args.command == "list":
        list_runs()
    elif args.command == "show":
        show_run(args.run_id)
    elif args.command == "compare":
        compare_runs(args.run_ids)


if __name__ == "__main__":
    main()
