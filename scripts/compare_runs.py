"""
Compare pipeline run statistics across logs/runs/*.jsonl.

Usage:
    py scripts/compare_runs.py                 # summarize all runs, most recent first
    py scripts/compare_runs.py --last 5         # only the 5 most recent
    py scripts/compare_runs.py --run <run_id>   # full per-dataset breakdown for one run
"""
import argparse
import json
from pathlib import Path

LOG_DIR = Path("logs/runs")


def load_run(path: Path) -> dict:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return {"path": path, "events": events}


def summarize(run: dict) -> dict:
    events = run["events"]
    start = next((e for e in events if e["event"] == "run_start"), None)
    end = next((e for e in events if e["event"] == "run_end"), None)
    units = [e for e in events if e["event"] in ("unit_complete", "unit_failed")]
    completed = [e for e in units if e["event"] == "unit_complete"]
    failed = [e for e in units if e["event"] == "unit_failed"]
    cached = [e for e in completed if e.get("cached")]

    durations = [e["duration_s"] for e in completed]
    return {
        "run_id": start["run_id"] if start else run["path"].stem,
        "git_commit": start.get("git_commit") if start else None,
        "total_units": start.get("total_units") if start else len(units),
        "completed": len(completed),
        "failed": len(failed),
        "cached": len(cached),
        "elapsed_s": end.get("elapsed_s") if end else None,
        "avg_unit_s": (sum(durations) / len(durations)) if durations else None,
        "failed_ids": [e["unit_id"] for e in failed],
    }


def print_summary_table(runs: list):
    print(f"{'run_id':<32} {'commit':<9} {'done/total':<12} {'failed':<7} {'cached':<7} {'elapsed':<10} {'avg/unit':<10}")
    print("-" * 100)
    for r in runs:
        s = summarize(r)
        done_total = f"{s['completed']}/{s['total_units']}"
        elapsed = f"{s['elapsed_s']:.0f}s" if s['elapsed_s'] is not None else "running"
        avg = f"{s['avg_unit_s']:.1f}s" if s['avg_unit_s'] is not None else "-"
        print(f"{s['run_id']:<32} {str(s['git_commit']):<9} {done_total:<12} {s['failed']:<7} {s['cached']:<7} {elapsed:<10} {avg:<10}")
        if s["failed_ids"]:
            print(f"  failed: {', '.join(s['failed_ids'])}")


def print_run_detail(run: dict):
    s = summarize(run)
    print(f"=== {s['run_id']} (commit {s['git_commit']}) ===")
    print(f"{'dataset':<20} {'status':<10} {'duration':<10} {'cached':<8} stages")
    for e in run["events"]:
        if e["event"] == "unit_complete":
            stages = ", ".join(f"{k}={v:.2f}s" for k, v in e.get("stage_timings_s", {}).items())
            print(f"{e['unit_id']:<20} {'ok':<10} {e['duration_s']:<10.2f} {str(e.get('cached', False)):<8} {stages}")
        elif e["event"] == "unit_failed":
            print(f"{e['unit_id']:<20} {'FAILED':<10} {'-':<10} {'-':<8} {e['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, help="only show the N most recent runs")
    parser.add_argument("--run", type=str, default=None, help="show full detail for one run_id")
    args = parser.parse_args()

    log_files = sorted(LOG_DIR.glob("*.jsonl"))
    if not log_files:
        print(f"No run logs found in {LOG_DIR}/")
        return

    if args.run:
        matches = [f for f in log_files if args.run in f.stem]
        if not matches:
            print(f"No run matching '{args.run}' found")
            return
        print_run_detail(load_run(matches[-1]))
        return

    runs = [load_run(f) for f in log_files]
    runs.sort(key=lambda r: r["path"].stat().st_mtime, reverse=True)
    if args.last:
        runs = runs[:args.last]
    print_summary_table(runs)


if __name__ == "__main__":
    main()
