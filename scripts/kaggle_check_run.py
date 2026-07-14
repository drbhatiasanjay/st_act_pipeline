"""
Check a Kaggle kernel's real status and, if it's finished, pull and summarize
the real log in one command -- replaces the ~5-step manual sequence repeated
many times this session (kernels status, PYTHONIOENCODING=utf-8 + kernels
logs, multiple greps for SHA/errors/circuit-breaker/progress).

Confirmed working commands this bundles (see CLAUDE.md's monitoring checklist
for the full reasoning):
- `kaggle kernels status` -- the only thing that's meaningful while RUNNING;
  `kernels output`/`kernels logs` return stale data mid-run.
- `kaggle kernels logs` (NOT `kernels output`, which only returns files
  written to /kaggle/working/, not the execution trace) -- needs
  PYTHONIOENCODING=utf-8 on Windows or it dies partway through with a
  'charmap' codec error. The output is a real JSON array of
  {stream_name, time, data} objects, not raw text -- parse it as such rather
  than regex-scraping the JSON-escaped text (confirmed: `json.loads()` on the
  whole response works directly).

Usage:
    python scripts/kaggle_check_run.py drbhatiasanjay/st-act-gpu-smoke-test
    python scripts/kaggle_check_run.py drbhatiasanjay/st-act-gpu-smoke-test --save-log out.txt
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def run_status(slug: str) -> str:
    result = subprocess.run(
        ["py", "-m", "kaggle", "kernels", "status", slug],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    # e.g. `owner/slug has status "KernelWorkerStatus.RUNNING"`
    match = re.search(r'"KernelWorkerStatus\.(\w+)"', result.stdout)
    return match.group(1) if match else result.stdout.strip()


def pull_log_lines(slug: str, save_path: Path) -> list[str]:
    """Fetch the real log and return it as a flat list of clean text lines.

    NOTE: PYTHONIOENCODING=utf-8 in the child env alone is NOT enough here --
    that only controls how the CHILD process (kaggle CLI) encodes its own
    output. subprocess.run(text=True) decodes those bytes back on the PARENT
    side using Python's own locale encoding (cp1252 on Windows) by default,
    regardless of the child's env -- confirmed as a real crash
    (UnicodeDecodeError on a real Kaggle log) before fixing it here.
    encoding="utf-8" passed directly to subprocess.run is what actually fixes
    the parent-side decode.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        ["py", "-m", "kaggle", "kernels", "logs", slug],
        capture_output=True, encoding="utf-8", errors="replace", env=env, check=True,
    )
    save_path.write_text(result.stdout, encoding="utf-8")

    entries = json.loads(result.stdout)
    lines = []
    for entry in entries:
        lines.extend(entry["data"].splitlines())
    return lines


def summarize(lines: list[str]) -> None:
    print(f"Total log lines: {len(lines)}")

    sha_line = next((line for line in lines if "Deployed code SHA:" in line), None)
    print(f"Deployed SHA: {sha_line.split('Deployed code SHA:')[-1].strip() if sha_line else 'not found'}")

    gpu_line = next((line for line in lines if re.search(r"- INFO: GPU: ", line)), None)
    print(f"GPU: {gpu_line.split('GPU:')[-1].strip() if gpu_line else 'not found'}")

    tb_idx = next((i for i, line in enumerate(lines) if "Traceback" in line), None)
    if tb_idx is not None:
        print(f"\n--- ERROR (Traceback at line {tb_idx}) ---")
        for line in lines[tb_idx:tb_idx + 10]:
            print(line)
    else:
        print("\nNo Traceback found.")

    circuit_breaker = next((line for line in lines if "Validation aborted" in line), None)
    if circuit_breaker:
        print(f"\n--- CIRCUIT BREAKER FIRED ---\n{circuit_breaker}")

    batch_lines = [line for line in lines if re.search(r"Batch \d+/\d+, Loss:", line)]
    if batch_lines:
        print(f"\nLast training batch progress: {batch_lines[-1].split(' - INFO: ')[-1]}")

    val_lines = [line for line in lines if "Validation - Edge Jaccard" in line]
    if val_lines:
        print(f"\nFinal validation result: {val_lines[-1].split(' - INFO: ')[-1]}")

    struct_zero = [
        line for line in lines
        if "TRAINING EPOCHS COMPLETED" in line or "Best validation score" in line
    ]
    for line in struct_zero:
        print(line.split(" - INFO: ")[-1] if " - INFO: " in line else line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slug", help="owner/kernel-name")
    parser.add_argument("--save-log", default=None, help="Where to save the full raw log (default: a temp file)")
    args = parser.parse_args()

    status = run_status(args.slug)
    print(f"Status: {status}")

    if status == "RUNNING":
        print("Still running -- no log to pull yet (kernels logs/output return stale data mid-run).")
        return

    save_path = Path(args.save_log) if args.save_log else Path(tempfile.gettempdir()) / f"{args.slug.split('/')[-1]}_log.txt"
    lines = pull_log_lines(args.slug, save_path)
    print(f"Full raw log saved to: {save_path}")
    print()
    summarize(lines)


if __name__ == "__main__":
    main()
