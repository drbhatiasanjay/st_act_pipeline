"""
Competitor research via the Kaggle CLI + Gemini API.

Pulls real public kernels and the live leaderboard for the Biohub cell-tracking
competition (the official Kaggle API, not web scraping -- the competition's
own pages are JS-rendered SPAs and return no usable content to a plain HTTP
fetch, confirmed directly before writing this script). Feeds the real
notebook source, the real leaderboard, and this repo's own architecture docs
to Gemini, asking for concrete technique deltas mapped to specific files --
not generic ML advice, and not anything not actually present in the pulled
notebooks.

Output is a dated markdown report in the repo root, following the same
naming convention as ISSUES_AND_FIXES_2026-07-12.md.

Usage:
    python scripts/research_competitors.py
    python scripts/research_competitors.py --top-n 10
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

from google import genai

MODEL = "gemini-3.1-pro-preview"
COMPETITION = "biohub-cell-tracking-during-development"
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "competitor_research"

# Quoted verbatim, not re-derived -- these are the real, current architecture
# and known-bugs record, so Gemini doesn't waste output re-suggesting fixes
# for things already tracked or re-explaining our own architecture back to us.
CONTEXT_FILES = ["PRD.md", ".claude/CLAUDE.md", "DEFERRED_IMPROVEMENTS.md"]

SYSTEM_PREAMBLE = """You are comparing REAL, PUBLISHED competitor solutions (pulled directly from
Kaggle via the official API, not paraphrased) against our own pipeline for the same competition.

Your job is to find concrete, actionable technique deltas -- not to write a generic ML advice
essay. Follow these rules strictly:

1. Every recommendation must cite which competitor notebook (by Kaggle ref) it came from, and must
   be mapped to a specific file/function in OUR repo (src/model.py, src/tracker.py, src/train.py,
   src/evaluation.py, run_pipeline.py) that it would change. Reject your own suggestion if you
   cannot name a specific file it affects.
2. Do NOT hallucinate techniques that are not actually present in the supplied notebook text. If
   you are inferring something from a title or a library import rather than reading the actual
   described method, say so explicitly ("inferred from import, not confirmed from description").
3. Weight signal by validation: a technique the notebook itself reports a concrete leaderboard
   score for is stronger evidence than one with no reported score. State the score when citing it.
4. Do not re-suggest anything already listed as a known bug or already-deferred idea in the
   supplied CLAUDE.md / DEFERRED_IMPROVEMENTS.md context -- check first, and if a competitor
   technique overlaps with something already tracked there, say so instead of repeating it as new.
5. Output as a priority-ordered list. Tag every item exactly one of:
   [ADOPT-CANDIDATE] -- concrete, notebook-grounded, not already tracked, worth a real look
   [ALREADY-DOING] -- we already do this, per the supplied context
   [NOT-APPLICABLE] -- present in a competitor notebook but doesn't fit our architecture/scope,
   explain why
   For each item: one-line summary, source (which notebook / which leaderboard fact), target file
   in our repo, and justification.
6. If you find nothing concrete after genuinely trying, say so plainly rather than inventing
   minor nitpicks to fill space."""


def run_kaggle(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "kaggle", *args]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    if result.returncode != 0:
        print(f"WARNING: `{' '.join(cmd)}` failed:\n{result.stderr}", file=sys.stderr)
    return result


def fetch_leaderboard() -> str:
    """Downloads the real current leaderboard CSV and returns its text.

    Fails loudly rather than degrading silently -- a Gemini synthesis built on a
    missing leaderboard would look identical to a real one, which is exactly the
    kind of silent-fallback failure mode this project has been burned by before.
    """
    zip_path = CACHE_DIR / f"{COMPETITION}.zip"
    zip_path.unlink(missing_ok=True)
    run_kaggle("competitions", "leaderboard", COMPETITION, "-d", "-p", str(CACHE_DIR))
    if not zip_path.exists():
        print("ERROR: leaderboard download did not produce a zip file -- aborting", file=sys.stderr)
        sys.exit(1)
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return zf.read(csv_name).decode("utf-8")


def list_top_kernels(top_n: int) -> list[dict]:
    """Real public kernels for this competition, sorted by vote count."""
    result = run_kaggle(
        "kernels", "list", "--competition", COMPETITION,
        "--sort-by", "voteCount", "--page-size", "100", "--csv",
    )
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        print("WARNING: kernel list came back empty", file=sys.stderr)
        return []
    header = lines[0].split(",")
    kernels = []
    for line in lines[1 : top_n + 1]:
        # kaggle's --csv output quotes fields containing commas; use csv module properly.
        import csv as csv_module
        import io
        row = next(csv_module.reader(io.StringIO(line)))
        kernels.append(dict(zip(header, row, strict=False)))
    return kernels


def pull_kernel_text(ref: str) -> str | None:
    """Downloads a kernel's real .ipynb and extracts markdown+code cell text.

    Returns None on failure -- callers must skip the notebook entirely rather
    than feeding Gemini a placeholder string it might mistake for real content.
    """
    dest = CACHE_DIR / ref.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    run_kaggle("kernels", "pull", ref, "-p", str(dest))
    ipynb_files = list(dest.glob("*.ipynb"))
    if not ipynb_files:
        print(f"WARNING: failed to pull {ref} -- skipping it entirely", file=sys.stderr)
        return None
    with open(ipynb_files[0], encoding="utf-8") as f:
        nb = json.load(f)
    parts = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if source.strip():
            parts.append(f"[{cell.get('cell_type')}]\n{source}")
    return "\n\n".join(parts)


def build_context_blocks() -> list[str]:
    blocks = []
    for rel_path in CONTEXT_FILES:
        path = REPO_ROOT / rel_path
        with open(path, encoding="utf-8") as f:
            content = f.read()
        blocks.append(f"## Our repo: {rel_path}\n```\n{content}\n```")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=6, help="Number of top-voted kernels to pull")
    parser.add_argument(
        "--out", default=None,
        help="Output markdown path (default: COMPETITOR_RESEARCH_<date>.md in repo root)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching leaderboard for {COMPETITION}...")
    leaderboard_csv = fetch_leaderboard()

    print(f"Listing top {args.top_n} public kernels by vote count...")
    kernels = list_top_kernels(args.top_n)
    if not kernels:
        print("ERROR: no kernels found -- aborting", file=sys.stderr)
        sys.exit(1)

    notebook_blocks = []
    for k in kernels:
        ref = k.get("ref", "")
        title = k.get("title", "")
        votes = k.get("totalVotes", "?")
        print(f"Pulling {ref} ({votes} votes)...")
        text = pull_kernel_text(ref)
        if text is None:
            continue
        notebook_blocks.append(
            f"## Competitor notebook: {ref}\nTitle: {title}\nVotes: {votes}\n\n{text[:20000]}"
        )

    if not notebook_blocks:
        print("ERROR: every kernel pull failed -- aborting rather than synthesizing from nothing", file=sys.stderr)
        sys.exit(1)

    context_blocks = build_context_blocks()

    full_prompt = (
        f"{SYSTEM_PREAMBLE}\n\n"
        f"# Real current leaderboard for {COMPETITION}\n```csv\n{leaderboard_csv}\n```\n\n"
        + "\n\n".join(context_blocks)
        + "\n\n# Real competitor notebooks (pulled via Kaggle API this run)\n\n"
        + "\n\n".join(notebook_blocks)
    )

    print("Calling Gemini for synthesis...")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents=full_prompt)

    out_path = Path(args.out) if args.out else REPO_ROOT / f"COMPETITOR_RESEARCH_{date.today().isoformat()}.md"
    header = (
        f"# Competitor Research — {date.today().isoformat()}\n\n"
        f"Generated by `scripts/research_competitors.py` "
        f"({datetime.now(UTC).isoformat()}Z) from {len(kernels)} real public kernels "
        f"pulled via the Kaggle API and the live leaderboard for `{COMPETITION}`. "
        f"Cross-check any [ADOPT-CANDIDATE] item against the actual pulled notebook "
        f"(cached under `.cache/competitor_research/`) before trusting it -- this report is "
        f"Gemini's synthesis, not independently re-verified.\n\n---\n\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + response.text)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
