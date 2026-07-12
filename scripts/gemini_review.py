"""
Adversarial code review via the Gemini API.

Sends a diff and/or specific files to Gemini with an adversarial-review
framing and prints its critique. Built to replace the manual copy-paste
round-trip (pasting fixes into the Gemini web app, pasting its answer back)
used earlier this session with a direct, scriptable cross-check.

Usage:
    python scripts/gemini_review.py --prompt "critique this fix for bugs" --diff
    python scripts/gemini_review.py --prompt "review this function for edge cases" --files src/train.py
    python scripts/gemini_review.py --prompt "..." --diff --diff-ref HEAD~3
"""

import argparse
import os
import subprocess
import sys

from google import genai

MODEL = "gemini-3.1-pro-preview"

SYSTEM_PREAMBLE = """You are performing an ADVERSARIAL code review. Your job is to find real
bugs, wrong assumptions, edge cases, and design flaws -- not to validate or praise the work.
Be direct and specific: cite exact file:line locations, explain the concrete failure scenario
(what input/state triggers it and what goes wrong), and rate your confidence (confirmed by
reading the code vs suspected). If you find nothing wrong after genuinely trying, say so
plainly rather than inventing minor nitpicks. Do not hedge with vague "consider..." suggestions
-- either it's a real problem or it isn't."""


def get_diff(diff_ref: str | None) -> str:
    # encoding="utf-8" explicitly: Windows' default locale encoding (cp1252)
    # chokes on non-ASCII characters (e.g. em-dashes) that real commit
    # diffs in this repo contain -- same class of issue hit earlier this
    # session with Kaggle log downloads needing PYTHONUTF8=1.
    cmd = ["git", "diff"] + ([diff_ref] if diff_ref else [])
    output = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    if not output.strip():
        # Working tree is clean -- fall back to the staged diff.
        cmd = ["git", "diff", "--cached"] + ([diff_ref] if diff_ref else [])
        output = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="The review question/framing")
    parser.add_argument("--files", nargs="*", default=[], help="File paths to include as context")
    parser.add_argument("--diff", action="store_true", help="Include `git diff` output")
    parser.add_argument(
        "--diff-ref", default=None,
        help="Base ref for the diff (e.g. HEAD~3); default is working-tree diff against HEAD"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    context_parts = []

    if args.diff:
        diff_output = get_diff(args.diff_ref)
        context_parts.append(f"## Git diff\n```diff\n{diff_output}\n```")

    for fpath in args.files:
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        context_parts.append(f"## File: {fpath}\n```python\n{content}\n```")

    if not context_parts:
        print("ERROR: no context provided -- pass --diff and/or --files", file=sys.stderr)
        sys.exit(1)

    full_prompt = (
        f"{SYSTEM_PREAMBLE}\n\n# Review request\n{args.prompt}\n\n"
        + "\n\n".join(context_parts)
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents=full_prompt)
    print(response.text)


if __name__ == "__main__":
    main()
