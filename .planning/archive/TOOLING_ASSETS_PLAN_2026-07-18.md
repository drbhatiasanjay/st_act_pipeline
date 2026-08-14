# Tooling/Process Assets Plan — for review and discussion (2026-07-18)

Purpose: a scoped plan for reusable assets surfaced during the P0-6 patch-review cycle
(v1→v4, commit `4901d45`), plus an honest checkpoint on competition progress. Written for
manual review and for pasting into the external reviewer session for a second opinion on
scope/priority — not yet implemented.

---

## 0. Context: why these came up

The P0-6 round ran 4 full adversarial-review patch cycles (v1 rejected, v2 rejected, v3
"CODE REVIEW: PASS" pending full-suite verification, v4 approved and integrated as `4901d45`).
Two real process incidents surfaced during that work that are worth turning into durable
assets rather than re-relying on memory next time:

1. A **self-caught bug in my own patch-generation procedure**: running the repository
   diagnostic suite between "stash the `training_log_smoke_test.csv` pollution" and
   "generate the patch via `git diff`" silently re-polluted that file, and it landed in the
   first v4 patch attempt as an unauthorized 15th file (the frozen scope was 14). Caught by
   grepping `diff --git` counts in the generated patch before delivery — a manual check, not
   an enforced one.
2. A **reproducible-looking native crash that turned out not to be a regression**: v4's
   `test_train_smoke.py::test_training_loop_smoke` segfaulted 4/4 times against 1/1 clean on
   baseline (small sample, looked damning). A controlled differential isolation (10 alternating
   fresh-process runs, 5 baseline / 5 v4) showed both sides crash at an *identical* 80% rate at
   an *identical* native crash site (`torch/autograd/graph.py:829`, PyTorch's own backward
   engine, in a file byte-identical between v3 and v4) — pre-existing local-environment
   flakiness, not a P0-6 regression. This is the third time this repo has hit this general
   class of native-crash-vs-real-bug question (see existing CLAUDE.md entries on the blosc2
   thread-count segfault and the earlier "concurrent torch job" segfault) — the methodology
   used this time was more rigorous than either prior instance and is worth keeping.

---

## 1. Proposed assets

### 1a. CLAUDE.md entry — patch-generation ordering bug
**What:** one new entry in the "Operational lessons" section of
`.claude/CLAUDE.md`, matching the existing style (concrete incident, root cause, fix,
how to recognize it again).
**Why:** this exact ordering trap (any command that touches `training_log_smoke_test.csv`
between the pre-generation stash and the `git diff` step) can recur any time a future patch
round happens to run the smoke test in between.
**Effort:** trivial (~15 min). **Risk:** none — pure documentation.

### 1b. Patch-generation guard script
**What:** `scripts/generate_review_patch.sh` (or `.py`) that encodes the full established
procedure atomically and *enforces* the invariant that broke this time:
1. `git stash push -- training_log_smoke_test.csv` (idempotent — no-op if already clean)
2. Assert `git status --short` matches an expected file list passed as an argument (fails
   loud if anything unexpected is dirty)
3. `git add -N <new files>` → `git diff --binary <baseline SHA> > <output patch>` → `git reset`
4. Re-run the same status assertion from step 2 — if it fails here (e.g. something repolluted
   the tree during step 3), refuse to emit the patch
5. Compute and print SHA-256 (cross-checked two ways)
**Why:** turns a bug I caught by manual vigilance into a bug the tooling can't produce.
**Effort:** small (~1-2 hours incl. testing against a real P0-x round).
**Risk:** low — used only for generating review artifacts, never touches the working tree
destructively (stash is reversible, everything else is read-only or additive).
**Open question:** should the "expected file list" be a repo-wide config (one frozen list per
active P0-phase) or a required CLI argument every time? Config is less typing but risks
going stale silently; CLI argument is more explicit but easy to fat-finger. Worth a second
opinion.

### 1c. Native-crash differential-isolation script
**What:** `scripts/diagnose_native_crash.sh <node_id> <dir_a> <dir_b> [n_iterations]` —
generalizes the ad hoc script written this session: alternates N fresh-process runs of a
given pytest node ID between two worktrees, logs exit code / classification (PASS / FAIL /
SEGFAULT) / wall time / python+torch version per run to a structured log, and prints a
summary table + crash-rate comparison at the end.
**Why:** this exact "is the crash mine or the environment's" question has now come up 3
times in this repo (blosc2 thread count, concurrent-torch-job contention, this session's
v3/v4 investigation) with no reusable tool — each time it's been solved by hand-written
one-off scripts. A parameterized version pays for itself on the next occurrence.
**Effort:** small-medium (~2-3 hours incl. the summary-table logic and README usage notes).
**Risk:** none to production code — pure diagnostic tooling, read-only against the target
worktrees (aside from running the test node itself, which may write to its own test-artifact
paths as it already does today).

### 1d. Memory entry — external adversarial-reviewer-loop pattern
**What:** one `feedback`-type entry in the global memory system capturing the pattern that
carried v1→v4 to a real, correct integration: exact-file-scope-lock (re-authorized via
explicit user confirmation on any expansion, never assumed), reporting derived *only* from
fresh detached worktrees (never the dev worktree's possibly-stale state), dual-hash
cross-checks (`sha256sum` + PowerShell `Get-FileHash`) on every delivered artifact, and a
fixed report template the reviewer could mechanically re-verify against.
**Why:** this is a genuinely reusable collaboration pattern for any future multi-round
external-review workflow (this project or elsewhere), independent of P0-6 specifically.
**Effort:** trivial (~10 min). **Risk:** none.

---

## 2. Sequencing question — the actual open decision

None of 1a–1d touch the model/training code at all. The real open question, given 66 days
remain to 2026-09-22 and the last confirmed real-model signal is a total training collapse
(`max_sigmoid: 0.00000221`, 2026-07-14 Kaggle run, commit `1c5c50f` — **before** the
DetectionLoss reweighting fix, the embryo-disjoint-split fix, and the edge-logits/
BCEWithLogitsLoss fix that have since landed), is:

**Do these 4 assets get built now, or does the very next action become the cheap local CPU
trace (~50 steps, no GPU, few minutes) verifying whether the DetectionLoss reweighting fix
actually reverses the collapse — before any of this tooling work?**

Arguments for tooling first: 1a/1d are ~25 minutes combined and directly reduce risk on the
*next* P0-phase review round, whenever it happens. Arguments for the trace first: none of
this tooling moves the competition score, and the collapse-fix verification has been the
identified next step since before this session started (see the now-superseded
`LOOP_ENGINEERING_APPROACH` planning note from earlier this session) and still hasn't run.

**No recommendation is being made here on purpose — this is exactly the kind of prioritization
call worth a second opinion from the reviewer session before committing time to either path.**

---

## 3. For the reviewer session (if pasted there)

Questions worth putting to the external reviewer alongside this doc:
1. Does 1b's "expected file list" design question (config vs. CLI arg) have an obvious right
   answer given how this project's scope-lock discipline has worked across P0-2 through P0-6?
2. Is there a real risk in deferring the collapse-verification trace further while building
   tooling, given the deadline runway?
3. Any additional guard the patch-generation script (1b) should enforce, given what actually
   went wrong in v1-v4, that isn't captured above?
