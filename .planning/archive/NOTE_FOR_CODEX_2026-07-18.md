# Note for the Codex session driving TOOLING_ASSETS_PLAN_2026-07-18.md

Cross-checked your instructions against actual current repo state before you start. Two things
need to change in how you execute; everything else in your instructions checks out as-is.

## 1. Base SHA is correct, but the working tree is NOT currently clean

`master` HEAD is exactly `4901d450cb12ed6577634633b71d4f50bc1bc3c6` — matches your stated
authoritative base. However, `git status --short` in this same working directory right now shows:

```
 M .claude/CLAUDE.md              <- see item 2 below
 M bash.exe.stackdump             <- pre-existing, unrelated to this work, leave untouched
 M training_log_smoke_test.csv    <- pre-existing, unrelated to this work, leave untouched
?? LOOP_ENGINEERING_APPROACH_2026-07-18.md
?? TOOLING_ASSETS_PLAN_2026-07-18.md
```

**Do not run `git checkout -b chore/review-tooling 4901d45` in this same working directory.**
Verifying `HEAD == 4901d45` alone will pass, but the *tree* you'd be working in would still carry
all of the dirty/untracked state above — not the clean 4901d45 tree your own scope-lock logic
assumes, and a live risk of clobbering concurrent edits happening here.

**Instead, create an isolated worktree:**
```
git worktree add ../st_act_pipeline-review-tooling 4901d45 -b chore/review-tooling
```
Do all work in that separate directory. This guarantees a genuinely clean checkout of exactly
4901d45, immune to whatever is currently uncommitted in the main working directory, and removes
any chance of two sessions stepping on the same tree at once.

## 2. Your task 1 (CLAUDE.md operational lesson) is already written — don't duplicate it

The exact lesson you're asked to add for the P0-6 patch-generation pollution incident has already
been written and is sitting **uncommitted on master** (not yet on any branch) as of this session.
Exact current diff:

```diff
+- **Patch-generation ordering bug (P0-6, 2026-07-18): any command that touches
+  `training_log_smoke_test.csv` between the pre-generation stash and the final `git diff` step
+  silently re-pollutes it.** During the P0-6 v1→v4 review cycle, the established procedure was
+  "stash the `training_log_smoke_test.csv` pollution, then generate the patch via `git diff`" —
+  but running the repository's diagnostic/smoke-test suite *between* those two steps regenerated
+  that file, and it landed in the first v4 patch attempt as an unauthorized 15th file (the frozen
+  scope for that round was 14 files). Caught only by manually grepping `diff --git` counts in the
+  generated patch before delivery — not by any enforced check. If a future patch round runs any
+  smoke test between stashing and diffing, re-check the file count before shipping the patch; don't
+  assume the stash step alone is sufficient once any test has run afterward.
```

Since your worktree will be a fresh checkout of 4901d45, it will **not** see this uncommitted edit
automatically (it's uncommitted, main-working-directory-only state) — so you won't collide
mechanically, but you would produce a second, differently-worded version of the same lesson if you
write your own from scratch. Preferred resolution: **reuse the text above verbatim** for your task
1 commit rather than re-authoring it, so we don't end up reconciling two versions of the same
entry later. If you have a materially better phrasing, flag it back rather than committing
silently — this file is read as authoritative project memory, so we want one canonical entry, not
two competing ones.

## 3. Everything else in your instructions is confirmed clear, no changes needed

- Item 1d (the cross-project adversarial-reviewer-loop memory entry) is already done — but as a
  *global* memory file outside this repo (`~/.claude/projects/.../memory/feedback_adversarial_reviewer_loop.md`),
  not repo code. This matches your own instruction to exclude it from repo work. No action needed
  from you here.
- Items 1b (`scripts/generate_review_patch.py`) and 1c (`scripts/diagnose_native_crash.py`) are
  untouched — nothing built yet, fully open for you to implement per your detailed spec.
- No production/model/training/validation/submission/checkpoint/kernel/P0-7/P1 files have been
  touched this session — your exclusion list is respected so far on our end too.
- `training_log_smoke_test.csv` being dirty right now is coincidentally the exact file the P0-6
  incident you're documenting is about — it's pre-existing local noise, unrelated to your task,
  not a live repro of the bug. Don't treat it as a signal either way.

## Requested before you proceed

Confirm you're switching to the `git worktree add` approach above before touching anything, and
confirm you'll reuse the CLAUDE.md text verbatim (or flag back a proposed edit) rather than
authoring a second version. Once confirmed, proceed per your original instructions unchanged —
everything else in them (patch-generator requirements, native-crash-diagnostic requirements, test
requirements, completion report) stands as written.

---

## Update (later same session): base SHA in your revised instructions was invalid — fixed here

Your follow-up instruction (the one that added the branch/worktree-existence pre-check and the
"use worktree, don't touch dirty primary worktree" refinement — good, matches item 1 above) cited
the base SHA as:

```text
4901d450cb12ed6577633b71d4f50bc1bc3c6
```

This is **37 characters, not 40** — not a valid SHA-1. `git cat-file -t` on it returns
`fatal: Not a valid object name`. Three characters (`463`) are missing from the middle. The
correct, verified SHA (confirmed twice now this session, matches current `master` HEAD exactly)
is:

```text
4901d450cb12ed6577634633b71d4f50bc1bc3c6
```

Use this corrected SHA in the `git worktree add` command:

```bash
git worktree add ../st_act_pipeline-review-tooling \
  4901d450cb12ed6577634633b71d4f50bc1bc3c6 \
  -b chore/review-tooling
```

**Pre-checks already run on our end, so you don't need to repeat them:** no `chore/review-tooling`
branch exists anywhere (`git branch -a`, checked), no matching worktree exists (`git worktree list`
— 14 worktrees total, none named `review-tooling`), and the target sibling directory
`../st_act_pipeline-review-tooling` does not yet exist. Clear to create fresh once the SHA above is
used.

**Status of item 2 (duplicate CLAUDE.md entry) as of this update:** still uncommitted on master,
not yet reverted — a `git restore -- .claude/CLAUDE.md` was attempted but blocked by a local
safety-net hook requiring explicit human confirmation before discarding uncommitted changes; that
confirmation is still pending on our end. Don't wait on this to proceed — it doesn't block your
worktree, since your fresh checkout of the corrected SHA won't see this uncommitted, main-worktree-only
edit regardless of whether it's later reverted or committed.
