# Codex project guidance

The canonical project instructions are in `.claude/CLAUDE.md`. Read that file and
`PRD.md` before planning or changing code. Do not duplicate or reinterpret their
domain rules here.

## Required workflow

- Preserve the working tree. Existing modified or untracked files belong to the user.
- Diagnose reported failures before implementing a fix.
- Add a regression test for bug fixes when practical; assertions must check exact
  behavior, shapes, values, units, or failure modes rather than only types or ranks.
- Treat physical units, anisotropy, scoring math, and scientific design claims as
  high-risk judgments. Follow the verification protocol in `.claude/CLAUDE.md`.
- Keep changes minimal and within the requested phase/scope.
- Never deploy, submit to Kaggle, start training, or spend GPU quota unless the user
  explicitly authorizes that action.
- Before reporting completion, run `powershell -ExecutionPolicy Bypass -File
  scripts/verify.ps1` and report any exclusions or failures honestly.

## Sources of truth

- Competition scope and phased roadmap: `PRD.md`
- Project-specific engineering and operational rules: `.claude/CLAUDE.md`
- Current resumable state: the newest applicable `SESSION_HANDOFF_*.md`
- Vendored scoring provenance: `REFERENCE_IMPLEMENTATION.md`

