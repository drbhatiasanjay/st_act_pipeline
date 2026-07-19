# Human-gated Claude and Codex automation

This repository uses GitHub as the control plane for bounded implementation and
independent review. The first version intentionally automates handoff and review,
not authority.

## Trust model

- A human starts each Claude write job with `workflow_dispatch`.
- The `agent-write` environment releases the Claude credential only after its
  configured approval gate.
- Claude works from an exact base SHA, opens or updates a draft PR, and posts a
  SHA-anchored `[CLAUDE_HANDOFF]` marker.
- A valid marker from a trusted collaborator automatically starts a read-only
  Codex review.
- Codex can post `CHANGES_REQUIRED` or `APPROVED — HUMAN_GATE_REQUIRED`; it cannot
  write the checkout, push, merge, deploy, train, invoke Kaggle, or start a wave.
- A human remains responsible for remediation dispatches, merge approval, later
  waves, deployment, and GPU execution.

## Required repository configuration

The workflows are dormant unless the repository variable
`AGENT_AUTOMATION_ENABLED` is exactly `true`.

Before enabling it:

1. Create the `agent-write` GitHub environment and configure required reviewers,
   prevent self-review, and disallow bypass where the repository plan supports it.
2. Store `ANTHROPIC_API_KEY` as an `agent-write` environment secret.
3. Store `OPENAI_API_KEY` as a repository or dedicated read-review environment
   secret. The Codex action proxies the key and receives read-only repository
   permissions.
4. Protect `master`: require a pull request, required deterministic checks,
   resolved conversations, and approval of the latest pushed commit. Disable
   administrative bypass where appropriate.
5. Keep GitHub auto-merge disabled. These workflows never call a merge API.
6. Install/configure the official Claude GitHub App if using its app-token path;
   otherwise the workflow uses its job-scoped `GITHUB_TOKEN`.

## Starting one wave

1. Record the bounded scope and invariants in an **Agent wave** issue.
2. Open **Actions → Agent - Claude build → Run workflow**.
3. Supply the wave name, exact base SHA, dedicated branch, and bounded task text.
4. Approve the `agent-write` environment gate.

When Claude posts a valid handoff marker, Codex review starts automatically. A
malformed marker, untrusted actor, cross-repository head, stale SHA, or duplicate
verdict is rejected before any model credential is exposed.

## Deliberately out of scope

- No bot-to-bot remediation loop. A human dispatches every write job.
- No automatic merge or next-wave launch.
- No invocation of `scripts/sync_kaggle_src.py`.
- No deployment, training, GPU run, Kaggle API call, or `GIT_SHA.txt` mutation.
- No execution of PR-controlled workflow code with write credentials.

## Official actions

- `anthropics/claude-code-action@v1`
- `openai/codex-action@v1`
- `actions/checkout@v5`
- `actions/github-script@v7`

