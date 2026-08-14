# Loop Engineering Approach — for this ML problem

Source concept: Lance Martin (Anthropic), "Designing Loops with Fable 5," June 2026. Primary
Anthropic URL not found directly; grounded via secondary write-ups (see Sources below).

## The six-part loop

Every working agent loop has these parts; failures trace to a missing one.

1. **Trigger** — cron, CI failure, webhook, or self-paced (`/loop`).
2. **Rules load** — pulls in frozen project knowledge (`CLAUDE.md`/`PRD.md`) + accumulated memory.
3. **Executor** — does *one bounded unit of work* ("fix the next failing test," not "make progress").
4. **Verifier** — grades the output in a **fresh, independent context window** — not the same
   agent self-critiquing.
5. **Memory write** — records what happened, distilled into reusable rules.
6. **Stop check** — done / escalate / park / go again.

## Key finding behind it

Verifier sub-agents running in an independent context caught ~73% of seeded issues vs. 7–33% for
self-critique in the same context. Mechanism: "the maker sees its own reasoning trail; the
verifier sees only the artifact and the rubric." On a Parameter Golf ML benchmark (train an
optimal model under 16MB in <10 min), Fable 5 run through this loop improved the pipeline ~6x more
than Opus 4.7 run without it, mainly by continuing to try structural changes instead of getting
stuck refining one small tweak.

## Where this already exists in this project

This repo's `CLAUDE.md` independently arrived at the same verifier-separation principle after
getting burned repeatedly ("never trust a sub-agent's 'done'/'tests pass' claim without
independently re-checking" — 6+ bugs caught this way). The GSD skill set already splits
planner/executor/verifier roles (`gsd-executor`, `gsd-verifier`, `gsd-plan-checker`), and the
training loop already has stop-checks (`TrainingLoop.train_epoch()`'s >50%-fallback-rate hard-fail,
`validate_epoch()`'s zero-node circuit breaker). The gap is that these pieces run ad hoc per phase
rather than as one continuously-cycling loop with memory carried forward automatically.

## Proposed mapping for this repo

| Loop part | Concrete piece here |
|---|---|
| Trigger | `/loop` skill (self-paced via `ScheduleWakeup`), scoped to **local-only** iteration — never auto-triggers a real Kaggle GPU run |
| Rules load | `PRD.md` + `CLAUDE.md` + memory files |
| Executor | one bounded change per cycle — e.g. "try the next detector threshold/NMS variant" or "implement the next unimplemented PRD §8 task" — dispatched as a scoped `Agent` call, not open-ended "improve the pipeline" |
| Verifier | a **separate** `Agent` call that only sees the diff + `src/evaluation.py` output, independently reruns the local eval harness, and reports the real score delta — no access to the executor's reasoning trail |
| Memory write | append to `DEFERRED_IMPROVEMENTS.md`/PRD changelog: what changed, score before/after, keep or revert |
| Stop check | plateau in local score over N cycles, GPU/time budget hit, or baseline (0.763) reached → escalate to human before spending a real Kaggle submission |

## Deliberate deviation from full autonomy

Given the project rule that submissions are scarce/rate-limited and Kaggle runs cost real
GPU-hours, the loop should hard-gate so the executor/verifier cycle only ever touches the
**local** evaluation harness. Any actual Kaggle training/submission run stays a manual, confirmed
step outside the loop — the trigger never fires one automatically.

## Sources

- [Claude Fable 5, Part 2: Loop Engineering – Ken Huang](https://kenhuangus.substack.com/p/claude-fable-5-part-2-loop-engineering)
- [Loop Engineering – Cobus Greyling](https://cobusgreyling.substack.com/p/loop-engineering)
- [Designing Loops with Fable 5: self-correction and cross-session memory](https://glean.smartcoder.ai/en/a/designing-loops-with-fable-5-self-correction-and-cross-sessi-p8jwfn)
- [Now what exactly is loop engineering? – Zyte](https://www.zyte.com/blog/now-what-exactly-is-loop-engineering-and-where-do-anthropics-fable-5-model-and-web-scraping-fit-in/)
