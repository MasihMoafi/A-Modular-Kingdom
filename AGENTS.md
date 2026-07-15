# Elpis Agent Map

## Start

- Read `VISION.md`; it is the stable product intent.
- Read `GUIDE.md`; it is the architecture source of truth.
- Read `REQUIREMENTS.md`; it separates confirmed requirements from open product choices.
- Read `FEATURES.json`; work only on `current_priority` unless Masih changes priority.
- Read `docs/AGENT_DISPATCH.md` before delegating work or creating a worktree.
- Read `SESSION_HANDOFF.md` when it exists, then verify its claims before continuing.
- Obey the global rules in `/home/masih/.codex/AGENTS.md`. The development-harness
  mirror is `/home/masih/Desktop/f/p/harnesses-I-use/skills/dev/`.
- Verify the repository state before editing; preserve unrelated user changes.
- Challenge unclear or solution-first requirements with `$challenge-requirements`
  before planning implementation.

## Context Discipline

- Load only the guide sections and upstream source files needed for the current task.
- Keep the active goal, changed files, verification, blocker, and next action visible.
- Summarize terminal output; do not carry raw logs once their result is known.
- After edits, retain the diff and verification result; reread file bodies on demand.
- Do not add slash commands unless Masih explicitly selects them.
- Worker agents must not edit `VISION.md`, `REQUIREMENTS.md`, `FEATURES.json`, or
  `SESSION_HANDOFF.md`; the coordinator owns those files.

## Definition Of Done

- Behavior is implemented, not merely documented.
- A feature becomes `verified` only when its acceptance check passes and evidence is
  added to `FEATURES.json`.
- Rust changes pass `cargo test`; Python changes pass the narrowest relevant test and
  `.venv/bin/python -m compileall -q src`.
- Known gaps and skipped checks are stated plainly.
