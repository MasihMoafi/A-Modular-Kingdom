# Elpis Session Handoff

## Goal

Ship Elpis `0.1.0` with internal RAG, portable context continuity, durable memory,
OpenAI subscription authentication, OpenRouter support, and a clean release path.

## Current State

- Canonical work is on `main`; the old prototype remains only under
  `archive/pre-cleanup-20260716`.
- Repository cleanup and the one-tool internal RAG path are accepted complete.
- Durable memory is Elpis-owned under `~/.elpis`, bounded, age-aware, diversity-ranked,
  recall-tracked, reviewable, and protected from one-off promotion.
- Elpis writes portable workspace `GOAL.md` and compact per-turn `ES.md` checkpoints.
  Fresh threads admit them automatically; exact resume keeps the native thread.
- Compaction stops if the portable continuity files cannot be safely synced first.
- `/status` reports admitted rules, goal, checkpoint, and memory summary with source,
  size, lifetime, and reason.
- OpenAI subscription remains the default authentication path. OpenRouter is a built-in,
  separately keyed provider selected with `--provider openrouter`.
- The binary is versioned `0.1.0`. Tagged builds publish a checksummed Linux x86_64
  release, and `scripts/install-elpis.sh` verifies and installs it atomically.
- The public README now reflects the current product instead of the obsolete prototype.

## Verification

- Local static checks passed: direct pinned Rust formatting, `git diff --check`, shell
  syntax, and workflow YAML parsing.
- No local Cargo or Rust compilation was run.
- GitHub Actions run `29523016397` is the authoritative remote build and focused test run
  for commit `ff86755`; record its final result before claiming acceptance.

## Next Action

When run `29523016397` passes, download and install its binary artifact. Verify locally:

1. `elpis --version` reports `0.1.0` and the amber Elpis interface launches.
2. `/status` shows the admitted context files and reasons.
3. Exact resume retains a task; a fresh thread continues leanly from `GOAL.md`/`ES.md`.
4. A related new session recalls a taught fact with provenance; an unrelated session does
   not receive it; review and reset remain available.
5. OpenRouter starts with its own key and does not request ChatGPT login.

After those checks, update `TASKS.md`, tag `v0.1.0`, and let the verified release workflow
publish the first binary. Do not add `/auto`, dreaming, extra providers, dictation, or
Codex pruning before this acceptance boundary.

## Boundaries

- Codex/ChatGPT login is authentication-only; Elpis owns context, memory, policy, and
  provider choice.
- Do not run Cargo locally.
- Do not start subagents without Masih approving a specific bounded task.
