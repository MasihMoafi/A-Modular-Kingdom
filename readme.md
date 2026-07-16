# Elpis

Elpis is a terminal coding agent designed to keep its purpose, context, and memory clear
across long tasks and new sessions.

It uses the proven Codex Rust execution foundation for terminal interaction, patches,
permissions, sandboxing, and ChatGPT authentication. Elpis owns the surrounding product:
retrieval, portable context, durable memory, provider choice, behavior, and visual identity.
Codex is an authentication and execution foundation, not a separately delegated Elpis
runtime.

The name comes from the Ancient Greek word for hope or expectation. Here it also describes
the goal of the project: an environment in which an agent can work clearly, truthfully, and
continuously without forcing the user to reconstruct the task after every session.

## What works

- A native Ratatui terminal interface with streaming commands, patches, permission modes,
  sandboxing, mouse selection, sessions, and compaction.
- ChatGPT subscription authentication inherited from Codex.
- A built-in OpenRouter provider using `OPENROUTER_API_KEY`, separate from ChatGPT login.
- One internal, read-only RAG service with `/rag` plus autonomous retrieval.
- Portable workspace continuity through compact `GOAL.md` and `ES.md` files.
- Exact thread resume or lean continuation through a fresh thread.
- A visible `/status` account of admitted context, including source, size, lifetime, and
  reason.
- Elpis-owned local memory with bounded artifacts, age-aware and diverse retrieval, recall
  tracking, and promotion only after repeated use across distinct contexts.

The context and memory foundations are implemented and are undergoing end-to-end release
acceptance. See [TASKS.md](TASKS.md) for the truthful current state.

## Install

Tagged releases publish a verified Linux x86_64 binary and checksum. From a checkout:

```bash
scripts/install-elpis.sh
```

The installer verifies the checksum and atomically installs `elpis` into
`~/.local/bin`. The first `v0.1.0` release will be tagged only after the acceptance checks
pass.

OpenAI subscription login remains the default. To use OpenRouter instead:

```bash
export OPENROUTER_API_KEY="your-key"
elpis --provider openrouter --model "provider/model"
```

The OpenRouter key is separate from ChatGPT login.

## Principles

- Be honest about what is implemented and what is only planned.
- Keep exact evidence on disk and model-visible context small.
- Preserve the user's active goal, decisions, constraints, verification, and next action.
- Treat memory as curated reusable knowledge, not a transcript dump.
- Keep authentication, provider choice, context, and memory boundaries explicit.
- Prefer small, reversible changes backed by focused checks.

## Repository map

- `codex-rs/` — contained Rust application and TUI.
- `src/` — the single-tool Python RAG service.
- `GUIDE.md` — product and architecture source of truth.
- `REQUIREMENTS.md` — accepted requirements and unresolved decisions.
- `TASKS.md` — release priorities and acceptance state.
- `docs/CONTEXT_AND_SESSIONS.md` — context and continuation contract.

## Development

Rust verification and Linux binary builds run through
`.github/workflows/embedded-elpis-linux.yml`; local Rust compilation is intentionally
avoided on the maintainer's workstation. The Python RAG service uses the project virtual
environment and focused tests under `tests/`.

Elpis is under active development and is not yet presented as a stable public release.

## License

Elpis is MIT licensed. The contained Codex-derived source retains its upstream Apache-2.0
notices and attribution.
