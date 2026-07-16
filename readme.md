# Elpis

[![Linux verification](https://github.com/MasihMoafi/Elpis/actions/workflows/embedded-elpis-linux.yml/badge.svg)](https://github.com/MasihMoafi/Elpis/actions/workflows/embedded-elpis-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A terminal coding agent that keeps the task intact.**

Elpis is designed for long coding sessions where an agent must preserve its purpose, admitted context, decisions, and useful memory without forcing the user to reconstruct the work after every session.

It uses the proven Codex Rust execution foundation for terminal interaction, patches, permissions, sandboxing, and ChatGPT authentication. Elpis owns the surrounding product: retrieval, portable context, durable memory, provider choice, behavior, and visual identity. Codex is an authentication and execution foundation, not a separately delegated Elpis runtime.

The name comes from the Ancient Greek word for hope or expectation. Here it describes the intended environment: an agent that can work clearly, truthfully, and continuously.

> **Release status:** under active development. The first `v0.1.0` release will be tagged only after the documented acceptance checks pass.

## The problem

Coding agents can execute quickly while losing the reason behind the work. Long sessions accumulate context, compaction hides details, and fresh sessions often begin with a manual reconstruction of goals, constraints, decisions, and verification.

Elpis treats continuity as a product boundary rather than an afterthought.

## What works

- Native Ratatui terminal interface with streaming commands, patches, permission modes, sandboxing, mouse selection, sessions, and compaction.
- ChatGPT subscription authentication inherited from Codex.
- Built-in OpenRouter support through `OPENROUTER_API_KEY`, separate from ChatGPT login.
- One internal, read-only RAG service with `/rag` and autonomous retrieval.
- Portable workspace continuity through compact `GOAL.md` and `ES.md` files.
- Exact thread resume or lean continuation through a fresh thread.
- Visible `/status` reporting for admitted context, including source, size, lifetime, and reason.
- Elpis-owned local memory with bounded artifacts, age-aware and diverse retrieval, recall tracking, and promotion only after repeated use across distinct contexts.

The context and memory foundations are implemented and undergoing end-to-end release acceptance. [TASKS.md](TASKS.md) is the truthful current-state record.

## How it works

```text
workspace evidence
      |
      v
compact portable context -----> terminal agent
      |                              |
      v                              v
bounded local memory <--------- verified work
```

Elpis keeps exact evidence on disk while admitting only compact, explainable context into the model. It separates thread resume, workspace continuity, retrieved evidence, and promoted memory so each source has a visible reason and lifetime.

## Install

Tagged releases publish a verified Linux x86_64 binary and checksum. From a checkout:

```bash
scripts/install-elpis.sh
```

The installer verifies the checksum and atomically installs `elpis` into `~/.local/bin`.

OpenAI subscription login remains the default. To use OpenRouter instead:

```bash
export OPENROUTER_API_KEY="your-key"
elpis --provider openrouter --model "provider/model"
```

The OpenRouter key is separate from ChatGPT login.

## Verification

Linux verification and binary builds run through [`.github/workflows/embedded-elpis-linux.yml`](.github/workflows/embedded-elpis-linux.yml). The Python RAG service has focused tests under `tests/`.

Release acceptance is tracked in [TASKS.md](TASKS.md), with supporting evidence in [`docs/`](docs/). A green badge means the configured workflow passed; it does not mean every planned feature is complete.

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

Rust verification and Linux binary builds run in GitHub Actions. Local Rust compilation is intentionally avoided on the maintainer's workstation. The Python RAG service uses the project virtual environment and focused tests under `tests/`.

Elpis is not yet presented as a stable public release.

## License

Elpis is MIT licensed. The contained Codex-derived source retains its upstream Apache-2.0 notices and attribution.
