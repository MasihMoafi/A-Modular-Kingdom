# Elpis

[![Linux verification](https://github.com/MasihMoafi/Elpis/actions/workflows/embedded-elpis-linux.yml/badge.svg)](https://github.com/MasihMoafi/Elpis/actions/workflows/embedded-elpis-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A terminal coding agent built to continue the work, not restart it.**

Elpis exists because long agent sessions lose shape. Goals get buried in transcripts,
decisions disappear after compaction, and a new session often begins with the user explaining
the same task again.

Elpis keeps the active goal, admitted context, decisions, verification, and next action
explicit. Exact evidence stays on disk. Model-visible context stays small.

It contains the Codex Rust execution foundation for terminal interaction, patches,
permissions, sandboxing, and ChatGPT authentication. Elpis owns its retrieval, portable
context, local memory, provider support, behavior, and interface.

> **Current state:** under active development. `v0.1.0` will be tagged after the release
> checks in [TASKS.md](TASKS.md) pass.

## What works

- Native Ratatui terminal interface with streaming commands, patches, permission modes,
  sandboxing, mouse selection, sessions, and compaction.
- ChatGPT subscription authentication inherited from Codex.
- OpenRouter support through `OPENROUTER_API_KEY`, separate from ChatGPT login.
- Claude Sonnet and Gemini Pro/Flash launcher shortcuts through OpenRouter.
- One internal, read-only RAG service with `/rag` and autonomous retrieval.
- Portable workspace state through compact `GOAL.md` and `ES.md` files.
- Exact thread resume or lean continuation in a fresh thread.
- `/status` reporting for admitted context: source, size, lifetime, and reason.
- Local bounded memory with age-aware retrieval, diversity, recall tracking, promotion after
  repeated use across distinct contexts, and an archive for deleted or faded facts.

The context and memory foundations compile and pass focused remote checks, but their
user-visible end-to-end acceptance is still pending. Claude and Gemini currently use
OpenRouter compatibility routes; native vendor adapters are not implemented. The distinctive
amber continuity interface is specified but only partially implemented. [TASKS.md](TASKS.md)
is the current-state record.

## The working model

```text
exact workspace evidence
          |
          v
compact admitted context -----> terminal agent
          |                           |
          v                           v
bounded local memory <--------- verified work
```

Elpis separates four things that are usually mixed together:

1. the current thread;
2. portable workspace state;
3. retrieved evidence;
4. reusable local memory.

Each source should have a reason for being present and a clear lifetime.

## Install

Tagged releases publish a Linux x86_64 binary and checksum. From a checkout:

```bash
scripts/install-elpis.sh
```

The installer verifies the checksum and installs `elpis` into `~/.local/bin` atomically.

OpenAI subscription login is the default. OpenRouter is separate:

```bash
export OPENROUTER_API_KEY="your-key"
elpis --provider openrouter --model "provider/model"
```

Compatibility shortcuts:

```bash
elpis --provider claude
elpis --provider gemini
elpis --provider gemini-flash
```

These shortcuts use OpenRouter and are not native Anthropic or Google adapters.

## Verification

Linux verification and binary builds run through
[`.github/workflows/embedded-elpis-linux.yml`](.github/workflows/embedded-elpis-linux.yml).
Ordinary changes run focused first-release checks and build the Elpis binary. Exhaustive
inherited TUI/app-server regression runs nightly, manually, and for tagged releases.

The Python retrieval service has focused tests under `tests/`. Release acceptance is tracked
in [TASKS.md](TASKS.md). The measured build and dependency-reduction plan is documented in
[`docs/BUILD_AND_REDUCTION_AUDIT.md`](docs/BUILD_AND_REDUCTION_AUDIT.md). A green workflow
badge means that workflow passed. It does not mean unfinished work is finished.

## Principles

- Say what is implemented and what is not.
- Keep exact evidence on disk and admitted context small.
- Preserve the active goal, decisions, constraints, verification, and next action.
- Treat memory as selected reusable knowledge, not stored conversation.
- Keep authentication, provider, context, and memory boundaries visible.
- Prefer small changes with focused checks.

## Repository map

- `codex-rs/` — Rust application and TUI.
- `src/` — the single-tool Python retrieval service.
- `GUIDE.md` — product and architecture source of truth.
- `REQUIREMENTS.md` — accepted requirements and unresolved decisions.
- `TASKS.md` — release work and acceptance state.
- `docs/CONTEXT_AND_SESSIONS.md` — context and continuation contract.
- `docs/UI_IDENTITY.md` — distinctive UI contract and implementation status.
- `docs/BUILD_AND_REDUCTION_AUDIT.md` — build baseline and measured subtraction plan.

## Development

Rust verification and Linux binary builds run in GitHub Actions. Local Rust compilation is
intentionally avoided on the maintainer's workstation. The Python service uses the project
virtual environment and focused tests under `tests/`.

Elpis is not yet presented as a stable public release.

## License

Elpis is MIT licensed. The contained Codex-derived source retains its upstream Apache-2.0
notices and attribution.
