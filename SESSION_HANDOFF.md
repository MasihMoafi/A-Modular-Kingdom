# Elpis Session Handoff

## Goal

Stabilize the imported Codex foundation as canonical Elpis, then make it lighter by
removing obsolete and unwanted surfaces one bounded, verified deletion at a time.

## Current State

- Done: the old dirty prototype is preserved exactly on
  `archive/legacy-prototype-20260716` at `e043d6a`.
- Done: `main` fast-forwards to the contained Codex-derived Elpis implementation from
  `agent/elpis-embedded-launcher`.
- Done: the pinned `codex-rs/` workspace builds a user-facing `elpis` binary without
  donor-clone access. The remotely built binary is installed at
  `/home/masih/.local/bin/elpis`.
- Inherited from Codex: command/file lifecycle rendering, permission modes, sandboxing,
  mouse handling, native sessions, native compaction, and ChatGPT authentication.
- Parked: Gemini/runtime-boundary work remains isolated on `agent/runtime-boundary` and
  is not current priority.
- Done: commit `dd36256` removes the superseded root `tui/`, old Python agent main,
  and obsolete Debian builder after reference checks proved the canonical binary and
  live RAG MCP do not use them. The ignored old TUI build cache was also removed.
- Verified: GitHub Actions run `29449900034` passed the remote Rust checks after that
  deletion. Its artifact is installed with SHA-256
  `bcb9dc5c5402e15cbb670210c31b94ccac58255bf7aaafca0b48831f4a395f8b` and launched
  successfully as `Elpis · Codex runtime` from canonical `main`.
- Blocked: none.

## Evidence

- Foundation acceptance: `docs/FOUNDATION_CODEX_BASELINE_EVIDENCE.md`.
- Donor-independent launch: `docs/LAUNCH_SMOKE_EVIDENCE.md`.
- Embedded build/install: `docs/EMBEDDED_ELPIS_EVIDENCE.md`.
- Dirty legacy state has no uncommitted files after archival.
- No local Rust compilation is permitted because it disrupts Masih's workstation;
  use the remote workflow for Rust checks.

## Next Action

Inventory the remaining non-`codex-rs/` Python/MCP, memory, RAG, tools, packaging, and
documentation surfaces as keep, remove, or decide-later. Select the next smallest
proven-unused surface; do not delete another Codex capability without an acceptance
check that names what must remain visible.

## Do Not Repeat

- Do not treat the archived prototype's missing rendering, permissions, or mouse
  behavior as gaps in the imported foundation.
- Do not extend Gemini/runtime work before foundation subtraction is stable.
- Do not rewrite inherited Codex capabilities from scratch.
- Do not compile Rust locally.
