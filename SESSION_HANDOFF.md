# Elpis Session Handoff

## Goal

Ship a clean first Elpis release whose distinctive foundations are internal RAG,
provider-neutral context/session continuity, and durable memory.

## Current State

- `main` is five commits ahead of `origin/main` at `fceb981`; nothing has been pushed.
- The Gemini cleanup was useful: it restored the required RAG proxy helper and removed
  obsolete `VectorIndex` code, unused notebook splitting functions, `/test-approval`,
  and `/exit`. The Python import and compile checks pass.
- The merged worker worktree and branch were removed. Only the main worktree remains;
  the intentional `archive/pre-cleanup-20260716` branch preserves old history.
- `TASKS.md` replaces the rejected JSON tracker and orders work as foundational,
  important, and nice-to-have.
- The Codex-derived TUI, permissions, tool rendering, sessions, compaction, mouse
  selection, and ChatGPT authentication remain the execution foundation.
- The Python MCP exposes only `query_knowledge_base` and now advertises it as read-only;
  `/rag` routing exists but its live workspace/path/autonomous acceptance checks are
  incomplete.
- No new memory implementation was found in the working tree, commits, branches, or
  recent reflog. The interrupted memory task made no local project change.
- Important correction: `codex-rs/memories/` already provides an enabled-by-default
  Rust pipeline for rollout extraction, consolidation, retrieval, citations, and local
  memory artifacts. Elpis-specific OpenClaw behavior has not been integrated or tested.
- First-release provider scope is OpenAI subscription plus OpenRouter. Additional
  providers are important later; `/auto` is a nice-to-have.

## Verification

- `.venv/bin/python -m compileall -q src`: passed.
- `PYTHONPATH=src .venv/bin/python` import of `rag.qdrant_backend`: passed.
- `.venv/bin/python -m pytest -q tests/test_rag_mcp_host.py`: passed.
- MCP initialize and `tools/list`: passed with exactly one read-only RAG tool.
- `git diff --check`: passed.
- No local Cargo or Rust compilation was run.

## Current Changes

- Replaced `FEATURES.json` with `TASKS.md`.
- Updated agent routing and requirement documents to use the truthful release tiers.
- These documentation changes are not yet committed.

## Next Action

Complete foundational task F2 with three live acceptance checks: direct workspace
query, explicit-path query, and autonomous retrieval for a broad question.

Do not continue memory implementation until the existing Codex Rust pipeline and the
current OpenClaw pipeline have been compared and Masih approves the behavioral contract.

## Boundaries

- Preserve Codex/ChatGPT login as authentication only; do not delegate the Elpis runtime
  to Codex.
- Do not claim planned features are implemented.
- Do not run Cargo locally.
- Do not start subagents unless Masih sanctions a specific task.
