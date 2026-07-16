# Elpis Session Handoff

## Goal

Ship a clean first Elpis release whose distinctive foundations are internal RAG,
provider-neutral context/session continuity, and durable memory.

## Current State

- `main` is ten commits ahead of `origin/main` at `108793c`; nothing has been pushed.
- The Gemini cleanup was useful: it restored the required RAG proxy helper and removed
  obsolete `VectorIndex` code, unused notebook splitting functions, `/test-approval`,
  and `/exit`. The Python import and compile checks pass.
- The merged worker worktree and branch were removed. Only the main worktree remains;
  the intentional `archive/pre-cleanup-20260716` branch preserves old history.
- `TASKS.md` replaces the rejected JSON tracker and orders work as foundational,
  important, and nice-to-have.
- The Codex-derived TUI, permissions, tool rendering, sessions, compaction, mouse
  selection, and ChatGPT authentication remain the execution foundation.
- Repository cleanup and internal RAG are accepted as complete by Masih.
- The Python MCP exposes only the read-only `query_knowledge_base` tool; `/rag` routing
  and autonomous retrieval are retained behavior.
- The interrupted external memory task made no local project change. The current local
  implementation adds distinct recall-context tracking and a 3-recall/2-context gate
  before new short-term evidence may enter durable `MEMORY.md`.
- Important correction: `codex-rs/memories/` already provides an enabled-by-default
  Rust pipeline for rollout extraction, consolidation, retrieval, citations, and local
  memory artifacts. The first OpenClaw-derived promotion behavior is integrated locally
  but not remotely compiled or accepted.
- First-release provider scope is OpenAI subscription plus OpenRouter. Additional
  providers are important later; `/auto` is a nice-to-have.

## Verification

- `.venv/bin/python -m compileall -q src`: passed.
- `PYTHONPATH=src .venv/bin/python` import of `rag.qdrant_backend`: passed.
- `.venv/bin/python -m pytest -q tests/test_rag_mcp_host.py`: passed.
- MCP initialize and `tools/list`: passed with exactly one read-only RAG tool.
- Direct backend explicit-path RAG query: passed in 8.6 seconds with correct sourced
  context/session results.
- Direct backend workspace RAG query: passed in 8.5 seconds with sourced project results.
- `git diff --check`: passed.
- No local Cargo or Rust compilation was run.
- Rust memory changes have only static review and `git diff --check`; remote compilation
  and focused tests remain required.

## Recent Changes

- `5b92ae1` replaces `FEATURES.json` with `TASKS.md` and truthful release tiers.
- `6d60288` advertises RAG as read-only; `54623f6` records backend acceptance.
- `fa94cd1` adds distinct recall tracking and the durable promotion gate.
- The worktree is clean; none of these local commits have been pushed.

## Next Action

Remotely compile and run focused state/memory tests for the recall-promotion slice, then
complete bounded storage, age-based fading, diverse retrieval, provenance, and memory
review/deletion.

Do not add dream narratives, cron scheduling, an MCP memory adapter, temporal decay, or
MMR until the recall-promotion slice passes. Decay and MMR belong in the later retrieval
layer, not in the consolidation scheduler.

## Boundaries

- Preserve Codex/ChatGPT login as authentication only; do not delegate the Elpis runtime
  to Codex.
- Do not claim planned features are implemented.
- Do not run Cargo locally.
- Do not start subagents unless Masih sanctions a specific task.
