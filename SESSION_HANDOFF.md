# Elpis Session Handoff

## Goal

Stabilize the contained Codex foundation, restore fast startup, and then completely
remove only Masih's approved unwanted features without losing retained Codex behavior.

## Current State

- Canonical code baseline before this checkpoint: `main` at `419384d`; `origin/main`
  matches.
- The installed `elpis` is the contained Codex-derived TUI, not the archived prototype
  and not code loaded from the donor Codex clone.
- Verified foundation behavior includes ChatGPT authentication, streaming tool/file
  rendering, permission modes, sandboxing, mouse selection, sessions, and compaction.
- `b135e7a` simplified the visible command surface; `695c6a8` formatted `/settings`;
  `419384d` removed startup tips and hid `/test-approval`.
- Important correction: most unwanted command variants and dispatch paths still exist.
  Hiding them is not complete deletion and is not accepted as done.
- User-observed regression: `elpis` takes about five seconds to become usable; the old
  prototype took under one second.
- PyTorch is not part of the Rust TUI. It belongs to the separate Python RAG service;
  it now loads only after an explicit RAG query.
- Found and corrected in the working tree: `src/agent/host.py` was the old unified MCP,
  exposed 21 unrelated tools, imported memory/Qdrant before handshake, and referenced
  nonexistent `rag.fetch_1` and `rag.fetch_2` modules.
- The replacement is a minimal standard-library MCP exposing only
  `query_knowledge_base`, backed by the real `rag.fetch` module on demand.
- Measured: old host import 1.64 seconds/105 MB; new handshake 0.04 seconds/13 MB;
  profiled TUI first frame 0.049 seconds. Masih confirmed the fresh launch is fast.
- Historical cause: the one-tool host already existed on the archived prototype branch,
  but the Codex-foundation transition inherited an older unified host from the embedded
  launcher line. The regression was missed during migration; it was not Codex overhead.
- RAG indexing occurs only after an explicit query when no persisted index exists. It
  loads Torch/models and scans up to 100 files then; it is not Elpis startup work.
- Done in the current subtraction: deleted the unreferenced `src/memory`, `src/tools`,
  old query parser, and old tool-execution helper. `pyproject.toml` now packages only
  the RAG host, RAG implementation, and its proxy helper and declares only RAG runtime
  dependencies. Codex owns the removed general agent capabilities.
- Gemini/runtime-boundary work remains parked on `agent/runtime-boundary` and is the
  least important current work.

## Approved Command And Feature Contract

Completely remove these user-facing features and their dedicated implementation:

- experimental UI, except keep **Keep computer awake** under `/settings`;
- personality, `/vim`, `/plan`, `/pets`, `/usage`, `/mention`, `/raw`, `/archive`,
  `/memories`, memory debug commands, and `/test-approval`;
- startup tips, including the remote test announcement;
- `/exit` while retaining `/quit`.

Retain or rename exactly as follows:

- retain `/agent`, `/skills`, `@` file attachment, permissions, mouse selection, and
  all Codex-quality command/file rendering;
- `/stop` becomes `/kill`, with Ctrl+K invoking it only when the composer is empty;
- `/keymap` becomes `/hotkeys`; `/delete` becomes `/del`;
- `/settings` contains Keep computer awake only; it remains off by default.

Do not delete shared token accounting, memory/session primitives, or editor behavior
merely because an unwanted slash command used them. Remove a complete product feature,
not unrelated foundations.

## UI And Future Features

- UI change is visual-only for now: colors and styling, with no content/layout feature
  redesign. Preserve what Codex shows and how it behaves.
- Add dictation to the roadmap. Audit contained Codex first for an existing option;
  otherwise design a consent-based Whisper path that inserts editable, unsent text.
- Research Kiro from official sources before borrowing anything; verify source access,
  language, license, and claimed speed instead of assuming them.

## Evidence And Constraints

- Latest remote Rust CI: run `29486136493` passed for `419384d`.
- Installed binary SHA-256:
  `0793a5179bfd844fcf755f7370bed757a36e4aac3925b6985b9d1fbc2b64cf01`.
- Do not compile Rust locally; it disrupts Masih's workstation. Use the remote workflow.
- No subagents are active. Use one worktree per genuinely parallel task, but keep the
  immediate blocker with the main agent.
- Startup fix is committed as `3e06042`.
- Current deletion changes: `pyproject.toml`, `src/agent/query_parser.py`,
  `src/agent/tool_execution.py`, `src/memory/`, `src/tools/`, `GUIDE.md`,
  `FEATURES.json`, and this handoff.
- Verification: MCP initialize and tools/list pass with exactly one tool and the narrow
  schema; `.venv/bin/python -m compileall -q src`, `jq -e . FEATURES.json`, and
  `git diff --check` pass. No Rust build is required for this Python service change.
- Startup performance is accepted. No push has been performed.

## Next Action

Commit the proven-unused Python tool and memory deletion as its own checkpoint. Then
choose whether the next bounded task is the appearance-only UI pass or continued Codex
feature subtraction; do not mix those changes in one commit.

## After Startup Is Understood

Remove the approved features in small, complete, remotely verified commits. Start with
the lowest-risk residue (`tooltips` and `test-approval`), then command-only paths, then
broader features such as plan/personality/raw/vim. Keep `FEATURES.json` truthful after
each step: hidden is not deleted, and CI passing is not user acceptance.

## Do Not Repeat

- Do not revive or compare against the archived prototype as if it were current code.
- Do not call hidden commands deleted.
- Do not attribute Rust TUI startup cost to PyTorch without evidence.
- Do not restore the old unified MCP, memory tools, or its other unrelated tools to
  `elpis-rag`; it has one tool only.
- Do not change UI content or layout while performing the visual identity pass.
- Do not prioritize Gemini or rewrite inherited Codex capabilities.
