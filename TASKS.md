# Elpis Tasks

This is the project task source of truth. A feature is complete only after its stated
behavior is implemented and verified; vision documents may describe the intended
product, but must not claim unfinished behavior is available.

## Foundational — required for the first release

### F1. Clean canonical repository — complete

- Done: restored the required RAG proxy helper; removed the obsolete `VectorIndex`,
  unused notebook splitting functions, `/test-approval`, `/exit`, and dead tooltip data.
- Proof: clean worktree, narrow Python checks, and remote Rust checks for Rust removals.

### F2. Internal RAG — complete

- `/rag <query>` and `/rag <path> -- <query>` exist.
- The MCP advertises its sole RAG tool as read-only, allowing the inherited Codex
  scheduler to run it beside other read-only exploration.
- Backend workspace and explicit-path queries both return sourced chunks; the
  explicit-path query correctly answered from `docs/CONTEXT_AND_SESSIONS.md`.
- Proof: one visible successful example for each path plus one autonomous example.

### F3. Context and session continuity — implemented, acceptance pending

- Keep goals, constraints, decisions, changed files, evidence, and the next action.
- Expire raw searches, reads, and command output after retaining a compact conclusion
  and source pointer.
- Support truthful exact resume and lean continuation across sessions and providers.
- Elpis mirrors goal changes into a portable
  workspace `GOAL.md` and writes a compact `ES.md` checkpoint after every completed turn.
  `ES.md` keeps the latest result, changed paths, command outcomes, and a pointer to the
  exact provider transcript without copying raw diffs or command output.
- Fresh threads automatically admit `GOAL.md` and `ES.md`; exact resumes retain the native
  thread. Elpis syncs the portable checkpoint before compaction and stops if that safety
  write fails.
- `/status` shows admitted rules, goal, checkpoint, and memory summary sources with their
  sizes, lifetimes, and reasons.
- Proof: resume one task exactly and one task leanly without replaying irrelevant work.

### F4. Durable memory — in progress

- The imported Codex foundation already contains an enabled-by-default Rust memory
  pipeline for extraction, consolidation, retrieval, citations, and local artifacts.
- Elpis has not yet adopted the selected OpenClaw behaviors or established an
  Elpis-owned provider-neutral memory contract.
- Implemented and remotely verified: distinct recall-context tracking and a promotion
  gate requiring three recalls across two contexts. Weak one-off memories remain
  searchable as short-term evidence instead of entering `MEMORY.md`.
- Implemented, pending remote verification: Elpis-owned artifact/database roots,
  age-based fading, diverse exact retrieval, semantic RAG fallback with exact-read
  provenance, and hard 30,000/10,000-character limits for durable memory/summary.
- Review/delete path: memory remains inspectable as ordinary files under
  `~/.elpis/memories`; granular edits remove stale entries, while the confirmed reset-all
  action clears Elpis memory without deleting Codex data.
- Next: remote verification and the end-to-end memory acceptance check.
- Proof: teach a project fact, recall it in a related new session with provenance, omit
  it from an unrelated session, and allow review/deletion.

### F5. First-release provider and authentication boundary — partial

- First-release scope: OpenAI subscription and OpenRouter.
- Preserve Codex/ChatGPT login as authentication only; Elpis owns context, memory,
  session policy, and provider choice.
- Proof: complete and resume one task through each first-release path.

### F6. Release readiness — in progress

- Install, authenticate, launch, and complete a first task from a clean environment.
- Implemented pending verification: the Elpis binary reports `0.1.0`; tagged builds
  publish a checksummed Linux x86_64 release asset, and the installer verifies then
  atomically installs it.
- Proof: clean-machine acceptance without repository-specific paths.

## Important — valuable after the foundation

- Additional providers: add provider adapters using proven Pi/OpenClaw patterns.
- Codex pruning: remove only proven-unneeded crates and product surfaces while retaining
  execution, approvals, sandboxing, sessions, compaction, and the TUI.
- Behavioral enforcement: apply concise creator/project rules across runtimes.
- Visual identity: amber Elpis styling without losing Codex information or interaction.
- Dictation: consent-based speech input inserted as editable, unsent text.

## Nice-to-have

- `/auto` model routing with a visible choice, reason, and manual override.
- Scheduled memory review or dreaming-style reports.
- Rich themes, animation, and additional presentation polish.

## Current Action

Finish remote verification, install the verified binary, and run the context and memory
acceptance checks. Then close the OpenAI/OpenRouter boundary and clean-install release
path. Do not add dream narratives, scheduling, or an MCP memory adapter to the first
release.
