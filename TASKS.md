# Elpis Tasks

This is the project task source of truth. A feature is complete only after its stated
behavior is implemented and verified; vision documents may describe the intended
product, but must not claim unfinished behavior is available.

## Foundational — required for the first release

### F1. Clean canonical repository — in progress

- Done: restored the required RAG proxy helper; removed the obsolete `VectorIndex`,
  unused notebook splitting functions, `/test-approval`, `/exit`, and dead tooltip data.
- Remaining: reconcile stale documentation and continue only evidence-backed cleanup.
- Proof: clean worktree, narrow Python checks, and remote Rust checks for Rust removals.

### F2. Internal RAG — partial

- `/rag <query>` and `/rag <path> -- <query>` exist.
- The MCP advertises its sole RAG tool as read-only, allowing the inherited Codex
  scheduler to run it beside other read-only exploration.
- Remaining: verify both paths against the live tool, verify autonomous retrieval for a
  broad question, and ensure exact file reads are used before edits.
- Proof: one visible successful example for each path plus one autonomous example.

### F3. Context and session continuity — not started

- Keep goals, constraints, decisions, changed files, evidence, and the next action.
- Expire raw searches, reads, and command output after retaining a compact conclusion
  and source pointer.
- Support truthful exact resume and lean continuation across sessions and providers.
- Proof: resume one task exactly and one task leanly without replaying irrelevant work.

### F4. Durable memory — investigation paused

- The imported Codex foundation already contains an enabled-by-default Rust memory
  pipeline for extraction, consolidation, retrieval, citations, and local artifacts.
- Elpis has not yet adopted the selected OpenClaw behaviors or established an
  Elpis-owned provider-neutral memory contract.
- Next: compare the existing Rust pipeline with OpenClaw's pruning, pre-compaction
  flush, dated notes, retrieval, and promotion; then approve the smallest integration
  design before implementation.
- Proof: teach a project fact, recall it in a related new session with provenance, omit
  it from an unrelated session, and allow review/deletion.

### F5. First-release provider and authentication boundary — partial

- First-release scope: OpenAI subscription and OpenRouter.
- Preserve Codex/ChatGPT login as authentication only; Elpis owns context, memory,
  session policy, and provider choice.
- Proof: complete and resume one task through each first-release path.

### F6. Release readiness — not started

- Install, authenticate, launch, and complete a first task from a clean environment.
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

Complete F2's three live acceptance examples. Do not begin memory implementation until
the existing Codex and OpenClaw memory pipelines have been compared and Masih approves
the resulting behavioral contract.
