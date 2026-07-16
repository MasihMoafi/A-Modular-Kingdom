# Elpis Product Requirements

This file preserves Masih's requirements across sessions. It separates confirmed
requirements from proposed solutions and unresolved questions. A feature is not
"implemented" until its user-visible behavior is proven.

## Working Agreement

- Convert scattered product direction into a cohesive, prioritized list.
- Do not treat every question or exploratory idea as an approved task.
- Put unclear choices under **Open Decisions** and discuss them before making a
  choice that materially changes the product.
- Point out drift, contradictions, unnecessary complexity, and better options.
- Keep required work ahead of nice-to-haves.
- Record verification evidence, not confidence.

## Confirmed Requirements

### R1. Elpis is the provider-neutral TUI and agent environment

Elpis presents one coding-agent interface whose model and runtime can be OpenAI/Codex,
Gemini, Claude, or another supported system. Elpis owns the surrounding TUI, runtime
selection, context projection, durable memory, provider-neutral session mirror,
behavioral policy, and evidence.

The selected runtime may own its low-level model loop and native capabilities. When
Codex is selected, Codex may own its native tools, thread, and compaction; Elpis must
keep its surrounding control layer and make this ownership visible. Authentication
alone must never silently choose Codex as the runtime for another provider.

Codex is the Rust implementation foundation and source donor. Required code is copied
into Elpis; the finished product must not depend on the separate Codex clone directory.
Codex/ChatGPT subscription login authenticates Codex-backed OpenAI use. It must not make
Codex the universal Elpis runtime or transfer Elpis-wide context, memory, session, and
behavior policy to Codex.

### R2. Consequential actions are controlled and visible

File changes and commands must follow an explicit permission policy. The interface
must show what the agent is reading, changing, or running; show useful progress and
results; and preserve evidence such as changed paths, diffs, command status, and
verification. A compact display must not hide what actually happened.

### R3. Context is deliberately managed inside a session

Elpis must know exactly what the model receives. Rules, goals, selected files,
conversation, tool output, and memory must have visible sources and sizes. Old searches,
file reads, command output, and failed attempts must leave the next model request after
their useful conclusion is recorded. Full evidence remains available on disk.

### R4. Work survives between sessions without replaying everything

Elpis must preserve the active goal, accepted requirements, decisions, changed files,
verification, blockers, and next action. It must support both exact continuation and a
lean continuation made from a checkpoint plus recent conversation. Starting a new
session must not silently lose project intent.

### R5. Memory is curated, searchable, and distinct from session state

Memory stores reusable facts, preferences, decisions, and proven procedures. It is not
a transcript dump. Detailed notes remain searchable on disk; only a small relevant
selection enters a model request. Durable memory should be reviewed or promoted by a
clear rule, with source information and a way to remove stale entries.

### R6. User behavior and project rules are enforceable

Applicable `AGENTS.md`, product requirements, project guidance, and selected behavioral
rules must reach the model and action layer. Hard safety rules must be enforced by code,
not merely described in a prompt.

### R7. Claims require proof

Documentation must distinguish current behavior from intended behavior. Each required
feature needs a focused user-visible check before it is marked working.

### R8. Internal RAG is available directly and autonomously

`/rag <query>` searches the current workspace, while `/rag <path> -- <query>` targets
a selected folder. The active runtime may also call the same RAG tool autonomously when
broad semantic retrieval would reduce context load. RAG identifies relevant chunks and
source paths; code edits must still use exact search or file reads for current positions.

For broad repository questions that need both semantic discovery and exact evidence,
Elpis should run at most one RAG query concurrently with exact search or file reads.
Each activity must remain separately visible and correctly paired with its result. Skip
speculative RAG for named-file lookups and simple edits; timeouts or cancellation must
not block exact work or admit stale RAG output into later context.

Elpis's primary visual identity uses amber, between orange and yellow, not purple.

## First-Release Work, In Order

1. **Clean the canonical repository**
   - The pinned foundation import is complete; do not rebuild its mature behavior.
   - Remove only code proven obsolete or explicitly approved for removal.
   - Keep the repository, documents, branches, and worktrees understandable.
2. **Verify internal RAG**
   - Verify direct workspace and path queries and autonomous use for broad discovery.
   - Mark retrieval read-only so Codex may schedule it beside exact reads or searches.
   - Require exact current-file evidence before code edits.
3. **Implement the context and session engine**
   - Full transcript on disk; small model-visible working set.
   - Expiring tool output, compact records, a visible context list, checkpoints,
     compaction, exact/lean continuation, fork, and rollback.
4. **Implement the memory foundation**
   - Reconcile the inherited Codex Rust memory pipeline with the selected OpenClaw
     behaviors before writing new machinery.
   - Provide curated long-term memory, dated notes, selective retrieval,
     pre-compaction flush, provenance, review, and deletion.
5. **Verify the first-release provider boundary**
   - Support OpenAI subscription and OpenRouter for the first release.
   - Elpis owns context, memory, session policy, and provider choice.
6. **Ship the first release**
   - Install, authenticate, launch, recover, and complete a task from a clean machine.

## Current Verified Truth

| Area | Current state |
| --- | --- |
| Canonical source | `main` is the contained Codex-derived Elpis foundation. The former prototype is preserved as a named historical tip inside `archive/pre-cleanup-20260716`. |
| Installed command | `elpis` resolves to `/home/masih/.local/bin/elpis`, built remotely from this repository. |
| ChatGPT login and Codex turn | Working through the imported native Codex implementation. |
| Commands, patches, and activity display | Inherited from Codex and exercised in the authenticated foundation acceptance turn. |
| Permission modes and sandboxing | Inherited from Codex; no Elpis reimplementation is required. A focused all-mode acceptance matrix remains useful before release. |
| Mouse selection/copy | Inherited from the Codex TUI; Masih confirmed the old prototype limitation does not apply to this foundation. |
| Native sessions and compaction | Inherited for Codex-owned threads. Elpis-wide provider-neutral continuity is not implemented. |
| Context list and compact records | Specified in docs; not implemented. |
| Compaction/checkpoints/lean continuation | Specified in docs; not implemented. |
| Inherited Codex memory | A substantial Rust extraction, consolidation, retrieval, citation, and artifact pipeline is present and enabled by default; Elpis acceptance has not been run. |
| OpenClaw-style memory | Not integrated. OpenClaw adds pruning, guarded compaction, dated append-only flushes, hybrid retrieval, and scored promotion that must be reconciled with the inherited Rust pipeline. |
| `/auto` model routing | Nice-to-have and not implemented. Product documents may describe the intended behavior but must not claim availability. |
| Gemini/other runtimes | Experimental Gemini ACP work is preserved as a named historical tip inside `archive/pre-cleanup-20260716`; it is not merged into `main`. |

## Proposed State Layout

This is a proposal, not yet approved:

- `REQUIREMENTS.md`: project requirements and unresolved product choices.
- `~/.elpis/.../sessions/`: complete append-only event history.
- `~/.elpis/.../checkpoints/`: goal, decisions, changes, verification, blocker, next action.
- `~/.elpis/.../GOAL.md`: current persistent goal and acceptance criteria.
- `~/.elpis/.../MEMORY.md`: compact, curated long-term memory.
- `~/.elpis/.../memory/YYYY-MM-DD.md`: detailed working notes and session summaries,
  searched when relevant rather than always injected.
- `~/.elpis/.../DREAMS.md`: optional later review surface for suggested promotions.

## Open Decisions

### D1. Foundation migration shape

Resolved: the pinned Codex Rust foundation lives under `codex-rs/`; `main` uses it as
the canonical Elpis baseline, and the former prototype is preserved as a named
historical tip inside `archive/pre-cleanup-20260716`. Continue by subtraction.

### D2. Persistent goal

Recommended: keep one active goal with measurable acceptance criteria, while checkpoints
record previous goals and outcomes. Decide whether Elpis may update the goal from clear
conversation or only after explicit user confirmation.

### D3. Automatic memory writes

Recommended: automatically write detailed session notes and pre-compaction notes, but
require stronger evidence or user review before promoting them into compact long-term
memory. Decide how much automatic promotion is acceptable.

### D4. Default continuation

Recommended: exact resume while the context is healthy; lean continuation when context
pressure is high or the user requests it. The switch and its evidence should be visible.

### D5. Automatic model-routing policy — deferred

`/auto` is not required for the first release. If implemented later, its classifier,
tier boundaries, eligible model pool, fallback behavior, and manual override must remain
visible.

## Nice-To-Haves After The Foundation

- Dream narratives and scheduled "dreaming" reports. Scored promotion itself belongs
  in the required memory foundation; the narrative metaphor does not.
- Visual context map.
- Rich themes, animation, and additional presentation polish beyond clear action events.
- Agent personalities beyond enforceable project behavior.
- Voice input/output and background scheduled work.
- `/auto` model routing.
