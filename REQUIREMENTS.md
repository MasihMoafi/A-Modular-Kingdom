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

## Required Work, In Order

1. **Adopt the proven Codex foundation inside Elpis**
   - Pin a known upstream revision and preserve its license notices.
   - Copy the required Rust workspace into Elpis; do not link to the donor directory.
   - Keep the proven TUI/event, command, patch, permission, sandbox, session, and test
     paths working before removing unwanted features.
2. **Match Codex's visible actions and exact permission behavior**
   - Render command start, live output, completion/failure, file patches, and changed
     file summaries using the copied Codex lifecycle and rendering code.
   - Preserve Codex's built-in Read Only, Default, and Full Access presets and its
     underlying approval/sandbox rules. Full Access is the explicit no-prompt mode;
     Default permits workspace work but asks for escalation; Read Only requires approval
     for edits or internet access.
3. **Create the provider boundary**
   - Keep OpenAI/Codex as the first working adapter with subscription authentication.
   - Add Gemini and Claude adapters without duplicating the TUI, tools, permissions,
     context, sessions, or memory engine.
4. **Implement the context and session engine**
   - Full transcript on disk; small model-visible working set.
   - Expiring tool output, compact receipts, visible context ledger, checkpoints,
     compaction, exact/lean continuation, fork, and rollback.
5. **Implement the memory foundation**
   - Curated long-term memory, dated working notes, search and selective retrieval,
     pre-compaction memory flush, provenance, review, and deletion.
6. **Complete required coding-agent capabilities**
   - Web research, image/file inputs where supported, interruption/cancellation,
     reliable errors, configuration, and recovery after restart.

## Current Verified Truth

| Area | Current state |
| --- | --- |
| ChatGPT login | Working through installed Codex. |
| Default model turn | Working through Codex app-server. Codex owns the low-level turn; Elpis currently supplies only a thin surrounding TUI. |
| Streaming answer | Working for Codex agent-message text. |
| File/command permissions | Partial. The prototype requests Codex's Default policy (`on-request` + workspace write), so in-workspace writes may correctly proceed without a prompt. Its `/yolo` switch does not reconfigure the thread and is therefore not a faithful mode switch. |
| File/command activity display | Incomplete. The action occurred visibly, but most structured command/file events are ignored or poorly rendered, so the display is much less informative than Codex. |
| Session save/list | Basic Elpis JSON session files exist. |
| Resume/fork | Working only authoritatively through delegated Codex threads; non-Codex mode replays rendered messages. |
| Context file selection | Partial. Explicit files can be injected, and unchanged selected files are not repeatedly appended by the TUI. |
| Context pruning | Display/session-message trimming exists, but it does not shrink the delegated Codex model context. |
| Context ledger and receipts | Specified in docs; not implemented. |
| Compaction/checkpoints/lean continuation | Specified in docs; not implemented. |
| Retrieval MCP | Working: one local `query_knowledge_base` tool. |
| Long-term memory | Not active in the current Codex path. Older Python memory modules remain, but the live MCP does not expose them. |
| OpenClaw-style memory | Not implemented. Source inspection confirms OpenClaw combines pruning, guarded compaction, dated append-only flushes, hybrid retrieval, and scored long-term promotion. Elpis currently has none of that complete pipeline. |

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

The direction is settled: use Codex's Rust foundation and subtract. Before moving code,
choose whether to replace the present repository tree in one migration branch or place
the pinned upstream foundation under a temporary internal directory and move Elpis
features onto it incrementally. The choice must preserve the current prototype until
the copied foundation reaches the same working milestone.

### D2. ES file

Clarify what `ES` means and whether it should be one file per session, one rolling
project file, or a folder. Recommended role: a human-readable checkpoint/session-summary
surface, not the long-term memory store and not the full transcript.

### D3. Persistent goal

Recommended: keep one active goal with measurable acceptance criteria, while checkpoints
record previous goals and outcomes. Decide whether Elpis may update the goal from clear
conversation or only after explicit user confirmation.

### D4. Automatic memory writes

Recommended: automatically write detailed session notes and pre-compaction notes, but
require stronger evidence or user review before promoting them into compact long-term
memory. Decide how much automatic promotion is acceptable.

### D5. Default continuation

Recommended: exact resume while the context is healthy; lean continuation when context
pressure is high or the user requests it. The switch and its evidence should be visible.

## Nice-To-Haves After The Foundation

- Dream narratives and scheduled "dreaming" reports. Scored promotion itself belongs
  in the required memory foundation; the narrative metaphor does not.
- Visual context map.
- Rich themes, animation, and additional presentation polish beyond clear action events.
- Agent personalities beyond enforceable project behavior.
- Voice input/output and background scheduled work.
