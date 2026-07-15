# Context And Session Contract

## User Promise

Elpis should remember the work without forcing the next model call to reread the work's
entire history. Exact artifacts stay on disk; the active context carries only what is
needed to decide and act.

## Context Lifecycle

Every admitted item has a lifetime:

- **durable:** applicable rules, the active goal, and explicit user constraints;
- **task:** decisions, changed paths, current blockers, and verification evidence;
- **turn:** searches, directory listings, file reads, command output, and failed probes.

Turn material expires after it answers its question. Before eviction, Elpis creates a
receipt containing the useful conclusion, source pointer, effect on the plan, and any
verification result. Exact file contents and logs are reread from their durable source
when needed; they are not kept merely because the agent once saw them.

The planned context ledger must show each active item, its source, approximate size,
lifetime, admission reason, and replacement receipt. Removing an item from the visible
TUI is not sufficient; it must be omitted from the next model-visible request.

The temporary Codex adapter owns its bootstrap thread, so compacting the Elpis display
alone cannot reduce that thread's model context. The Elpis context engine must assemble
the intended working set before every turn and project it into the selected runtime.
For a Codex-owned thread, Elpis must also coordinate native compaction or deliberately
start a lean replacement thread; it must not pretend that a smaller mirror changed
Codex's internal thread state.

## Session Persistence

Elpis separates four kinds of continuity:

1. The Elpis thread preserves immediate conversational continuity. During bootstrap,
   Elpis stores and resumes the temporary Codex app-server thread ID.
2. The workspace preserves exact code, documents, diffs, and other artifacts.
3. A checkpoint preserves the task state: goal, constraints, decisions, changed files,
   verification, blocker, and next action.
4. Memory preserves only reusable knowledge that should survive beyond the task.

Two continuation modes serve different needs:

- **Exact resume:** resume the runtime thread when its accumulated context remains useful.
- **Lean continuation:** compact the runtime or begin a fresh thread from the checkpoint
  plus a small recent-turn suffix when exploration and raw output have become a burden.

A checkpoint is written at a user-requested handoff, a clear phase boundary, before
automatic compaction, or when context pressure crosses the configured threshold. The
checkpoint never embeds full logs or file bodies; it points to them and records the
smallest useful conclusion.

## Resume Contract

A fresh agent reads `AGENTS.md`, `GUIDE.md`, and `SESSION_HANDOFF.md`, then verifies the
working tree and the last recorded check. It performs the single next action rather than
repeating completed exploration. The handoff is replaced when state advances and removed
when the objective is complete.

## Implementation Order

1. Add compact receipts for search, read, command, and file-change events.
2. Add the visible context ledger and omit expired turn material from new requests.
3. Add checkpoint generation and user-controlled exact/lean continuation.
4. Connect curated memory only after the checkpoint boundary is reliable.
