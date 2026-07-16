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
compact record containing the useful conclusion, source pointer, effect on the plan, and any
verification result. Exact file contents and logs are reread from their durable source
when needed; they are not kept merely because the agent once saw them.

The context list shows each active portable item, its source, approximate size,
lifetime, admission reason, and replacement record. Removing an item from the visible
TUI is not sufficient; it must be omitted from the next model-visible request.

The contained Codex runtime owns exact thread history. Elpis coordinates native
compaction and admits its portable goal and checkpoint when starting a fresh lean thread;
it does not pretend that changing the visible display changed an existing thread.

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

Elpis stores the current portable workspace goal and checkpoint under
`~/.elpis/context/workspaces/<readable-workspace-key>/`:

- `GOAL.md` mirrors goal changes from either the user or agent;
- `ES.md` is replaced after each completed turn with the latest result, changed paths,
  command outcomes, and the provider-thread pointer.

The provider transcript remains the exact evidence source. These Markdown files are the
small, editable continuation surface and must not become a second transcript.

## Genius Context Management Design (LCM & Decoupling)

These core design specs represent the reference architecture for context and session management, drawn from OpenClaw's lossless mechanics:

1. **Lossless Context Management (LCM) Pipeline:**
   * **Context Reconstruction Engine:** Evaluates the remaining token budget after each turn (via an `afterTurn` handler) and queries a SQLite database (`lcm.db`) where historical turns and tool executions are stored.
   * **Lazy-Loading Context:** Workspace context files and memories are dynamically indexed and vector-embedded, then lazily injected into the prompt based on active topics, rather than dumping large files blindly.

2. **Pre-Compaction Memory Flush (The Silent Turn):**
   * **Active Threshold Trigger:** When active tokens approach the `reserveTokens` floor, a user-invisible "Pre-compaction memory flush" prompt triggers.
   * **Filesystem Write & Truncation:** Important decisions and state updates are synced to the filesystem before the older conversational transcript is truncated or summarized.

3. **Compaction Model Decoupling (`compaction.model`):**
   * **Decoupled Roles:** Primary reasoning is routed to a premium model (e.g., Claude 3.5 Opus / Gemini Pro), while background history compaction and summarization are routed to a cheaper, faster model (e.g., Sonnet or local) via `compaction.model`, preserving context space and token budgets.

4. **Vector-Based Cross-Session Indexing:**
   * **Global Context Index:** A dedicated SQLite database (`memory-vectors.db`) tracks embeddings of session transcripts and files across separate workspace threads, allowing semantic queries to span disconnected sessions.

## Resume Contract

A fresh agent reads `AGENTS.md`, `GUIDE.md`, and `SESSION_HANDOFF.md`, then verifies the
working tree and the last recorded check. It performs the single next action rather than
repeating completed exploration. The handoff is replaced when state advances and removed
when the objective is complete.

## Acceptance Order

1. Verify exact resume retains the native thread.
2. Verify a fresh thread receives only the portable goal and compact checkpoint.
3. Verify `/status` names every portable source and its admission reason.
4. Verify curated memory appears only for a related task and remains absent otherwise.
