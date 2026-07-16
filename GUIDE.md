# Elpis Technical Guide

## Purpose

Elpis is a provider-neutral coding-agent TUI. It does not try to make a new foundation
model. It gives agents backed by OpenAI/Codex, Gemini, Claude, or another model one
coherent local interface by managing:

- the instructions and project knowledge admitted into working context;
- the boundary between transient context and durable memory;
- tools, edits, commands, permissions, and their evidence;
- continuity across turns, compaction, and sessions;
- a model-independent TUI where consequential actions remain inspectable.

The intended result is assimilation: whichever model provider is selected, the agent
adopts the user's harness, workspace, memory, and control policy without being buried
under them.

## Product Value

Elpis should create value in four ways:

1. **Context sovereignty:** the user can see and control what enters the agent's
   working set. Selecting a file is an intentional context operation, not decoration.
2. **Reliable continuity:** sessions preserve goals, decisions, changes, and evidence,
   while disposable logs and stale file bodies fall away.
3. **Safe agency:** edits and commands use explicit sandbox and approval contracts.
   The UI shows what is proposed and records what happened.
4. **Runtime choice:** Elpis keeps one surrounding control environment while allowing
   the model provider and low-level agent runtime to change explicitly.

Elpis is not primarily a provider switcher, a transcript viewer, or a collection of
slash commands. A command belongs in the product only when the user deliberately
selects it and its behavior has a stable contract.

## Source Map

Treat upstream behavior as evidence, not inspiration copied from memory.

### Codex: execution and interface reference

Primary source is the local clone at `/home/masih/Desktop/f/p/others/codex`:

- App-server: `/home/masih/Desktop/f/p/others/codex/codex-rs/app-server`
- App-server protocol: `/home/masih/Desktop/f/p/others/codex/codex-rs/app-server-protocol`
- Rust TUI: `/home/masih/Desktop/f/p/others/codex/codex-rs/tui`
- Core agent runtime: `/home/masih/Desktop/f/p/others/codex/codex-rs/core`

The clone's origin is [openai/codex](https://github.com/openai/codex). Use the remote
only for provenance or an explicitly requested update; do not browse it when the local
clone can answer the question.

Codex is the contained implementation foundation for thread/turn/item semantics,
streaming, file changes, command execution, approvals, sandboxing, `@` interactions,
and TUI ergonomics. The pinned Rust workspace and tests live under `codex-rs/`, with
Apache-2.0 notices preserved. Elpis does not load code from, or require, the separate
Codex clone directory at runtime.

The active foundation strategy is **fork and subtract**: keep the pinned Codex
execution and TUI paths intact, remove unwanted
OpenAI-specific product surfaces in small tested steps, and introduce a provider
subsystem only after the foundation is stable. Do not revive the archived hand-grown
prototype as the active runtime.

### OpenClaw: context and continuity reference

Primary source is the current shallow clone at
`/home/masih/Desktop/f/p/others/openclaw-upstream`. The older May 26 source archive is
`/home/masih/Desktop/f/p/others/openclaw-main`.
The primary clone was refreshed from the official repository on 2026-07-16 at
`dd58667b` (source version `2026.7.2`).
Read implementation and tests, not only explanatory documents. The main source areas
are:

- live context pruning: `src/agents/agent-hooks/context-pruning/`;
- pre-compaction memory flush: `src/auto-reply/reply/memory-flush.ts` and
  `agent-runner-memory.ts`;
- guarded compaction: `src/agents/agent-hooks/compaction-safeguard.ts`;
- search and retrieval: `extensions/memory-core/src/memory/`;
- dated notes, long-term promotion, size budgets, and dreaming:
  `extensions/memory-core/src/flush-plan.ts`, `short-term-promotion.ts`,
  `memory-budget.ts`, and `dreaming.ts`.

The upstream project is [openclaw/openclaw](https://github.com/openclaw/openclaw).
The initialized but empty-history `/home/masih/Desktop/f/repos/openclaw-main` is not an
upstream source clone and is not the Elpis reference tree.

OpenClaw's useful memory behavior is a pipeline, not a special Markdown filename:
ephemeral pruning keeps a live turn small; guarded compaction preserves continuity;
pre-compaction flush writes dated append-only notes; hybrid search retrieves only
relevant excerpts; repeated useful recalls may be promoted into bounded long-term
memory. Dreaming is optional scheduled review and promotion on top of that foundation.

Elpis's concrete context-receipt and session-checkpoint contract lives in
`docs/CONTEXT_AND_SESSIONS.md`.

## Runtime Architecture

```text
User
  -> Elpis TUI (presentation, selection, approvals, context visibility)
  -> Elpis control layer (runtime choice, context, memory, session mirror, policy)
       -> selected agent runtime
            -> Codex app-server (owns a Codex turn and its native tools/thread)
            -> Elpis embedded runtime (provider-neutral direct model path)
            -> external/ACP runtime (Claude CLI, Gemini CLI, or another harness)
       -> Elpis retrieval services
  -> Workspace + durable Elpis state (~/.elpis)
```

The current `elpis` executable is built from the contained `codex-rs/` foundation. It
uses the existing ChatGPT/Codex authentication and native Codex turn implementation;
it does not launch code from the donor clone or the archived prototype.

Runtime ownership is explicit. When Codex is selected, Codex may own the low-level
model loop, native shell/file tools, native thread, and native compaction. Elpis still
owns the surrounding product: runtime selection, context projection, durable memory,
provider-neutral session mirror, behavioral policy, approvals bridge, and visible TUI.
Selecting another runtime must not silently route through Codex.

The Codex account and runtime boundaries are specified in
`docs/AUTHENTICATION_BOUNDARY.md`. Authentication alone must never silently select a
runtime; the active model and runtime owner must be visible.

## Context Contract

Context is a budgeted working set, not the session archive.

### 1. Durable prefix

Load the smallest stable routing layer: applicable `AGENTS.md`, this guide, and a
short skill index. Detailed rules are loaded only when a task triggers them. An
instruction file should behave like a table of contents, not a knowledge dump.

### 2. Turn input

Send the new user message plus explicitly requested context. An `@file` mention is
an explicit refresh. A pinned checklist file is injected once and again only when its
content changes. Never append the same unchanged file body on every turn.

### 3. Exploration expires

Searches, directory listings, file reads, probes, and unsuccessful paths are temporary
working material. Once they have answered the current question, remove their raw output
from the model-visible context. Keep only the useful conclusion, a pointer to its source,
and the consequence for the next action.

Do not turn `GUIDE.md` into an exploration log. Promote one distilled fact only when it is
durable enough to change how future agents should work on Elpis; replace stale guidance
instead of accumulating discoveries.

### 4. Tool and file receipts

Keep full events in the on-disk transcript. The model-visible working set should
replace stale bulky results with compact receipts containing:

- operation and target;
- success/failure and exit status;
- concise result or error summary;
- changed paths and a diff/stat reference;
- verification performed;
- a pointer for rereading the full artifact when needed.

After a file edit, preserve the path, semantic change, diff, and verification. Do not
retain the entire old and new file merely because the agent touched them.

### 5. Pruning and compaction

- **Pruning is ephemeral:** trim or replace old tool results for the next request;
  leave the durable transcript intact.
- **Compaction is persistent:** summarize older conversation into a checkpoint that
  preserves the goal, constraints, decisions, changed files, verification, blockers,
  and next action.
- Before compaction, flush genuinely reusable facts to durable memory.
- Keep a recent-turn suffix verbatim so the agent does not wake up inside a summary.

### 6. Memory

Memory is curated cross-session knowledge, not a transcript mirror. Store stable user
preferences, project facts, decisions, and proven procedures. Retrieve only relevant
entries for the current task and make provenance visible.

### 7. Measurement

Prefer runtime-reported token usage and context-window size. Character division is a
fallback estimate only. Track injected sources and bytes/tokens by category so the
user can answer: "Why is this in context?"

## Session Semantics

An API call does not receive previous messages by magic. Either Elpis resends the chosen
working context, or a provider stores a thread and reconstructs it. Elpis must keep its
own provider-neutral session record and context ledger even when a provider also offers
thread IDs. Resume, fork, rollback, and compaction must therefore have Elpis semantics,
with provider thread IDs treated as adapter-specific state rather than the project truth.

Reasoning tokens count toward usage, but hidden reasoning is not a useful transcript
to carry forward verbatim. Preserve decisions and evidence. Streamed tool events and
large outputs also should not remain indefinitely in the model-visible working set.

## Current State

Verified foundation:

- `main` contains the pinned Codex Rust workspace under `codex-rs/` and builds the
  user-facing `elpis` executable;
- ChatGPT authentication, streaming, commands, patches, permission modes, sandboxing,
  mouse interaction, native sessions, and native compaction are inherited from Codex;
- a real authenticated turn ran a command and created a file, and launch checks proved
  no donor-clone runtime dependency;
- the old hand-grown TUI, Python agent state, and Gemini/runtime-boundary experiments
  are preserved as named tips inside `archive/pre-cleanup-20260716`; they are not part
  of canonical `main`.

Current stabilization has two separate lanes: complete the approved feature subtraction
in small remotely verified commits, and give the inherited Ratatui interface an
appearance-only Elpis identity without changing its information or behavior. The visual
identity pass is the immediate user-visible task. Context design, memory design, `/auto`
routing, and other feature additions remain later work.

The currently installed build includes the command-surface changes from `b135e7a` and
the startup-tip/test-command changes from `419384d`. Most unwanted commands were made
undiscoverable, not fully deleted from the Rust implementation; this does **not** satisfy
the approved subtraction requirement. Complete removal must proceed feature by feature
without deleting shared machinery required by retained behavior.

### Startup performance

Typing `elpis` previously took about five seconds before the TUI was usable; the archived
prototype took under one second. The regression is now resolved. Do not describe PyTorch
as part of the Rust TUI: it belongs to the separate Python RAG service and is loaded only
for an explicit RAG query. Keep the startup target below one second, with optional
services forbidden from blocking the first usable frame.

The July 16 startup audit found that the registered `elpis-rag` process was still the
old unified MCP: it advertised 21 unrelated memory, shell, web, speech, document, and
RAG tools and imported the memory/Qdrant stack before answering Codex. The working-tree
replacement exposes only `query_knowledge_base` and answers the MCP handshake using
Python's standard library. Its measured handshake is about 0.04 seconds, a profiled
Elpis launch scheduled its first frame in 0.049 seconds, and Masih confirmed that the
fresh launch is fast. Startup performance is accepted.

The one-tool host was not a new product decision: it already existed on the historical
legacy-prototype tip now preserved inside `archive/pre-cleanup-20260716`. During the
Codex-foundation transition, canonical `main` inherited an older unified host from the
embedded-launcher line where extra tools defaulted on, while the minimal host remained
on the prototype line. This was a migration regression, not a Codex TUI cost.

RAG indexing is not startup work. `rag.fetch` is imported only after an explicit query.
On that first query it loads Torch and embedding models, then loads an existing persisted
index or scans and indexes the requested folder when no usable index exists. If indexing
is later moved to a background job, it must not block or visually take over the TUI; the
previous usable index should remain available until replacement is complete.

The Python service is RAG-only. Its retained source packages are `src/agent` for the
one-tool MCP host, `src/rag` for retrieval, and `src/utils` for RAG's proxy handling.
The superseded Python memory, shell, web, speech, document, parser, and tool-execution
modules are deleted; Codex owns those general agent capabilities. Future Elpis memory
and dictation work must be designed at the Elpis product layer, not restored inside the
RAG MCP.

### Product scope already decided

- The active terminal interface is Codex's contained Rust TUI and already uses Ratatui.
  Elpis does not need a UI-framework rewrite for the visual identity pass.
- UI work is appearance-only for now: colors and styling may change, but Codex-quality
  content, rendering, and interactions must remain.
- Dictation is a required future feature. First audit the contained Codex source for an
  existing speech-to-text path; otherwise specify a visible, consent-based Whisper
  integration. Dictation must insert editable text and must not auto-submit it.
- Kiro may be researched for reusable ideas only after its actual source availability,
  language, license, and performance claims are verified. Do not assume it is open
  source or copyable.

## Engineering Rules

- Read this guide before architectural work; read only the source sections relevant
  to the current task.
- Keep upstream protocol handling version-aware. Generate schemas from the installed
  Codex version when exact message shapes matter.
- Do not add slash commands without explicit user selection.
- Do not call a temporary directory a sandbox. State the actual isolation boundary.
- Preserve user-visible behavior with focused tests for protocol and context changes.
- Record what is implemented separately from what is intended.
- Treat `main` as canonical. Read archive branches only when historical prototype
  behavior is specifically needed.
- Keep one normal local worktree. Create a temporary worktree only for an approved,
  genuinely parallel task; remove it after its useful commit is merged or archived.
- Push accepted canonical checkpoints to `main`. Historical or abandoned branch tips
  belong in the single `archive/pre-cleanup-20260716` history branch, not in a growing
  set of active-looking branches.

## Verification

Use `.github/workflows/embedded-elpis-linux.yml` for Rust formatting, focused tests,
build, executable identity, and artifact verification. Do not compile Rust locally on
Masih's workstation. Run only narrow non-Rust checks locally.
