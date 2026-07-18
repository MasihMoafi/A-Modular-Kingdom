# Elpis Technical Guide

## Product Thesis

> You put an agent into an Elpis, and it becomes Elpis. Be Elpis, my friend.

Elpis is the environment an agent enters and assimilates into. The selected model or
agent runtime may change, but the user's goals, working style, durable knowledge,
context policy, evidence, and behavioral boundaries continue coherently. Elpis is both
a state and a direction: it is never fully complete, and each verified change should
make the environment clearer, more capable, and easier for its creator to control.

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

Elpis should create value in five ways:

1. **Assimilation:** a selected runtime adopts the creator's applicable instructions,
   goals, context, memory, and behavioral rules, rather than the user adapting to it.
2. **Context sovereignty:** the user can see and control what enters the agent's
   working set. Selecting a file is an intentional context operation, not decoration.
3. **Reliable continuity:** sessions preserve goals, decisions, changes, and evidence
   across model changes, compaction, and restarts, while disposable logs and stale
   file bodies fall away instead of replaying an ever-growing transcript.
4. **Safe, transparent agency:** edits and commands use explicit sandbox and approval
   contracts. The UI shows what is proposed and records what happened; Elpis does not
   claim success when it has only hidden or documented a gap.
5. **Runtime choice and user ownership:** Elpis keeps one surrounding control
   environment while allowing the model provider and low-level agent runtime to
   change explicitly. Durable state is inspectable, editable, exportable, and not
   tied to one model provider or agent runtime.

Elpis is not primarily a provider switcher, a transcript viewer, or a collection of
slash commands. A command belongs in the product only when the user deliberately
selects it and its behavior has a stable contract.

Elpis is not distinguished by having another terminal chat interface — existing
projects already provide excellent model access, tools, permissions, terminal
rendering, and agent loops, and Elpis reuses those implementations. It is
distinguished by the five values above.

## Desired Output: First Public Release

Ship an installable terminal product, not a repository demo, in which a user can:

1. authenticate and deliberately select a supported model and runtime;
2. see which runtime owns the turn and which capabilities Elpis retains;
3. give the agent a task under Read Only, Default, or Full Access permissions;
4. watch readable commands, output, file changes, diffs, failures, and verification;
5. inspect and control the exact working context admitted by Elpis;
6. resume later from the same goal, decisions, changes, and relevant memory without
   replaying irrelevant history;
7. switch to at least one non-Codex runtime while retaining Elpis-owned continuity;
8. install and complete a first real coding task from a clean environment.

The release is successful only when these behaviors pass their acceptance checks from
a clean checkout and the distinctive context/continuity behavior is visibly better than
using the selected runtime alone.

Not required for the first release: messaging channels, scheduled automation, voice,
or desktop applications; dream narratives or autonomous skill creation; support for
every provider or every upstream Codex/OpenClaw feature; a new implementation where a
proven upstream component can satisfy the contract.

## Proof Standard

A feature is real only when its user-visible acceptance check passes and the evidence
is recorded. Documentation, hidden code, or a plausible architecture is not proof. The
defining evaluation: can a fresh supported runtime enter Elpis, receive the right
current goal and relevant history, obey the creator's rules, perform visible work under
the chosen permission mode, and resume later without irrelevant context? `TASKS.md` is
the current-state record against this standard.

## Requirements

This section preserves confirmed product requirements. A feature is not implemented
until its user-visible behavior is verified; current implementation state lives in
`TASKS.md`, not here, so the two documents do not drift against each other.

### Working Agreement

- Keep required work ahead of speculative features.
- Challenge unnecessary complexity and solution-first requests.
- Record evidence rather than confidence.
- Prefer small, reversible changes and the smallest useful verification.

### Confirmed Requirements

**R1. Provider-neutral Elpis environment** — Elpis owns the TUI, provider/runtime
selection, context projection, durable memory, provider-neutral continuity, behavioral
policy, permissions bridge, and evidence. A selected runtime may own its low-level
model loop and native tools, but authentication must never silently transfer
Elpis-owned state or product identity to that runtime.

**R2. Visible and controlled agency** — Commands and file changes follow explicit
permission and sandbox policies. The interface must preserve changed paths, diffs,
command status, failures, and verification evidence.

**R3. Deliberate context lifecycle** — Elpis must know what the model receives. Rules,
goal, selected files, conversation, tool output, and memory have visible sources,
sizes, reasons, and lifetimes. Stale exploration leaves the next request only after
its conclusion and exact evidence pointer are retained. A length threshold alone is not
a complete context policy.

**R4. Exact and lean continuity** — The active goal, decisions, constraints, changed
files, verification, blockers, and next action survive restarts. Elpis supports exact
native-thread resume and lean continuation from a compact portable checkpoint.

**R5. Curated memory** — Memory stores reusable facts and proven procedures, not
transcripts. Promotion requires repeated useful recall across distinct contexts.
Memory remains searchable, attributable, reviewable, deletable, and bounded. Deleted
or faded facts enter a searchable archive before baseline reset; archive failure must
stop the reset.

**R6. Enforceable creator and project rules** — Applicable `AGENTS.md`, project
requirements, and behavioral rules reach the model and action layer. Hard safety rules
are enforced by code where prompts are insufficient.

**R7. Claims require proof** — Documentation separates implemented behavior, remote
tests, and outstanding user acceptance. Design documents and hidden code are not proof.

**R8. Internal read-only RAG** — `/rag <query>` searches the workspace and
`/rag <path> -- <query>` targets a folder. The runtime may call the same read-only tool
autonomously for broad discovery. Exact current-file evidence remains required before
editing.

**R9. Proportionate, measured development cycle** — Ordinary changes receive focused
first-release checks. Exhaustive inherited TUI/app-server regression runs nightly,
manually, and for releases unless the change directly touches that surface. CI must not
edit source or create status-only commits. Dependency deletion follows Cargo timing
evidence and product optionality, not crate names.

**R10. Distinctive continuity-first identity** — Elpis uses a cyan visual identity
(superseding an earlier amber design) and visibly separates runtime, model, context,
memory, permissions, and evidence. The [UI Identity](#ui-identity) section below is the
acceptance contract, not proof that every surface it describes already ships.

### First-Release Order

1. Keep the canonical repository and verification cycle clean.
2. Preserve accepted internal RAG behavior.
3. Finish memory acceptance, including archive, review, deletion, and reset.
4. Finish exact/lean context and session acceptance; replace the length-only cleaner.
5. Verify authenticated OpenAI and OpenRouter task/resume paths.
6. Implement the persistent identity line and coherent cyan foundation.
7. Install and complete a real task from a clean environment, then tag `v0.1.0`.

### State Layout

- `~/.elpis/context/workspaces/<workspace>/GOAL.md` — active goal.
- `~/.elpis/context/workspaces/<workspace>/ES.md` — compact latest checkpoint.
- `~/.elpis/memories/MEMORY.md` — curated durable memory.
- `~/.elpis/memories/archive.md` — append-only faded/deleted evidence.
- `~/.elpis/state/memories_1.sqlite` — recall, promotion, and consolidation state.
- Provider transcripts and workspace artifacts remain the exact evidence sources.

### Deferred Decisions

- Whether goal changes require explicit confirmation.
- Default threshold for switching from exact to lean continuation.
- Native Anthropic and Google adapter order.
- `/auto`, dreaming reports, voice, rich animation, and scheduled work.

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

Elpis's concrete context-record and session-checkpoint contract lives in
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

### 4. Compact tool and file records

Keep full events in the on-disk transcript. The model-visible working set should
replace stale bulky results with compact records containing:

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
own provider-neutral session record and context list even when a provider also offers
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

Current foundation work is memory plus context acceptance. Elpis-owned memory storage,
promotion, bounded retrieval, portable `GOAL.md`/`ES.md`, lean fresh-thread admission,
pre-compaction protection, and the visible admitted-context list are implemented pending
remote verification. Visual identity, provider expansion, pruning, and `/auto` routing
follow the foundational memory/context acceptance checks.

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

### Memory ownership

Elpis memory artifacts default to `~/.elpis/memories` and memory database state defaults
to `~/.elpis/state`. Codex configuration, authentication, threads, logs, and goals remain
separate. Resetting Elpis memory must not delete Codex memory or state. Unconfigured
non-TUI consumers retain the inherited Codex-home fallback for compatibility.

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
