# Elpis Product Requirements

This file preserves confirmed product requirements and current proof. A feature is not
implemented until its user-visible behavior is verified.

## Working Agreement

- Keep required work ahead of speculative features.
- Challenge unnecessary complexity and solution-first requests.
- Record evidence rather than confidence.
- Prefer small, reversible changes and the smallest useful verification.

## Confirmed Requirements

### R1. Provider-neutral Elpis environment

Elpis owns the TUI, provider/runtime selection, context projection, durable memory,
provider-neutral continuity, behavioral policy, permissions bridge, and evidence. A selected
runtime may own its low-level model loop and native tools, but authentication must never
silently transfer Elpis-owned state or product identity to that runtime.

### R2. Visible and controlled agency

Commands and file changes follow explicit permission and sandbox policies. The interface
must preserve changed paths, diffs, command status, failures, and verification evidence.

### R3. Deliberate context lifecycle

Elpis must know what the model receives. Rules, goal, selected files, conversation, tool
output, and memory have visible sources, sizes, reasons, and lifetimes. Stale exploration
leaves the next request only after its conclusion and exact evidence pointer are retained.
A length threshold alone is not a complete context policy.

### R4. Exact and lean continuity

The active goal, decisions, constraints, changed files, verification, blockers, and next
action survive restarts. Elpis supports exact native-thread resume and lean continuation from
a compact portable checkpoint.

### R5. Curated memory

Memory stores reusable facts and proven procedures, not transcripts. Promotion requires
repeated useful recall across distinct contexts. Memory remains searchable, attributable,
reviewable, deletable, and bounded. Deleted or faded facts enter a searchable archive before
baseline reset; archive failure must stop the reset.

### R6. Enforceable creator and project rules

Applicable `AGENTS.md`, project requirements, and behavioral rules reach the model and action
layer. Hard safety rules are enforced by code where prompts are insufficient.

### R7. Claims require proof

Documentation separates implemented behavior, remote tests, and outstanding user acceptance.
Design documents and hidden code are not proof.

### R8. Internal read-only RAG

`/rag <query>` searches the workspace and `/rag <path> -- <query>` targets a folder. The
runtime may call the same read-only tool autonomously for broad discovery. Exact current-file
evidence remains required before editing.

### R9. Proportionate, measured development cycle

Ordinary changes receive focused first-release checks. Exhaustive inherited TUI/app-server
regression runs nightly, manually, and for releases unless the change directly touches that
surface. CI must not edit source or create status-only commits. Dependency deletion follows
Cargo timing evidence and product optionality, not crate names.

### R10. Distinctive continuity-first identity

Elpis uses an amber visual identity and visibly separates runtime, model, context, memory,
permissions, and evidence. `docs/UI_IDENTITY.md` is the acceptance contract, not proof that
those surfaces already ship.

### R11. Claude Code as a selectable runtime

Confirmed by Masih 2026-07-18. Claude Code (authenticated via its own subscription login, not
an extracted/reused credential) becomes a second selectable runtime alongside Codex, not just
another model under the existing runtime:

- Appears in the `/model` picker as its own provider group, with its model shown beneath it —
  same grouped pattern already proposed for other providers.
- A dedicated `/claude-code` command switches the active session to it directly.
- Foundation exists (`codex-rs/claude-bridge`, tested, CI-green — spawns
  `claude -p --output-format stream-json` and parses real captured output): see
  `agent/claude-runtime-bridge`. Not yet wired into any UI; today Elpis's TUI is hard-wired to
  exactly one backend (`codex_app_server_client::AppServerClient`). Introducing the actual
  runtime-selection abstraction (deciding Codex-app-server vs. this subprocess bridge, touched
  in `ChatWidget`/`App` construction) is the confirmed next step.

## First-Release Order

1. Keep the canonical repository and verification cycle clean.
2. Preserve accepted internal RAG behavior.
3. Finish memory acceptance, including archive, review, deletion, and reset.
4. Finish exact/lean context and session acceptance; replace the length-only cleaner.
5. Verify authenticated OpenAI and OpenRouter task/resume paths.
6. Implement the persistent identity line and coherent amber foundation.
7. Install and complete a real task from a clean environment, then tag `v0.1.0`.

## Current Verified Truth

| Area | Current state |
| --- | --- |
| Canonical source | `main` contains the Codex-derived Rust foundation; the former prototype is archived. |
| Execution, patches, permissions, sandboxing | Inherited from Codex and exercised in the foundation acceptance turn. |
| Internal RAG | Accepted complete for workspace, explicit-path, and autonomous read-only retrieval. |
| Portable context | `GOAL.md`, `ES.md`, fresh-thread admission, `/status`, and focused remote checks exist; exact/lean user acceptance remains. |
| Context cleaner | Partial: tool outputs over 1,000 characters are replaced, but lifecycle semantics, conclusions, pointers, and focused tests are missing. |
| Durable memory | Roots, promotion, bounds, ranked retrieval, and focused remote tests exist; related/unrelated recall and review/reset acceptance remain. |
| Memory archive | Implemented; the build-cycle branch propagates archive failures and adds a deleted-line regression. |
| Provider launcher | OpenAI, OpenRouter, Bedrock, Ollama, LM Studio, and Claude/Gemini OpenRouter aliases are tested at configuration level. |
| Native Claude/Gemini | Not implemented. Current shortcuts use OpenRouter. |
| Provider-aware `/model` UI | Not implemented. |
| UI identity | Naming and runtime title exist; amber styling, identity line, continuity event, and evidence-first completion remain. |
| Remote build | Run `29534784054` passed; the pre-optimization cycle was about 21 minutes and the artifact was 102,988,260 bytes. |
| `/auto` routing | Deferred and not implemented. |

## State Layout

- `~/.elpis/context/workspaces/<workspace>/GOAL.md` — active goal.
- `~/.elpis/context/workspaces/<workspace>/ES.md` — compact latest checkpoint.
- `~/.elpis/memories/MEMORY.md` — curated durable memory.
- `~/.elpis/memories/archive.md` — append-only faded/deleted evidence.
- `~/.elpis/state/memories_1.sqlite` — recall, promotion, and consolidation state.
- Provider transcripts and workspace artifacts remain the exact evidence sources.

## Deferred Decisions

- Whether goal changes require explicit confirmation.
- Default threshold for switching from exact to lean continuation.
- Native Anthropic and Google adapter order.
- `/auto`, dreaming reports, voice, rich animation, and scheduled work.
