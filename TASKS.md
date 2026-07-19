# Elpis Tasks

This is the project task source of truth. A feature is complete only after its stated
behavior is implemented and verified; vision documents describe intended behavior but must
not claim unfinished behavior is available.

## Foundational — required for the first release

### F1. Clean canonical repository — complete

- Restored the required RAG proxy helper.
- Removed the obsolete `VectorIndex`, unused notebook splitting functions,
  `/test-approval`, `/exit`, and dead tooltip data.
- Proof: clean canonical `main`, focused Python checks, and remote Rust checks.

### F2. Internal RAG — complete

- `/rag <query>` and `/rag <path> -- <query>` exist.
- The MCP advertises one read-only RAG tool, allowing concurrent scheduling beside exact
  read-only exploration.
- Workspace and explicit-path queries return sourced chunks.
- Proof: visible workspace, explicit-path, and autonomous examples.

### F3. Context and session continuity — implemented; acceptance pending

- `GOAL.md` mirrors the active workspace goal and `ES.md` records the latest compact result,
  changed paths, command outcomes, and exact provider-transcript pointer.
- Fresh threads admit the portable goal/checkpoint; exact resume retains the native thread.
- Elpis syncs portable continuity before compaction and stops when that safety write fails.
- `/status` reports admitted rules, goal, checkpoint, and memory summary with size, lifetime,
  and reason.
- Remote verification: the focused continuity tests and Linux build passed in run
  `29534784054`.
- Remaining acceptance: resume one real task exactly and one leanly without replaying
  irrelevant work; verify `/status` against the actual next model request.
- The tool-output cleaner is lifecycle-aware (`core/src/context_cleaner.rs`): older outputs
  over 1,200 characters are reduced to bounded head/tail excerpts with a durable
  `rollout://tool-call/<id>` evidence pointer, the two newest outputs stay intact, evictions
  surface in `/status`, and focused tests cover the behavior.
- Remaining cleaner gap: the retained excerpt is positional (head/tail), not a semantic
  conclusion of the output.

### F4. Durable memory — implemented; end-to-end acceptance pending

- Elpis-owned memory artifacts and database state live under `~/.elpis`.
- Implemented and remotely verified: recall-context tracking, promotion after three recalls
  across two contexts, age-aware diverse retrieval, semantic fallback, provenance, and hard
  30,000/10,000-character durable-memory/summary limits.
- Weak one-off memories remain searchable evidence instead of entering `MEMORY.md`.
- Review/delete remains file-based; reset-all clears Elpis memory without deleting Codex
  threads or state.
- Build-cycle branch: deleted/faded lines are archived before baseline reset, archive write
  failures stop the reset, and a focused archive regression is included.
- Remaining acceptance: teach a fact, recall it with provenance in a related new session,
  omit it from an unrelated session, review/delete it, and verify reset behavior locally.

### F5. First-release provider and authentication boundary — partial

- OpenAI subscription authentication remains the default Codex-backed path.
- OpenRouter is separately keyed through `OPENROUTER_API_KEY`.
- Launcher/configuration tests cover OpenAI, OpenRouter, Bedrock, Ollama, and LM Studio IDs.
- `--provider claude`, `gemini`, and `gemini-flash` select OpenRouter family aliases; the
  native adapters are the separate `anthropic` and `google-gemini` providers.
- Native Anthropic Messages and Google Gemini generateContent wire APIs are implemented
  (PR #46) with request translation, SSE streaming, and mock-server tests in
  `core/src/chat_completions.rs`; live vendor acceptance is pending.
- Remaining first-release acceptance: complete and resume one task through OpenAI and
  OpenRouter, proving Elpis-owned goal, context, memory, permissions, and evidence survive.
- The `/model` surface now uses the Elpis `Choose a mind` naming and shows provider,
  protocol, route, and credential labels.
- Live provider switching mid-session is unimplemented at the protocol layer:
  `ThreadSettingsUpdateParams` (`app-server-protocol/src/protocol/v2/thread.rs`) carries
  `model` but no provider field, while the paired `ThreadSettings` read/notification struct
  does carry `model_provider`. Provider choice still happens only at launch (`--provider`);
  a live switch needs an explicit protocol/runtime slice, not a cosmetic selector.

### F6. Release readiness and build cycle — in progress

- The binary reports `0.1.0`.
- Run `29534784054` passed and uploaded the verified Linux x86_64 artifact.
- Tagged builds publish a checksummed release asset; the installer verifies and atomically
  installs it.
- The pre-optimization verified run took about 21 minutes and produced a 102,988,260-byte
  artifact.
- Build-cycle branch removes CI source mutation and status-file commits, enables incremental
  cross-commit cache reuse, keeps focused checks on ordinary changes, moves exhaustive
  inherited regression to nightly/manual/tag runs, and uploads Cargo timing evidence.
- Remaining acceptance: compare the optimized runtime to the baseline, install the verified
  artifact from a clean environment, authenticate, launch, and complete a first task.

### F7. Distinctive Elpis UI/UX — design complete; implementation partial

- Implemented: Elpis naming, the mature inherited Ratatui interaction model, a persistent
  cyan identity header (model, context-used percent, location — commit `49cd113`,
  superseding the earlier amber design), suppression of the conflicting inherited footer
  status line, the Context Ledger with per-file `skills/dev` rows, and the `Choose a mind`
  `/model` naming (commit `bae7108`).
- Not yet implemented: the signature continuity event (only a generic eviction notice
  exists, now naming what survived — commit `de4ed6f`, `"Survived: goal, checkpoint, and
  admitted rules (see /status)."` — but not distinguishing resume/compaction/provider-change
  events), the evidence-first completion hierarchy, and a render-verified
  context-accounting consistency check. GUIDE.md's UI Identity section is a contract, not
  proof.
- Proof required: a new user watches a task cross compaction or provider change and can state
  what survived, what expired, which runtime acted, and where evidence lives.
- Known bug, fixed 2026-07-18: `continuity_sources()` built the dev-skills path from a
  directory name (`skills-i-use`) that no longer existed on disk, so `/skills` fell back to
  built-in skills instead of Masih's own, and the Context Ledger showed only global sources.
  Root cause was a stale path string, not a ledger rendering bug.
- Ledger dev-skills rows: `skills/dev/*.md` files are now enumerated dynamically as
  individual, independently toggleable rows admitted by default (commit `2f85ce3`, with a
  focused regression in `core/src/elpis_context.rs`). Remote verification is in progress;
  not accepted until the CI run passes and a terminal render check confirms the rows.

### F8. Claude Code as a selectable runtime (R11) — first slice landed

- `codex-rs/claude-bridge`: a subprocess bridge to the Claude Code CLI's `--print`
  non-interactive mode (`claude -p --output-format stream-json`), with a typed event
  parser built from an empirical capture (Claude Code 2.1.214), tested, CI-green.
- `/claude-code` command and an `ActiveRuntime` (`Codex` | `ClaudeCode`) session enum
  exist (`codex-rs/tui/src/chatwidget/runtime_selection.rs`) and switching prints a
  confirmation in the transcript.
- Turn submission is actually routed through `claude-bridge` when the active runtime is
  Claude Code (`codex-rs/tui/src/chatwidget/claude_code_turn.rs`), not just recorded:
  whole-message text-in/text-out only — no incremental streaming render (unlike Codex's
  `StreamController`), and `tool_use`/`tool_result` content blocks are not inspected or
  bridged to Elpis's approval/diff/permission surfaces. A Claude Code turn that calls a
  tool shows nothing in the TUI until the process exits, then only its final text (if
  any). `codex-rs/claude-bridge`'s own parser doesn't decode those block shapes yet
  either, so this is a source-level limitation, not just an integration gap.
- Not yet implemented: Claude Code does not appear in the `/model` picker as its own
  provider group (`codex-rs/tui/src/chatwidget/model_popups.rs` has no Claude Code
  entries) — `/claude-code` is currently the only way to select it.
- Claude Code authenticates via its own subscription login, separate from Codex/ChatGPT
  and from the native/OpenRouter provider credentials in F5.

## Reduction campaign

- Completed process subtraction: remove broad inherited regression from every ordinary push,
  remove CI auto-formatting, and remove the self-mutating build-status commit.
- Bounded code subtraction: the issue #32 removal of the inert `debug-m-drop` and
  `debug-m-update` placeholders was recovered from the unmerged `agent/product-integration`
  branch and cherry-picked onto `agent/context-ledger-ui` (PR #49 closed as superseded);
  lands with that branch after CI passes.
- Measured candidates are recorded in `docs/BUILD_AND_REDUCTION_AUDIT.md`.
- Do not delete arbitrary workspace crates: first prove they are reachable from the Elpis
  binary and optional under the product requirements.

## Important — after the first-release foundation

- Agent-owned post-turn context pruning, full contract (Masih's ace in the hole; see
  `docs/CONTEXT_AND_SESSIONS.md`): agent-authored compact turn outcome record, deterministic
  validation and expiry of exploratory traces from the next request, fail-closed
  preservation, plus the evidence-first completion hierarchy rendering that same record.
  The deterministic first pass (1,200-char receipts, whitespace cleanup, evidence pointers)
  is merged; the outcome-record engine is not. Fable-owned.
- Default tool integration, evaluated 2026-07-19: `rtk` 0.43.0 (token-optimizing command
  proxy — candidate engine for pruning's compact action receipts), `codebase-memory-mcp`
  (already reachable through inherited `~/.codex/config.toml`; decide default-on wiring and
  the SessionStart hint), and `ponytail` (minimal-code discipline; ship as a default
  `skills/dev` rule).
- Context Ledger parity with `design-prototype.png` (kept in Masih's local files; not in the
  repo): grouped sections (files/memory/instructions/evidence), per-row token counts,
  include/exclude-all keys, and the "why included" panel. Delegable; spec is the prototype.
- Interactive clarifying questions: before ambiguous or costly work, Elpis presents a
  structured selectable prompt (question, options, multi-select) instead of silently
  assuming, and records the chosen answer in the session evidence.
- Live vendor acceptance of the implemented native Anthropic and Google adapters (see F5).
- Additional provider/runtime adapters using proven Pi/OpenClaw patterns.
- Behavioral enforcement across runtimes.
- Dictation with visible consent and editable, unsent text.
- Further Codex subtraction, one measured capability at a time.
- Workspace RAG enhancements: an interactive path prompt on `/rag` (Enter defaults to the
  terminal's current working directory with a configurable folder-depth/token guardrail
  against scanning something like `node_modules`), a natural-language RAG trigger so the
  agent can invoke retrieval without an explicit `/rag` command, and keeping RAG defaults
  from mixing active workspace source with the global memory `archive.md`.
- `/deep-research`: an autonomous mode combining structured RAG, web search, and recursive
  crawling to build reference context before proposing edits.
- Provider-grouped `/model` picker: list models grouped by provider (OpenAI, native
  Anthropic/Gemini, OpenRouter families) instead of a flat list.
- LSP-backed code intelligence for the active runtime: real language-server queries
  (go-to-definition, precise references, live diagnostics) instead of grep/text search.
  No confirmed LSP client exists in any runtime currently bridged into Elpis; scope as its
  own investigation before committing.
- A **Dynamic Context Files Panel** was proposed here in an earlier draft of this backlog;
  it is already implemented as the Context Ledger (see F7) and is not a future item.

## Nice-to-have

- `/auto` routing with a visible choice, reason, and manual override. A proposed shape:
  classify by complexity and route easy edits/summaries to a low-cost fast model, medium
  work to a mid-tier model, and architectural/multi-file work to a frontier model.
- Scheduled memory review or dreaming-style reports.
- Rich themes and animation beyond the first coherent cyan identity.
- Elpis Family Tree: a hierarchical multi-agent framework where a coordinator runtime
  delegates scoped sub-tasks to worker runtimes in parallel worktrees/branches, with a
  code-level token/cost harness to bound runaway loops.
- Messaging adapters (Telegram/Discord) using OpenClaw/Pi connection patterns, so Elpis can
  run as a daemon connected to channels.
- A `/elpis` poetry easter egg: a cyan-themed display of stylized lyrics or poems.

## Current Action

1. Install the verified `main` binary (run `29663596709`) and finish acceptance: fixed
   `ES.md` lean continuation, `/claude-code` runtime smoke, cyan header and context
   accounting consistency. Already accepted live on 2026-07-18/19: dev-skills ledger rows
   and toggles, `/status`, exact resume, memory teach/related-recall/unrelated-omission,
   OpenAI end-to-end task. OpenRouter leg deferred (use a free model; low priority).
2. Masih approves or rejects the installed build; release tagging happens only after his
   approval.
3. Cargo-timing first pass (2026-07-19): top costs are first-party crates (codex-core
   39.7s, tui 22.1s, config 19.7s, app-server 17.8s); no dominant third-party dependency.
   Select one bounded deletion from `docs/BUILD_AND_REDUCTION_AUDIT.md` candidates only.
4. R11 (Claude Code as a selectable runtime) — foundation merged to `main`, not complete.
   Done and CI-verified: `codex-rs/claude-bridge` crate (spawns
   `claude -p --output-format stream-json`, parses real captured event schema);
   `/claude-code` command switches `ActiveRuntime`; submitting a message while active
   routes through the bridge and renders plain-text replies
   (`tui/src/chatwidget/claude_code_turn.rs`). Never run live by a human, only CI with a
   fake `claude` binary (`CLAUDE_BRIDGE_BINARY_OVERRIDE_ENV`-style override — check the
   crate for the exact env var name). Remaining, not started: (a) `/model` picker must
   show Claude Code as its own provider group with its model listed beneath, matching
   the pattern already used for other providers; (b) tool-call/permission event
   rendering — `claude-bridge`'s own doc comments say `tool_use`/`tool_result` JSON
   shapes were never empirically captured, only plain-text `assistant`/`result` events
   were; a turn where Claude uses a tool currently shows "used tools, not rendered yet"
   instead of the tool call. Next step for (b): capture a real `claude -p --verbose
   --output-format stream-json` run that actually uses a tool, add typed variants for
   the observed shape, then bridge to Elpis's existing approval/diff UI reused from the
   Codex path — don't invent a parallel one.
