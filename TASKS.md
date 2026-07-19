# Elpis Tasks

This is the project task source of truth. A feature is complete only after its stated
behavior is implemented and verified; vision documents describe intended behavior but must
not claim unfinished behavior is available.

Version map: **Foundational = v0.1** (publish gate), **Important = v0.2**,
**Nice-to-have = v0.3+**. An item ships in v0.1 only if it is listed under Foundational.

## Foundational — required for the first release (v0.1)

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
- Cleaner visibility gap (Masih, 2026-07-19: "I have not yet seen it work"): the cleaner
  is wired into the live request path (`client.rs` calls `clean_transient_tool_outputs`),
  but nothing in the TUI ever demonstrates it. Acceptance: a session with a long tool
  output must show the eviction in `/status` and the user must be able to see the receipt
  replacing the raw output. Until a user can watch it happen, it does not count as done.
  Masih's concrete ask: a per-turn "context saved" metric — how many bytes/tokens the
  cleaner removed from the next request — visible as a number and bar in the header,
  ledger, or `/context`, so a user can watch context go *down*, not only up.

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
- Context Ledger prototype-parity slice is implemented in PR #55: real admitted sources are
  grouped as files/memory/instructions/evidence, row/section/total counts are explicitly
  estimated tokens, `g i`/`g e` toggle all selectable rows, and `w` shows the selected
  source's inclusion reason. CI run `29682921080` passed; installed terminal render
  acceptance remains pending.

### F8. Claude Code as a selectable runtime (R11) — text bridge verified live; takeover mode built, awaiting live acceptance

- `codex-rs/claude-bridge`: a subprocess bridge to the Claude Code CLI's `--print`
  non-interactive mode (`claude -p --output-format stream-json`), with a typed event
  parser built from an empirical capture (Claude Code 2.1.214), tested, CI-green.
- `/claude-code` command and an `ActiveRuntime` (`Codex` | `ClaudeCode`) session enum
  exist (`codex-rs/tui/src/chatwidget/runtime_selection.rs`) and switching prints a
  confirmation in the transcript.
- Live acceptance 2026-07-19 (tmux-driven, installed 11:54 binary): `/claude-code` then a
  text prompt returned the exact reply `READY_E2E` rendered in the TUI. Text routing works
  end to end on the installed build.
- Why it looked broken to Masih: the installed binary predates commit `0084fcf` and still
  prints the stale "full routing is not implemented yet" notice on switch; the header keeps
  showing the Codex model (`gpt-5.5`) while Claude Code is active; and `/model` has no
  Claude Code entry. Fixes: reinstall from a post-`0084fcf` build (triggered as run
  `29681801846`); header must display the active runtime; picker entry is delegated
  (LEDGER/PICKER worker branch).
- Turn submission is actually routed through `claude-bridge` when the active runtime is
  Claude Code (`codex-rs/tui/src/chatwidget/claude_code_turn.rs`), not just recorded:
  whole-message text-in/text-out only — no incremental streaming render (unlike Codex's
  `StreamController`), and `tool_use`/`tool_result` content blocks are not inspected or
  bridged to Elpis's approval/diff/permission surfaces. A Claude Code turn that calls a
  tool shows nothing in the TUI until the process exits, then only its final text (if
  any). `codex-rs/claude-bridge`'s own parser doesn't decode those block shapes yet
  either, so this is a source-level limitation, not just an integration gap.
- Implemented in PR #55: `/model` shows a `CLAUDE CODE` provider/runtime group with an
  honest `Account default` row because the CLI subscription chooses the actual model.
  Selecting it reuses the existing `ActiveRuntime::ClaudeCode` switch path; selecting a
  Codex model switches back to Codex. CI run `29682921080` passed; installed picker render
  acceptance remains pending.
- Claude Code authenticates via its own subscription login, separate from Codex/ChatGPT
  and from the native/OpenRouter provider credentials in F5.
- Direction change 2026-07-19 (Masih): the primary interactive path is now **takeover
  mode** — `/claude-code` suspends the Elpis TUI and launches the user's real `claude`
  CLI in the same terminal (exactly as if the user typed `claude`), restoring Elpis on
  exit. This gives 100% genuine Claude Code UX with zero re-implementation; Elpis state
  enters the session through Claude Code's own extension points (workspace CLAUDE.md /
  `--append-system-prompt`, hooks, MCP). This is the Ollama pattern: Ollama never
  re-implemented Claude Code's UI — Claude Code's backend is swappable via its
  Anthropic-compatible API endpoint, and the real CLI does the rest. Fable-owned.
- Takeover mode implemented 2026-07-19 (branch `agent/claude-code-takeover`):
  `/claude-code` now sends `AppEvent::LaunchClaudeCodeTakeover`; the app layer reuses the
  external-editor suspend machinery (`Tui::with_restored(RestoreMode::Full, ..)`) to
  fully restore the terminal, run `claude` with inherited stdio in the workspace cwd,
  and restore the TUI on exit. The runtime picker's `Account default` row keeps the
  in-Elpis text bridge. Acceptance: Masih runs `/claude-code` on the fresh build and
  lands in real Claude Code, exits, and is back in Elpis. Continuity injection via
  Claude Code extension points (CLAUDE.md / `--append-system-prompt`) is the follow-up,
  not part of this gate.
- The existing text bridge remains for headless/scripted turns. Full
  `tool_use`/`tool_result` bridging onto Elpis approval surfaces is deprioritized —
  only worth building if takeover mode proves insufficient.
- Forking Claude Code itself is not an option: it is closed-source (an obfuscated
  JavaScript CLI), unlike Codex (Apache-2.0 Rust). The correct architecture is exactly
  this bridge plus UX parity work (see the v0.2 Claude-parity item).

## Reduction campaign

- Completed process subtraction: remove broad inherited regression from every ordinary push,
  remove CI auto-formatting, and remove the self-mutating build-status commit.
- Bounded code subtraction: the issue #32 removal of the inert `debug-m-drop` and
  `debug-m-update` placeholders was recovered from the unmerged `agent/product-integration`
  branch and cherry-picked onto `agent/context-ledger-ui` (PR #49 closed as superseded);
  lands with that branch after CI passes.
- Measured candidates are recorded in `docs/BUILD_AND_REDUCTION_AUDIT.md`.
- New 2026-07-19: the installed binary is 356,440,368 bytes versus the 102,988,260-byte
  stripped baseline artifact — a 3.5× size regression despite `file` reporting it
  stripped. Investigate what the build-cycle CI changes (incremental compilation, profile
  edits) did to the release profile before any code deletion is credited or blamed.
- Do not delete arbitrary workspace crates: first prove they are reachable from the Elpis
  binary and optional under the product requirements.

## Important — after the first-release foundation (v0.2)

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
  `skills/dev` rule). A fourth candidate — a fast codebase-indexing bash script Masih
  linked in an earlier (lost) conversation — needs Masih to re-send the link before it can
  be evaluated.
- Context Ledger parity with `design-prototype.png` (kept in Masih's local files; not in the
  repo): grouped sections (files/memory/instructions/evidence), per-row token counts,
  include/exclude-all keys, and the "why included" panel are implemented in PR #55 and
  CI-green; installed terminal render acceptance remains.
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

- Context Ledger corrections, from Masih's 2026-07-19 session review (verified findings):
  the displayed numbers are file sizes in bytes, not tokens, and carry no unit label
  (`5.0k` for the 5,076-byte global `~/.codex/AGENTS.md` is accurate but unlabeled); the
  total (`29.4k admitted`) is the byte sum of admitted sources and legitimately differs
  per workspace because sources are per-project, but nothing explains that; a source
  capped by the 8,000-character admission limit still displays its full on-disk size.
  Required: label units (or convert to a token estimate), show the truncated-vs-full size
  when the cap applies, and state the workspace scope in the ledger header.
  PR #55 converts the rows and totals to explicitly estimated, admission-capped tokens;
  truncated-vs-full disclosure and an explicit workspace-scope label remain.
- Context Ledger `/add` completion: `/add <path>` already admits single files
  (`core/src/elpis_context.rs::add_continuity_source`), but the ledger shows no hint that
  it exists. Required: a "use /add to add a file or directory" hint line in the ledger
  panel, and directory support — `/add <dir>` opens a file chooser listing the directory's
  files so the user picks which to admit as individual toggleable rows.
- Claude Code UX parity, phase 1 (Masih explicitly wants Elpis to feel like Claude Code):
  the four permission modes cycled with one key (normal / auto-accept edits / plan /
  bypass) including an "auto" mode; composer mouse selection with Backspace deleting the
  selection; composer undo/redo (Ctrl+Z / Ctrl+Y). Investigate what the inherited Ratatui
  composer already supports before writing anything new.
- Distribution strategy ("publish everywhere", Masih 2026-07-19): package Elpis's
  differentiators — durable memory and post-turn pruning — as a Claude Code plugin
  (MCP server + hooks) and a Codex-compatible equivalent, publishable to their plugin
  marketplaces, so Claude/Codex users get Elpis features without leaving their tool.
  Separately: an Anthropic-API-compatible proxy endpoint served by Elpis, so the real
  Claude Code CLI can run *any* Elpis-routed model (the same mechanism Ollama uses).
  Release engineering: macOS and Windows binaries are built on GitHub Actions macOS/
  Windows runners — owning a Mac is not required; publish .deb + tarball + checksums,
  and report compressed download size per platform.
- Startup speed: audit the elpis launch path (config, sqlite, auth, MCP host) for work
  that blocks the first frame, using the audit method in
  `docs/BUILD_AND_REDUCTION_AUDIT.md`; the binary must feel faster than a Node CLI or
  the Rust choice is being wasted. Related: the 3.5× size regression above.
- `codebase-memory-mcp` heats Masih's CPU (~90°C) when indexing: it must not be wired
  default-on; if integrated, index lazily/throttled and never at session start.
- Spinner identity: replace inherited working-state words with Elpis language
  ("elpising…"), matching the cyan identity; small, delegable.
- Ship `tmux` as a recommended (not required) package alongside the `.deb`
  (`Recommends:` field), since tmux-driven render checks are the project's own acceptance
  method. Fable's position, stated for the record: a hard dependency would be wrong —
  Elpis runs fine without tmux; Masih can overrule.

## Nice-to-have (v0.3+)

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
- A `/elpis` poetry easter egg: a cyan-themed display of stylized lyrics or poems
  (implementation in flight on `agent/rag-ux-easter-egg`, PR #54).
- Claude Code UX parity, phase 2: managing running agents from the keyboard (Claude
  Code's left-arrow agent panel), mid-turn message queuing, mobile remote control of a
  session, and opt-in telemetry. Large, multi-slice work; explicitly not v0.1/v0.2.

## Current Action

1. Priority 1 — make Claude Code usable and believable inside Elpis (F8). Verified
   2026-07-19 by a live tmux-driven run of the installed binary: text routing works
   (`/claude-code` + prompt returned the exact requested reply in the TUI), but the
   installed 11:54 binary predates commit `0084fcf` and still prints the false "full
   routing is not implemented yet" notice, the header keeps showing the Codex model
   while Claude Code is active, and `/model` offers no Claude Code entry — so to a user
   the feature reads as absent. Actions: install the fresh post-fix build (run
   `29681801846`, triggered); make the header show the active runtime; merge the CI-green
   picker entry (PR #55, `agent/ledger-parity-model-picker`); then the Fable-owned core work —
   bridge `tool_use`/`tool_result` events onto the existing approval/diff UI, starting
   from an empirical capture of a tool-using `claude -p --verbose
   --output-format stream-json` run.
2. Masih approves or rejects the fresh installed build; release tagging happens only
   after his approval. Everything accepted on 2026-07-19 remains accepted: continuity
   (`ES.md`/`GOAL.md`, lean continuation, exact resume), memory
   teach/recall/omission, `/status`, dev-skills ledger rows, OpenAI end-to-end task,
   header/footer accounting agreement. OpenRouter leg deferred.
3. Worker branches in flight: `agent/rag-ux-easter-egg` (terra, PR #54) and
   `agent/ledger-parity-model-picker` (sol, PR #55, CI run `29682921080` green). Integrate
   one at a time; the coordinator merges, workers do not.
4. Cargo-timing first pass (2026-07-19): top costs are first-party crates (codex-core
   39.7s, tui 22.1s, config 19.7s, app-server 17.8s); no dominant third-party dependency.
   Also investigate the 3.5× installed-binary size regression (see Reduction campaign)
   before selecting one bounded deletion from `docs/BUILD_AND_REDUCTION_AUDIT.md`.
