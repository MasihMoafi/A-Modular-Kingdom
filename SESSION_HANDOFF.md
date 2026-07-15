# Elpis Session Handoff

## Goal

Build Elpis as a provider-neutral coding-agent TUI and environment. Start from Codex's
proven Rust foundation, subtract unwanted product surfaces, add replaceable OpenAI/Codex,
Gemini, and Claude adapters, and add an OpenClaw-derived context and memory system.

## Current State

- Prototype milestone: Codex app-server is the current OpenAI backend with ChatGPT authentication,
  `gpt-5.4-mini`, streaming, approvals, real token usage, and authoritative thread
  start/resume/fork. It launches the installed `codex` executable by name and does not
  use the separate Codex source clone at runtime.
- Corrected product boundary: Elpis is the surrounding TUI/context/memory/session/policy
  environment; the selected low-level runtime may own its native loop and tools.
- Corrected implementation direction: use a pinned Codex Rust revision as Elpis's
  internal foundation and subtract, rather than recreating mature features one by one.
- Verified current Codex source after fetching `origin/main`: the donor clone is 1,420
  commits behind and has unrelated local changes, so no pull/reset was performed.
- Verified Codex's exact permission presets and the source modules for command/file
  lifecycle rendering, approvals, sandboxing, and patch safety.
- Verified OpenClaw's memory implementation source: live pruning, guarded compaction,
  pre-compaction dated flush, hybrid retrieval, bounded promotion, and optional dreaming.
- Verified current OpenClaw revision `63aafd003d687e04bf4398bbd6f2abb583357bfa`
  in the new shallow clone at `/home/masih/Desktop/f/p/others/openclaw-upstream`.
  When Codex is selected, Codex owns the low-level loop, native tools/thread, and native
  compaction; OpenClaw retains its surrounding control layer and transcript mirror.
- Done: `VISION.md` preserves the GitHub README's assimilation thesis as the stable
  product intent and defines the first public release's desired user-visible output.
- Done: `FEATURES.json` is the supervised, acceptance-tested implementation queue with
  difficulty, dependencies, parallel safety, file scope, and suggested agent routing.
- Done: `docs/AGENT_DISPATCH.md` defines coordinator ownership, one-worktree-per-task,
  worker prompts, integration, and the current parallelism gate.
- Done: global `$challenge-requirements` skill created and validated at
  `/home/masih/.codex/skills/challenge-requirements`.
- Done: `docs/UPSTREAM_CAPABILITY_MAP.md` records what to study from Codex, OpenClaw,
  Pi, Hermes Agent, and OpenCode without treating upstream claims as Elpis features.
- Done: the Codex and OpenClaw references in `GUIDE.md` point to their local clones.
- Done: the Elpis MCP contains one tool, `query_knowledge_base`, with only `query` and
  optional `doc_path`. The 17 legacy tools and four optional duplicate/voice tools were
  deleted from `src/agent/host.py`; their eventual replacements belong in Elpis Rust.
- Done: `docs/CONTEXT_AND_SESSIONS.md` specifies exploration expiry, receipts, context
  ledger, exact resume, lean continuation, checkpoints, and memory boundaries.
- Done: `docs/AUTHENTICATION_BOUNDARY.md` separates status-only authentication,
  Elpis-owned direct model authentication, and deliberately selected Codex runtime use.
- Done: `scripts/codex-auth-status-smoke.sh` proves authenticated ChatGPT account status
  without starting a Codex thread or exposing credentials.
- Done: `REQUIREMENTS.md` records Masih's corrected product criteria, the verified
  feature truth, required implementation order, and unresolved choices.
- Done: `docs/CODEX_FOUNDATION_MIGRATION.md` pins the candidate Codex revision, maps the
  source modules to preserve, and defines the fork-and-subtract migration and first
  acceptance check.
- In progress: `foundation-codex-baseline` in `FEATURES.json`; migration worktree not yet created.
- Blocked: none.

## Evidence

- Changed for the latest step: `AGENTS.md`, `VISION.md`, `FEATURES.json`, `GUIDE.md`,
  `REQUIREMENTS.md`, `readme.md`, `docs/AUTHENTICATION_BOUNDARY.md`,
  `docs/CONTEXT_AND_SESSIONS.md`, `docs/CODEX_FOUNDATION_MIGRATION.md`,
  `docs/UPSTREAM_CAPABILITY_MAP.md`, `docs/AGENT_DISPATCH.md`,
  `tui/docs/ELPIS_INSPIRATION.md`, and this handoff.
  These are control/documentation corrections; no runtime code was changed.
- Minimal MCP smoke: one advertised tool, exactly two parameters, no prompts or
  resources, and successful retrieval from `GUIDE.md`.
- Codex MCP configuration no longer carries the obsolete `MCP_PROFILE` switch.
- Current verification: 10 Rust tests passed and
  `.venv/bin/python -m compileall -q src` passed.
- `FEATURES.json` parses successfully. Whole-worktree `git diff --check` remains noisy
  because pre-existing edits in `src/agent/main.py` contain trailing whitespace.
- Authentication smoke: installed Codex `0.144.4` returned account type `chatgpt` with
  OpenAI authentication required through `account/read`; no thread or turn was started.
- Earlier live verification: a resumed Codex thread recalled its prior codeword, ran
  `pwd`, and created the requested file through a `fileChange` event.
- Open risk: without the installed Codex executable, the current OpenAI prototype cannot
  act as a coding agent; the donor foundation has not yet been copied into Elpis.
- Open risk: deprecated Python memory/tool support modules remain in the repository but
  are no longer imported or exposed by the MCP; delete them only after checking for
  non-MCP callers.
- Open risk: the TUI's `/yolo` label is inaccurate and toggling it does not recreate or
  reconfigure the current Codex thread. The prototype always starts with Codex's Default
  policy (`on-request` plus workspace write).
- Worktree contains earlier user/agent changes and is intentionally uncommitted.

## Next Action

Create an isolated foundation worktree from the pinned Codex revision and make the
first vertical slice pass: Elpis branding/startup, arbitrary mouse selection, readable
command/file lifecycles, and exact Codex permission presets. Preserve the current dirty
prototype and both upstream source trees.

## Ordered Tasks

1. **Foundation:** pin and copy the required Codex Rust workspace into Elpis with
   attribution; preserve command/file rendering, permissions, sandboxing, tools,
   sessions, and their tests before subtracting features.
2. **Runtime boundary:** keep Codex working through subscription authentication, then
   define the Elpis embedded and external runtime contracts for Gemini and Claude paths.
3. **Context engine:** expire searches, reads, command output, and failed probes; retain
   receipts and expose the model-visible context ledger.
4. **Session engine:** persist Elpis-owned threads and checkpoints; implement exact
   resume, lean handoff, compaction, fork, and rollback.
5. **Memory engine:** add dated append-only notes, pre-compaction flush, hybrid retrieval,
   provenance, bounded long-term promotion, and deletion/review controls.

## Do Not Repeat

- Do not reimplement Codex file, shell, patch, or approval tools in Python.
- Do not make Elpis depend on `/home/masih/Desktop/f/p/others/codex` at runtime.
- Do not hand-recreate Codex's permissions, tools, and action rendering when its tested
  Apache-2.0 Rust implementation can be copied and adapted.
- Do not remove Codex/ChatGPT login; it is the required authentication boundary.
- Do not mistake a compact TUI rendering for reduced model context.
- Do not expose the full Python MCP profile to Codex.
- Do not retain raw searches, file reads, command output, or failed probes after their
  conclusion and source pointer have been recorded.
