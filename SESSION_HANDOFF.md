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
- Done: `foundation-codex-baseline` worktree at `/home/masih/Desktop/f/p/Elpis-foundation`
  on branch `agent/foundation-codex-baseline`.
- WIP: runtime-boundary work now has a Codex-only `--runtime` selector, an explicit
  ownership contract, dispatch through the existing Codex launch body, and focused
  contract tests. Gemini and Claude adapters and transport/authentication choices were
  deliberately not added.
- Done for the foundation import: the complete committed `codex-rs/` workspace from
  pinned revision `2e1607ee2fa8099a233df7437adee5f16a741905` is now contained in
  this worktree. Its Apache-2.0 `LICENSE`, `NOTICE`, provenance, upstream crate
  boundaries, and tests are preserved; no features have been subtracted.
- Done for the build baseline: Codex's documented OpenSSL prerequisite is present and
  the locked imported `codex-tui` package compiles successfully.
- Done for the launch smoke: `codex-rs/target/debug/codex-tui` starts from the
  worktree with no donor-path access. `--help` exits 0; a live launch runs for 4 s
  (killed by timeout). `strace` (617 syscalls) and `strings` confirm zero file opens
  from `/home/masih/Desktop/f/p/others/codex`. Evidence in
  `docs/LAUNCH_SMOKE_EVIDENCE.md`.
- Verified: `foundation-codex-baseline` passed its final user-visible acceptance
  check. The imported TUI completed an authenticated Codex turn
  `019f6616-842c-7b90-9b48-844406cd496f`, visibly ran `pwd`, created the requested
  harmless marker file, and displayed the verified working directory and exact
  marker content. The marker file was removed. Evidence in
  `docs/FOUNDATION_CODEX_BASELINE_EVIDENCE.md`.
- Blocked: none. The incomplete TUI test run is a known verification gap, not a blocker
  for the compilation baseline Masih accepted on 2026-07-15.
- Runtime-boundary verification is deferred: local Cargo compilation significantly
  disrupted Masih's workstation and was stopped at his request. This WIP must remain
  `in_progress` until its focused tests and behavior-preservation check run elsewhere or
  under an explicitly approved low-impact setup.

## Evidence

- Changed for the foundation checkpoint: imported `codex-rs/` at the pinned commit,
  retained its license/notice and added `codex-rs/ELPIS_UPSTREAM.md`; updated
  `docs/CODEX_FOUNDATION_MIGRATION.md` and this handoff. No imported runtime behavior
  was changed.
- Minimal MCP smoke: one advertised tool, exactly two parameters, no prompts or
  resources, and successful retrieval from `GUIDE.md`.
- Codex MCP configuration no longer carries the obsolete `MCP_PROFILE` switch.
- Current verification: 10 Rust tests passed and
  `.venv/bin/python -m compileall -q src` passed.
- Imported-source integrity matches the pinned committed `codex-rs/` tree exactly,
  excluding the added local provenance/license files and build output.
- `cargo metadata --manifest-path codex-rs/Cargo.toml --no-deps` passed, and
  `cargo test --manifest-path codex-rs/Cargo.toml -p codex-utils-approval-presets
  --locked` passed.
- Codex's pinned build setup requires `pkg-config` and OpenSSL development files.
  Ubuntu `libssl-dev` 3.0.13-0ubuntu3.11 was installed; `pkg-config --modversion
  openssl` now returns `3.0.13`.
- `CC=/usr/bin/cc CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc cargo
  build --manifest-path codex-rs/Cargo.toml -p codex-tui --locked` passed in 5m41s.
- Known test gap: plain `cargo test -p codex-tui --locked` first aborted on a stack
  overflow because it omitted Codex's documented `RUST_MIN_STACK=8388608`. The exact
  failed test passed with that setting. A full rerun then progressed through the 3,068
  library tests but hung in
  `ide_context::ipc::tests::fetch_ide_context_does_not_fall_back_after_primary_protocol_error`
  and was stopped. Per Masih's direction, do not install or compile `cargo-nextest`;
  accept successful TUI compilation as this baseline and retain the hang as a gap.
- `FEATURES.json` parses successfully. The edited migration and handoff documents pass
  `git diff --check`; the imported upstream tree retains intentional whitespace in
  fixtures, prompts, patches, and snapshots, so whole-import `diff --check` is noisy.
- Authentication smoke: installed Codex `0.144.4` returned account type `chatgpt` with
  OpenAI authentication required through `account/read`; no thread or turn was started.
- Earlier live verification: a resumed Codex thread recalled its prior codeword, ran
  `pwd`, and created the requested file through a `fileChange` event.
- Stable file boundaries for action rendering, permissions, and mouse selection are
  recorded in `docs/CODEX_FOUNDATION_MIGRATION.md`.
- Resolved: the imported foundation launch and donor-path runtime checks are complete;
  evidence in `docs/LAUNCH_SMOKE_EVIDENCE.md`. Full TUI test suite gap remains.
- Final foundation acceptance: the repository-built imported TUI used the existing
  authenticated Codex account to run `pwd`, create
  `.elpis-foundation-acceptance-test.txt`, and verify its exact
  `ELPIS_FOUNDATION_CODEX_BASELINE_OK` content. An independent check passed and the
  test file was removed. `FEATURES.json` now marks the feature `verified`.
- Open risk: deprecated Python memory/tool support modules remain in the repository but
  are no longer imported or exposed by the MCP; delete them only after checking for
  non-MCP callers.
- Open risk: the TUI's `/yolo` label is inaccurate and toggling it does not recreate or
  reconfigure the current Codex thread. The prototype always starts with Codex's Default
  policy (`on-request` plus workspace write).
- The foundation import is committed on `agent/foundation-codex-baseline`; inspect
  `git log -1` for the checkpoint hash. Build output under `codex-rs/target/` is ignored.
- Runtime-boundary WIP evidence is recorded in
  `docs/RUNTIME_BOUNDARY_WIP_EVIDENCE.md`. Static review and `git diff --check` passed;
  the added Rust tests were not executed, so no runtime-selection acceptance claim was
  made and `FEATURES.json` remains `in_progress`.

## Next Action

Stop at the runtime-boundary WIP checkpoint. Before expanding the contract or adding a
non-Codex adapter, run the focused tests and behavior-preservation check in an
environment that does not disrupt Masih's workstation.

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
- Do not install or compile `cargo-nextest`; successful imported-TUI compilation is the
  accepted baseline, and the hanging TUI test is a recorded gap.
- Do not retain raw searches, file reads, command output, or failed probes after their
  conclusion and source pointer have been recorded.
- Do not run Cargo on Masih's workstation for this WIP unless he explicitly approves a
  low-impact verification setup.
