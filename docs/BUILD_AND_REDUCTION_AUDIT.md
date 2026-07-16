# Elpis Build And Reduction Audit

## Baseline

The last verified pre-optimization run was GitHub Actions run `29534784054` for commit
`e841704e`. The runner started at 21:06:52 UTC and the result commit was written at
21:27:58 UTC: about 21 minutes end to end. The uploaded stripped Linux artifact was
102,988,260 bytes.

This is a cycle-time problem, but not evidence that every crate in `codex-rs/` is built.
`cargo build -p codex-tui --bin elpis` compiles the dependency graph reachable from the
Elpis TUI; unrelated workspace members do not materially affect that command merely by
existing.

## Root Causes

1. `codex-tui` is the mature Codex-derived product surface. Its active graph includes the
   core runtime, terminal rendering, authentication, permissions, sandboxing, sessions,
   MCP/RAG integration, skills/plugins, model management, and supporting libraries. A true
   cold build is therefore substantial.
2. The old workflow compiled overlapping test graphs on every `main` push: several memory
   crates, app-server integration tests, the complete TUI library test target, individual
   TUI tests, and the binary itself.
3. The cache key omitted the commit while GitHub caches are immutable. Later runs restored
   an old `target` snapshot but could not save the newly compiled state under the same key.
   `CARGO_INCREMENTAL=0` also prevented incremental reuse.
4. CI formatted source and committed a status file back to `main`. That automation required
   write permission, created repository noise, and mixed verification with source mutation.

## Changes Applied In The Build-Cycle Pass

- Restore strict `cargo fmt --check`; CI never edits source.
- Remove the self-mutating `.github/builds/latest-main.json` process.
- Cache Cargo downloads separately from compiled outputs.
- Key compiled outputs by toolchain, lockfile, and commit, with compatible restore prefixes.
- Enable incremental compilation and disable dev/test debug information in CI.
- Keep the first-release launcher, provider, memory, archive, bounds, retrieval, and binary
  checks on ordinary changes.
- Run the inherited app-server and complete TUI regression graph only on nightly, manual
  full-regression, and tagged-release runs.
- Generate and upload Cargo's HTML timing report with non-PR builds. The next reduction pass
  must use that report before removing dependencies.

## Reduction Candidates

### Proven safe or already bounded

- Remove the inert `debug-m-drop` and `debug-m-update` commands. They are hidden, labelled
  `DO NOT USE`, and only display a generic stub. This is tracked by issue #32.
- Keep broad TUI/app-server regression tests, but stop compiling their dev-only graph on
  every ordinary push. This removes process, not product capability.
- Keep the auxiliary `md-events` binary out of the Elpis build. It is not built by
  `--bin elpis`; deleting it would not explain the current build time.

### Measure before deleting

The next Cargo timing report should identify the most expensive reachable crates. Audit
these product surfaces in descending measured cost, then remove one bounded capability at a
time with an acceptance test:

- Codex Desktop handoff and cloud configuration;
- apps/connectors and plugin browsing;
- feedback upload and telemetry presentation;
- IDE integration and external-agent import;
- usage/account views, personality, plan mode, pets, raw mode, and Vim mode;
- theme and syntax-highlighting assets;
- image handling when no retained first-release interaction requires it.

Several of these have hidden slash commands but may still be reachable through keybindings,
settings, startup flows, or shared runtime code. Hiding a command is not proof that its
underlying dependency is removable.

### Do not delete for build-speed theatre

Do not remove arbitrary workspace members such as cloud tasks, V8 experiments, or sample
servers merely because they appear in the root workspace list. First prove that they are in
the Elpis binary's reachable dependency graph. Repository-size cleanup and build-time
cleanup are different campaigns.

## Feature Truth At This Audit

### UI/UX

Implemented: Elpis naming and the visible `Elpis · continuity runtime` title; the inherited
Ratatui interaction quality remains intact.

Not yet implemented: the coherent amber theme, persistent runtime/model/context/memory line,
provider-aware `Choose a mind` model picker, signature continuity event, and evidence-first
completion hierarchy. `docs/UI_IDENTITY.md` is a design contract, not proof of those surfaces.

### Memory And Context

Implemented and remotely tested: Elpis-owned memory roots, recall tracking and promotion,
artifact bounds, ranked retrieval, portable `GOAL.md`/`ES.md`, focused continuity tests, and
a verified Linux build. This pass adds a focused archive regression and makes archive write
failure block baseline reset rather than silently losing deleted memory.

Not complete: user-visible exact-resume, lean-continuation, related/unrelated recall, review,
reset, and compaction acceptance still must run locally. The current 1,000-character context
cleaner is a blunt deterministic filter without a focused test, age/turn distinction,
conclusion preservation, or source pointer; it does not yet satisfy the full context-cleaner
contract.

### Providers

Implemented and remotely tested at the launcher/configuration layer: OpenAI, OpenRouter,
Bedrock, Ollama, and LM Studio provider IDs; Claude Sonnet and Gemini Pro/Flash aliases route
through OpenRouter.

Not complete: Claude and Gemini are not native vendor adapters, the `/model` UI is not the
provider-aware `Choose a mind` surface, and end-to-end authenticated task/resume acceptance
has not been recorded for each first-release path.

## Next Reduction Gate

After this branch passes, compare its ordinary-change runtime with the 21-minute baseline and
inspect the uploaded Cargo timing report. Only then select the highest-cost optional product
surface, remove it in isolation, and prove retained execution, permissions, sandboxing,
sessions, compaction, context, memory, and RAG still work.
