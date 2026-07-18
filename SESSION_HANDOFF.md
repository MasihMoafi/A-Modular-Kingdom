# Elpis Session Handoff

## Goal

Ship Elpis `0.1.0` with internal RAG, portable context continuity, durable memory,
OpenAI subscription authentication, OpenRouter support, a distinctive continuity-first
interface, and a clean release path.

## Current State

- Canonical released work is on `main`; the build-cycle branch merged as PR #34, and the
  active engineering branch is `agent/context-ledger-ui`.
- Repository cleanup and the one-tool internal RAG path are accepted complete.
- Durable memory is Elpis-owned under `~/.elpis`, bounded, age-aware, diversity-ranked,
  recall-tracked, reviewable, and protected from one-off promotion.
- Elpis writes portable workspace `GOAL.md` and compact per-turn `ES.md` checkpoints.
  Fresh threads admit them automatically; exact resume keeps the native thread.
- Compaction stops if portable continuity cannot be synced first.
- `/status` reports admitted rules, goal, checkpoint, and memory summary with source, size,
  lifetime, and reason.
- OpenAI subscription remains the default authentication path. OpenRouter is separately
  keyed and selected explicitly.
- Claude Sonnet and Gemini Pro/Flash launcher shortcuts route through OpenRouter; native
  Anthropic (`--provider anthropic`) and Google Gemini (`--provider google-gemini`) adapters
  are implemented with streaming and mock-server tests (PR #46); live vendor acceptance is
  pending.
- The binary is versioned `0.1.0`. Tagged builds publish a checksummed Linux x86_64 release,
  and `scripts/install-elpis.sh` verifies and installs it atomically.
- UI identity is only partially implemented: Elpis naming and runtime title exist, while the
  amber system, persistent identity line, provider-aware model picker, continuity event, and
  evidence-first completion hierarchy remain to be built.

## Authoritative Verification

- GitHub Actions run `29534784054` passed the pre-optimization focused memory, context,
  provider, TUI, binary, and identity checks for commit `e841704e`.
- Its Linux artifact exists and is 102,988,260 bytes.
- The run took about 21 minutes from runner start to result commit.
- This proves the checked code paths compile and pass remotely. It does not replace the
  user-visible exact/lean continuation, related/unrelated recall, clean install, or provider
  acceptance checks.

## Active Branch Changes

- CI no longer formats or commits source.
- The self-mutating build-result file is removed.
- Cargo downloads and compiled outputs use separate caches; compiled output keys now allow
  each commit to save its updated incremental state while restoring compatible predecessors.
- Ordinary changes run the first-release launcher/provider/memory/archive checks and build
  one binary. Exhaustive inherited TUI/app-server regression moves to nightly, manual full,
  and tagged runs.
- Cargo timing HTML is uploaded for measured dependency reduction.
- Deleted memory lines are archived before baseline reset; archive write failures now stop
  reset rather than silently losing data; a focused regression covers this path.
- `docs/BUILD_AND_REDUCTION_AUDIT.md` records the measured causes and deletion candidates.
- WIP, unverified: the Context Ledger uses `Shift+Tab` and its selectable rows write an
  inspectable workspace `admission.toml`. The same registry now drives next-turn admission for
  `GOAL.md`, `ES.md`, applicable global/project `AGENTS.md` rules, and the configured
  `skills/dev` rules — now enumerated dynamically so every `skills/dev/*.md` file is its own
  toggleable row admitted by default. `memory_summary.md` and `archive.md` are not UI sources.
  The identity line and public slash-command list were narrowed to Elpis terminology. The
  issue #32 inert debug-command removal was cherry-picked onto this branch. No local Rust
  check was run; remote verification pending.

## Known Gaps

- The context cleaner is lifecycle-aware with focused tests, turn-age distinction, and a
  durable evidence pointer (`core/src/context_cleaner.rs`); its remaining gap is that the
  retained excerpt is positional (head/tail), not a semantic conclusion.
- Memory/context require end-to-end local acceptance.
- OpenAI/OpenRouter require authenticated task-and-resume acceptance.
- The native Anthropic/Gemini adapters lack live vendor acceptance; the `claude`/`gemini`
  launcher shortcuts remain OpenRouter aliases.
- `/model` is not yet the provider-aware `Choose a mind` surface.
- The Context Ledger implementation needs a remote or otherwise safe Rust test/render pass;
  do not call it accepted from static review.
- The current app-server settings update carries a model but not a provider, so a live provider
  switch needs an explicit protocol/runtime slice; do not present a cosmetic selector as support.

## Next Action

1. Open and verify the build-cycle branch through GitHub Actions.
2. Compare its ordinary-change runtime with the 21-minute baseline and inspect the Cargo
   timing artifact.
3. Merge only when the focused branch checks pass.
4. Install the verified binary and run exact/lean context, related/unrelated memory,
   OpenAI, and OpenRouter acceptance locally.
5. Run the focused Context Ledger regression and a terminal render check in GitHub Actions,
   then refine the selected `design-prototype.png` direction from evidence.

## Boundaries

- Codex/ChatGPT login is authentication-only; Elpis owns context, memory, policy, provider
  choice, and product identity.
- Do not claim design documents as implemented behavior.
- Do not delete workspace crates merely because their names look unrelated; prove reachability
  and optionality with Cargo timings and product acceptance.
- Do not run Cargo locally on the maintainer workstation.
