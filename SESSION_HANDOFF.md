# Elpis Session Handoff

## Goal

Ship Elpis `0.1.0` with internal RAG, portable context continuity, durable memory,
OpenAI subscription authentication, OpenRouter support, a distinctive continuity-first
interface, and a clean release path.

## Current State

- Canonical released work is on `main`; the active engineering branch is
  `agent/build-cycle-and-reduction-audit-v2`.
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
- Claude Sonnet and Gemini Pro/Flash launcher shortcuts currently route through OpenRouter;
  native Anthropic and Google adapters are not implemented.
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

## Known Gaps

- The 1,000-character context cleaner is a blunt filter without a focused test, turn-age
  distinction, compact conclusion, or source pointer. The full cleaner contract is not done.
- Memory/context require end-to-end local acceptance.
- OpenAI/OpenRouter require authenticated task-and-resume acceptance.
- Claude/Gemini are OpenRouter aliases, not native adapters.
- `/model` is not yet the provider-aware `Choose a mind` surface.
- The distinctive amber continuity UI is not implemented beyond naming/title.

## Next Action

1. Open and verify the build-cycle branch through GitHub Actions.
2. Compare its ordinary-change runtime with the 21-minute baseline and inspect the Cargo
   timing artifact.
3. Merge only when the focused branch checks pass.
4. Install the verified binary and run exact/lean context, related/unrelated memory,
   OpenAI, and OpenRouter acceptance locally.
5. Begin the UI pass with the persistent identity line and amber styling foundation.

## Boundaries

- Codex/ChatGPT login is authentication-only; Elpis owns context, memory, policy, provider
  choice, and product identity.
- Do not claim design documents as implemented behavior.
- Do not delete workspace crates merely because their names look unrelated; prove reachability
  and optionality with Cargo timings and product acceptance.
- Do not run Cargo locally on the maintainer workstation.
