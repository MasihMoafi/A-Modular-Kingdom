# Upstream Capability Map

Elpis should reuse proven implementations and reserve original work for its distinctive
context, memory, continuity, assimilation, and supervision layer.

| Source | Proven capability to reuse or study | Elpis boundary |
| --- | --- | --- |
| Codex | Rust TUI, native coding loop, commands, patches, permissions, sandboxing, streaming events, threads, compaction | Foundation for Codex-quality execution and presentation; copied code must live inside Elpis |
| OpenClaw | Runtime/provider separation, Codex harness bridge, context projection, session mirror, pruning, memory flush, retrieval, promotion, channels | Primary TypeScript reference for the surrounding Elpis control layer |
| Pi | Small composable TypeScript packages for multi-provider APIs, agent state, TUI, and coding CLI | Study provider-neutral interfaces and extension simplicity; Pi does not supply Codex-level built-in permissions |
| Hermes Agent | Provider choice, TUI, cross-session search, user modeling, skill learning, scheduled work, multiple execution backends | Study the closed learning loop and user-facing memory controls; verify implementation before adopting claims |
| OpenCode | Multi-provider coding product, read-only/build agents, subagents, polished installation and desktop/TUI delivery | Study routing, product packaging, agent modes, and release experience |

## Current Source Locations

- Codex: `/home/masih/Desktop/f/p/others/codex`, fetched `origin/main` at
  `2e1607ee2fa8099a233df7437adee5f16a741905` on 2026-07-15. Its working tree has
  unrelated local changes and must not be reset.
- OpenClaw current shallow clone: `/home/masih/Desktop/f/p/others/openclaw-upstream`,
  revision `63aafd003d687e04bf4398bbd6f2abb583357bfa`.
- OpenClaw older May 26 source archive: `/home/masih/Desktop/f/p/others/openclaw-main`.

## OpenClaw's Codex Ownership Model

When Codex is selected, Codex owns the low-level model loop, native tools, native thread,
and native compaction. OpenClaw retains model/runtime selection, channels, session files,
the transcript mirror, projected context, OpenClaw dynamic tools, approvals, and output
delivery. Elpis should adopt this explicit layered ownership model.

No capability should enter `TASKS.md` as implemented merely because an upstream
project has it. It becomes an Elpis feature only after the Elpis acceptance check passes.
