# Elpis

> You put an agent into an Elpis, and it becomes Elpis. Be Elpis, my friend.

Elpis is an agent environment for context, memory, behavior, runtime choice, evidence,
and cross-session continuity. A selected runtime—Codex, Gemini, Claude, or another
supported agent—enters the user's environment and assimilates into its current goal,
applicable rules, durable knowledge, and working style.

## Why It Exists

Agent quality depends on more than the model. Long transcripts, repeated file bodies,
raw tool output, stale instructions, and weak session boundaries consume context and
hide the current goal. Elpis gives the user and agent an explicit working-set policy:

- load durable guidance only when it applies;
- inject `@` resources intentionally and pinned resources only when changed;
- keep full evidence on disk while shrinking stale model-visible tool results;
- preserve goals, decisions, diffs, verification, and blockers across compaction;
- route edits and commands through visible sandbox and approval contracts.

The intended outcome is not another generic chat client. It is an environment in
which an agent can work with greater continuity, restraint, and transparency.

## Current Runtime

The Rust TUI currently uses Codex app-server as its first working runtime:

- ChatGPT subscription authentication through the installed Codex CLI;
- `gpt-5.4-mini` as the low-cost default model;
- persisted Codex threads with authoritative resume and fork operations;
- streamed responses and runtime-reported token usage;
- Codex workspace sandboxing and approval requests;
- explicit and changed-only context-file injection.

The Python MCP exposes one local retrieval tool. Codex currently owns the low-level
agent turn and native coding capabilities. Elpis supplies a thin TUI around it; the
distinctive Elpis context, memory, behavior, supervision, and runtime layers are not yet
implemented.

## Run

Prerequisites: Rust, an installed Codex CLI, and an authenticated ChatGPT account.

```bash
codex login status
cd tui
cargo run
```

If Codex is signed out, run `codex login` and complete the browser flow. The current
runtime authenticates through Codex app-server and must never render or log credentials.
See [docs/AUTHENTICATION_BOUNDARY.md](docs/AUTHENTICATION_BOUNDARY.md).

## Project Eye

Read [VISION.md](VISION.md) for the stable product intent, [FEATURES.json](FEATURES.json)
for the supervised implementation queue, and [GUIDE.md](GUIDE.md) for architecture and
current truth. Agents start from [AGENTS.md](AGENTS.md).

## Name

*Elpis* is the Greek personification of hope or expectation, remembered as what
remained in Pandora's jar. Here it names a direction rather than a finished state:
each change should make the agent's environment clearer, safer, and more coherent.

## Status

Elpis is a prototype, not a shippable product. Codex login and model turns work, but
action rendering, exact permissions, mouse selection, runtime parity, context lifecycle,
memory, behavioral assimilation, and clean installation remain unfinished. Status is
tracked by user-visible acceptance checks in [FEATURES.json](FEATURES.json).
