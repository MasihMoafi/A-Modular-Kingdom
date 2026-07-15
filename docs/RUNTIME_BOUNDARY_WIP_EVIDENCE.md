# Runtime Boundary WIP Evidence

## Status

This checkpoint is **WIP and not verified**. It must not be used as evidence that
runtime selection or non-Codex adapters satisfy their user-visible acceptance check.

## Challenged Scope

The smallest approved runtime-boundary slice is to preserve the existing Codex path
while making its dispatch and ownership boundary explicit. Gemini and Claude transport,
authentication, adapters, live switching, and capability emulation remain deferred
because they require separate product decisions and acceptance tests.

## Implemented WIP

- `--runtime` defaults to the only implemented value, `codex`.
- TUI startup dispatches that value to the unchanged Codex launch body.
- The contract records that Codex owns its native turn, tools, thread, and compaction,
  while Elpis retains context projection, durable memory, session mirroring, behavioral
  policy, and evidence.
- Focused unit tests cover the default, explicit Codex selection, rejection of
  unimplemented Gemini and Claude values, and the ownership split.

## Review And Deferred Verification

- Static review confirmed the diff is limited to the TUI CLI, launch seam, runtime
  contract, focused tests, and project evidence.
- `git diff --check` passed before the WIP commit.
- Cargo compilation and tests were attempted but stopped before completion at Masih's
  request because local Rust compilation significantly disrupted the workstation.
- No Cargo command was rerun after that instruction. The tests remain unexecuted and
  the feature remains `in_progress`, not `verified`.
