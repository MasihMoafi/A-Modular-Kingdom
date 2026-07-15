# Runtime Boundary WIP Evidence

## Status

The focused Codex runtime-boundary slice is **verified**. The broader runtime-selection
feature remains `in_progress`: Gemini and Claude adapters have not been implemented or
accepted through a user-visible turn.

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

## Verification

- Static review confirmed the diff is limited to the TUI CLI, launch seam, runtime
  contract, focused tests, and project evidence.
- `git diff --check` passed before the WIP commit.
- No Cargo command was rerun on Masih's workstation.
- GitHub Actions run
  [29442993318](https://github.com/MasihMoafi/Elpis/actions/runs/29442993318)
  compiled the locked `codex-tui` library-test target at commit `033c00f` and passed
  all four focused runtime-boundary tests.
- Compilation took about 6.5 minutes; the focused tests took about 5 seconds. The
  remote cache was saved for later runs.
- This proves the Codex boundary and rejection of unimplemented adapters; it does not
  prove the full multi-runtime user-visible acceptance check.
