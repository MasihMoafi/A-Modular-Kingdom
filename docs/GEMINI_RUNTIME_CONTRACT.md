# Gemini Runtime Contract

## Desired Outcome

The user can explicitly select Gemini, complete a visible turn, close Elpis, and resume
the same Elpis session without routing the turn through Codex.

## Challenged Decisions

| Candidate | Decision | Reason |
| --- | --- | --- |
| Reuse `src/agent/main.py` | Remove from the runtime path | It is a hand-grown API-key loop with duplicated tool decisions, non-streaming Gemini output, absolute paths, and no declared `google-genai` dependency. |
| Call Gemini's model API directly | Defer | It would require Elpis to rebuild a mature agent loop, tools, approvals, and sessions before proving a second runtime. |
| Spawn Gemini CLI in headless JSON mode | Defer | It can stream events but does not provide the bidirectional session, approval, cancellation, and file-proxy contract available through ACP. |
| Use Gemini CLI through ACP | Keep | Installed Gemini CLI `0.50.0` exposes JSON-RPC over stdio, authentication, new/load session, prompt, cancellation, model/mode changes, MCP, and a proxied filesystem. |
| Implement Claude simultaneously | Defer | One non-Codex runtime proves the boundary. A second adapter before that proof adds interface churn without new evidence. |

## First Adapter Contract

### Launch and authentication

- Elpis launches `gemini --acp` by executable name in the selected workspace.
- Elpis checks and records the executable version and reports a clear error if Gemini
  CLI or required ACP capabilities are unavailable.
- Gemini's ACP `authenticate` flow owns Gemini credentials. Elpis stores no Gemini
  token, and Codex/ChatGPT authentication is never consulted.

### Transport and lifecycle

- Communication is newline-delimited JSON-RPC 2.0 over the child process's standard
  input and output.
- The first slice supports `initialize`, `authenticate`, `session/new`,
  `session/prompt`, `session/cancel`, and `session/load`.
- Elpis records the Gemini session ID only as adapter state. The Elpis session and its
  provider-neutral transcript remain the project truth.
- Unexpected process exit, malformed protocol messages, and unsupported capabilities
  become visible errors; they must not fall back to Codex.

### Ownership

| Capability | Owner |
| --- | --- |
| Low-level model turn and Gemini-native conversation | Gemini CLI |
| Gemini-native tool choice and compaction | Gemini CLI |
| Runtime selection and visible owner label | Elpis |
| Context projection, durable memory, and behavioral policy | Elpis |
| Provider-neutral session mirror and evidence | Elpis |
| Filesystem boundary and permission enforcement exposed through ACP | Elpis |

Gemini approval modes are not treated as sufficient enforcement. Its own documentation
says plan mode is not yet fully functional; Elpis must enforce its permission preset at
the ACP filesystem/tool boundary. Exact permission parity remains the separate
`permission-presets` feature.

## First Acceptance Check

1. Launch Elpis with Gemini explicitly selected; the screen names Gemini as turn owner.
2. Start an ACP session and send a harmless prompt containing a unique codeword.
3. Show the streamed answer in the shared Elpis transcript and store the Gemini session
   ID only in adapter state.
4. Exit, resume the Elpis session through `session/load`, and ask for the codeword.
5. Confirm the correct answer and prove that no Codex process, thread, or credential was
   used for either turn.

This check proves runtime selection and continuity only. Command/file rendering and
permission parity retain their own acceptance checks.

## Explicit Non-Goals

- No reuse of the legacy Python Gemini loop.
- No Gemini API key requirement imposed by Elpis.
- No silent fallback to Codex or another provider.
- No live runtime switch in the middle of a turn.
- No Claude adapter, direct Gemini SDK adapter, or full capability parity in this slice.

## Source Evidence

- Installed runtime: `gemini` resolves to `@google/gemini-cli` version `0.50.0`.
- Installed ACP guide:
  `/usr/local/node/lib/node_modules/@google/gemini-cli/bundle/docs/cli/acp-mode.md`.
- Existing legacy paths inspected: `src/agent/main.py` and `tui/src/main.rs`.

