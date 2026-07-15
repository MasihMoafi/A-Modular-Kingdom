# Codex/ChatGPT Authentication Boundary

## Decision

Elpis retains Codex/ChatGPT login and may deliberately select the Codex agent runtime.
Authentication and runtime selection are separate decisions.

The supported boundary has three parts:

1. For a status-only authentication check, Elpis uses only the Codex app-server v2 account RPCs:
   `account/read`, `account/login/start`, `account/login/cancel`, and the
   `account/login/completed` and `account/updated` notifications. It must not send
   `thread/*`, `turn/*`, file, command, approval, or tool requests through this
   authentication connection.
2. An Elpis-owned direct OpenAI runtime may port the required Rust authentication component from the
   local `codex-login` crate. That component owns browser/device login, Codex credential
   storage compatibility, token refresh, and transient bearer-token access for the
   Elpis-owned model adapter. The token must not be rendered, logged, returned to the
   TUI, or persisted in a second Elpis credential store.
3. When the user selects the Codex runtime, Codex app-server may own that low-level
   model loop, native tools, native thread, and native compaction. Elpis retains runtime
   selection, context projection, durable memory, session mirroring, behavioral policy,
   approval bridging, and presentation. The UI must show that Codex owns the turn.

The stable app-server `account/read` response intentionally reports account state
without returning a token. The legacy `getAuthStatus` request can return a token with
`includeToken`, but Codex marks that API deprecated in favor of `account/read`; Elpis
therefore must not build its native model adapter on that RPC.

A direct path dependency on the separate Codex clone is never a finished boundary.
Elpis may vendor/adapt Codex source or manage a compatible app-server binary, but it
must not load source from the donor directory. Selecting Gemini, Claude, or another
runtime must not implicitly delegate its turn to Codex.

## Upstream Evidence

- Account protocol: `others/codex/codex-rs/app-server-protocol/src/protocol/v2/account.rs`
- Account processor: `others/codex/codex-rs/app-server/src/request_processors/account_processor.rs`
- Reusable auth component: `others/codex/codex-rs/login/`
- Deprecation marker: `others/codex/codex-rs/app-server-protocol/src/protocol/common.rs`

All paths above are relative to `/home/masih/Desktop/f/p/`.

## Status Smoke

Run from the Elpis repository:

```bash
scripts/codex-auth-status-smoke.sh
```

The smoke initializes app-server and sends exactly one `account/read` request. It
prints only the account type and whether OpenAI authentication is required; it does
not print email, tokens, account IDs, or raw response payloads. It does not start a
thread or model turn.
