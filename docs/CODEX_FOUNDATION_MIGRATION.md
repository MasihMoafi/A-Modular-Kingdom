# Codex Foundation Migration

## Decision

Elpis will use **fork and subtract**, not continue rebuilding Codex features inside the
current monolithic TUI. Codex is a source donor; Elpis will contain the required code
and will not read or execute code from the donor clone directory at runtime.

Pinned candidate revision:

- repository: `openai/codex`;
- revision: `2e1607ee2fa8099a233df7437adee5f16a741905`;
- license: Apache-2.0; retain `LICENSE`, `NOTICE`, and applicable source notices;
- local donor: `/home/masih/Desktop/f/p/others/codex`.

The donor working tree has unrelated edits. Import only committed content from the
pinned revision; never copy its working-tree state or pull/reset over those edits.

## Imported Baseline

The committed `codex-rs/` tree from the pinned revision is imported at the repository
root. Its Apache-2.0 `LICENSE` and `NOTICE` are retained inside that directory, and
`codex-rs/ELPIS_UPSTREAM.md` records provenance. After the imported foundation passed
its authenticated milestone, the superseded root `tui/` prototype was archived on
the legacy-prototype tip now held inside `archive/pre-cleanup-20260716` and removed from
canonical `main`.

No feature subtraction or Elpis-specific renaming is part of this baseline import.
Crate and module boundaries remain upstream-shaped so the later action-rendering,
permission, and mouse-selection work can stay isolated and retain upstream tests.

## Product Boundary

Keep one Elpis-owned execution environment:

```text
Elpis TUI and control layer
  -> context, memory, provider-neutral session mirror, behavior, evidence
  -> selected runtime
       -> Codex app-server using Codex/ChatGPT subscription authentication
       -> Elpis embedded provider-neutral runtime
       -> external runtime such as Claude CLI or Gemini CLI
```

Each runtime declares what it owns. Elpis always retains its control-layer state, but
native tools, permissions, and compaction may remain owned by the selected runtime and
be bridged into Elpis rather than reimplemented.

## Preserve First

Import these proven behaviors with their existing tests before subtracting features:

| Behavior | Principal Codex source |
| --- | --- |
| Permission types and profiles | `codex-rs/protocol/src/protocol.rs`, `protocol/src/models.rs`, `utils/approval-presets/` |
| Patch safety and writable-root checks | `codex-rs/core/src/safety.rs`, `core/src/tools/handlers/apply_patch.rs`, `apply-patch/` |
| Shell lifecycle and running processes | `codex-rs/core/src/tools/handlers/shell.rs`, `core/src/tools/runtimes/shell/`, `exec/` |
| Sandbox enforcement | `codex-rs/core/src/tools/sandboxing.rs`, `sandboxing/`, `linux-sandbox/`, `execpolicy/` |
| Command event rendering | `codex-rs/tui/src/chatwidget/command_lifecycle.rs`, `exec_cell/`, `exec_state.rs` |
| File/patch event rendering | `codex-rs/tui/src/chatwidget/tool_lifecycle.rs`, `history_cell/patches.rs`, `diff_render.rs` |
| Approval interface | `codex-rs/tui/src/chatwidget/permissions_menu.rs`, `permission_popups.rs`, `bottom_pane/approval_overlay.rs` |
| Event routing and replay | `codex-rs/tui/src/chatwidget/protocol.rs`, `replay.rs`, app-server protocol item types |
| Session/thread storage | `codex-rs/rollout/`, `thread-store/`, `state/` |
| OpenAI login and refresh | `codex-rs/login/` and the narrow auth dependencies it requires |
| Provider definitions | `codex-rs/model-provider/`, `model-provider-info/`, `core/src/client.rs` |

Codex's present provider definition supports the OpenAI Responses wire format. It is
not by itself a native Gemini/Claude abstraction. Elpis therefore needs an agent-runtime
contract plus optional provider adapters, similar to OpenClaw's separation of model,
provider, and runtime.

## Stable Task Boundaries

Keep these ownership seams intact until the baseline acceptance check passes:

| Task | Primary files | Preserved contract and tests |
| --- | --- | --- |
| Action rendering | `codex-rs/tui/src/chatwidget/command_lifecycle.rs`, `tool_lifecycle.rs`, `exec_state.rs`, `exec_cell/`, `history_cell/patches.rs`, `diff_render.rs` | Own command/file lifecycle projection and rendered cells. Keep colocated unit tests and `snapshots/` fixtures with any change. Treat `chatwidget/protocol.rs` as the shared event-routing seam. |
| Permissions | `codex-rs/protocol/src/protocol.rs`, `protocol/src/models.rs`, `utils/approval-presets/`, `core/src/safety.rs`, `core/src/tools/sandboxing.rs`, `sandboxing/`, `linux-sandbox/`, `execpolicy/`, `tui/src/app_server_session.rs`, `tui/src/chatwidget/permissions_menu.rs`, `permission_popups.rs`, `bottom_pane/approval_overlay.rs` | Own permission types, preset selection, enforcement, and approval UI. Preserve crate tests plus approval snapshots; do not alter rendering lifecycle files while changing policy. |
| Mouse selection and copy | `codex-rs/tui/src/tui.rs`, `tui/event_stream.rs`, `app/input.rs`, `app/event_dispatch.rs`, `app_event.rs`, `chatwidget.rs` raw-output methods, `history_cell/mod.rs` raw lines, `insert_history.rs` | Codex deliberately skips mouse events and does not enable mouse capture, leaving selection to the terminal. Raw scrollback supplies copy-faithful lines. Preserve `history_cell/tests.rs`, raw-mode chatwidget tests/snapshots, and terminal-mode tests. |

`chatwidget.rs` and `chatwidget/protocol.rs` are shared seams, not general cleanup areas.
Rendering owns lifecycle routing; mouse-selection work owns only raw-output state and
copy-friendly transcript projection. Coordinate before changing either seam.

## Exact Permission Baseline

Copy Codex's semantics before adding Elpis-specific policy:

- **Read Only:** may read workspace files; edits or internet require approval.
- **Default:** may read/edit within the workspace and run commands; internet or work
  outside the workspace requires approval.
- **Full Access:** no approval prompts; filesystem and internet restrictions are off.

The current prototype's `/yolo` flag is not this behavior: it changes only how received
approval requests are answered and does not reconfigure the running thread.

## Migration Sequence

1. Preserve the current prototype as a runnable checkpoint without rewriting its dirty
   worktree history.
2. Import the pinned Codex workspace into an isolated migration branch/worktree using
   committed files only, with license and provenance intact.
3. Rename/package it as Elpis while keeping upstream tests green.
4. Remove cloud/product surfaces that are outside Elpis only in small test-backed steps.
5. Replace Codex branding and configuration with Elpis equivalents.
6. Introduce the runtime contract; keep Codex behavior as the first reference runtime.
7. Move the existing Elpis product rules, retrieval, context visibility, and desired UI
   changes onto the new foundation.
8. Add Gemini, then Claude, against shared runtime and provider contract tests.
9. Add the OpenClaw-derived context/session/memory pipeline after the execution baseline
   is stable.

## First Acceptance Check

In Elpis, ask the OpenAI/Codex provider to create a small file and run one command under
each permission preset. The screen must show the action start, useful live state,
completion or failure, and file diff. Prompt/no-prompt behavior must match Codex's three
presets exactly. The test must run from Elpis without accessing the donor clone path.

## Not Yet Approved As Requirements

- which Codex cloud, apps, realtime, and experimental surfaces remain;
- whether provider switching is per session or allowed during a live session;
- whether Claude/Gemini use direct APIs, their subscription CLI authentication, or both;
- the final visual redesign beyond faithful, readable action rendering.
