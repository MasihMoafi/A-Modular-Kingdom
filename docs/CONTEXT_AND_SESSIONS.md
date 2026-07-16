# Context And Session Contract

## User Promise

Elpis should preserve the work without forcing the next model call to reread its entire
history. Exact artifacts stay on disk; active context carries only what is needed to decide
and act.

## Implementation Status

Implemented and covered by focused remote checks:

- portable workspace `GOAL.md` and per-turn `ES.md`;
- fresh-thread admission of the portable goal and checkpoint;
- exact native-thread resume;
- pre-compaction synchronization that stops on write failure;
- `/status` reporting for portable sources, sizes, lifetimes, and reasons;
- Elpis-owned bounded memory roots, retrieval, promotion, and archive storage.

Still requiring end-to-end acceptance:

- one real exact resume and one lean continuation;
- proof that `/status` matches the next model-visible request;
- related versus unrelated memory admission;
- memory review, deletion, reset, and compaction behavior.

The current deterministic cleaner replaces function/tool output text longer than 1,000
characters. It is an interim implementation: it does not distinguish current evidence from
stale exploration, retain a compact conclusion and source pointer, or have a focused
regression test. Earlier design notes about a lossless context database, background
summarizer, decoupled compaction model, or global vector session index are not implemented.

## Context Lifecycle

Every admitted item has a lifetime:

- **durable:** rules, active goal, and explicit constraints;
- **task:** decisions, changed paths, blockers, and verification;
- **turn:** searches, reads, command output, and failed probes.

Turn material expires only after its question is answered. Before eviction, Elpis must retain
operation, target, conclusion, status, changed paths, and an exact evidence pointer. Removing
an item from the visible TUI is insufficient; it must be absent from the next request.

## Session Persistence

Elpis separates:

1. the native runtime thread;
2. exact workspace artifacts;
3. a compact portable checkpoint;
4. reusable cross-session memory.

- **Exact resume:** continue the native thread while its accumulated context remains useful.
- **Lean continuation:** begin a fresh thread from the portable goal/checkpoint and a small
  recent suffix.

Portable state lives under `~/.elpis/context/workspaces/<workspace>/`:

- `GOAL.md` mirrors the active goal;
- `ES.md` records the latest result, changed paths, command outcomes, next action, and
  provider-thread pointer.

Provider transcripts remain the exact evidence source. The portable files must never become
a second transcript.

## Context Cleaner Acceptance

A compliant cleaner must:

1. preserve the newest result until it has been used;
2. identify stale exploration by lifetime/turn rather than length alone;
3. retain a compact conclusion and source pointer;
4. keep raw output on disk;
5. prove evicted material is absent from the next request;
6. stop when safe checkpointing cannot be completed.

The current length-only replacement should be replaced, not expanded with more arbitrary
thresholds.

## Memory And Archive Contract

Active durable memory is bounded under `~/.elpis/memories`. Promotion requires repeated
useful recall across distinct contexts. Weak and detailed evidence remains searchable rather
than always entering the prompt.

Deleted or age-faded lines must be appended to `archive.md` before baseline reset. Archive
write failure is consequential and must block reset. Archive retrieval remains selective;
the entire archive is never injected by default.

## Acceptance Order

1. Exact resume retains the native thread.
2. Lean continuation receives only the portable goal/checkpoint and recent suffix.
3. `/status` names every admitted portable source and reason.
4. The cleaner removes stale raw output while preserving a useful record and evidence pointer.
5. Related memory appears with provenance; unrelated memory stays absent.
6. Review, deletion, archive, and reset work without losing exact evidence.
