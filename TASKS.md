# Elpis Tasks

This is the current task source of truth. Keep it short. Completed release history is
archived in `docs/TASKS_V0_1_ARCHIVE.md`.

## How priority works

Every product task belongs to one of three importance levels:

- **Foundational:** Elpis would lose its purpose, reliability, or basic usability without it.
- **Important:** materially improves Elpis after the foundation is solid.
- **Nice-to-have:** useful optional work that must not delay foundational polish.

These levels describe importance, not difficulty, and they are not release numbers.
Easy, Medium, and Hard are separate task-difficulty labels proposed for `/auto`, where
Elpis would choose an appropriate model for a task. They do not replace the importance
levels above.

No new feature work starts while the Current Action is open unless Masih explicitly
changes priority.

## Current Action — make the existing UI solid

**Importance:** Foundational

**Status:** active

**First target:** Context Ledger

The current Elpis works well enough for daily use. The priority is now to perfect the
features already present before adding new ones.

Known Context Ledger problems:

- On 2026-07-23, pressing Tab with a draft in the composer submitted the message instead
  of opening the Context Ledger.
- The current ledger layout and interaction are not good enough.
- After the key-routing bug is fixed, review the ledger with Masih and choose whether to
  improve it or remove it. Do not assume that decision.

Acceptance:

1. Pressing Tab never submits the draft or behaves like Enter.
2. With text in the composer, opening and closing the ledger preserves the draft exactly.
3. The ledger can be navigated without accidental message submission.
4. Masih reviews the resulting ledger and either accepts the improved design or chooses
   its removal.
5. Focused Rust tests and the required Rust test suite pass.

Non-goals:

- Do not add `/auto`, agent controls, `/multi-task`, voice, LSP, or other new features.
- Do not redesign unrelated screens during the ledger fix.

## Foundational

### F1. Reliable current baseline — active

- Fix bugs in existing behavior before adding features.
- Polish confusing or weak UI one area at a time, starting with the Context Ledger.
- Preserve working context continuity, pruning, memory, RAG, provider, permission, and
  session behavior while polishing the product.

### F2. Context, continuity, memory, and RAG — shipped

- Portable context, exact resume, lean continuation, dual-layer pruning, durable memory,
  and local RAG are implemented.
- New defects in these systems are foundational regressions.

### F3. Provider and permission boundary — shipped

- OpenAI subscription auth is the default path.
- Anthropic, Gemini, and OpenRouter paths are available through their supported adapters.
- Runtime/provider identity and approval controls are visible.

### F4. Release and installation baseline — shipped

- `v0.1.0` is the release tag (re-tagged 2026-07-23 after history cleanup; hash deliberately not pinned here).
- CI builds and verifies the Linux release artifact.
- The release installer verifies the downloaded artifact before replacing the binary.

## Important

### I1. Easier installation and distribution — pending

Improve installation and distribution after the current baseline is polished. Keep one
clear supported path and verify it in a clean environment.

### I2. Careful Rust subtraction — ongoing maintenance

Continue deleting inherited Codex code only when reachability and behavior checks prove
Elpis does not need it. Small, measured removals are preferred over broad deletion.

### I3. Performance guardrails — monitor

Startup already feels fast in current daily use, so there is no active startup project.
Binary-size reduction is also not an active feature. Measure release builds in CI and
open focused work only if startup time or release size regresses.

## Nice-to-have

These are wanted ideas, but they are optional until the current product is polished:

- `/auto`: classify work as Easy, Medium, or Hard and choose a suitable model.
- Agent controls and `/multi-task`: run and inspect several agents, potentially as a
  visible task graph.
- Structured interactive clarification inside Elpis.
- Voice input.
- LSP-backed code intelligence.
- Further UI improvements after the Context Ledger.
- Remote messaging, scheduling, mobile control, and opt-in telemetry.

Old-data cleanup is not active work. It would mean previewing and then removing stale
caches, duplicate evidence, old checkpoints, and other data that is no longer needed,
without touching current sessions or authoritative memory. Add it only if storage growth
becomes a real problem and Masih approves the exact retention rules.
