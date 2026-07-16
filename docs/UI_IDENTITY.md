# Elpis UI Identity

Elpis should feel unique because the interface exposes what Elpis uniquely owns: runtime
identity, admitted context, durable memory, continuity, permissions, and evidence.

> The model may change; the work continues.

## Implementation Status

Implemented:

- Elpis product naming;
- the visible `Elpis · continuity runtime` title;
- the mature inherited Ratatui interaction/rendering foundation;
- `/status` context-source reporting.

Not yet implemented:

- a coherent adaptive amber palette across the TUI;
- the persistent runtime/model/context/memory identity line;
- the provider-aware `Choose a mind` model surface;
- a signature continuity event for resume, compaction, and provider changes;
- an evidence-first completion hierarchy.

This document is the design and acceptance contract. Examples below are not shipping behavior
until their implementation and user-visible checks pass.

## Visual Language

- **Primary:** warm amber between orange and yellow.
- **Neutral:** terminal foreground and muted gray for provider/runtime detail.
- **Success:** verified outcomes only.
- **Warning:** context pressure, expiring evidence, permission escalation, or provider
  degradation.
- **Error:** failed actions and broken continuity.

Amber identifies Elpis-owned state; vendor details remain neutral.

## Persistent Identity Line

```text
ELPIS  runtime: codex  model: gpt-…  context: 41%  memory: 6 sources  mode: default
```

Runtime and model provider must remain distinct. Authentication must never imply ownership.

## Core Surfaces

### Choose a mind

`/model` should lead with a provider-aware layer showing display name, provider, runtime path,
context window, tool support, credential source, and whether the path is native,
compatibility-based, local, or external.

### Context ledger

`/status` should evolve into a compact row-per-source ledger:

```text
source              size       lifetime       reason
GOAL.md             1.2k       persistent     active objective
ES.md               2.8k       session        lean continuation
memory:provider     640b        turn           relevant project fact
read:src/main.rs    9.4k        expiring       current edit target
```

### Continuity event

Resume, compaction, provider switch, and lean continuation should produce one recognizable
amber event stating what remained, what expired, where exact evidence lives, and which
runtime owns the next turn.

### Evidence-first completion

A generated claim and verified result must look different. Completion should separate claim,
changed paths, commands/status, tests/evidence, and unresolved gaps.

## Delivery Order

1. Persistent identity line and Elpis-owned terminology.
2. Adaptive amber styling foundation.
3. Provider-aware `Choose a mind` model surface.
4. Context ledger refinement.
5. Signature continuity event.
6. Evidence-first completion hierarchy.
7. Optional animation/themes only after acceptance.

## Acceptance

A new user watches one task cross compaction or provider change and can explain:

1. which runtime performed each turn;
2. which goal, context, and memories survived;
3. what expired;
4. what changed and was verified;
5. where exact evidence can be inspected.
