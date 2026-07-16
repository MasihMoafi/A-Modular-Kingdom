# Elpis UI Identity

Elpis should not look unique by decorating Codex. It should feel unique because the interface continuously exposes what Elpis uniquely owns: runtime identity, admitted context, durable memory, continuity, permissions, and evidence.

## First-release design rule

> The model may change; the work continues.

Every visual choice should make that promise more legible. Preserve the mature Ratatui interaction model and rendering quality; change hierarchy, language, and emphasis rather than rebuilding the interface.

## Visual language

- **Primary:** warm amber between orange and yellow.
- **Neutral:** terminal foreground and muted gray for inherited runtime detail.
- **Success:** reserved for verified outcomes, never merely completed generation.
- **Warning:** context pressure, expiring evidence, permission escalation, or provider degradation.
- **Error:** failed actions and broken continuity only.

Amber identifies Elpis-owned state. Provider/runtime-specific details remain neutral so the product never visually collapses into one vendor.

## Persistent identity line

The first usable frame should make these boundaries visible without opening a menu:

```text
ELPIS  runtime: codex  model: gpt-…  context: 41%  memory: 6 sources  mode: default
```

The line must distinguish runtime from model provider. Authentication must never silently imply runtime ownership.

## Core surfaces

### Choose a mind

`/model` should begin with a small provider-aware layer rather than an undifferentiated model dump. Each option should show:

- display name and provider;
- runtime path: native, OpenRouter, local, or external;
- context window and known tool support;
- credential source;
- a clear warning when the path is compatibility-based rather than native.

Changing the model should explicitly state that Elpis context, memory, permissions, and session continuity remain in force.

### Context ledger

`/status` should evolve into a compact ledger with one row per admitted source:

```text
source              size       lifetime       reason
GOAL.md             1.2k       persistent     active objective
ES.md               2.8k       session        lean continuation
memory:provider     640b        turn           relevant project fact
read:src/main.rs    9.4k        expiring       current edit target
```

The important interaction is not a graph. It is the ability to answer: why is this present, when will it leave, and what evidence survives after it leaves?

### Continuity event

Resume, compaction, provider switch, and lean continuation should produce one recognizable amber event card containing:

- what remained;
- what expired;
- where exact evidence lives;
- which runtime now owns the next turn.

This is Elpis's signature moment and should be demoable in under ninety seconds.

### Evidence-first completion

A generated answer and a verified result must look different. The completion cell should separate:

- claim;
- changed paths;
- commands and status;
- tests or acceptance evidence;
- unresolved gaps.

## Language

Use Elpis language for Elpis-owned surfaces: session, goal, context, memory, permission, evidence, continuation, runtime. Use Codex, Claude, Gemini, OpenRouter, and other names only for the component that actually owns that behavior.

Avoid generic AI ornament. No glowing brain, mystical loading narrative, or decorative provider logos as the primary identity. The distinctiveness comes from visible continuity and control.

## Delivery order

1. Persistent identity line and Elpis-owned terminology.
2. Provider-aware `Choose a mind` model surface.
3. Context ledger refinement.
4. Signature continuity event.
5. Evidence-first completion hierarchy.
6. Optional animation and themes only after these interactions pass acceptance.

## Acceptance

A new user should be able to watch one task cross a compaction or provider change and correctly explain:

1. which runtime performed each turn;
2. which goal, context, and memories survived;
3. what was discarded;
4. what was changed and verified;
5. where the exact evidence can be inspected.
