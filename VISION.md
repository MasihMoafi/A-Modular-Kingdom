# Elpis Vision

## Product Thesis

> You put an agent into an Elpis, and it becomes Elpis. Be Elpis, my friend.

Elpis is the environment an agent enters and assimilates into. The selected model or
agent runtime may change, but the user's goals, working style, durable knowledge,
context policy, evidence, and behavioral boundaries continue coherently.

Elpis is both a state and a direction. It is never fully complete; each verified change
should make the environment clearer, more capable, more harmonious, and easier for its
creator to control.

## What Makes Elpis Distinct

Elpis is not distinguished by having another terminal chat interface. Existing projects
already provide excellent model access, tools, permissions, terminal rendering, and
agent loops. Elpis should reuse those implementations.

Elpis is distinguished by:

1. **Assimilation:** a selected runtime adopts the creator's applicable instructions,
   goals, context, memory, and behavioral rules.
2. **Context sovereignty:** the user can see why information entered the working set,
   how much space it uses, how long it lives, and what replaced it when it expired.
3. **Continuity:** work survives model changes, compaction, restarts, and new sessions
   without replaying an ever-growing transcript.
4. **Transparent agency:** actions, permissions, changes, failures, and verification are
   visible. Elpis does not claim success when it has only hidden or documented a gap.
5. **User ownership:** durable state is inspectable, editable, exportable, and not tied
   to one model provider or agent runtime.

## Desired Output: First Public Release

Ship an installable terminal product—not a repository demo—in which a user can:

1. authenticate and deliberately select a supported model and runtime;
2. see which runtime owns the turn and which capabilities Elpis retains;
3. give the agent a task under Read Only, Default, or Full Access permissions;
4. watch readable commands, output, file changes, diffs, failures, and verification;
5. inspect and control the exact working context admitted by Elpis;
6. resume later from the same goal, decisions, changes, and relevant memory without
   replaying irrelevant history;
7. switch to at least one non-Codex runtime while retaining Elpis-owned continuity;
8. install and complete a first real coding task from a clean environment.

The release is successful only when these behaviors pass their acceptance checks from
a clean checkout and the distinctive context/continuity behavior is visibly better than
using the selected runtime alone.

### Not Required For The First Release

- messaging channels, scheduled automation, voice, or desktop applications;
- dream narratives or autonomous skill creation;
- support for every provider or every upstream Codex/OpenClaw feature;
- a new implementation where a proven upstream component can satisfy the contract.

## Layered Runtime Ownership

Elpis owns the surrounding environment: TUI, runtime selection, context assembly,
durable memory, provider-neutral session mirror, behavioral policy, and evidence.

The selected runtime owns the low-level turn it is designed to execute. When Codex is
selected, Codex may own its model loop, native tools, native thread, and native
compaction. Elpis projects its context into that turn, bridges approvals and tools,
mirrors the visible transcript, and preserves Elpis-owned continuity. Other runtimes
may divide ownership differently, but the boundary must always be explicit.

## Proof Standard

A feature is real only when its user-visible acceptance check passes and the evidence
is recorded. Documentation, hidden code, or a plausible architecture is not proof.

The defining evaluation is: can a fresh supported runtime enter Elpis, receive the
right current goal and relevant history, obey the creator's rules, perform visible
work under the chosen permission mode, and resume later without irrelevant context?
