# Context and sessions

Elpis separates the working conversation from durable, user-approved context. This
keeps a session useful after its immediate transcript has been reduced instead of
letting every request inherit an ever-growing history.

## What is retained

The session keeps the active goal and the evidence needed to continue it: changes,
verification, blockers, and the next action. User-visible sources are admitted
deliberately rather than being silently copied from every file in a checkout.

## Reduction is continuity, not deletion

When the live conversation must be reduced, Elpis preserves a compact checkpoint that
records the current state and the evidence behind it. The next turn can resume from
that checkpoint without presenting a stale summary as fresh repository state.

## Operating guidance

Verify mutable repository state before acting. Treat an admitted source as context,
not permission to overwrite a newer user instruction. For a screenshot-led first
run, return to the [visual walkthrough](visual-walkthrough.md).
