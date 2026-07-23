# Memory

Elpis uses bounded local memory so recurring project knowledge can remain available
without turning every past conversation into prompt baggage.

## Provenance and recall

Memory entries carry the source that supports them. Recall is a discovery aid: when a
fact can have changed, Elpis verifies it against the current workspace before relying
on it. This distinguishes a useful prior observation from a current truth.

## Promotion and limits

Short-term evidence is not automatically a permanent rule. Repeated, useful patterns
can be promoted, while weak or stale material remains separate. The archive is
fail-closed so unavailable memory never causes the agent to invent retained state.

Read [context and sessions](context-and-sessions.md) for how memory fits into the
live-session reduction model.
