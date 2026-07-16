# Agent Prompt: Elpis Codebase Reduction Audit

Work in `/home/masih/Desktop/f/p/Elpis`. Read `AGENTS.md`, `GUIDE.md`,
`SESSION_HANDOFF.md`, and `TASKS.md` first. Verify the checkout before making any
claim.

## Goal

Produce an evidence-backed plan to make Elpis smaller, clearer, and cheaper to start or
run without removing retained Codex-quality behavior. This is an audit first, not a
blanket refactor.

## Hard Boundaries

- Do not edit code, create branches/worktrees, commit, push, or open a PR during the
  audit.
- Do not run Cargo or compile Rust on the local workstation.
- Do not infer that code is unused from its name or from a hidden slash command. Prove
  call sites, ownership, configuration reachability, tests, and runtime purpose.
- Preserve ChatGPT/Codex login, streaming, shell/file tools, approvals, sandboxing,
  sessions, compaction, mouse selection, `/agent`, `/skills`, and `@` attachment.
- Preserve the one-tool `elpis-rag` boundary. Do not restore deleted Python tools.
- Do not mix appearance changes, feature deletion, and architecture work.

## Audit

1. Map the active launch path from the `elpis` command to the first usable Ratatui frame.
2. Identify startup work that blocks that frame and distinguish measured cost from
   speculation. Use existing evidence or the installed binary only.
3. Find duplicate, unreachable, obsolete, or Elpis-unwanted code. For each candidate,
   name the exact path and symbol, references, tests, dependencies, user-visible effect,
   removal risk, and smallest verification.
4. Separate inherited Codex machinery that Elpis still needs from dedicated product
   surfaces Masih approved for deletion.
5. Identify large dependencies or modules only when repository evidence shows they are
   part of the active build or runtime.

## Required Output

Write one concise report with three ranked sections:

- **Remove now:** proven unused or already superseded; low risk.
- **Investigate:** promising but not yet proven; state the missing evidence.
- **Keep:** heavy-looking code that supports retained behavior; explain why.

For every removal candidate include difficulty (`easy`, `medium`, or `hard`), expected
benefit, exact acceptance test, and one bounded commit-sized task. Recommend only the
single best first removal. Stop after the report and wait for Masih to approve what may
be deleted.

After approval, implement only one selected candidate per commit and use the remote Rust
workflow for verification. Never turn this audit into an open-ended cleanup campaign.
