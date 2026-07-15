# Agent Dispatch

## Rule

Use one coordinator and one worktree per implementation task. The coordinator owns
`VISION.md`, `REQUIREMENTS.md`, `FEATURES.json`, architecture decisions, task ordering,
integration, and the final acceptance decision. Worker agents implement bounded tasks;
they do not redefine the product.

Do not run two agents against the same files or an unresolved shared interface. More
agents increase speed only when tasks are genuinely independent.

## Difficulty Routing

| Difficulty | Characteristics | Preferred worker |
| --- | --- | --- |
| Easy | One localized behavior, known solution, low-risk change, narrow test | Jules |
| Medium | Several files, bounded design choice, adaptation of an existing pattern | Flash 3.5 or Jules with a precise task |
| Hard | Architecture, runtime ownership, security, permissions, context/memory semantics, migration, cross-cutting interfaces | Main high-reasoning model |

Escalate a task when investigation reveals a broader interface or product decision.
Do not let a worker quietly expand scope.

## Worktree Workflow

1. Start only from the shared committed control baseline.
2. Select one task from `FEATURES.json` whose dependencies are `verified`.
3. Create one branch and worktree named for that task.
4. Give the worker the task fields, exact file scope, non-goals, and acceptance test.
5. Require the worker to return changed files, checks run, evidence, risks, and commit.
6. Review and integrate one branch at a time. Run the acceptance check after integration.
7. The coordinator alone updates feature status to `verified`.

Example after the control baseline is committed:

```bash
git worktree add ../Elpis-wt-terminal-selection -b agent/terminal-selection main
```

Remove a worktree only after its branch is integrated or intentionally abandoned.

## Worker Prompt Contract

Every delegated prompt must contain:

```text
Task ID:
Desired user-visible behavior:
Why it is needed:
Allowed files:
Forbidden scope:
Dependencies already verified:
Acceptance test:
Required checks:
Return: summary, changed files, verification, risks, commit hash.
```

If any field is missing or contradictory, challenge the requirement before coding.

## Current Parallelism Gate

The Codex-derived foundation establishes shared runtime, event, permission, and TUI
interfaces. Until that baseline exists, do not delegate changes that would target those
same interfaces. Safe parallel work is limited to isolated research, tests that do not
assume an interface, and small corrections in files explicitly excluded from the
foundation migration.
