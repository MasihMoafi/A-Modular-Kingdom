# Project Agent Rules

## Start Here
- Read this file, then read: `progress.md`.
- Verify cwd: `/home/masih/Desktop/p/A-Modular-Kingdom`.
- Baseline check: Run `./init.sh` to configure the local environment and verify sanity.

## Work Rules
- Work on one task/feature at a time.
- Keep changes scoped to python modules, tests, or documentation.
- Preserve key RAG and memory tool APIs: `query_knowledge_base`, `save_memory`, `search_memories`.
- Use `uv` for package management and `pytest` for testing.
- Always run background processes or indexers wrapped with `scripts/thermal_runner.py` to prevent core CPU overheating.

## Definition of Done
- Acceptance criteria met for the targeted feature.
- Verification run: Run `pytest` to confirm package integrity.
- Evidence recorded in: `progress.md`.
- Known risks or skipped checks are explicitly listed.
