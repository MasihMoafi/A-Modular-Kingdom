# Elpis Tasks

This is the project task source of truth. A feature is complete only after its stated
behavior is implemented and verified; vision documents describe intended behavior but must
not claim unfinished behavior is available.

## Foundational — required for the first release

### F1. Clean canonical repository — complete

- Restored the required RAG proxy helper.
- Removed the obsolete `VectorIndex`, unused notebook splitting functions,
  `/test-approval`, `/exit`, and dead tooltip data.
- Proof: clean canonical `main`, focused Python checks, and remote Rust checks.

### F2. Internal RAG — complete

- `/rag <query>` and `/rag <path> -- <query>` exist.
- The MCP advertises one read-only RAG tool, allowing concurrent scheduling beside exact
  read-only exploration.
- Workspace and explicit-path queries return sourced chunks.
- Proof: visible workspace, explicit-path, and autonomous examples.

### F3. Context and session continuity — implemented; acceptance pending

- `GOAL.md` mirrors the active workspace goal and `ES.md` records the latest compact result,
  changed paths, command outcomes, and exact provider-transcript pointer.
- Fresh threads admit the portable goal/checkpoint; exact resume retains the native thread.
- Elpis syncs portable continuity before compaction and stops when that safety write fails.
- `/status` reports admitted rules, goal, checkpoint, and memory summary with size, lifetime,
  and reason.
- Remote verification: the focused continuity tests and Linux build passed in run
  `29534784054`.
- Remaining acceptance: resume one real task exactly and one leanly without replaying
  irrelevant work; verify `/status` against the actual next model request.
- Known gap: the current 1,000-character tool-output cleaner is a blunt deterministic filter.
  It lacks a focused regression test, turn-age distinction, compact conclusion, and source
  pointer, so the full dynamic context-cleaner contract is not complete.

### F4. Durable memory — implemented; end-to-end acceptance pending

- Elpis-owned memory artifacts and database state live under `~/.elpis`.
- Implemented and remotely verified: recall-context tracking, promotion after three recalls
  across two contexts, age-aware diverse retrieval, semantic fallback, provenance, and hard
  30,000/10,000-character durable-memory/summary limits.
- Weak one-off memories remain searchable evidence instead of entering `MEMORY.md`.
- Review/delete remains file-based; reset-all clears Elpis memory without deleting Codex
  threads or state.
- Build-cycle branch: deleted/faded lines are archived before baseline reset, archive write
  failures stop the reset, and a focused archive regression is included.
- Remaining acceptance: teach a fact, recall it with provenance in a related new session,
  omit it from an unrelated session, review/delete it, and verify reset behavior locally.

### F5. First-release provider and authentication boundary — partial

- OpenAI subscription authentication remains the default Codex-backed path.
- OpenRouter is separately keyed through `OPENROUTER_API_KEY`.
- Launcher/configuration tests cover OpenAI, OpenRouter, Bedrock, Ollama, and LM Studio IDs.
- `--provider claude`, `gemini`, and `gemini-flash` select OpenRouter family aliases; these
  are compatibility routes, not native Anthropic or Google adapters.
- Remaining first-release acceptance: complete and resume one task through OpenAI and
  OpenRouter, proving Elpis-owned goal, context, memory, permissions, and evidence survive.
- The provider-aware `Choose a mind` `/model` surface is not implemented.

### F6. Release readiness and build cycle — in progress

- The binary reports `0.1.0`.
- Run `29534784054` passed and uploaded the verified Linux x86_64 artifact.
- Tagged builds publish a checksummed release asset; the installer verifies and atomically
  installs it.
- The pre-optimization verified run took about 21 minutes and produced a 102,988,260-byte
  artifact.
- Build-cycle branch removes CI source mutation and status-file commits, enables incremental
  cross-commit cache reuse, keeps focused checks on ordinary changes, moves exhaustive
  inherited regression to nightly/manual/tag runs, and uploads Cargo timing evidence.
- Remaining acceptance: compare the optimized runtime to the baseline, install the verified
  artifact from a clean environment, authenticate, launch, and complete a first task.

### F7. Distinctive Elpis UI/UX — design complete; implementation partial

- Implemented: Elpis naming, `Elpis · continuity runtime`, and the mature inherited Ratatui
  interaction model.
- Design contract: warm amber identity, persistent runtime/model/context/memory line,
  provider-aware model selection, context ledger, continuity event, and evidence-first
  completion hierarchy.
- Not yet implemented: the coherent amber styling and the five defining product surfaces
  above. `docs/UI_IDENTITY.md` is a contract, not proof.
- Proof required: a new user watches a task cross compaction or provider change and can state
  what survived, what expired, which runtime acted, and where evidence lives.
- Known bug, fixed 2026-07-18: `continuity_sources()` built the dev-skills path from a
  directory name (`skills-i-use`) that no longer existed on disk, so `/skills` fell back to
  built-in skills instead of Masih's own, and the Context Ledger showed only global sources.
  Root cause was a stale path string, not a ledger rendering bug.
- Remaining ledger gap: dev-skills-folder contents (`skills/dev/*`) must render as individual,
  independently toggleable rows in the Context Ledger, admitted by default — not collapsed
  into a single "global files" entry. Not yet verified after the path fix.

## Reduction campaign

- Completed process subtraction: remove broad inherited regression from every ordinary push,
  remove CI auto-formatting, and remove the self-mutating build-status commit.
- Bounded code subtraction selected next: issue #32 removes the inert `debug-m-drop` and
  `debug-m-update` placeholders.
- Measured candidates are recorded in `docs/BUILD_AND_REDUCTION_AUDIT.md`.
- Do not delete arbitrary workspace crates: first prove they are reachable from the Elpis
  binary and optional under the product requirements.

## Important — after the first-release foundation

- Interactive clarifying questions: before ambiguous or costly work, Elpis presents a
  structured selectable prompt (question, options, multi-select) instead of silently
  assuming, and records the chosen answer in the session evidence.
- Native Anthropic and Google protocol adapters.
- Additional provider/runtime adapters using proven Pi/OpenClaw patterns.
- Behavioral enforcement across runtimes.
- Dictation with visible consent and editable, unsent text.
- Further Codex subtraction, one measured capability at a time.

## Nice-to-have

- `/auto` routing with a visible choice, reason, and manual override.
- Scheduled memory review or dreaming-style reports.
- Rich themes and animation beyond the first coherent amber identity.

## Current Action

1. Make the build-cycle and archive branch green and compare its ordinary-change duration
   with the 21-minute baseline.
2. Inspect the uploaded Cargo timing report and select the highest-cost optional dependency
   surface for one bounded deletion.
3. Install the verified binary and run context, memory, OpenAI, and OpenRouter acceptance.
4. Implement the selected [`design-prototype.png`](design-prototype.png) direction: persistent
   identity line plus the cyan Context Ledger. Its `GOAL.md` and `ES.md` controls must govern
   next-turn admission before claiming unique UI/UX is complete.
