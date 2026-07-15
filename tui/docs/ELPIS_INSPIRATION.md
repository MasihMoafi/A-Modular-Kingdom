# Elpis Inspiration & Feature List

> Historical brainstorm. Some source names and conclusions below were speculative and
> are not implementation truth. Use `VISION.md`, `FEATURES.json`, and
> `docs/UPSTREAM_CAPABILITY_MAP.md` for the current verified direction.

This is a brainstormed list of features extracted from OpenClaw and OpenAI's Codex CLI for Elpis.

## 1. Context & Memory Management (from OpenClaw)
OpenClaw is heavily focused on aggressively pruning its context window to prevent terminal bloat and agent chatter from killing performance:
* **Tail-Context Preservation (`diagnostic-session-context.ts`)**: Instead of passing the entire historical log, it only reads the tail end of the text buffers. If it exceeds a certain length, it simply truncates the top (`truncated: true`), effectively keeping a sliding window of only the most recent context.
* **Payload Pruning (`KEEP_PRUNED_CONNECTIONS`)**: It actively truncates large payload returns (like massive JSONs or logs) and replaces the overflow with `...<truncated>`.
* **Simple Completion Wrappers (`prepareSimpleCompletionModelForAgent`)**: For tasks that don't need the massive agent toolset, it strips away the heavy system prompt and tools, executing a "simple completion" just to extract the pure signal from noisy user input.

## 2. TUI & Agentic Features (from Codex CLI)
Codex is written in Rust (like the Elpis TUI) and uses a robust, agent-first terminal interface. Features to study or implement:
* **Execution Policies (`execpolicy.md`)**: A granular permission system in the TUI that determines whether the agent can run commands automatically, or if it requires interactive user approval (Y/n) for risky commands.
* **Sandboxed Execution (`sandbox.md`)**: A feature that runs terminal commands in an isolated environment (like a temporary workspace or Docker container) so the agent can safely experiment with code before applying it to the main project.
* **Slash Commands (`slash_commands.md`)**: Interactive TUI macros that trigger specific agent behaviors. For example, typing `/plan` forces the agent into a planning state before coding, or `/goal` puts it into a relentless loop to accomplish a massive task.
* **Skills Framework (`skills.md`)**: A modular architecture where the agent can dynamically load "skills" (custom tools or prompts) based on the workspace it is in.

## 3. Next Steps for Elpis
1. **Unify V1 and V2 RAG**: Create the single tunable Master RAG engine.
2. **Implement the TUI Sliding Window**: Build the Rust equivalent of OpenClaw's context tailing, so Elpis naturally truncates the oldest parts of the conversation.
3. **Build the Execution Policy Engine**: Add the Codex-style interactive approval block for risky commands.
