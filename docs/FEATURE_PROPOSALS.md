# Elpis Feature Proposals & Prioritization

This document tracks ideas and proposals for future Elpis versions, classifying them according to the repository's priority framework:

1. **Foundational (Priority 1):** Necessary for basic execution and safety; release blocking.
2. **Important (Priority 2):** Crucial value-adds for productivity and routing; next-release scope.
3. **Nice-to-have (Priority 3):** Delight features, easter eggs, and visual polish.

---

## 1. RAG & Deep Research

### [Important] Workspace RAG Enhancements
* **Interactive Path Prompt**: When `/rag` is executed, the UI prompts the user to enter a search path.
  * If a path is selected, RAG is scoped to that path.
  * If the user simply presses Enter without entering a path, RAG defaults to searching the terminal's Current Working Directory (CWD) and its subdirectories.
  * **Search Guardrail**: Implement a configurable folder depth and token hard limit for CWD searches to prevent scanning massive directories (e.g. `node_modules` or root).
* **Natural Language RAG Trigger**: The user should not have to manually run `/rag` commands. They can use natural language (e.g., "use RAG to search my project for X") and the agent must autonomously invoke RAG tools efficiently.
* **Archive Separation**: Ensure RAG defaults do not mix active workspace source code files with faded memories inside the global `archive.md` file.

### [Important] `/deep-research` Option
* An autonomous researcher mode that uses structured RAG, web search, and recursive crawling to build comprehensive reference context before proposing code edits.

---

## 2. Model & Team Orchestration

### [Important] Provider-Grouped `/model` Picker
* In the `/model` selection popup, models must be grouped and listed cleanly by provider:
  * **OpenAI Models** (e.g. latest GPT-4o, GPT-o1, etc.)
  * **Claude Models** (e.g. Claude 3.5 Sonnet, Claude 3 Opus)
  * **OpenRouter Models** (a list of the latest models like Kimi 3, GLM 2.5, Gemini, DeepSeek, with proper names and routing)

### [Important] Intelligent `/auto` Cost Routing
* An automatic classifier that routes incoming queries to different models based on complexity:
  * **Easy (e.g. syntax fixes, summaries)**: Low-cost fast model (e.g., Gemini Flash, Claude Haiku, Ollama).
  * **Medium (e.g. feature addition, refactoring)**: Mid-tier model.
  * **Hard (e.g. architectural design, multi-file integration)**: Frontier intelligence models (e.g., Claude Opus, Gemini Pro).

### [Nice-to-have] Elpis Family Tree (Multi-Agent Swarm)
* A hierarchical multi-agent execution framework to handle large-scale, long-running tasks autonomously:
  * **Father Elpis**: The coordinator runtime using the highest intelligence and context length model. It monitors goals, verifies results, and delegates sub-tasks.
  * **Baby Elpises**: Specialized worker runtimes spawned in parallel worktrees (or branches) to implement scoped changes under Father Elpis's steering.
  * **Cost Harness**: Code-level tokens/cost limits to prevent runaway loops in autonomous agents.

---

## 3. Interfaces & Channels

### [Important] Dynamic Context Files Panel
* A dynamic sidebar or panel in the TUI that displays all files currently admitted as context.
* The user can easily select files and toggle them on/off to manually adjust the active context.
* Some context files are automatically added by default based on the active user task.

### [Important] LSP-Backed Code Intelligence For The Active Runtime
* Give whichever coding agent is active inside Elpis (Codex, or a future Claude/other bridge)
  real language-server queries — go-to-definition, precise references, live diagnostics —
  instead of relying on grep/text search for code navigation.
* Note: this is not "replicate what Claude Code does" — no confirmed LSP client exists in any
  current runtime being bridged into Elpis. Scope as its own investigation before committing.

### [Nice-to-have] Messaging Adapters (Telegram / Discord)
* Porting OpenClaw/Pi messaging connection patterns so that Elpis can be run as a daemon connected to channels like Telegram.

### [Nice-to-have] `/elpis` Poetry Easter Egg
* A cute, amber-themed visual command `/elpis` that displays stylized lyrics or poems (such as *Hindsight* by Anathema), replacing the word "love" with "Elpis" in a beautiful text frame.
