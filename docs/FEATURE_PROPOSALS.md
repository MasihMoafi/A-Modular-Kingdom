# Elpis Feature Proposals & Prioritization

This document tracks ideas and proposals for future Elpis versions, classifying them according to the repository's priority framework:

1. **Foundational (Priority 1):** Necessary for basic execution and safety; release blocking.
2. **Important (Priority 2):** Crucial value-adds for productivity and routing; next-release scope.
3. **Nice-to-have (Priority 3):** Delight features, easter eggs, and visual polish.

---

## 1. RAG & Deep Research

### [Important] Workspace RAG Enhancements
* **Directory Prompting**: When running `/rag <query>` without an explicit directory, the UI should ask the user to specify a directory (with a recommended default), or fall back to searching the terminal's Current Working Directory (CWD) and its subdirectories.
* **Search Guardrail**: Implement a configurable folder depth and token hard limit for CWD searches to prevent scanning unintended large directories (e.g. `node_modules` or root).
* **Archive Separation**: Ensure `/rag` defaults do not mix active workspace source code files with faded memories inside the global `archive.md` file.

### [Important] `/deep-research` Option
* An autonomous researcher mode that uses structured RAG, web search, and recursive crawling to build comprehensive reference context before proposing code edits.

---

## 2. Model & Team Orchestration

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

### [Nice-to-have] Messaging Adapters (Telegram / Discord)
* Porting OpenClaw/Pi messaging connection patterns so that Elpis can be run as a daemon connected to channels like Telegram.

### [Nice-to-have] `/elpis` Poetry Easter Egg
* A cute, amber-themed visual command `/elpis` that displays stylized lyrics or poems (such as *Hindsight* by Anathema), replacing the word "love" with "Elpis" in a beautiful text frame.
