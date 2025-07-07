# Memory System Documentation

This document outlines the three versions of the memory system, from a simple key-value store to a sophisticated, multi-stage memory architecture.

---

## Version 1: `simple.py` - The Foundational Memory

This version is a command-line tool that provides the most basic memory functions: adding and searching for information. It is the foundation upon which all later versions are built.

### Architecture

*   **Single File:** All logic is contained within `simple.py`.
*   **`SimpleMemory` Class:** This class is responsible for all memory operations.
*   **ChromaDB Backend:** It uses a local ChromaDB database to store memories as vectors, allowing for semantic search.
*   **Command-Line Interface (CLI):** The user interacts with the memory system through a simple CLI with commands like `add`, `search`, `clear`, and `exit`.

### How It Works

1.  The user enters a command, such as `add "The sky is blue"`.
2.  The `SimpleMemory` class takes the text, converts it into a vector embedding (using ChromaDB's default embedding model), and stores it in the database with a unique ID.
3.  When the user enters `search "what color is the sky?"`, the system converts the query into a vector and searches the database for the most semantically similar memories.

### Limitations

*   **No Intelligence:** This version has no "brain." It cannot decide what is important to remember, nor can it update or correct existing memories. It simply stores whatever it is told.
*   **No Context:** It cannot understand the context of a conversation. Every `add` and `search` is a separate, isolated event.
*   **Redundancy:** It will happily store the same fact multiple times, leading to a cluttered and inefficient memory.

---

## Version 2: `v2.py` - The "Smart" Memory

This version introduces a "brain" to the memory system. It uses a Large Language Model (LLM) to analyze new information and decide whether to add it as a new fact, update an existing fact, or ignore it.

### Architecture

*   **Two-Part System:**
    *   **The "Notebook" (`SimpleMemory` class):** This is the same simple storage system from Version 1. It has no intelligence.
    *   **The "Brain" (`SmartMemory` class):** This new class contains the intelligence. It uses an LLM to make decisions.
*   **LLM-Powered Decisions:** The `SmartMemory` class uses the `ollama` library to interact with a local LLM.
*   **Tool-Based Logic:** The LLM is given a set of "tools" (`AddMemory`, `UpdateMemory`, `NoOperation`) and is forced to choose one when presented with a new fact. This provides a structured way to manage the memory.

### How It Works

1.  The user enters `add "I made a mistake, my car is red, not blue."`.
2.  The `SmartMemory` "brain" takes this new fact and first searches the "notebook" for similar memories (e.g., it might find a memory that says "my car is blue").
3.  It then presents both the old memory and the new fact to the LLM, asking it to choose a tool.
4.  The LLM, seeing that the new fact is a correction, chooses the `UpdateMemory` tool.
5.  The "brain" then executes the LLM's decision, commanding the "notebook" to update the old, incorrect memory with the new, correct information.

### Improvements Over V1

*   **Intelligence:** The system can now understand the *relationship* between new and old information.
*   **Correction and Updating:** It can correct mistakes, which is a critical step towards a robust memory.
*   **Reduced Redundancy:** The `NoOperation` tool allows the system to ignore duplicate information.

---

## Version 3: The Current System - The Agentic Memory

This is the most advanced version, a complete, multi-stage memory architecture inspired by the `mem0` paper. It separates the agent, the tools, and the memory logic into different files and introduces a sophisticated, two-stage memory consolidation process.

### Architecture

*   **Three-File System:**
    *   `interactive_agent_V2_worked.py`: The main agent that the user interacts with. It contains the primary chat loop and the master prompt that guides the agent's behavior.
    *   `mcp_host.py`: A background "host" process that exposes the memory tools (`save_fact`, `search_memories`, `list_all_memories`) to the agent. This separation allows for more complex and robust tool management.
    *   `mem0_memory.py`: The core memory logic. This is the most intelligent part of the system.
*   **Two-Stage Memory Consolidation:** This is the key innovation.
    1.  **Fact Extraction:** When a conversation turn is saved, the `mem0_memory.py` module first uses a "hyper-critical fact extractor" prompt to pull out only the most essential, enduring facts from the raw text.
    2.  **Memory Consolidation:** It then takes each of these pure facts and, just like in Version 2, uses an LLM to decide whether to `ADD` it as a new memory, `UPDATE` an existing one, or `NOOP` (ignore it) if it's a duplicate.
*   **Agentic Prompting:** The agent's main prompt is now a direct command, forcing it to treat its memory as the single source of truth and to correct the user if they say something that contradicts a known fact.
*   **Memory Inspection:** The user can, at any time, use the `/memory` command to see a human-readable list of every fact the agent has stored in its mind, providing full transparency.

### How It Works

1.  **User:** "The DNS button on my modem is broken."
2.  **Agent:** (Responds normally). The conversation turn is sent to the memory host.
3.  **Memory Host:** The `_extract_facts` function pulls out the fact: "The user's modem has a DNS button that is broken." This fact is **ADDED** to the database.
4.  **User:** "I made a mistake. There is no DNS button; there is a DSL button instead."
5.  **Agent:** (Responds normally). This new conversation turn is sent to the memory host.
6.  **Memory Host:**
    *   The `_extract_facts` function pulls out the new, critical fact: "There is no DNS button on the modem; there is a DSL button."
    *   The `add` function then takes this new fact and searches for similar memories, finding the old, incorrect fact about the DNS button.
    *   The `_decide_memory_operation` function sees that the new fact is a correction and chooses the **UPDATE** operation, replacing the old memory with the new, correct one.
7.  **User (in a new session):** "The DNS button on my modem is blinking."
8.  **Agent:** The agent searches its memory and finds the corrected fact ("There is no DNS button..."). Its master prompt commands it to trust this memory, and it responds: "Based on your previous clarification, it seems you may be referring to the **DSL button**..."

### Improvements Over V2

*   **Robustness:** The two-stage process makes the memory far more accurate, as it only saves pure, essential facts.
*   **Clarity:** The separation of the agent, host, and memory logic makes the system more organized and easier to maintain.
*   **Transparency:** The `/memory` command gives the user a direct window into the agent's mind.
*   **Truthfulness:** The new, forceful agent prompt ensures the agent is not just "helpful" but is a reliable source of truth based on its memory.
