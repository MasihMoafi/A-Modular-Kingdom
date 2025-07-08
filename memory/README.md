# Memory Module

This module contains the V3 Agentic Memory system, inspired by the mem0 paper. Its purpose is to create a robust and accurate long-term memory for the agent by using a two-stage consolidation process.

## `core.py` - The Memory Core

This file contains the `Mem0` class, which is the central component of the memory system.

### Architecture

-   **Backend**: Uses a local ChromaDB vector database to store and retrieve memories.
-   **LLM-Powered**: Leverages a Large Language Model (`qwen3:8b`) for intelligent processing.

### Two-Stage Memory Consolidation

When a new conversation turn is passed to the `add` method, it undergoes a two-stage process before being committed to memory:

1.  **Fact Extraction (`_extract_facts`)**:
    -   The raw conversation text is first sent to an LLM with a "hyper-critical fact extractor" prompt.
    -   The LLM's job is to pull out only the most essential, enduring facts (e.g., "The user's car is red") and ignore conversational filler, greetings, or questions.
    -   This ensures that only high-quality, self-contained information proceeds to the next stage.

2.  **Consolidation (`_decide_memory_operation`)**:
    -   Each extracted fact is then processed individually.
    -   The system searches for semantically similar memories that already exist in the database.
    -   The new fact and the similar existing memories are presented to the LLM with a "memory consolidation" prompt.
    -   The LLM is forced to make a decision by choosing one of three operations:
        -   `ADD`: The fact is new and should be added to the database.
        -   `UPDATE`: The fact is a correction or update to an existing memory. The system updates the old entry with the new information.
        -   `NOOP`: The fact is a duplicate or redundant and should be ignored.
    -   This process prevents memory clutter and allows the agent to self-correct and refine its knowledge over time.

### Methods

-   `add(conversation: str)`: The main entry point for adding new information. Triggers the two-stage consolidation process.
-   `search(query: str)`: Searches the memory for relevant facts.
-   `get_all_memories()`: Retrieves all memories currently in the database.
