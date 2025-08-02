# Agent Module

This module contains the core client-server logic for the agent, following the Model Context Protocol (MCP).

## Architecture

The agent operates on a client-server model to separate the interactive frontend from the tool-hosting backend.

### `host.py` - The MCP Server

-   **Purpose**: Acts as a stable, background process that hosts the agent's tools. It initializes and manages the Memory System (`Mem0`) and the RAG pipeline, exposing their functionalities through the MCP.
-   **Execution**: This script is not run directly. It is started as a subprocess by `main.py`.
-   **Tools Exposed**:
    -   `save_fact`: Saves a conversation turn to the memory system.
    -   `search_memories`: Searches the memory system for relevant facts.
    -   `list_all_memories`: Retrieves all facts currently stored in memory.
    -   `query_knowledge_base`: Queries the RAG system for information from external documents.

### `main.py` - The MCP Client

-   **Purpose**: This is the interactive agent that the user communicates with. It acts as the "brain," handling user interaction, decision-making, and calling tools on the `host.py` server.
-   **Execution**: This is the main entry point to start the agent. Run `python3 main.py` from this directory.
-   **Core Logic**:
    1.  Starts `host.py` as a background server process.
    2.  Enters a loop to accept user input.
    3.  **Autonomous Decision-Making**:
        -   First, it searches its internal memory for context relevant to the user's query.
        -   It then uses an LLM to classify the query. If the query is determined to be about "organic chemistry," it autonomously calls the `query_knowledge_base` tool.
    4.  **Response Synthesis**: It combines the context from its internal memory and any information retrieved from the knowledge base to form a comprehensive prompt for the LLM.
    5.  **Interaction**: It presents the LLM's final response to the user.
    6.  **Learning**: It saves the full conversation turn to its memory via the `save_fact` tool.
