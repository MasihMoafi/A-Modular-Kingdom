# A-Modular-Kingdom

A-Modular-Kingdom is an advanced, agentic framework built on the Model Context Protocol (MCP). It features a sophisticated, self-correcting memory system and a powerful, extensible tool-use architecture.

## Core Architecture

The project follows a strict client-server architecture to ensure stability and modularity.

-   **Agent (`agent/`)**: The "brain" of the system. It contains the interactive client that the user communicates with and the MCP server that hosts the tools.
-   **Memory (`memory/`)**: A V3 agentic memory system that uses a two-stage process (fact extraction and consolidation) to build a robust and accurate knowledge base from conversations.
-   **RAG (`rag/`)**: A V2 Retrieval-Augmented Generation pipeline. It uses a hybrid FAISS/BM25 retriever and a CrossEncoder re-ranker to provide the agent with deep knowledge from external documents.
-   **Tools (`tools/`)**: An empty directory intended for housing future tools to extend the agent's capabilities.

## How It Works

1.  The user runs `agent/main.py`.
2.  `main.py` (the MCP client) starts `agent/host.py` (the MCP server) as a background process.
3.  The `host.py` server initializes the Memory and RAG systems, making them available as tools.
4.  The user interacts with the agent in `main.py`.
5.  The agent autonomously decides which tools to use (e.g., search its memory, query the RAG system) based on the user's query.
6.  The agent calls the tools on the `host.py` server via MCP.
7.  The agent synthesizes the information from its tools and memory to provide an intelligent response.
8.  The conversation is saved to the memory system, which intelligently processes and integrates the new information.

## Setup & Run

To run the agent, execute the following command from the project root:

```bash
python3 agent/main.py
```
