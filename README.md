# A Modular Kingdom (AMK) & Elpis

**A Modular Kingdom (AMK)** is a highly optimized, local-first ecosystem that merges an advanced Python-based Retrieval-Augmented Generation (RAG) backend with a Rust-based Terminal UI (Elpis). 

The goal of this project is to build an intelligent, lightweight agentic framework capable of reasoning across massive local codebases and documents at lightning speed, entirely free from cloud-dependency or heavy memory bloat.

---

## Architecture Overview

The system is split into two distinct, communicating layers:

### 1. The RAG Backend (Python Core)
The long-term knowledge retrieval engine, capable of scoring and retrieving deeply buried context across thousands of documents.
* **Unified Hybrid Search:** Uses a highly optimized Reciprocal Rank Fusion (RRF) engine (`score = weight / (k + rank)`) to seamlessly combine dense vector similarities (Semantic) with BM25 (Keyword exact-match).
* **Configurable Tunability:** Easily tweak `ensemble_weights` to bias the retrieval towards exact terminology or abstract semantics based on the dataset.
* **Local First:** Driven by Qdrant (SQLite cache) and Ollama (`qwen3-embedding:8b`).
* **Flexible Reranking:** Supports both lightweight CrossEncoders (e.g., `ms-marco`) for speed, or generative LLMs for deep-reasoning judgements. 

*Experiment Note: Current evaluations demonstrate a 93.3% success rate at retrieving exact needles in 1000-character chunks.*

### 2. Elpis / AMK-TUI (Rust Frontend)
Elpis is the terminal UI and the active "Brain" of the operation. While the Python backend handles long-term RAG, Elpis handles **Short-term Memory** and Agentic actions.
* **Context Preservation:** Heavily inspired by OpenClaw, Elpis manages the context window dynamically using sliding-tail preservation, actively pruning repetitive logs and truncating massive payloads to maintain sub-second response times.
* **Execution Policies:** Built-in Codex-style interactive blocks `[Y/n]` that pause execution and demand user approval before firing risky shell operations.
* **Signal Extraction:** Uses a barebones LLM wrapper to strip noise from raw user prompts, isolating the pure action item before triggering complex multi-tool workflows.

---

## RAG vs Memory

A critical distinction in this project's architecture:
- **RAG (`src/rag`)** is our *Long-Term Storage*. It is slow to index, but lightning-fast to query. It is purely for retrieving documentation.
- **Memory (`amk-tui`)** is our *Working Memory*. It is the active context window the LLM sees. It is actively scrubbed, truncated, and managed by Elpis to ensure the LLM never gets "terminal bloat".

---

## Future Integrations
1. Complete the network bridge between Elpis and the Python AMK Core.
2. Implement robust TUI macro slash commands (`/plan`, `/goal`).
3. Add visual architecture diagrams demonstrating the pipeline flow.
