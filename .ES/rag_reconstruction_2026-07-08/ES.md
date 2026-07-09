# Session State

- **Date:** 2026-07-08
- **DI (Declared Intent):** Reconstruct the A-Modular-Kingdom (AMK) project step-by-step, starting with rigorous testing of RAG V1 and V2, and preparing the MCP host for `amk-tui` integration according to the Elpis vision.
- **DO (Desired Output):** A structured test series executed surgically (ablation style), robust RAG configuration, and fixed connectivity between the AMK core and `amk-tui`.
- **Learned lessons:** [Pending]
- **Useful notes:** The hierarchical multi-agent layer is bypassed for now; the sole focus is on RAG and Memory capabilities acting as a robust system layer (Elpis).

## Approved Experiment Objectives Log

### Experiment 1: Retrieval Primacy (Completed)
- **Objective:** Test pure, raw retrieval capability (needle-in-a-haystack) without LLM reranking interference.
- **Constraints:** `top_k=5` strictly. No reranker models. Local `qwen3-embedding:8b` only. 
- **Architecture:** Unified Hybrid Search. Merged former V1 (Weighted Ensemble) and V2 (Unweighted RRF) into a single, highly configurable engine natively running RRF with tunable weights `(score = weight / (k + rank))`.
- **Outcome:** Successfully retrieved 5 chunks for all 30 queries. Final evaluation achieved a 93.3% success rate at 1000-character chunk sizing, proving the unified architecture is pristine.

### Clarification for Future Agents (RAG vs Memory)
**CRITICAL:** RAG and Memory are distinctly different systems in this project.
1. **RAG (Long-term Knowledge Retrieval):** Handled by `src/rag/core.py`. This is the Hybrid Search (Qdrant Vector + BM25) for pulling permanent documentation and deep knowledge.
2. **Memory (Dynamic Context Management):** Handled by the Elpis TUI backend. This refers to the active, sliding-window truncation of the LLM context (e.g., pruning repetitive logs or massive payloads) inspired by OpenClaw's `KEEP_PRUNED_CONNECTIONS` to preserve speed and prevent terminal bloat.

### Future Integration (AMK-TUI / Elpis)
The next session will focus on integrating Elpis (the Rust TUI) with the unified Python RAG. Features explicitly targeted from Codex/OpenClaw include:
- Execution Policies (`[Y/n]` blocks).
- Tail-Context Preservation (sliding window memory).
- Signal Extraction (filtering noise before agent processing).
