# Experiment Log: Napoleon Retrieval Primacy

**Date:** 2026-07-08 / 2026-07-09
**Target Document:** `document.pdf` (Napoleon Biography)

## Objective
Test pure, raw retrieval capability (finding the needle-in-a-haystack) without LLM reranking interference. The goal is to prove whether the embedding model + bm25 can successfully pull the necessary context into the top 5 chunks.

## Constraints
- **Retrieval Limit:** `top_k=5` strictly.
- **Reranking:** Disabled. No LLM reranker models active.
- **Embedding Model:** Local `qwen3-embedding:8b` via Ollama.
- **Chunking:** Reduced to Size 300, Overlap 50 (to prevent NotebookLM overload).

## Architecture
- **Unified Hybrid Search:** Vector Search + BM25 Fusion using configurable Reciprocal Rank Fusion (RRF) and tunable weights.

## Status & Evaluation
- **Execution:** Completed. The final run utilized a 1000-character chunk size with 150 overlap.
- **Evaluation:** Evaluated by internal AI agent auditing the `results_unified.md` file against the expected answers. 
- **Results:**
  - **Unified Hybrid Architecture Success Rate:** 93.33% (26 fully present, 4 partial, 0 missing)

## Critical Lesson Learned
Finding the "sweet spot" for chunk sizing is the an important factor in RAG accuracy. 
- At **300 characters**, the chunks were too small, literally amputating semantic context; perfect result for keyword queries.
- At **1000 characters** (with 150 overlap), the chunks perfectly captured full philosophical paragraphs and semantic meaning, rocketing the success rate to 93% without overwhelming the context window.

**Conclusion:** 
A 1000-character chunk size is the golden ratio for this architecture. Furthermore, the Unified Hybrid Search (Vector + BM25) eliminated all 0% misses. We will proceed with this single unified architecture and 1000-character chunks for future experiments as our base-line.
