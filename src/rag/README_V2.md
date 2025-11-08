# RAG Module

This module contains the Retrieval-Augmented Generation (RAG) pipeline. It is a V2 implementation designed to provide the agent with deep knowledge from external documents.

## `core_2.py` - The RAG Pipeline

This file contains the `RAGPipeline` class, which is the core of the retrieval engine.

### Architecture

-   **Vector Database**: Uses `Qdrant` for fast and efficient vector search with persistence.
-   **Embeddings**: `embeddinggemma` via Ollama (768-dimensional vectors).
-   **Hybrid Retrieval**: Combines two search methods for superior results:
    1.  **Vector Search (Qdrant)**: Finds documents that are semantically similar to the query.
    2.  **Lexical Search (BM25)**: Finds documents that contain the exact keywords from the query.
-   **Re-ranking**: After the initial retrieval, a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is used to re-rank the results for the highest possible relevance.

### Configuration

- **chunk_size**: 700 (optimal for coherent context)
- **chunk_overlap**: 100 (ensures continuity between chunks)
- **rerank_top_k**: 5 (returns 5 best results after reranking)

## `fetch_2.py` - The Control Panel

This file provides the entry point to the RAG system.

-   **Configuration (`RAG_CONFIG`)**: A single dictionary that centralizes all settings for the pipeline, such as model names, file paths, chunking parameters, and retriever weights. This allows for easy tuning and experimentation.
-   **Singleton Pattern (`get_rag_pipeline`)**: This function ensures that the RAG pipeline, which is computationally expensive to initialize, is loaded only once. Subsequent calls retrieve the existing instance, saving time and resources.
-   **Main Function (`fetchExternalKnowledge`)**: The simple, primary function that the agent's host calls to query the RAG system.
