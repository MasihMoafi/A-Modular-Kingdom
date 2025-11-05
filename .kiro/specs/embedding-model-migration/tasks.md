# Implementation Plan

- [x] 1. Create Ollama embedding interface
  - Create `A-Modular-Kingdom/rag/embeddings.py` with `OllamaEmbeddingFunction` class
  - Implement `embed_query()` method using `ollama.embeddings()` API
  - Implement `embed_documents()` method for batch processing
  - Add error handling for Ollama connection failures
  - _Requirements: 1.1, 1.4_

- [ ]* 1.1 Write unit tests for embedding function
  - Test single query embedding returns correct dimension
  - Test batch document embedding
  - Test error handling when Ollama unavailable
  - _Requirements: 1.1, 1.4_

- [x] 2. Update RAG configuration
  - Modify `RAG_CONFIG_V3` in `fetch_3.py` to use `embeddinggemma`
  - Add `embed_provider` field to config
  - Update config documentation
  - _Requirements: 3.1, 3.2_

- [x] 3. Implement model change detection
  - Create `_save_model_info()` method in `RAGPipelineV3` to save model metadata
  - Create `_load_model_info()` method to read existing model metadata
  - Add model comparison logic in `_load_or_create_database()`
  - Trigger automatic re-indexing when model changes
  - _Requirements: 2.1, 2.2, 3.3_

- [x] 4. Update RAG system to use Ollama embeddings
  - Replace `SentenceTransformerEmbeddings` import in `core_3.py`
  - Update `RAGPipelineV3.__init__()` to use `OllamaEmbeddingFunction`
  - Ensure embedding function interface remains compatible with `VectorIndex`
  - Test RAG search with sample queries
  - _Requirements: 1.1, 1.2, 2.3_

- [x] 5. Update memory system to use Ollama embeddings
  - Import ChromaDB's Ollama embedding function in `memory/core.py`
  - Update `Mem0._get_client_and_collection()` to configure Ollama embeddings
  - Ensure collection creation uses Ollama embedding function
  - Test memory search operations
  - _Requirements: 1.3, 2.3, 3.2_

- [x] 6. Add comprehensive error handling
  - Add Ollama availability check on host startup
  - Provide clear error messages for missing `embeddinggemma` model
  - Add connection timeout handling
  - Log embedding operations for debugging
  - _Requirements: 1.4, 4.3_

- [x] 7. Test MCP host integration
  - Start `host.py` and verify no startup errors
  - Test `query_knowledge_base` tool with sample query
  - Test `search_memories` tool with sample query
  - Test `save_memory` and verify embedding works
  - Verify no timeouts or hangs during operations
  - _Requirements: 4.1, 4.2, 4.3_

- [ ]* 7.1 Test with multiple MCP clients
  - Configure and test connection from Claude Code
  - Configure and test connection from another MCP client
  - Verify all 14 tools are accessible
  - _Requirements: 4.1_
