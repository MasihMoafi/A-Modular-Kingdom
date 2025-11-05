# Design Document

## Overview

This design migrates the embedding layer from SentenceTransformer to Ollama's `embeddinggemma` model. The migration involves creating a unified embedding interface, updating both RAG and memory systems, and ensuring automatic re-indexing when the model changes.

## Architecture

### Current Architecture
```
RAG System → SentenceTransformerEmbeddings → VectorIndex
Memory System → ChromaDB (default embeddings)
```

### New Architecture
```
RAG System → OllamaEmbeddings → VectorIndex
Memory System → ChromaDB (OllamaEmbeddings)
```

### Key Changes
1. Replace `SentenceTransformerEmbeddings` with Ollama embedding function
2. Configure ChromaDB to use Ollama embeddings
3. Add model change detection for automatic re-indexing
4. Maintain existing VectorIndex interface

## Components and Interfaces

### 1. Ollama Embedding Function

**Location:** `A-Modular-Kingdom/rag/embeddings.py` (new file)

**Purpose:** Provide a unified embedding interface using Ollama

**Interface:**
```python
class OllamaEmbeddingFunction:
    def __init__(self, model: str = "embeddinggemma"):
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
```

**Key Methods:**
- `embed_query()`: Single text embedding for queries
- `embed_documents()`: Batch embedding for documents
- Error handling for Ollama connection issues

### 2. RAG System Updates

**Location:** `A-Modular-Kingdom/rag/core_3.py`

**Changes:**
- Replace `SentenceTransformerEmbeddings` import with `OllamaEmbeddingFunction`
- Update `RAGPipelineV3.__init__()` to use Ollama embeddings
- Add model tracking in database metadata
- Implement re-indexing trigger on model change

**Modified Config:**
```python
RAG_CONFIG_V3 = {
    "embed_model": "embeddinggemma",
    "embed_provider": "ollama",  # new field
    # ... rest of config
}
```

### 3. Memory System Updates

**Location:** `A-Modular-Kingdom/memory/core.py`

**Changes:**
- Add Ollama embedding function to ChromaDB initialization
- Update `Mem0.__init__()` to configure ChromaDB with Ollama embeddings
- Ensure embedding consistency across memory operations

**Implementation:**
```python
from chromadb.utils import embedding_functions

# Create Ollama embedding function for ChromaDB
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="embeddinggemma",
    url="http://localhost:11434/api/embeddings"
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)
```

### 4. Model Change Detection

**Location:** `A-Modular-Kingdom/rag/core_3.py`

**Purpose:** Detect when embedding model changes and trigger re-indexing

**Logic:**
1. Store current model name in `{persist_dir}/model_info.json`
2. On initialization, compare stored model with config model
3. If different, set `force_reindex=True` automatically
4. Update stored model after successful indexing

## Data Models

### Model Info Metadata
```json
{
  "embed_model": "embeddinggemma",
  "embed_provider": "ollama",
  "last_indexed": "2025-02-10T12:00:00Z",
  "num_documents": 150,
  "vector_dimension": 768
}
```

### Embedding Function Interface
```python
# Compatible with both SentenceTransformer and Ollama
embedding_fn: Callable[[str], List[float]]
```

## Error Handling

### Ollama Connection Errors
- **Error:** Ollama server not running
- **Handling:** Provide clear error message with instructions to start Ollama
- **Fallback:** None (fail fast with helpful message)

### Embedding Dimension Mismatch
- **Error:** Existing vectors have different dimensions than new model
- **Handling:** Automatic re-indexing triggered
- **User Notification:** Log message indicating re-indexing in progress

### Model Not Available
- **Error:** `embeddinggemma` model not pulled in Ollama
- **Handling:** Provide error message with `ollama pull embeddinggemma` command
- **Fallback:** None (user must pull model)

## Testing Strategy

### Unit Tests
- Test `OllamaEmbeddingFunction.embed_query()` returns correct dimension
- Test `OllamaEmbeddingFunction.embed_documents()` handles batches
- Test error handling when Ollama is unavailable

### Integration Tests
- Test RAG search with Ollama embeddings
- Test memory search with Ollama embeddings
- Test model change detection triggers re-indexing
- Test MCP host responds correctly to all tool calls

### Manual Testing
- Connect from Claude Code and verify all tools work
- Connect from Codex and verify all tools work
- Verify no timeouts or hangs during embedding operations
- Test with sample queries to ensure quality results

## Migration Path

### Phase 1: Create Embedding Interface
1. Create `embeddings.py` with `OllamaEmbeddingFunction`
2. Add tests for embedding function
3. Verify Ollama API compatibility

### Phase 2: Update RAG System
1. Modify `core_3.py` to use Ollama embeddings
2. Add model change detection
3. Test re-indexing with sample documents

### Phase 3: Update Memory System
1. Modify `core.py` to use Ollama embeddings
2. Test memory operations with new embeddings
3. Verify ChromaDB compatibility

### Phase 4: Integration Testing
1. Test full MCP host with all tools
2. Connect from multiple MCP clients
3. Verify performance and stability

## Performance Considerations

### Embedding Speed
- Ollama embeddings may be slower than SentenceTransformer
- Consider batch embedding for large document sets
- Monitor embedding latency during search operations

### Memory Usage
- Ollama runs as separate process (already running for LLM)
- No additional memory overhead from SentenceTransformer models
- ChromaDB memory usage remains the same

### Re-indexing Time
- Initial re-indexing required after migration
- Time depends on document count (estimate: ~1-2 docs/second)
- Progress logging to track re-indexing status

## Security Considerations

- Ollama runs locally (no external API calls)
- No sensitive data leaves the machine
- Same security posture as current implementation

## Future Enhancements

- Support for multiple embedding providers (configurable)
- Embedding caching for frequently used queries
- Parallel embedding for faster indexing
- Embedding quality metrics and monitoring
