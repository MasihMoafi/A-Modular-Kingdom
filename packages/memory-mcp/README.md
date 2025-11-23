# rag-mem

RAG and Memory tools exposed via Model Context Protocol (MCP).

## Features

- **RAG (Retrieval-Augmented Generation)**: Semantic search over your documents
  - Hybrid retrieval (vector + BM25)
  - CrossEncoder reranking (optional)
  - Support for PDF, Markdown, Python, JSON, Jupyter notebooks
- **Memory System**: Persistent fact/memory storage
  - BM25-based fast search
  - ChromaDB vector fallback
  - Simple CRUD operations
- **Multiple Embedding Providers**:
  - Ollama (default, requires Ollama running)
  - SentenceTransformers (`pip install rag-mem[local]`)
  - OpenAI (`pip install rag-mem[openai]`)
  - Anthropic/Voyage (`pip install rag-mem[anthropic]`)
  - Cohere (`pip install rag-mem[cohere]`)
- **LLM-Agnostic**: Works with any LLM client that supports MCP

## Installation

```bash
# Fast install with uv (recommended)
uv pip install rag-mem

# Or with pip
pip install rag-mem
```

**Default**: Uses [Ollama](https://ollama.ai) for embeddings (free, local, private).

```bash
# One-time Ollama setup:
ollama pull nomic-embed-text
```

**No Ollama?** Use offline embeddings instead:
```bash
pip install rag-mem[local]
export MEMORY_MCP_EMBED_PROVIDER=sentence-transformers
```

## Quick Start

### 1. Initialize Configuration

```bash
memory-mcp init
```

This creates `~/.memory-mcp/config.toml` with default settings.

### 2. Start the Server

```bash
# Basic server
memory-mcp serve

# With document paths for RAG
memory-mcp serve --docs ./documents ./notes

# With specific embedding provider
memory-mcp serve --embed-provider openai --embed-model text-embedding-3-small
```

### 3. Connect from Claude Desktop

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory-mcp",
      "args": ["serve", "--docs", "/path/to/your/documents"]
    }
  }
}
```

## Configuration

Configuration is loaded from (in order of precedence):
1. Environment variables (prefixed with `MEMORY_MCP_`)
2. Config file (`~/.memory-mcp/config.toml`)
3. Default values

### Environment Variables

```bash
export MEMORY_MCP_EMBED_PROVIDER=openai
export MEMORY_MCP_OPENAI_API_KEY=sk-...
export MEMORY_MCP_QDRANT_MODE=cloud
export MEMORY_MCP_QDRANT_URL=https://your-cluster.qdrant.io
export MEMORY_MCP_QDRANT_API_KEY=...
```

### Config File Example

```toml
# ~/.memory-mcp/config.toml

# Default: Ollama (free, local)
embed_provider = "ollama"
embed_model = "nomic-embed-text"

# Alternative: Offline (pip install rag-mem[local])
# embed_provider = "sentence-transformers"
# embed_model = "all-MiniLM-L6-v2"

# RAG settings
qdrant_mode = "local"
rag_chunk_size = 700
rag_top_k = 5
```

## Docker

```bash
# Build
docker build -t memory-mcp .

# Run with OpenAI embeddings
docker run -it --rm \
  -v ./documents:/docs:ro \
  -v ./data:/data \
  -e MEMORY_MCP_EMBED_PROVIDER=openai \
  -e MEMORY_MCP_OPENAI_API_KEY=sk-... \
  memory-mcp serve --docs /docs

# Run with Ollama (requires host network for Ollama access)
docker run -it --rm \
  --network host \
  -v ./documents:/docs:ro \
  -v ./data:/data \
  memory-mcp serve --docs /docs
```

## Available Tools

When connected via MCP, these tools are available:

### RAG Tools

- **`query_knowledge_base`**: Search indexed documents
  - `query`: Search query
  - `doc_path`: Optional specific document path
  - `top_k`: Number of results

### Memory Tools

- **`save_memory`**: Store text content
- **`save_fact`**: Store structured fact with metadata
- **`search_memories`**: Search stored memories
- **`delete_memory`**: Delete by ID
- **`list_all_memories`**: List all stored memories

## CLI Commands

```bash
# Initialize config
memory-mcp init

# Show current config
memory-mcp config

# Start MCP server
memory-mcp serve [--docs PATH...] [--embed-provider PROVIDER]

# Index documents (without starting server)
memory-mcp index PATH... [--force]
```

## Python API

```python
from memory_mcp import Settings, create_server
from memory_mcp.rag import RAGPipeline
from memory_mcp.memory import MemoryStore

# Custom settings
settings = Settings(
    embed_provider="openai",
    openai_api_key="sk-...",
)

# Use RAG directly
pipeline = RAGPipeline(
    settings=settings,
    document_paths=["./docs"],
)
pipeline.index()
results = pipeline.search("How does authentication work?")

# Use memory directly
memory = MemoryStore(settings)
memory.add("User prefers dark mode")
memories = memory.search("preferences")
```

## Custom Embedding Providers

Implement the `EmbeddingProvider` interface:

```python
from memory_mcp.embeddings.base import EmbeddingProvider

class MyEmbeddings(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        pass

    def embed_query(self, text: str) -> list[float]:
        # Your implementation
        pass
```

## Architecture

```
memory-mcp/
├── embeddings/     # Pluggable embedding providers
├── rag/            # RAG pipeline (Qdrant + BM25 + reranking)
├── memory/         # Memory store (ChromaDB + BM25)
├── server.py       # FastMCP server
├── config.py       # Pydantic settings
└── cli.py          # CLI entry point
```

## License

MIT
