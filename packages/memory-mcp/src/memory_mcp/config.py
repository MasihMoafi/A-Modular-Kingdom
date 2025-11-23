"""Configuration system for Memory MCP.

Configuration can be provided via:
1. Environment variables (prefixed with MEMORY_MCP_)
2. Config file (~/.memory-mcp/config.toml)
3. Programmatic override

Environment variables take precedence over config file values.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


CONFIG_DIR = Path.home() / ".memory-mcp"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def _load_toml_config() -> dict:
    """Load configuration from TOML file if it exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    return {}


class Settings(BaseSettings):
    """Memory MCP configuration settings.

    All settings can be overridden via environment variables with the
    MEMORY_MCP_ prefix (e.g., MEMORY_MCP_EMBED_PROVIDER=openai).
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding configuration
    embed_provider: Literal[
        "ollama", "sentence-transformers", "openai", "anthropic", "cohere"
    ] = Field(
        default="sentence-transformers",
        description="Embedding provider. Default requires: pip install rag-mem[local]",
    )
    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for embeddings",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    # API keys (for cloud providers)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic/Voyage API key",
    )
    cohere_api_key: str | None = Field(
        default=None,
        description="Cohere API key",
    )

    # Qdrant configuration (for RAG)
    qdrant_mode: Literal["local", "cloud", "memory"] = Field(
        default="local",
        description="Qdrant storage mode: local (disk), cloud (Qdrant Cloud), memory (in-memory)",
    )
    qdrant_path: str = Field(
        default="~/.memory-mcp/qdrant_data",
        description="Path for local Qdrant storage",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant Cloud URL (required if mode=cloud)",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant Cloud API key (required if mode=cloud)",
    )

    # ChromaDB configuration (for memory)
    chroma_path: str = Field(
        default="~/.memory-mcp/chroma_data",
        description="Path for ChromaDB persistent storage",
    )

    # RAG configuration
    rag_chunk_size: int = Field(
        default=700,
        description="Document chunk size for RAG indexing",
    )
    rag_chunk_overlap: int = Field(
        default=100,
        description="Overlap between chunks",
    )
    rag_top_k: int = Field(
        default=5,
        description="Number of results to return from RAG queries",
    )
    rag_rerank: bool = Field(
        default=True,
        description="Enable cross-encoder reranking",
    )
    rag_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )

    # Memory configuration
    memory_collection: str = Field(
        default="memories",
        description="ChromaDB collection name for memories",
    )

    # Server configuration
    server_name: str = Field(
        default="memory-mcp",
        description="MCP server name",
    )

    # Hardware configuration
    device: str | None = Field(
        default=None,
        description="Device for local models (cpu, cuda, mps). Auto-detected if not set.",
    )

    @field_validator("qdrant_path", "chroma_path")
    @classmethod
    def expand_path(cls, v: str) -> str:
        """Expand ~ in paths."""
        return str(Path(v).expanduser())

    @classmethod
    def from_toml(cls, overrides: dict | None = None) -> "Settings":
        """Create settings from TOML config file with optional overrides."""
        config = _load_toml_config()
        if overrides:
            config.update(overrides)
        return cls(**config)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    This function loads settings once and caches them for the lifetime
    of the process. Use Settings() directly if you need fresh settings.
    """
    toml_config = _load_toml_config()
    return Settings(**toml_config)


def init_config_dir():
    """Initialize the configuration directory with default config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        default_config = '''# Memory MCP Configuration
# See https://github.com/MasihMoafi/A-Modular-Kingdom/tree/main/packages/memory-mcp

# Embedding provider - CHOOSE ONE:
#
# Option 1: Local with SentenceTransformers (offline, ~2GB download first time)
#   pip install rag-mem[local]
embed_provider = "sentence-transformers"
embed_model = "all-MiniLM-L6-v2"

# Option 2: Ollama (free, requires Ollama running)
# embed_provider = "ollama"
# embed_model = "nomic-embed-text"
# ollama_base_url = "http://localhost:11434"

# Option 3: OpenAI (paid API)
#   pip install rag-mem[openai]
# embed_provider = "openai"
# embed_model = "text-embedding-3-small"
# openai_api_key = "sk-..."

# Other API keys
# anthropic_api_key = "..."
# cohere_api_key = "..."

# Qdrant settings (for RAG vector storage)
qdrant_mode = "local"  # local, cloud, or memory
# qdrant_path = "~/.memory-mcp/qdrant_data"
# qdrant_url = "https://..."  # For cloud mode
# qdrant_api_key = "..."  # For cloud mode

# ChromaDB settings (for memory storage)
# chroma_path = "~/.memory-mcp/chroma_data"

# RAG settings
rag_chunk_size = 700
rag_chunk_overlap = 100
rag_top_k = 5
rag_rerank = true
# rag_reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Memory settings
# memory_collection = "memories"

# Hardware (auto-detected if not set)
# device = "cuda"  # cpu, cuda, mps
'''
        CONFIG_FILE.write_text(default_config)
        return True
    return False
