"""Tests for configuration system."""

import os
from pathlib import Path

import pytest

from memory_mcp.config import Settings


def test_default_settings():
    """Test that default settings are loaded correctly."""
    settings = Settings()

    assert settings.embed_provider == "ollama"
    assert settings.embed_model == "nomic-embed-text"
    assert settings.qdrant_mode == "local"
    assert settings.rag_chunk_size == 700
    assert settings.rag_rerank is True


def test_settings_from_env(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("MEMORY_MCP_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_MCP_RAG_TOP_K", "10")

    settings = Settings()

    assert settings.embed_provider == "openai"
    assert settings.rag_top_k == 10


def test_path_expansion():
    """Test that paths with ~ are expanded."""
    settings = Settings(qdrant_path="~/test/qdrant")

    assert "~" not in settings.qdrant_path
    assert str(Path.home()) in settings.qdrant_path


def test_invalid_embed_provider():
    """Test that invalid embed provider raises error."""
    with pytest.raises(ValueError):
        Settings(embed_provider="invalid_provider")


def test_qdrant_modes():
    """Test valid qdrant modes."""
    for mode in ["local", "cloud", "memory"]:
        settings = Settings(qdrant_mode=mode)
        assert settings.qdrant_mode == mode
