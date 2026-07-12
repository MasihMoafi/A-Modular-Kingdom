import os
import pytest
import sys
from unittest.mock import patch

# We don't want to load actual models when testing resolve_path,
# so we mock RAGPipeline loading when importing fetch
with patch.dict('sys.modules'):
    # Mock some problematic dependencies to avoid heavy imports
    sys.modules['src.rag.core'] = type('MockCore', (), {'RAGPipelineV2': type('MockPipeline', (), {})})
    sys.modules['src.rag.core_2'] = type('MockCore2', (), {'RAGPipelineV2': type('MockPipeline', (), {})})
    from src.rag.fetch import resolve_path

def test_resolve_path_empty():
    """Test that resolving an empty path returns None."""
    assert resolve_path("") is None
    assert resolve_path(None) is None

def test_resolve_path_shortcuts(monkeypatch):
    """Test that common shortcuts like desktop, documents, downloads are resolved."""
    # Mock expanduser to return a predictable path
    def mock_expanduser(path):
        return path.replace("~", "/home/mockuser")
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    def mock_isabs(path):
        return path.startswith("/")
    monkeypatch.setattr(os.path, "isabs", mock_isabs)

    assert resolve_path("desktop") == "/home/mockuser/Desktop"
    assert resolve_path(" DOCUMENTS ") == "/home/mockuser/Documents"
    assert resolve_path("downloads") == "/home/mockuser/Downloads"

def test_resolve_path_expanduser(monkeypatch):
    """Test that ~ is expanded to the user's home directory."""
    def mock_expanduser(path):
        return path.replace("~", "/home/mockuser")
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    def mock_isabs(path):
        return path.startswith("/")
    monkeypatch.setattr(os.path, "isabs", mock_isabs)

    assert resolve_path("~/myfolder/file.txt") == "/home/mockuser/myfolder/file.txt"

def test_resolve_path_expandvars(monkeypatch):
    """Test that environment variables in paths are expanded."""
    monkeypatch.setenv("TEST_RAG_DIR", "rag_dir_value")

    path_with_var = "$TEST_RAG_DIR/data"
    resolved = resolve_path(path_with_var)

    assert os.path.isabs(resolved)
    assert resolved.endswith(os.path.join("rag_dir_value", "data"))

def test_resolve_path_relative_to_absolute():
    """Test that a relative path is converted to an absolute path."""
    relative = "some/relative/path"
    resolved = resolve_path(relative)

    assert os.path.isabs(resolved)
    assert resolved == os.path.abspath(relative)

def test_resolve_path_absolute_unchanged():
    """Test that an absolute path remains unchanged."""
    absolute = "/a/completely/absolute/path"
    resolved = resolve_path(absolute)

    assert resolved == absolute
