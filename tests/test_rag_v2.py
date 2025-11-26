"""
Test RAG v2 functionality
"""
import os
import sys
import time
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'docs')


def test_rag_v2_basic():
    """Test basic RAG v2 functionality with mixed file types"""
    result = fetchExternalKnowledge("machine learning", doc_path=FIXTURES_PATH)
    assert result is not None
    assert len(result) > 0
    assert "machine learning" in result.lower() or "neural" in result.lower()


def test_rag_v2_ipynb_support():
    """Test that .ipynb files are indexed and searchable"""
    result = fetchExternalKnowledge("sklearn", doc_path=FIXTURES_PATH)
    assert result is not None
    assert "sklearn" in result.lower() or "scikit" in result.lower()


def test_rag_v2_python_files():
    """Test Python file indexing"""
    result = fetchExternalKnowledge("numpy", doc_path=FIXTURES_PATH)
    assert result is not None
    assert "numpy" in result.lower()


def test_rag_v2_timing():
    """Measure indexing and query time"""
    start = time.time()
    result = fetchExternalKnowledge("PyTorch", doc_path=FIXTURES_PATH)
    elapsed = time.time() - start

    print(f"\nRAG V2 Timing: {elapsed:.2f}s")
    assert elapsed < 30  # Should be fast with small dataset
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
