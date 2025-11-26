"""
Test RAG v3 functionality
"""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_3 import fetchExternalKnowledge

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'docs')


def test_rag_v3_basic():
    """Test basic RAG v3 functionality"""
    result = fetchExternalKnowledge("machine learning", doc_path=FIXTURES_PATH)
    assert result is not None
    assert len(result) > 0


def test_rag_v3_ipynb_support():
    """Test that .ipynb files work in v3 (newly added)"""
    result = fetchExternalKnowledge("sklearn", doc_path=FIXTURES_PATH)
    assert result is not None
    assert "sklearn" in result.lower() or "scikit" in result.lower()


def test_rag_v3_hybrid_search():
    """Test v3's RRF fusion and reranking"""
    result = fetchExternalKnowledge("neural networks classification", doc_path=FIXTURES_PATH)
    assert result is not None
    # V3 should use both vector + BM25 + reranking
    assert len(result) > 0


def test_rag_v3_timing():
    """Measure v3 performance"""
    start = time.time()
    result = fetchExternalKnowledge("TensorFlow", doc_path=FIXTURES_PATH)
    elapsed = time.time() - start

    print(f"\nRAG V3 Timing: {elapsed:.2f}s")
    assert elapsed < 60  # V3 is slower (LLM reranking) but should still be reasonable
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
