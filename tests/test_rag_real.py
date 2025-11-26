"""
Test RAG with REAL documents (Napoleon.pdf, project docs)
"""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledgeV3 as fetchV3

# Real document paths
NAPOLEON_PDF = os.path.join(os.path.dirname(__file__), '..', 'src', 'rag', 'files', 'Napoleon.pdf')
REAL_DOCS = os.path.join(os.path.dirname(__file__), 'fixtures', 'real_docs')


class TestRAGV2Real:
    """Test RAG V2 with real documents"""

    def test_napoleon_battles(self):
        """Query Napoleon.pdf for battles"""
        result = fetchV2("What battles did Napoleon fight?")
        assert result is not None
        assert len(result) > 100
        # Should mention specific battles
        result_lower = result.lower()
        assert any(battle in result_lower for battle in ['waterloo', 'austerlitz', 'battle', 'war'])

    def test_napoleon_timeline(self):
        """Query Napoleon.pdf for dates"""
        result = fetchV2("When was Napoleon born?")
        assert result is not None
        assert '1769' in result or 'born' in result.lower()

    def test_real_project_docs(self):
        """Query real project documentation"""
        result = fetchV2("anthropic prompt engineering", doc_path=REAL_DOCS)
        assert result is not None
        assert len(result) > 100
        # Should find content from anthropics-prompt-eng-interactive-tutorial.md
        assert any(word in result.lower() for word in ['prompt', 'claude', 'anthropic'])

    def test_claude_instructions(self):
        """Query claude-agent-instructions.md"""
        result = fetchV2("claude agent", doc_path=REAL_DOCS)
        assert result is not None
        assert 'claude' in result.lower() or 'agent' in result.lower()


class TestRAGV3Real:
    """Test RAG V3 with real documents"""

    def test_napoleon_battles_v3(self):
        """V3: Query Napoleon.pdf for battles"""
        result = fetchV3("What battles did Napoleon fight?")
        assert result is not None
        assert len(result) > 100
        result_lower = result.lower()
        assert any(battle in result_lower for battle in ['waterloo', 'austerlitz', 'battle', 'war'])

    def test_napoleon_empire(self):
        """V3: Query about Napoleon's empire"""
        result = fetchV3("Tell me about Napoleon's empire")
        assert result is not None
        assert any(word in result.lower() for word in ['france', 'empire', 'emperor', 'napoleon'])

    def test_project_requirements_v3(self):
        """V3: Query project requirement documents"""
        result = fetchV3("zigzag design requirements", doc_path=REAL_DOCS)
        assert result is not None
        assert 'zigzag' in result.lower() or 'design' in result.lower()

    def test_forex_project_v3(self):
        """V3: Query forex_project.md"""
        result = fetchV3("forex", doc_path=REAL_DOCS)
        assert result is not None
        assert 'forex' in result.lower()


class TestRAGPerformanceReal:
    """Performance benchmarks with REAL documents"""

    def test_v2_speed_real(self):
        """Measure V2 speed with real docs"""
        start = time.time()
        result = fetchV2("Napoleon")
        elapsed = time.time() - start

        print(f"\nV2 Real Document Query: {elapsed:.2f}s")
        assert result is not None
        assert elapsed < 60  # Should complete within 60s (indexing + query)

    def test_v3_speed_real(self):
        """Measure V3 speed with real docs"""
        start = time.time()
        result = fetchV3("Napoleon")
        elapsed = time.time() - start

        print(f"\nV3 Real Document Query: {elapsed:.2f}s")
        assert result is not None
        assert elapsed < 60  # Should complete within 60s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
