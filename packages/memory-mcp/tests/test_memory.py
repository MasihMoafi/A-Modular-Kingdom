"""Tests for memory store."""

import tempfile
from pathlib import Path

import pytest

from memory_mcp.config import Settings
from memory_mcp.memory import MemoryStore


@pytest.fixture
def temp_settings():
    """Create settings with temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Settings(chroma_path=tmpdir)


@pytest.fixture
def memory_store(temp_settings):
    """Create a memory store with temporary storage."""
    return MemoryStore(temp_settings)


def test_add_and_search(memory_store):
    """Test adding and searching memories."""
    memory_id = memory_store.add("User prefers dark mode")

    assert memory_id is not None
    assert len(memory_id) == 36

    results = memory_store.search("dark mode", k=1)

    assert len(results) == 1
    assert "dark mode" in results[0]["content"]


def test_add_with_metadata(memory_store):
    """Test adding memory with metadata."""
    memory_id = memory_store.add(
        "Python is the user's favorite language",
        metadata={"category": "preferences", "confidence": 0.9},
    )

    results = memory_store.search("favorite language")

    assert len(results) >= 1


def test_delete(memory_store):
    """Test deleting a memory."""
    memory_id = memory_store.add("Temporary memory")

    assert memory_store.count() == 1

    success = memory_store.delete(memory_id)

    assert success is True
    assert memory_store.count() == 0


def test_get_all(memory_store):
    """Test retrieving all memories."""
    memory_store.add("First memory")
    memory_store.add("Second memory")
    memory_store.add("Third memory")

    all_memories = memory_store.get_all()

    assert len(all_memories) == 3


def test_count(memory_store):
    """Test counting memories."""
    assert memory_store.count() == 0

    memory_store.add("Memory 1")
    assert memory_store.count() == 1

    memory_store.add("Memory 2")
    assert memory_store.count() == 2


def test_search_no_results(memory_store):
    """Test searching with no matching results."""
    memory_store.add("Python programming")

    results = memory_store.search("javascript frameworks", k=3)

    assert isinstance(results, list)
