import sys
from unittest.mock import patch, MagicMock

# Mock out heavy ML dependencies to avoid environment issues
mock_modules = {
    'ollama': MagicMock(),
    'fitz': MagicMock(),
    'sentence_transformers': MagicMock(),
    'torch': MagicMock(),
    'src.rag.qdrant_backend': MagicMock(),
}

with patch.dict('sys.modules', mock_modules):
    from src.rag.core import chunk_text

def test_chunk_text_empty_or_none():
    """Test that empty or None text returns an empty list."""
    assert chunk_text("") == []
    assert chunk_text(None) == []

def test_chunk_text_too_short():
    """Test that text shorter than 50 characters (excluding whitespace) returns empty list."""
    assert chunk_text("a" * 50) == []
    # Strings with <= 50 non-whitespace chars
    assert chunk_text(" " * 51) == []

def test_chunk_text_single_chunk():
    """Test text that fits exactly within one chunk or is just over the 50 char threshold."""
    text = "a" * 51
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_multiple_chunks_overlap():
    """Test text that requires splitting and overlapping."""
    text = "a" * 100
    # chunk 1: [0:60], next start = 60 - 20 = 40
    # chunk 2: [40:100], next start = 100 (since 100 == len(text))
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=20)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 60
    assert chunks[1] == "a" * 60

def test_chunk_text_skips_whitespace_chunks():
    """Test that chunks composed entirely of whitespace (or <= 50 non-whitespace) are skipped."""
    # chunk 1: 51 'a's, chunk 2: 51 spaces
    text = ("a" * 51) + (" " * 51)
    chunks = chunk_text(text, chunk_size=51, chunk_overlap=0)
    # Second chunk should be stripped and its length will be 0 (<= 50)
    assert len(chunks) == 1
    assert chunks[0] == "a" * 51

def test_chunk_text_safety_limit():
    """Test that the safety limit of 10001 chunks is respected to prevent infinite loops."""
    # A string of length 10100
    text = "a" * 10100
    # chunk_size=52, overlap=51 means it advances by 1 char each time
    chunks = chunk_text(text, chunk_size=52, chunk_overlap=51)
    # Loop breaks when len(chunks) > 10000, which means it will be exactly 10001
    assert len(chunks) == 10001
