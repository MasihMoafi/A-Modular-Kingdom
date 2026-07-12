import pytest
from src.rag.core import chunk_text

def test_chunk_text_empty_or_small():
    """Test text that is empty or smaller than the threshold."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("a" * 49) == []

def test_chunk_text_exact_size():
    """Test text that exactly fits the chunk size."""
    text = "a" * 100
    # Provide a smaller text to meet > 50 condition but let's test chunk_size 100
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "a" * 100

def test_chunk_text_overlap():
    """Test multiple chunks and correct overlap behavior."""
    text = "a" * 50 + "b" * 50 + "c" * 50
    # len(text) = 150
    # chunk_size = 100, overlap = 50
    # first chunk: [0:100] = a*50 + b*50
    # start = 100 - 50 = 50
    # second chunk: [50:150] = b*50 + c*50
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 50 + "b" * 50
    assert chunks[1] == "b" * 50 + "c" * 50

def test_chunk_text_large():
    """Test chunk text with larger text size."""
    # Build a repetitive string of numbers
    text = "".join(str(i % 10) for i in range(250))
    # chunk_size = 100, chunk_overlap = 20
    # 0 -> 100
    # 80 -> 180
    # 160 -> 250 (len is 90)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 3
    assert chunks[0] == text[0:100]
    assert chunks[1] == text[80:180]
    assert chunks[2] == text[160:250]

def test_chunk_small_chunk_discard():
    """Test that chunks smaller than or equal to 50 characters are not returned."""
    # Text length 150
    # chunk_size = 100, overlap = 50
    # First chunk [0:100] is kept
    # Second chunk [50:150] is kept
    text = "a" * 50 + "b" * 50 + "c" * 50
    # What if end is reached and remaining is small?
    # Say chunk_size = 100, overlap = 0, length = 120
    # First chunk [0:100] (len 100) -> kept
    # Next start = 100. Chunk [100:120] (len 20) -> NOT kept (<= 50)
    text2 = "x" * 120
    chunks = chunk_text(text2, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == "x" * 100

    # Let's verify exactly 50 chars is discarded, and 51 is kept
    text3 = "y" * 150
    chunks_50 = chunk_text(text3, chunk_size=100, chunk_overlap=50)
    # start 0 -> 100 (len 100)
    # start 50 -> 150 (len 100)
    assert len(chunks_50) == 2

    # To get a 50 char chunk at the end:
    # 100 size, 0 overlap, length 150
    # start 0:100 (100)
    # start 100:150 (50) -> discarded
    chunks_exact_50 = chunk_text("z" * 150, chunk_size=100, chunk_overlap=0)
    assert len(chunks_exact_50) == 1
    assert chunks_exact_50[0] == "z" * 100

    chunks_51 = chunk_text("w" * 151, chunk_size=100, chunk_overlap=0)
    assert len(chunks_51) == 2
    assert chunks_51[1] == "w" * 51

def test_chunk_text_safety_limit():
    """Test the safety limit of 10000 chunks."""
    # Create a scenario that would generate > 10000 chunks
    # We won't actually pass 10000 * 50 characters, as that's 500k.
    # Actually wait, 500k is fine to construct.
    text = "a" * (50 * 10001)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=90)
    # Each iteration advances by 10.
    # The condition is `if len(chunks) > 10000: break`
    # This means the length will be exactly 10001 when it breaks.
    assert len(chunks) == 10001
