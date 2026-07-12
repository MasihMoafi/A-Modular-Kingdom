import pytest
from src.rag.notebook_chunker import should_split_cell

def test_should_split_cell_less_than_max():
    """Test when cell content length is less than max_chunk_size."""
    cell = {"content": "a" * 1000}
    assert not should_split_cell(cell, max_chunk_size=2000)

def test_should_split_cell_equal_to_max():
    """Test when cell content length is exactly equal to max_chunk_size."""
    cell = {"content": "a" * 2000}
    assert not should_split_cell(cell, max_chunk_size=2000)

def test_should_split_cell_greater_than_max():
    """Test when cell content length is greater than max_chunk_size."""
    cell = {"content": "a" * 2001}
    assert should_split_cell(cell, max_chunk_size=2000)

def test_should_split_cell_default_max_chunk_size():
    """Test should_split_cell with default max_chunk_size (2000)."""
    cell_small = {"content": "a" * 2000}
    cell_large = {"content": "a" * 2001}
    assert not should_split_cell(cell_small)
    assert should_split_cell(cell_large)

def test_should_split_cell_custom_max_chunk_size():
    """Test should_split_cell with a custom max_chunk_size."""
    cell = {"content": "a" * 500}
    assert not should_split_cell(cell, max_chunk_size=500)
    assert should_split_cell(cell, max_chunk_size=499)

def test_should_split_cell_empty_content():
    """Test when cell content is empty."""
    cell = {"content": ""}
    assert not should_split_cell(cell, max_chunk_size=2000)

def test_should_split_cell_missing_content():
    """Test when cell dictionary is missing the 'content' key. Should raise KeyError."""
    cell = {}
    with pytest.raises(KeyError):
        should_split_cell(cell)
