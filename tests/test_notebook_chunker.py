import pytest
from unittest.mock import patch
from rag.notebook_chunker import extract_cells_from_notebook

def test_extract_cells_from_notebook_nonexistent_file():
    """Test that a non-existent file returns an empty list."""
    result = extract_cells_from_notebook("non_existent_notebook_file_xyz123.ipynb")
    assert result == []

def test_extract_cells_from_notebook_mock_exception():
    """Test that an exception during open returns an empty list."""
    with patch("builtins.open", side_effect=Exception("Mocked open exception")):
        result = extract_cells_from_notebook("fake_notebook.ipynb")
        assert result == []

def test_extract_cells_from_notebook_json_error():
    """Test that an invalid json file raises an exception which is caught and returns an empty list."""
    with patch("builtins.open"):
        with patch("json.load", side_effect=ValueError("Invalid JSON")):
            result = extract_cells_from_notebook("fake_notebook.ipynb")
            assert result == []
