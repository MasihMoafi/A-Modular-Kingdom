import pytest
import os
from src.rag.notebook_chunker import extract_cells_from_notebook

def test_extract_cells_nonexistent_file():
    """Test that reading a non-existent file returns an empty list."""
    result = extract_cells_from_notebook("nonexistent_notebook_file_xyz.ipynb")
    assert result == []

def test_extract_cells_invalid_json(tmp_path):
    """Test that reading a file with invalid JSON returns an empty list."""
    invalid_file = tmp_path / "invalid.ipynb"
    invalid_file.write_text("This is not valid JSON content {", encoding="utf-8")

    result = extract_cells_from_notebook(str(invalid_file))
    assert result == []
