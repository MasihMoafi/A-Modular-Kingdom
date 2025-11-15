"""
Jupyter Notebook Chunker for RAG System
Extracts cells from .ipynb files with proper metadata preservation
"""
import json
import re
from typing import List, Dict, Any


def extract_cells_from_notebook(notebook_path: str) -> List[Dict[str, Any]]:
    """
    Extract individual cells from Jupyter notebook with rich metadata.

    Args:
        notebook_path: Path to .ipynb file

    Returns:
        List of dicts with keys: content, source, cell_type, cell_id, exercise_number, chunk_id
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error loading notebook {notebook_path}: {e}")
        return []

    cells = notebook.get('cells', [])
    extracted_cells = []

    for idx, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        cell_id = cell.get('id', f'cell_{idx}')

        # Join source lines (notebooks store source as list of strings)
        source_lines = cell.get('source', [])
        if isinstance(source_lines, list):
            content = ''.join(source_lines)
        else:
            content = str(source_lines)

        # Skip empty cells
        if not content.strip():
            continue

        # Extract exercise number from content
        exercise_number = extract_exercise_number(content)

        # Build metadata
        metadata = {
            'content': content,
            'source': notebook_path,
            'cell_type': cell_type,
            'cell_id': cell_id,
            'cell_index': idx,
            'exercise_number': exercise_number,
            'chunk_id': idx  # Each cell is a chunk
        }

        extracted_cells.append(metadata)

    return extracted_cells


def extract_exercise_number(content: str) -> str:
    """
    Extract exercise number from cell content.

    Patterns:
    - "# GRADED CELL: exercise 1"
    - "### Exercise 2:"
    - "Exercise 3 - "
    """
    patterns = [
        r'#\s*GRADED\s+CELL:\s*exercise\s+(\d+)',
        r'###\s*Exercise\s+(\d+)',
        r'Exercise\s+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def should_split_cell(cell: Dict[str, Any], max_chunk_size: int = 2000) -> bool:
    """
    Determine if a cell should be split into smaller chunks.

    Args:
        cell: Cell metadata dict
        max_chunk_size: Maximum characters per chunk

    Returns:
        True if cell should be split
    """
    content_length = len(cell['content'])
    return content_length > max_chunk_size


def split_large_cell(cell: Dict[str, Any], chunk_size: int = 1500, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split a large cell into smaller chunks while preserving metadata.

    Args:
        cell: Cell metadata dict
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunk dicts with updated chunk_id
    """
    content = cell['content']
    chunks = []

    # Simple sliding window chunking
    start = 0
    chunk_counter = 0

    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]

        # Create new chunk with same metadata
        chunk_metadata = cell.copy()
        chunk_metadata['content'] = chunk_text
        chunk_metadata['chunk_id'] = f"{cell['cell_index']}.{chunk_counter}"
        chunk_metadata['is_partial'] = True
        chunk_metadata['partial_index'] = chunk_counter

        chunks.append(chunk_metadata)

        # Move start position
        start = end - chunk_overlap
        chunk_counter += 1

        # Prevent infinite loop
        if start >= len(content) or chunk_counter > 100:
            break

    return chunks


def process_notebook_for_rag(notebook_path: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
    """
    Complete pipeline: extract cells, optionally split large ones.

    Args:
        notebook_path: Path to .ipynb file
        max_chunk_size: Max size before splitting

    Returns:
        List of processed chunks ready for indexing
    """
    cells = extract_cells_from_notebook(notebook_path)

    processed_chunks = []

    for cell in cells:
        if should_split_cell(cell, max_chunk_size):
            # Split large cell
            sub_chunks = split_large_cell(cell, chunk_size=1500, chunk_overlap=200)
            processed_chunks.extend(sub_chunks)
        else:
            # Keep cell as-is
            processed_chunks.append(cell)

    return processed_chunks
