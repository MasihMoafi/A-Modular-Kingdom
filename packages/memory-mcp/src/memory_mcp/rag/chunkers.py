"""Document chunking utilities for RAG."""

import json
import re
from typing import Any


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    source: str = "",
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        source: Source file path for metadata

    Returns:
        List of chunk dictionaries with content and metadata
    """
    if not text or len(text.strip()) < 50:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if len(chunk_text.strip()) > 50:
            chunks.append(
                {
                    "content": chunk_text,
                    "source": source,
                    "chunk_id": chunk_id,
                }
            )

        start = end - chunk_overlap if end < len(text) else len(text)
        chunk_id += 1

        if chunk_id > 10000:
            break

    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file using PyMuPDF."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        return re.sub(r"\s+", " ", full_text).strip()
    except ImportError:
        raise ImportError("PDF support requires pymupdf. Install with: pip install pymupdf")
    except Exception:
        return ""


def extract_text_from_json(json_path: str) -> str:
    """Extract text from JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def flatten_json(obj, prefix=""):
            text_parts = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        text_parts.extend(flatten_json(value, f"{prefix}{key}."))
                    else:
                        text_parts.append(f"{prefix}{key}: {value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    text_parts.extend(flatten_json(item, f"{prefix}[{i}]."))
            else:
                text_parts.append(str(obj))
            return text_parts

        return "\n".join(flatten_json(data))
    except Exception:
        return ""


def process_notebook(
    notebook_path: str,
    max_chunk_size: int = 2000,
) -> list[dict[str, Any]]:
    """Extract cells from Jupyter notebook with metadata.

    Args:
        notebook_path: Path to .ipynb file
        max_chunk_size: Maximum size before splitting cells

    Returns:
        List of chunk dictionaries ready for indexing
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except Exception:
        return []

    cells = notebook.get("cells", [])
    processed_chunks = []

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        cell_id = cell.get("id", f"cell_{idx}")

        source_lines = cell.get("source", [])
        if isinstance(source_lines, list):
            content = "".join(source_lines)
        else:
            content = str(source_lines)

        if not content.strip():
            continue

        exercise_number = _extract_exercise_number(content)

        metadata = {
            "content": content,
            "source": notebook_path,
            "cell_type": cell_type,
            "cell_id": cell_id,
            "cell_index": idx,
            "exercise_number": exercise_number,
            "chunk_id": idx,
        }

        if len(content) > max_chunk_size:
            sub_chunks = _split_large_cell(metadata, chunk_size=1500, chunk_overlap=200)
            processed_chunks.extend(sub_chunks)
        else:
            processed_chunks.append(metadata)

    return processed_chunks


def _extract_exercise_number(content: str) -> str | None:
    """Extract exercise number from cell content."""
    patterns = [
        r"#\s*GRADED\s+CELL:\s*exercise\s+(\d+)",
        r"###\s*Exercise\s+(\d+)",
        r"Exercise\s+(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _split_large_cell(
    cell: dict[str, Any],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split a large cell into smaller chunks."""
    content = cell["content"]
    chunks = []
    start = 0
    chunk_counter = 0

    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]

        chunk_metadata = cell.copy()
        chunk_metadata["content"] = chunk_text
        chunk_metadata["chunk_id"] = f"{cell['cell_index']}.{chunk_counter}"
        chunk_metadata["is_partial"] = True
        chunk_metadata["partial_index"] = chunk_counter

        chunks.append(chunk_metadata)

        start = end - chunk_overlap
        chunk_counter += 1

        if start >= len(content) or chunk_counter > 100:
            break

    return chunks
