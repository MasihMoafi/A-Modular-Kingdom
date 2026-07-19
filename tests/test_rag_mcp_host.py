from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.host import _TOOL


def test_rag_tool_is_advertised_as_read_only() -> None:
    assert _TOOL["name"] == "query_knowledge_base"
    assert _TOOL["annotations"] == {
        "title": "Search Elpis knowledge base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }


def test_normalize_path_prevents_traversal() -> None:
    from agent.host import _normalize_path
    import pytest

    with pytest.raises(ValueError, match="Path is outside the workspace"):
        _normalize_path("../../etc/passwd")

    with pytest.raises(ValueError, match="Path is outside the workspace"):
        _normalize_path("/etc/passwd")
