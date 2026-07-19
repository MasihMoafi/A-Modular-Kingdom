from pathlib import Path
import json
import sys
import tempfile
import types


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import host


def test_default_workspace_call_excludes_global_archive() -> None:
    original_workspace = host.workspace_root
    original_import = host.importlib.import_module
    captured = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            host.workspace_root = Path(directory)
            host.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.update(kwargs) or "ok"
            )
            assert json.loads(host.query_knowledge_base("find context")) == {"result": "ok"}
    finally:
        host.workspace_root = original_workspace
        host.importlib.import_module = original_import

    assert captured["doc_path"] == str(Path(directory).resolve())
    assert captured["exclude_global_archive"] is True


def test_explicit_path_keeps_its_scope() -> None:
    original_import = host.importlib.import_module
    original_workspace = host.workspace_root
    captured = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            host.workspace_root = Path(directory)
            host.importlib.import_module = lambda _name: types.SimpleNamespace(
                fetchExternalKnowledge=lambda **kwargs: captured.update(kwargs) or "ok"
            )
            host.query_knowledge_base("find context", directory)
            assert captured["doc_path"] == str(Path(directory).resolve())
    finally:
        host.importlib.import_module = original_import
        host.workspace_root = original_workspace

    assert captured["exclude_global_archive"] is False


def test_node_modules_scope_has_clear_recovery() -> None:
    original_workspace = host.workspace_root
    try:
        with tempfile.TemporaryDirectory() as directory:
            host.workspace_root = Path(directory)
            unsafe = Path(directory) / "node_modules"
            unsafe.mkdir()
            output = host.query_knowledge_base("find context", str(unsafe))
    finally:
        host.workspace_root = original_workspace

    assert "will not scan dependency or build folders" in output
    assert "Choose the project source folder instead" in output


def test_depth_limit_has_clear_recovery() -> None:
    original = host.os.environ.get("ELPIS_RAG_MAX_DEPTH")
    original_workspace = host.workspace_root
    host.os.environ["ELPIS_RAG_MAX_DEPTH"] = "1"
    try:
        with tempfile.TemporaryDirectory() as directory:
            host.workspace_root = Path(directory)
            nested = Path(directory) / "one" / "two"
            nested.mkdir(parents=True)
            output = host.query_knowledge_base("find context", directory)
    finally:
        host.workspace_root = original_workspace
        if original is None:
            host.os.environ.pop("ELPIS_RAG_MAX_DEPTH", None)
        else:
            host.os.environ["ELPIS_RAG_MAX_DEPTH"] = original

    assert "configured 1-folder depth limit" in output
    assert "ELPIS_RAG_MAX_DEPTH" in output
