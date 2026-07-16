#!/usr/bin/env python
# coding: utf-8
"""Minimal stdio MCP host for Elpis local RAG."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

repo_root = Path(__file__).resolve().parents[2]
project_root = repo_root / "src"
workspace_root = Path(
    os.environ.get("ELPIS_WORKSPACE_ROOT")
    or os.environ.get("AMK_WORKSPACE_ROOT")
    or os.getcwd()
).expanduser().resolve()
sys.path.insert(0, str(project_root))

_SERVER_NAME = "elpis-rag"
_SERVER_VERSION = "0.1.0"
_use_content_length_framing = False
_initialized = False

_TOOL = {
    "name": "query_knowledge_base",
    "description": (
        "Run local RAG over the launch workspace or supplied files or directory. "
        "Use this autonomously when broad semantic discovery would avoid loading many files "
        "into context. For code changes, follow retrieved source paths with exact search or "
        "file reads to obtain current line positions."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for the knowledge base",
            },
            "doc_path": {
                "type": "string",
                "default": "",
                "description": "Optional documents directory",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def _read_payload() -> str | None:
    global _use_content_length_framing

    first = sys.stdin.buffer.readline()
    if not first:
        return None

    if not first.lower().startswith(b"content-length:"):
        return first.decode("utf-8", errors="replace")

    _use_content_length_framing = True
    content_length = None
    line = first
    while line not in (b"\r\n", b"\n"):
        text = line.decode("ascii", errors="ignore").strip()
        if ":" in text:
            key, value = text.split(":", 1)
            if key.strip().lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError:
                    return None
        line = sys.stdin.buffer.readline()
        if not line:
            return None

    if content_length is None or content_length < 0:
        return None
    return sys.stdin.buffer.read(content_length).decode("utf-8", errors="replace")


def _write_message(message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
    if _use_content_length_framing:
        data = payload.encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def _result(request_id: Any, result: dict[str, Any]) -> None:
    _write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: Any, code: int, message: str) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def _normalize_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return str(candidate.resolve())


def query_knowledge_base(
    query: str,
    doc_path: str = "",
) -> str:
    """Load and run RAG only after an explicit tool call."""
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return json.dumps({"error": "query must be a non-empty string"})

    normalized_doc_path = _normalize_path(doc_path) if doc_path.strip() else ""
    if not normalized_doc_path:
        normalized_doc_path = str(workspace_root)

    try:
        module = importlib.import_module("rag.fetch")
        fetch = getattr(module, "fetchExternalKnowledge", None)
        if not callable(fetch):
            return json.dumps({"error": "rag.fetch has no fetchExternalKnowledge function"})

        if normalized_doc_path:
            value = fetch(query=query, doc_path=normalized_doc_path)
        else:
            value = fetch(query=query)
        return json.dumps({"result": value})
    except Exception as exc:
        return json.dumps({"error": f"RAG query failed: {exc}"})


def _handle_request(message: dict[str, Any]) -> None:
    global _initialized

    method = message.get("method")
    request_id = message.get("id")

    if request_id is None:
        return
    if method == "initialize":
        params = message.get("params") or {}
        protocol_version = params.get("protocolVersion", "2025-06-18")
        _result(
            request_id,
            {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
            },
        )
        _initialized = True
        return
    if method == "ping":
        _result(request_id, {})
        return
    if not _initialized:
        _error(request_id, -32002, "Server is not initialized")
        return
    if method == "tools/list":
        _result(request_id, {"tools": [_TOOL]})
        return
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != _TOOL["name"]:
            _error(request_id, -32601, "Unknown tool")
            return
        arguments = params.get("arguments") or {}
        try:
            output = query_knowledge_base(
                query=arguments.get("query", ""),
                doc_path=arguments.get("doc_path", ""),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            output = json.dumps({"error": f"Invalid arguments: {exc}"})
        _result(
            request_id,
            {
                "content": [{"type": "text", "text": output}],
                "isError": False,
            },
        )
        return
    _error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    while True:
        payload = _read_payload()
        if payload is None:
            return
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            _handle_request(message)


if __name__ == "__main__":
    main()
