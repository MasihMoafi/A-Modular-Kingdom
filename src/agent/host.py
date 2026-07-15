#!/usr/bin/env python
"""Minimal Elpis MCP server: one local retrieval tool."""

import asyncio
import importlib
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version as package_version

from mcp.server.fastmcp import FastMCP
from pydantic import Field


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(repo_root, "src")
workspace_root = os.path.realpath(os.environ.get("ELPIS_WORKSPACE_ROOT", os.getcwd()))
sys.path.insert(0, project_root)

_LOG_STDERR = os.environ.get("MCP_LOG_STDERR", "0").lower() in {"1", "true", "yes", "y", "on"}
_LOG_FILE = os.environ.get("MCP_LOG_FILE", "").strip()
_DEBUG_PROTOCOL = os.environ.get("MCP_DEBUG_PROTOCOL", "0").lower() in {"1", "true", "yes", "y", "on"}


def _elog(message: str) -> None:
    if _LOG_STDERR:
        sys.stderr.write(message)
        sys.stderr.flush()
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(message)
        except Exception:
            pass


from utils.proxy import restore_http as _restore_proxy

_restore_proxy()

mcp = FastMCP("elpis_context")


@mcp.tool(
    name="query_knowledge_base",
    description="Retrieve relevant passages from a local file, directory, or the launch workspace.",
)
def query_knowledge_base(
    query: str = Field(description="Question to answer from local knowledge"),
    doc_path: str = Field(default="", description="Optional file or directory; defaults to the launch workspace"),
) -> str:
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "Query must not be empty."})

    target = doc_path.strip() if isinstance(doc_path, str) else ""
    if not target:
        target = workspace_root
    elif not os.path.isabs(target) and not target.startswith("~"):
        target = os.path.join(workspace_root, target)

    try:
        module = importlib.import_module("rag.fetch")
        retrieve = getattr(module, "fetchExternalKnowledge", None)
        if not callable(retrieve):
            return json.dumps({"error": "The maintained RAG retriever is unavailable."})

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = retrieve(query=query.strip(), doc_path=target)

        if stdout_buffer.tell() or stderr_buffer.tell():
            _elog(
                f"[HOST] Suppressed RAG output: stdout={stdout_buffer.tell()} "
                f"stderr={stderr_buffer.tell()} chars\n"
            )
        return json.dumps({"result": result, "source": os.path.realpath(os.path.expanduser(target))})
    except Exception as exc:
        _elog(f"[HOST] Retrieval failed: {exc}\n")
        return json.dumps({"error": f"Local retrieval failed: {exc}"})


import mcp.types as mcp_types
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS


# Keep the synchronous loop used by the existing host; it supports both newline JSON
# and Content-Length framing without letting library output corrupt MCP stdout.
server_info = mcp_types.Implementation(name="elpis_context", version=package_version("mcp"))
server_capabilities = mcp_types.ServerCapabilities(
    tools=mcp_types.ToolsCapability(listChanged=False),
    resources=None,
    prompts=None,
)

initialized = False
use_content_length_framing = False


def _write_jsonrpc(message: mcp_types.JSONRPCMessage) -> None:
    payload = message.model_dump_json(by_alias=True, exclude_none=True)
    if use_content_length_framing:
        data = payload.encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    else:
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()


def _read_jsonrpc_payload() -> str | None:
    global use_content_length_framing

    first = sys.stdin.buffer.readline()
    if not first:
        return None
    if _DEBUG_PROTOCOL:
        _elog(f"[HOST][proto] first_line={first[:120]!r}\n")

    if first.lower().startswith(b"content-length:"):
        use_content_length_framing = True
        header_lines = [first]
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            header_lines.append(line)
            if line in (b"\r\n", b"\n"):
                break

        content_length = None
        for raw in header_lines:
            text = raw.decode("ascii", errors="ignore").strip()
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            if key.strip().lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError:
                    return None

        if content_length is None or content_length < 0:
            return None
        body = sys.stdin.buffer.read(content_length)
        if len(body) != content_length:
            return None
        return body.decode("utf-8", errors="replace")

    return first.decode("utf-8", errors="replace")


def _send_error(request_id: mcp_types.RequestId, code: int, message: str) -> None:
    error = mcp_types.JSONRPCError(
        jsonrpc="2.0",
        id=request_id,
        error=mcp_types.ErrorData(code=code, message=message),
    )
    _write_jsonrpc(mcp_types.JSONRPCMessage(error))


def _send_result(request_id: mcp_types.RequestId, result: mcp_types.ServerResult) -> None:
    response = mcp_types.JSONRPCResponse(
        jsonrpc="2.0",
        id=request_id,
        result=result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    _write_jsonrpc(mcp_types.JSONRPCMessage(response))


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

try:
    while True:
        payload = _read_jsonrpc_payload()
        if payload is None:
            break

        try:
            message = mcp_types.JSONRPCMessage.model_validate_json(payload)
        except Exception:
            continue

        if not isinstance(message.root, mcp_types.JSONRPCRequest):
            continue

        request_id = message.root.id
        try:
            request = mcp_types.ClientRequest.model_validate(
                message.root.model_dump(by_alias=True, mode="json", exclude_none=True)
            )
        except Exception:
            _send_error(request_id, mcp_types.INVALID_PARAMS, "Invalid request parameters")
            continue

        try:
            match request.root:
                case mcp_types.InitializeRequest(params=params):
                    requested = params.protocolVersion
                    protocol = (
                        requested
                        if requested in SUPPORTED_PROTOCOL_VERSIONS
                        else mcp_types.LATEST_PROTOCOL_VERSION
                    )
                    result = mcp_types.InitializeResult(
                        protocolVersion=protocol,
                        capabilities=server_capabilities,
                        serverInfo=server_info,
                        instructions=None,
                    )
                    _send_result(request_id, mcp_types.ServerResult(result))
                    initialized = True

                case mcp_types.PingRequest():
                    _send_result(request_id, mcp_types.ServerResult(mcp_types.EmptyResult()))

                case mcp_types.ListToolsRequest():
                    if not initialized:
                        raise RuntimeError("Received tools/list before initialization")
                    listed = loop.run_until_complete(mcp.list_tools())
                    result = mcp_types.ListToolsResult(tools=listed)
                    _send_result(request_id, mcp_types.ServerResult(result))

                case mcp_types.CallToolRequest(params=params):
                    if not initialized:
                        raise RuntimeError("Received tools/call before initialization")
                    output = loop.run_until_complete(mcp.call_tool(params.name, params.arguments or {}))
                    if isinstance(output, mcp_types.CallToolResult):
                        result = output
                    elif isinstance(output, tuple) and len(output) == 2:
                        content, structured = output
                        result = mcp_types.CallToolResult(
                            content=list(content), structuredContent=structured, isError=False
                        )
                    elif isinstance(output, dict):
                        result = mcp_types.CallToolResult(
                            content=[], structuredContent=output, isError=False
                        )
                    else:
                        result = mcp_types.CallToolResult(
                            content=list(output), structuredContent=None, isError=False
                        )
                    _send_result(request_id, mcp_types.ServerResult(result))

                case mcp_types.ListResourcesRequest():
                    result = mcp_types.ListResourcesResult(resources=[])
                    _send_result(request_id, mcp_types.ServerResult(result))

                case mcp_types.ListResourceTemplatesRequest():
                    result = mcp_types.ListResourceTemplatesResult(resourceTemplates=[])
                    _send_result(request_id, mcp_types.ServerResult(result))

                case mcp_types.ListPromptsRequest():
                    result = mcp_types.ListPromptsResult(prompts=[])
                    _send_result(request_id, mcp_types.ServerResult(result))

                case _:
                    _send_error(
                        request_id,
                        mcp_types.METHOD_NOT_FOUND,
                        f"Unsupported method: {message.root.method}",
                    )
        except Exception as exc:
            _send_error(request_id, mcp_types.INTERNAL_ERROR, str(exc))
finally:
    loop.close()
