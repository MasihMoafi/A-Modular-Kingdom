#!/usr/bin/env python
# coding: utf-8
import os
os.environ["TQDM_DISABLE"] = "1"  # Disable tqdm via env var
# Prefer offline-first HF/Transformers to avoid tool timeouts in restricted envs.
# Users can override these before starting the server.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import io

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.join(repo_root, "src")
sys.path.insert(0, project_root)
import json
import importlib
import glob
import subprocess
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Dict, List
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from memory.scoped_manager import ScopedMemoryManager
from memory.memory_config import MemoryScope

# Default MCP surface area: RAG + scoped memory.
# Extra tools are optional behind env flags to keep startup/tooling lean.
_EXPOSE_EXTRA_TOOLS = os.environ.get("MCP_EXPOSE_EXTRA_TOOLS", "0").lower() in ("1", "true", "yes", "y", "on")
_EXPOSE_RESOURCES = os.environ.get("MCP_EXPOSE_RESOURCES", "0").lower() in ("1", "true", "yes", "y", "on")

# Initialize FastMCP
mcp = FastMCP("unified_knowledge_agent_host")
scoped_memory = None

_LOG_STDERR = os.environ.get("MCP_LOG_STDERR", "0").lower() in ("1", "true", "yes", "y", "on")
_LOG_FILE = os.environ.get("MCP_LOG_FILE", "").strip()
_DEBUG_PROTOCOL = os.environ.get("MCP_DEBUG_PROTOCOL", "0").lower() in ("1", "true", "yes", "y", "on")


def _elog(msg: str) -> None:
    try:
        if _LOG_STDERR:
            sys.stderr.write(msg)
            sys.stderr.flush()
        if _LOG_FILE:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg)
    except Exception:
        pass


def _call_tool_safely(func, *args, **kwargs):
    """Run tool code while suppressing accidental stdout/stderr writes and catching crashes."""
    out_buf = StringIO()
    err_buf = StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            result = func(*args, **kwargs)
    except Exception as e:
        _elog(f"[HOST] Tool {func.__name__} crashed: {e}\n")
        return json.dumps({"error": f"{func.__name__} failed: {str(e)}"})

    leaked = out_buf.getvalue().strip()
    if leaked:
        _elog(f"[HOST] Suppressed stdout leak ({len(leaked)} chars)\n")
    leaked_err = err_buf.getvalue().strip()
    if leaked_err:
        _elog(f"[HOST] Suppressed stderr leak ({len(leaked_err)} chars)\n")
    return result


def _call_in_subprocess(module_name: str, function_name: str, payload: dict, timeout: int = 120) -> str:
    """Isolate unstable/native tool code so it can't kill the MCP transport."""
    runner = (
        "import json, importlib\n"
        f"module = importlib.import_module({module_name!r})\n"
        f"func = getattr(module, {function_name!r})\n"
        f"payload = {repr(payload)}\n"
        "result = func(**payload)\n"
        "print(result if isinstance(result, str) else json.dumps(result))\n"
    )
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"  # Use cached models, don't hit huggingface.co
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pp}" if existing_pp else project_root

    proc = subprocess.run(
        [sys.executable, '-c', runner],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    if proc.returncode != 0:
        return json.dumps({
            'success': False,
            'error': f'{module_name}.{function_name} failed',
            'stderr': proc.stderr[-1000:]
        })
    return proc.stdout.strip() or json.dumps({'success': False, 'error': 'No output from subprocess'})


try:
    # Clear stale Qdrant lock files from previous crashes
    import glob as _glob
    _mem_root = os.path.join(repo_root, ".modular_kingdom")
    for _lock in _glob.glob(os.path.join(_mem_root, "**", ".lock"), recursive=True):
        os.remove(_lock)
        _elog(f"[HOST] Removed stale lock: {_lock}\n")

    scoped_memory = ScopedMemoryManager(project_root=repo_root)
    # Pre-initialize all scopes to avoid race conditions on first access
    for _scope in MemoryScope:
        try:
            scoped_memory._get_instance(_scope)
        except Exception:
            pass
    _elog("[HOST] Scoped memory initialized successfully\n")
except Exception as e:
    _elog(f"[HOST] WARNING: Memory system disabled: {e}\n")
    scoped_memory = None

# Restore HTTP(S) proxy for RAG subprocesses (SOCKS breaks httpx)
from utils.proxy import restore_http as _restore_proxy
_restore_proxy()

@mcp.tool(
    name="save_fact",
    description="Save structured facts to the memory system with automatic processing and fact extraction from the provided content"
)
def save_fact(
    fact_data: Dict = Field(description="Dictionary containing 'content' key with the fact to save")
) -> str:
    if scoped_memory is None:
        return json.dumps({"error": "Scoped memory not initialized."})
    try:
        content_to_save = fact_data.get('content')
        if not content_to_save:
            return json.dumps({"error": "No 'content' field in fact JSON."})
        memory_id = scoped_memory.save(content_to_save)
        return json.dumps({"status": "success", "id": memory_id, "message": f"Fact saved: '{content_to_save}'"})
    except Exception as e:
        return json.dumps({"error": f"Error saving fact: {str(e)}"})

@mcp.tool(
    name="save_memory",
    description="Save content to scoped memory (global or project). Prefix with #global:rule, #global:pref, #persona, #project:context, or let system infer scope."
)
def save_direct_memory(
    content: str = Field(description="Text to save (with optional #scope:type prefix)"),
    scope: str = Field(default="", description="Explicit scope: 'global_rules', 'global_preferences', 'project_context', etc.")
) -> str:
    """Save content to scoped memory."""
    if scoped_memory is None:
        return json.dumps({"error": "Scoped memory not initialized."})
    try:
        # Parse scope from content prefix if present
        target_scope = None
        if scope:
            try:
                target_scope = MemoryScope(scope)
            except ValueError:
                pass
        
        # Check for prefix in content
        if not target_scope and content.startswith("#"):
            from memory.memory_config import MemoryConfig
            config = MemoryConfig()
            parsed_scope, cleaned_content = config.parse_scope_prefix(content)
            if parsed_scope:
                target_scope = parsed_scope
                content = cleaned_content
        
        memory_id = scoped_memory.save(content, scope=target_scope)
        scope_name = target_scope.value if target_scope else "inferred"
        return json.dumps({"status": "success", "scope": scope_name, "id": memory_id})
    except Exception as e:
        return json.dumps({"error": f"Error saving scoped memory: {str(e)}"})

@mcp.tool(
    name="set_global_rule",
    description="Set a permanent global rule that persists across all projects and sessions"
)
def set_global_rule(
    rule: str = Field(description="The rule/instruction to save globally")
) -> str:
    """Convenience tool to save global rules."""
    if scoped_memory is None:
        return json.dumps({"error": "Scoped memory not initialized."})
    try:
        memory_id = scoped_memory.save(rule, scope=MemoryScope.GLOBAL_RULES)
        return json.dumps({"status": "success", "scope": "global_rules", "id": memory_id})
    except Exception as e:
        return json.dumps({"error": f"Error setting global rule: {str(e)}"})

@mcp.tool(
    name="delete_memory",
    description="Delete a specific memory from the memory system using its unique identifier"
)
def delete_memory(
    memory_id: str = Field(description="The unique ID of the memory to delete")
) -> str:
    """Delete a memory by its ID (searches all scopes)."""
    if scoped_memory is None:
        return json.dumps({"error": "Scoped memory not initialized."})
    try:
        for scope in MemoryScope:
            try:
                scoped_memory.delete(memory_id, scope)
                return json.dumps({"status": "success", "message": f"Memory {memory_id} deleted from {scope.value}"})
            except KeyError:
                continue  # Not found in this scope
            except Exception as e:
                sys.stderr.write(f"[HOST] Delete error in {scope.value}: {e}\n")
                continue  # Log but try next scope
        return json.dumps({"error": f"Memory {memory_id} not found in any scope"})
    except Exception as e:
        return json.dumps({"error": f"Error deleting memory: {str(e)}"})

@mcp.tool(
    name="search_memories",
    description="Search scoped memories with priority (global rules → preferences → personas → project). Returns scope metadata."
)
def search_memories(
    query: str = Field(description="Search query"),
    top_k: int = Field(default=3, description="Results per scope"),
    scope_filter: str = Field(default="", description="Optional: search only specific scope")
) -> str:
    if scoped_memory is None:
        return json.dumps([{"error": "Scoped memory not initialized."}])
    try:
        scopes_to_search = None
        if scope_filter:
            try:
                scopes_to_search = [MemoryScope(scope_filter)]
            except ValueError:
                pass
        
        results = scoped_memory.search(query, k=top_k, scopes=scopes_to_search)
        formatted = [
            {
                "content": res["content"],
                "scope": res.get("metadata", {}).get("scope", "unknown"),
                "id": res["id"]
            }
            for res in results
        ]
        return json.dumps(formatted)
    except Exception as e:
        return json.dumps([{"error": f"Error searching scoped memories: {str(e)}"}])

@mcp.tool(
    name="query_knowledge_base",
    description="Use RAG to explore codebases efficiently without loading entire files into context. Answers 'how does X work?', 'what's in folder Y?', architectural questions. Works on .py, .md, .ipynb, .js, .ts, etc. 98% less context than reading files. Use BEFORE reading files for exploration tasks. Supports v1, v2, v3 (v2 recommended)."
)
def query_knowledge_base(
    query: str = Field(description="The search query for the knowledge base"),
    version: str = Field(default="v2", description="RAG version to use: 'v1', 'v2', or 'v3'"),
    doc_path: str = Field(default="", description="Optional path to a specific documents directory"),
    files: List[str] = Field(default=[], description="Optional list of specific file paths to index and search")
) -> str:
    """
    A tool to query the external knowledge base (RAG system) with selectable versions.
    """
    _elog(f"[HOST] RAG tool called with query: '{query[:20]}...', version: {version}, path: '{doc_path}', files: {len(files) if files else 0}\n")

    try:
        # Smart defaults:
        # - If neither `doc_path` nor `files` were provided, index this repo by default.
        # - Treat relative paths as relative to the repo root (not process CWD).
        repo_default = repo_root
        shortcut_words = {"desktop", "documents", "downloads"}

        if (not doc_path) and (not files):
            doc_path = repo_default

        if doc_path and isinstance(doc_path, str):
            dp = doc_path.strip()
            if dp and (not os.path.isabs(dp)) and (not dp.startswith("~")) and (dp.lower() not in shortcut_words):
                doc_path = os.path.join(repo_default, dp)

        if files:
            normalized = []
            for f in files:
                if not f or not isinstance(f, str):
                    continue
                fp = f.strip()
                if not fp:
                    continue
                if (not os.path.isabs(fp)) and (not fp.startswith("~")):
                    fp = os.path.join(repo_default, fp)
                normalized.append(fp)
            files = normalized

        # Map version to module name
        module_map = {
            "v1": "rag.fetch",
            "v2": "rag.fetch_2",
            "v3": "rag.fetch_3"
        }

        module_name = module_map.get(version)
        if not module_name:
            return json.dumps({"error": f"Invalid RAG version '{version}'. Must be 'v1', 'v2', or 'v3'."})

        # IMPORTANT: Keep RAG in-process so models/pipelines can be cached across calls.
        # The previous per-call subprocess approach was reliable for isolation, but it
        # makes cold-start latency huge and frequently exceeds client tool timeouts.
        _elog(f"[HOST] Calling fetchExternalKnowledge from {module_name} in-process\n")

        payload = {"query": query}
        if files and len(files) > 0:
            payload["file_list"] = files
        elif doc_path and isinstance(doc_path, str):
            payload["doc_path"] = doc_path

        module = importlib.import_module(module_name)
        func = getattr(module, "fetchExternalKnowledge", None)
        if not callable(func):
            return json.dumps({"error": f"Module '{module_name}' has no callable fetchExternalKnowledge(query, ...)"})

        def _run_rag():
            # Keep signature explicit to avoid silent arg mismatches.
            if "file_list" in payload:
                return func(query=payload["query"], file_list=payload["file_list"])
            if "doc_path" in payload:
                return func(query=payload["query"], doc_path=payload["doc_path"])
            return func(query=payload["query"])

        results = _call_tool_safely(_run_rag)

        _elog(f"[HOST] RAG ({version}) returned {len(str(results)) if results else 0} chars\n")

        return json.dumps({"result": results})

    except (ModuleNotFoundError, AttributeError) as e:
        error_msg = f"Could not find or load 'fetchExternalKnowledge' function in module for version '{version}': {str(e)}"
        _elog(f"[HOST] RAG error: {error_msg}\n")
        return json.dumps({"error": error_msg})
    except Exception as e:
        import traceback
        error_msg = f"Error querying knowledge base on host with version {version}: {str(e)}"
        _elog(f"[HOST] RAG error: {error_msg}\n")
        _elog(f"[HOST] Traceback: {traceback.format_exc()}\n")
        return json.dumps({"error": error_msg})

@mcp.tool(
    name="list_all_memories",
    description="Retrieve and list all saved memories from the memory system for review and management"
)
def list_all_memories() -> str:
    """Retrieve and list all saved memories from the memory system for review and management"""
    _elog("[HOST] list_all_memories called\n")
    if not scoped_memory:
        return json.dumps({"error": "Memory system not initialized"})
    
    try:
        all_memories = {}
        
        for scope in MemoryScope:
            try:
                memories = scoped_memory.list_all(scope)
                all_memories[scope.value] = memories
            except Exception as e:
                _elog(f"[HOST] Error listing memories for scope {scope}: {e}\n")
        
        return json.dumps(all_memories, indent=2)
    except Exception as e:
        _elog(f"[HOST] Critical error in list_all_memories: {e}\n")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})

@mcp.tool(
    name="health_check",
    description="Check MCP server health and component status"
)
def health_check() -> str:
    """Return status of all MCP components."""
    status = {
        "server": "ok",
        "memory": "ok" if scoped_memory else "disabled",
    }
    # Test RAG import
    try:
        from rag.fetch_2 import _DEPENDENCIES_OK
        status["rag"] = "ok" if _DEPENDENCIES_OK else "deps_missing"
    except Exception as e:
        status["rag"] = f"error: {str(e)}"
    return json.dumps(status)

if _EXPOSE_EXTRA_TOOLS:
    from tools.code_exec import run_code
    from tools.web_search import perform_web_search

    @mcp.tool(
        name="code_execute",
        description="Execute Python code in a sandboxed subprocess and return stdout/stderr"
    )
    def code_execute(
        code: str = Field(description="Python code to execute"),
        timeout_seconds: int = Field(default=15, description="Execution timeout in seconds")
    ) -> str:
        try:
            return _call_tool_safely(run_code, code=code, timeout_seconds=timeout_seconds)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    @mcp.tool(
        name="analyze_media",
        description="Analyze image/video files with a local multimodal model via Ollama (e.g., gemma3:4b)"
    )
    def analyze_media(
        model: str = Field(default="gemma3:4b", description="Ollama model id to use"),
        paths: list[str] = Field(description="List of absolute file paths to media files")
    ) -> str:
        try:
            from tools.vision import analyze_media_with_ollama
            return _call_tool_safely(analyze_media_with_ollama, model=model, paths=paths)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    @mcp.tool(
        name="web_search",
        description="Search the web using DuckDuckGo and return relevant results"
    )
    def web_search(
        query: str = Field(description="The search query")
    ) -> str:
        try:
            return _call_tool_safely(perform_web_search, query)
        except Exception as e:
            return json.dumps({"error": f"Web search failed: {str(e)}"})

    @mcp.tool(
        name="browse_web",
        description="Control a persistent browser. Actions: navigate (open URL), click (by CSS selector or x,y coords), type (text into selector or focused element), press (key like Enter/Tab), screenshot (capture current page), wait (wait ms for page to load), get_text (extract page text), close. Browser stays open between calls."
    )
    def browse_web(
        action: str = Field(description="Action: navigate, click, type, press, screenshot, wait, get_text, close"),
        url: str = Field(default="", description="URL for navigate action"),
        selector: str = Field(default="", description="CSS selector for click/type"),
        text: str = Field(default="", description="Text for type action"),
        x: int = Field(default=None, description="X coordinate for click"),
        y: int = Field(default=None, description="Y coordinate for click"),
        key: str = Field(default="", description="Key for press action (e.g. Enter, Tab)"),
        ms: int = Field(default=2000, description="Milliseconds for wait action"),
        headless: bool = Field(default=False, description="Run browser headless")
    ) -> str:
        """Control browser with discrete actions. Browser persists between calls."""
        try:
            from tools.browser_agent_playwright import browser_action
            return browser_action(action=action, url=url, selector=selector,
                                  text=text, x=x, y=y, key=key, headless=headless, ms=ms)
        except Exception as e:
            return json.dumps({"error": f"Browser action failed: {str(e)}"})

    @mcp.tool(
        name="text_to_speech",
        description="Convert text to speech using various TTS engines. Can play audio directly or save to file."
    )
    def tts_tool(
        text: str = Field(description="Text to convert to speech"),
        engine: str = Field(default="pyttsx3", description="TTS engine to use: pyttsx3, gtts, or kokoro"),
        voice: str = Field(default="", description="Voice name/ID to use (engine-specific)"),
        speed: float = Field(default=200, description="Speech rate for pyttsx3"),
        play_audio: bool = Field(default=True, description="Whether to play audio immediately")
    ) -> str:
        """Convert text to speech and optionally play it."""
        try:
            kwargs = {
                "engine": engine,
                "speed": speed,
                "play_audio": play_audio
            }
            if voice:
                kwargs["voice"] = voice

            return _call_in_subprocess("tools.tts", "text_to_speech", {"text": text, **kwargs}, timeout=60)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    @mcp.tool(
        name="speech_to_text",
        description="Convert speech to text using microphone recording or existing audio file"
    )
    def stt_tool(
        duration: int = Field(default=5, description="Recording duration in seconds (ignored if file_path provided)"),
        engine: str = Field(default="whisper", description="STT engine: whisper or speech_recognition"),
        model: str = Field(default="base", description="Model size for Whisper: tiny, base, small, medium, large"),
        language: str = Field(default="", description="Language code (e.g., en, es, fr)"),
        file_path: str = Field(default="", description="Path to existing audio file to transcribe")
    ) -> str:
        """Record audio and convert it to text, or transcribe an existing audio file."""
        try:
            kwargs = {
                "duration": duration,
                "engine": engine,
                "model": model
            }
            if language:
                kwargs["language"] = language
            if file_path:
                kwargs["file_path"] = file_path

            return _call_in_subprocess("tools.stt", "speech_to_text", kwargs, timeout=max(60, duration + 45))
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

if _EXPOSE_RESOURCES:
    # --- RESOURCES for @ functionality ---
    @mcp.resource("docs://documents", mime_type="application/json")
    def list_documents() -> str:
        """Returns list of available document IDs for @ mentions."""
        try:
            files_dir = os.path.join(project_root, "rag", "files")
            if not os.path.exists(files_dir):
                return json.dumps([])

            pattern = os.path.join(files_dir, "*")
            files = glob.glob(pattern)
            doc_ids = [os.path.basename(f) for f in files if os.path.isfile(f)]
            return json.dumps(doc_ids)
        except Exception:
            return json.dumps([])

    @mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")
    def get_document_content(doc_id: str) -> str:
        """Returns the content of a specific document."""
        try:
            files_dir = os.path.join(project_root, "rag", "files")
            file_path = os.path.join(files_dir, doc_id)

            if not os.path.exists(file_path):
                raise ValueError(f"Document {doc_id} not found")

            if doc_id.lower().endswith(".pdf"):
                import fitz

                doc = fitz.open(file_path)
                text = "".join(page.get_text() for page in doc)
                doc.close()
                return text

            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: Could not retrieve content for {doc_id}: {str(e)}"

def _warm_rag_v2_background() -> None:
    # Important: never print to stdout; MCP stdio transport uses stdout for protocol messages.
    try:
        from rag import fetch_2 as _rag_v2
        _ = _rag_v2.get_rag_pipeline()
        _elog("[HOST] RAG v2 warmed (background)\n")
    except Exception as _e:
        _elog(f"[HOST] RAG warm skipped/failed: {_e}\n")
    finally:
        pass


# Do not block MCP initialization on model loads/indexing.
# Default is OFF because warm-up can be slow and can also require network (e.g. vector DB).
if os.environ.get("MCP_WARM_RAG_ON_START", "0").lower() in ("1", "true", "yes", "y", "on"):
    try:
        import threading

        threading.Thread(target=_warm_rag_v2_background, daemon=True).start()
    except Exception as _e:
        _elog(f"[HOST] RAG warm thread failed: {_e}\n")

_elog("[HOST] Entering MCP stdio event loop\n")

import asyncio
from importlib.metadata import version as _pkg_version

import mcp.types as _mcp_types
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS as _SUPPORTED_PROTOCOL_VERSIONS

# This server does NOT use the SDK's stdio transport because stdin async-IO is
# unreliable in this environment. We run a tiny synchronous JSON-RPC loop and
# delegate the actual tool implementations to FastMCP.
_server_info = _mcp_types.Implementation(
    name="unified_knowledge_agent_host",
    version=_pkg_version("mcp"),
)
_server_capabilities = _mcp_types.ServerCapabilities(
    tools=_mcp_types.ToolsCapability(listChanged=False),
    resources=_mcp_types.ResourcesCapability(subscribe=False, listChanged=False),
    prompts=_mcp_types.PromptsCapability(listChanged=False),
)

_initialized = False

_use_content_length_framing = False


def _write_jsonrpc(message: _mcp_types.JSONRPCMessage) -> None:
    payload = message.model_dump_json(by_alias=True, exclude_none=True)
    if _use_content_length_framing:
        data = payload.encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    else:
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()


def _read_jsonrpc_payload() -> str | None:
    """
    Support both:
    - newline-delimited JSON (legacy stdio clients)
    - Content-Length framed JSON (rmcp / LSP-style transport)
    """
    global _use_content_length_framing

    first = sys.stdin.buffer.readline()
    if not first:
        return None
    if _DEBUG_PROTOCOL:
        _elog(f"[HOST][proto] first_line={first[:120]!r}\n")

    lower = first.lower()
    if lower.startswith(b"content-length:"):
        _use_content_length_framing = True

        header_lines = [first]
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            header_lines.append(line)
            if line in (b"\r\n", b"\n"):
                break
        if _DEBUG_PROTOCOL:
            _elog(f"[HOST][proto] header_lines={len(header_lines)}\n")

        content_length = None
        for raw in header_lines:
            try:
                text = raw.decode("ascii", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            if ":" not in text:
                continue
            k, v = text.split(":", 1)
            if k.strip().lower() == "content-length":
                try:
                    content_length = int(v.strip())
                except ValueError:
                    content_length = None

        if content_length is None or content_length < 0:
            if _DEBUG_PROTOCOL:
                _elog("[HOST][proto] missing/invalid content-length\n")
            return None

        remaining = content_length
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = sys.stdin.buffer.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        return body.decode("utf-8", errors="replace")

    # Newline-delimited JSON
    return first.decode("utf-8", errors="replace")


def _send_error(req_id: _mcp_types.RequestId, code: int, message: str) -> None:
    err = _mcp_types.JSONRPCError(
        jsonrpc="2.0",
        id=req_id,
        error=_mcp_types.ErrorData(code=code, message=message),
    )
    _write_jsonrpc(_mcp_types.JSONRPCMessage(err))


def _send_result(req_id: _mcp_types.RequestId, result: _mcp_types.ServerResult) -> None:
    resp = _mcp_types.JSONRPCResponse(
        jsonrpc="2.0",
        id=req_id,
        result=result.model_dump(by_alias=True, mode="json", exclude_none=True),
    )
    _write_jsonrpc(_mcp_types.JSONRPCMessage(resp))


_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

try:
    while True:
        payload = _read_jsonrpc_payload()
        if payload is None:
            break

        try:
            msg = _mcp_types.JSONRPCMessage.model_validate_json(payload)
        except Exception:
            # Can't respond without a request id.
            if _DEBUG_PROTOCOL:
                _elog(f"[HOST][proto] failed to parse jsonrpc payload (len={len(payload) if payload else 0})\n")
            continue

        if not isinstance(msg.root, _mcp_types.JSONRPCRequest):
            # Notifications/responses from the client are ignored (Codex sends initialized notification).
            continue

        req_id = msg.root.id
        if _DEBUG_PROTOCOL:
            _elog(f"[HOST][proto] request method={getattr(msg.root,'method',None)!r} id={req_id!r}\n")
        try:
            req = _mcp_types.ClientRequest.model_validate(
                msg.root.model_dump(by_alias=True, mode="json", exclude_none=True)
            )
        except Exception:
            _send_error(req_id, _mcp_types.INVALID_PARAMS, "Invalid request parameters")
            continue

        try:
            match req.root:
                case _mcp_types.InitializeRequest(params=params):
                    requested = params.protocolVersion
                    protocol = (
                        requested if requested in _SUPPORTED_PROTOCOL_VERSIONS else _mcp_types.LATEST_PROTOCOL_VERSION
                    )
                    init = _mcp_types.InitializeResult(
                        protocolVersion=protocol,
                        capabilities=_server_capabilities,
                        serverInfo=_server_info,
                        instructions=None,
                    )
                    _send_result(req_id, _mcp_types.ServerResult(init))
                    _initialized = True

                case _mcp_types.PingRequest():
                    _send_result(req_id, _mcp_types.ServerResult(_mcp_types.EmptyResult()))

                case _mcp_types.ListToolsRequest():
                    if not _initialized:
                        raise RuntimeError("Received tools/list before initialization")
                    tools = _loop.run_until_complete(mcp.list_tools())
                    _send_result(req_id, _mcp_types.ServerResult(_mcp_types.ListToolsResult(tools=tools)))

                case _mcp_types.CallToolRequest(params=params):
                    if not _initialized:
                        raise RuntimeError("Received tools/call before initialization")
                    args = params.arguments or {}
                    out = _loop.run_until_complete(mcp.call_tool(params.name, args))
                    # FastMCP returns:
                    # - CallToolResult (already in protocol form), or
                    # - list[ContentBlock] for unstructured output, or
                    # - tuple(list[ContentBlock], dict) for structured output, or
                    # - dict for structured output.
                    if isinstance(out, _mcp_types.CallToolResult):
                        result = out
                    elif isinstance(out, tuple) and len(out) == 2:
                        content, structured = out
                        result = _mcp_types.CallToolResult(
                            content=list(content),
                            structuredContent=structured,
                            isError=False,
                        )
                    elif isinstance(out, dict):
                        result = _mcp_types.CallToolResult(content=[], structuredContent=out, isError=False)
                    else:
                        result = _mcp_types.CallToolResult(content=list(out), structuredContent=None, isError=False)
                    _send_result(req_id, _mcp_types.ServerResult(result))

                case _mcp_types.ListResourcesRequest():
                    resources = _loop.run_until_complete(mcp.list_resources())
                    _send_result(req_id, _mcp_types.ServerResult(_mcp_types.ListResourcesResult(resources=resources)))

                case _mcp_types.ReadResourceRequest(params=params):
                    contents = _loop.run_until_complete(mcp.read_resource(params.uri))
                    _send_result(req_id, _mcp_types.ServerResult(_mcp_types.ReadResourceResult(contents=contents)))

                case _mcp_types.ListPromptsRequest():
                    prompts = _loop.run_until_complete(mcp.list_prompts())
                    _send_result(req_id, _mcp_types.ServerResult(_mcp_types.ListPromptsResult(prompts=prompts)))

                case _mcp_types.GetPromptRequest(params=params):
                    prompt = _loop.run_until_complete(mcp.get_prompt(params.name, params.arguments))
                    _send_result(req_id, _mcp_types.ServerResult(prompt))

                case _mcp_types.ListResourceTemplatesRequest():
                    templates = _loop.run_until_complete(mcp.list_resource_templates())
                    _send_result(
                        req_id, _mcp_types.ServerResult(_mcp_types.ListResourceTemplatesResult(resourceTemplates=templates))
                    )

                case _:
                    _send_error(req_id, _mcp_types.METHOD_NOT_FOUND, f"Unsupported method: {msg.root.method}")
        except Exception as e:
            _send_error(req_id, _mcp_types.INTERNAL_ERROR, str(e))
finally:
    try:
        _loop.close()
    except Exception:
        pass
