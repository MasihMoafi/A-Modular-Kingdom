#!/usr/bin/env python
# coding: utf-8

import os
import sys
import asyncio
import nest_asyncio
import traceback
import json
import argparse
import warnings
import re
from typing import List, Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

# --- Initial Setup ---
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

try:
    # Prefer classic namespace to avoid deprecation noise in langchain>=1.0.
    from langchain_classic.memory import ConversationBufferWindowMemory
except Exception:
    from langchain.memory import ConversationBufferWindowMemory
from mcp import ClientSession, stdio_client, StdioServerParameters
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

# --- Get the absolute path to the host.py script ---
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
HOST_PATH = os.path.join(AGENT_DIR, "host.py")
REPO_ROOT = os.path.dirname(os.path.dirname(AGENT_DIR))
from memory.path_policy import resolve_memory_base

DEFAULT_MEMORY_BASE = resolve_memory_base(project_root=REPO_ROOT)
os.environ.setdefault("MEMORY_BASE_PATH", DEFAULT_MEMORY_BASE)

nest_asyncio.apply()

# Runtime-selected LLM backend (defaults to local Ollama).
LLM_MODEL = "qwen3:8b"
genai_client = None
types = None
ollama = None


def initialize_llm(provider: str = "ollama", model: Optional[str] = None):
    """Initialize LLM backend.

    Defaults to local Ollama to keep CLI usage simple and offline-first.
    Gemini is available only when explicitly requested.
    """
    global LLM_MODEL, genai_client, types, ollama

    selected = (provider or "ollama").strip().lower()
    if selected == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Either set it and use --provider gemini, "
                "or run with --provider ollama."
            )
        from google import genai
        from google.genai import types as gemini_types

        genai_client = genai.Client(api_key=api_key)
        types = gemini_types
        LLM_MODEL = "gemini"
        ollama = None
        print("Using Gemini API")
        return

    # Default path: local Ollama.
    import ollama as _ollama
    ollama = _ollama

    LLM_MODEL = (model or os.getenv("OLLAMA_MODEL") or "qwen3:8b").strip()
    print(f"Using Ollama ({LLM_MODEL})")

class DocumentCompleter(Completer):
    def __init__(self):
        self.resources = []
        self.commands = [
            '/memory', '/help', '/tools', '/files', '/browser_automation', '/rag',
            '/read', '/write', '/mkmd', '/mkdocx', '/exec', '/web', '/crawl',
            '/memory_status', '/memory_migrate', '/memory_cleanup', '/quit'
        ]
    
    def update_resources(self, resources: List[str]):
        self.resources = resources
    
    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        
        # Handle slash commands
        if text_before_cursor.startswith('/'):
            prefix = text_before_cursor[1:]
            for cmd in self.commands:
                if cmd[1:].startswith(prefix.lower()):
                    yield Completion(
                        cmd[1:],
                        start_position=-len(prefix),
                        display=cmd,
                        display_meta="Command",
                    )
            return
        
        # Handle @ mentions
        if "@" in text_before_cursor:
            last_at_pos = text_before_cursor.rfind("@")
            prefix = text_before_cursor[last_at_pos + 1:]
            
            for resource_id in self.resources:
                if resource_id.lower().startswith(prefix.lower()):
                    yield Completion(
                        resource_id,
                        start_position=-len(prefix),
                        display=resource_id,
                        display_meta="Document",
                    )


def _extract_text_content(result) -> str:
    try:
        if not result or not getattr(result, "contents", None):
            return ""
        return getattr(result.contents[0], "text", "") or ""
    except Exception:
        return ""


def _extract_tool_text(result) -> str:
    try:
        if result and getattr(result, "content", None):
            first = result.content[0]
            text = getattr(first, "text", "")
            if text:
                return text
    except Exception:
        pass
    try:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            if isinstance(structured, str):
                return structured
            return json.dumps(structured)
    except Exception:
        pass
    return ""


def _normalize_memory_listing(payload) -> list[dict]:
    normalized: list[dict] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"id": "N/A", "content": item, "scope": "unknown"})
        return normalized

    if isinstance(payload, dict):
        # Host returns {scope_name: [memories...]}.
        for scope_name, items in payload.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    mem = dict(item)
                    mem.setdefault("scope", scope_name)
                    normalized.append(mem)
                elif isinstance(item, str):
                    normalized.append({"id": "N/A", "content": item, "scope": scope_name})
        return normalized

    return normalized


async def _list_docs_from_resource(session: ClientSession) -> list[str]:
    docs = await session.read_resource("docs://documents")
    raw = _extract_text_content(docs)
    data = json.loads(raw) if raw else []
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _list_docs_from_filesystem() -> list[str]:
    files_dir = os.path.join(os.path.dirname(AGENT_DIR), "rag", "files")
    if not os.path.isdir(files_dir):
        return []
    return sorted(
        name for name in os.listdir(files_dir)
        if os.path.isfile(os.path.join(files_dir, name))
    )


async def _list_documents(session: ClientSession) -> list[str]:
    try:
        return await _list_docs_from_resource(session)
    except Exception:
        return _list_docs_from_filesystem()


async def _read_document_content(session: ClientSession, doc_id: str) -> str:
    try:
        doc_resource = await session.read_resource(f"docs://documents/{doc_id}")
        return _extract_text_content(doc_resource)
    except Exception:
        files_dir = os.path.join(os.path.dirname(AGENT_DIR), "rag", "files")
        file_path = os.path.join(files_dir, doc_id)
        if not os.path.isfile(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


def _resolve_user_path(path_str: str) -> str:
    expanded = os.path.expanduser(path_str.strip())
    if os.path.isabs(expanded):
        return os.path.realpath(expanded)
    return os.path.realpath(os.path.join(REPO_ROOT, expanded))


def _is_safe_write_target(path_str: str) -> bool:
    target = os.path.realpath(path_str)
    allowed_roots = [os.path.realpath(REPO_ROOT), os.path.realpath("/tmp")]
    return any(target == root or target.startswith(root + os.sep) for root in allowed_roots)


async def _prompt_multiline(prompt_session: PromptSession, header: str) -> str:
    print(header)
    print("Finish with a single line: .done")
    lines = []
    while True:
        try:
            line = await prompt_session.prompt_async("")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == ".done":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _capabilities_hint() -> str:
    return (
        "I can do that here. Use:\n"
        "- /write <path> <content>      (create/overwrite files)\n"
        "- /mkmd <path> <content>       (create markdown)\n"
        "- /mkdocx <path> <content>     (create docx)\n"
        "- /read <path>                 (read files)\n"
        "- /exec <python>               (run Python in sandbox)\n"
        "- /web <query>                 (web search)\n"
        "- /crawl <query>               (search + crawl + RAG)\n"
        "- /rag \"<query>\" [path] [v2]  (local RAG search)\n"
        "- /browser_automation          (browser task)\n"
        "- /memory_status               (see memory stores)\n"
        "- /memory_migrate              (migrate legacy memories)\n"
        "- /memory_cleanup              (archive legacy stores)\n"
        "You can also say: create docs/notes.md with content ... or create report.docx with content ..."
    )


def _is_capability_question(text: str) -> bool:
    t = text.lower()
    asks_ability = any(x in t for x in ["can you", "are you able", "do you have", "can u"])
    touches_ops = any(
        x in t
        for x in [
            "write file",
            "create file",
            "generate file",
            "read file",
            "search web",
            "internet",
            "run code",
            "execute code",
            "browser",
            "memory",
            "crawl",
        ]
    )
    return asks_ability and touches_ops


def _parse_nl_file_request(text: str):
    lower = text.lower()
    if not any(x in lower for x in ["create", "write", "generate", "make", "save"]):
        return None

    path_match = re.search(r'["\']([^"\']+\.(?:md|markdown|docx))["\']', text, re.IGNORECASE)
    if path_match:
        raw_path = path_match.group(1).strip()
    else:
        plain = re.search(r'((?:~|/|\./|\.\./)?[A-Za-z0-9_\-./]+?\.(?:md|markdown|docx))\b', text, re.IGNORECASE)
        raw_path = plain.group(1).strip() if plain else ""
    if not raw_path:
        return None

    suffix = os.path.splitext(raw_path)[1].lower()
    if suffix not in (".md", ".markdown", ".docx"):
        return None

    title = ""
    title_match = re.search(r'(?:title|titled)\s+["\']([^"\']+)["\']', text, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()

    content = ""
    content_match = re.search(
        r'(?:with|containing)\s+(?:content|text|body)\s*[:=]?\s*(.+)$',
        text,
        re.IGNORECASE,
    )
    if content_match:
        content = content_match.group(1).strip()
    elif ":" in text:
        parts = text.split(":", 1)
        if len(parts) == 2:
            content = parts[1].strip()

    if len(content) >= 2 and ((content[0] == content[-1] == '"') or (content[0] == content[-1] == "'")):
        content = content[1:-1]

    return {"path": raw_path, "suffix": suffix, "content": content, "title": title}


def _split_path_and_content(arg_text: str) -> tuple[str, str]:
    raw = (arg_text or "").strip()
    if not raw:
        return "", ""
    for marker in ("<<<", ":::"):
        if marker in raw:
            path_part, content_part = raw.split(marker, 1)
            return path_part.strip(), content_part.strip()
    parts = raw.split(maxsplit=1)
    path_part = parts[0]
    content_part = parts[1].strip() if len(parts) > 1 else ""
    return path_part, content_part


def _parse_json_dict(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        # Tolerate noisy tool output by extracting the first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                payload = json.loads(candidate)
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}
        return {}


def _extract_urls_from_web_payload(payload: dict) -> list[str]:
    urls = []
    items = payload.get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if url.startswith("http://") or url.startswith("https://"):
                urls.append(url)
    if not urls:
        for match in re.finditer(r"https?://[^\s)]+", str(payload.get("results", ""))):
            urls.append(match.group(0).rstrip(".,;"))
    deduped = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


async def _web_crawl_rag_context(
    session: ClientSession,
    available_tools: set[str],
    query: str,
    version: str = "v2",
    max_pages: int = 5,
    max_depth: int = 1,
) -> tuple[str, list[str]]:
    web_text = ""
    rag_text = ""
    crawled_files: list[str] = []

    if "web_search" not in available_tools and available_tools:
        return "", []

    search_result = await session.call_tool("web_search", {"query": query})
    search_payload = _parse_json_dict(_extract_tool_text(search_result))
    if search_payload.get("error"):
        return f"\n--- External Knowledge (Web) ---\n{search_payload.get('error')}", []

    web_text = str(search_payload.get("results", "")).strip()
    urls = _extract_urls_from_web_payload(search_payload)

    if ("crawl_web" in available_tools or not available_tools) and (urls or query):
        crawl_params = {
            "query": query if not urls else "",
            "urls": urls[:max_pages],
            "max_pages": max_pages,
            "max_depth": max_depth,
            "same_domain_only": True,
        }
        crawl_result = await session.call_tool("crawl_web", crawl_params)
        crawl_payload = _parse_json_dict(_extract_tool_text(crawl_result))
        if not crawl_payload.get("error"):
            files = crawl_payload.get("files", [])
            if isinstance(files, list):
                crawled_files = [str(x) for x in files if str(x).strip()]

    if crawled_files and ("query_knowledge_base" in available_tools or not available_tools):
        rag_result = await session.call_tool(
            "query_knowledge_base",
            {"query": query, "version": version, "files": crawled_files[:max_pages]},
        )
        rag_payload = _parse_json_dict(_extract_tool_text(rag_result))
        rag_text = str(rag_payload.get("result", "")).strip()

    blocks = []
    if web_text:
        blocks.append(f"--- External Knowledge (Web Search) ---\n{web_text[:6000]}")
    if rag_text:
        blocks.append(f"--- External Knowledge (Crawled Web via RAG) ---\n{rag_text}")
    return ("\n\n" + "\n\n".join(blocks)) if blocks else "", crawled_files

async def main(think_level=None):
    print("--- Intelligent Agent ---")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", HOST_PATH],
        env=dict(os.environ),
    )
    
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available_tools = set()
            try:
                tool_result = await session.list_tools()
                available_tools = {tool.name for tool in getattr(tool_result, "tools", [])}
            except Exception:
                available_tools = set()
            
            # Setup prompt_toolkit with dropdown
            completer = DocumentCompleter()
            kb = KeyBindings()
            
            @kb.add("@")
            def _(event):
                buffer = event.app.current_buffer
                buffer.insert_text("@")
                if buffer.document.is_cursor_at_the_end:
                    buffer.start_completion(select_first=False)
                    
            @kb.add("/")
            def _(event):
                buffer = event.app.current_buffer
                if buffer.document.is_cursor_at_the_end and not buffer.text:
                    buffer.insert_text("/")
                    buffer.start_completion(select_first=False)
                else:
                    buffer.insert_text("/")


            # Enter: if line ends with '\\', insert newline instead of sending
            @kb.add("enter")
            def _(event):
                buf = event.app.current_buffer
                text = buf.text
                if text.endswith("\\") and buf.document.is_cursor_at_the_end:
                    # replace trailing backslash with newline
                    buf.delete_before_cursor(count=1)
                    buf.insert_text("\n")
                else:
                    buf.validate_and_handle()
            
            prompt_session = PromptSession(
                completer=completer,
                key_bindings=kb,
                style=Style.from_dict({
                    "prompt": "#aaaaaa",
                    "completion-menu.completion": "bg:#222222 #ffffff",
                    "completion-menu.completion.current": "bg:#444444 #ffffff",
                }),
                complete_while_typing=True,
                complete_in_thread=True,
            )
            interactive_input = sys.stdin.isatty()
            # Short-term memory (windowed) with fixed k=50
            stm = ConversationBufferWindowMemory(k=50, return_messages=False)
            
            # Load available documents for dropdown
            try:
                doc_list = await _list_documents(session)
                completer.update_resources(doc_list)
                print(f"\nLoaded {len(doc_list)} documents for @ completion")
            except Exception as e:
                print(f"Could not load document list: {e}")
            
            print("\nAgent is ready. Type 'exit' to quit. Use @ to see document dropdown.")

            while True:
                try:
                    try:
                        user_input = await prompt_session.prompt_async("\n> ")
                    except (EOFError, KeyboardInterrupt):
                        print("\nExiting...")
                        break
                    if user_input.lower() in ('exit', '/exit', '/quit'):
                        break
                    if not user_input.strip():
                        continue
                    if _is_capability_question(user_input):
                        print(_capabilities_hint())
                        continue

                    nl_file_req = None if user_input.startswith('/') else _parse_nl_file_request(user_input)
                    if nl_file_req:
                        try:
                            path = nl_file_req["path"]
                            suffix = nl_file_req["suffix"]
                            title = nl_file_req.get("title", "")
                            body = nl_file_req.get("content", "")
                            if not body:
                                if not interactive_input:
                                    print("File operation error: missing content in non-interactive mode.")
                                    continue
                                body = await _prompt_multiline(prompt_session, f"Enter content for {path}")

                            if suffix == ".docx":
                                tool_name = "create_docx"
                                params = {"path": path, "title": title, "body": body, "overwrite": True}
                            elif suffix in (".md", ".markdown"):
                                tool_name = "create_markdown"
                                params = {"path": path, "title": title, "body": body, "overwrite": True}
                            else:
                                tool_name = "write_file"
                                params = {"path": path, "content": body, "overwrite": True}

                            if tool_name not in available_tools and available_tools:
                                print(f"{tool_name} tool is unavailable on this host.")
                                continue

                            result = await session.call_tool(tool_name, params)
                            raw = _extract_tool_text(result)
                            payload = json.loads(raw)
                            if payload.get("error"):
                                print(f"File operation error: {payload['error']}")
                            else:
                                print(f"✅ File created: {payload.get('path', path)}")
                        except Exception as e:
                            print(f"File operation error: {e}")
                        continue

                    # Handle # for direct memory saving
                    if user_input.startswith('#'):
                        raw = user_input.strip()
                        scoped_prefixes = (
                            "#global:",
                            "#project:",
                            "#persona:",
                        )
                        # Keep scope-prefixed form intact so host can parse scope.
                        if raw.lower().startswith(scoped_prefixes):
                            content_to_save = raw
                        else:
                            content_to_save = raw[1:].strip()
                        if content_to_save:
                            print("📝 Saving directly to memory...")
                            await session.call_tool('save_memory', {'content': content_to_save})
                            print("✅ Saved to memory!")
                        continue

                    # Handle slash commands
                    if user_input.startswith('/'):
                        command_line = user_input[1:].strip()
                        if not command_line:
                            continue
                        command_parts = command_line.split(maxsplit=1)
                        command = command_parts[0].lower()
                        command_arg = command_parts[1] if len(command_parts) > 1 else ""
                        
                        if command == 'help':
                            print("""Available commands:
 - /help - Show this help
 - /quit - Exit
 - /tools - List all available tools  
 - /memory - List and manage memories
 - /memory_status - Show active and legacy memory stores
 - /memory_migrate - Preview/apply legacy memory migration
 - /memory_cleanup - Archive legacy memory directories (safe cleanup)
 - /files - List available files
 - /rag "<query>" [path] [version] - Search documents with RAG
 - /web <query> - Force web search tool
 - /crawl <query> - Search web, crawl pages, then run RAG over crawled pages
 - /browser_automation <task> [--headed] - Run a browser task
 - /read <path> - Read a local file
 - /write <path> <content> - Write/create a file
 - /mkmd <path> <content> - Create markdown file
 - /mkdocx <path> <content> - Create docx file
 - /exec - Execute Python via MCP code_execute
 - @filename - Access file content (e.g., @Napoleon.pdf)
 - #message - Save message directly to memory""")
                            continue

                        elif command in ('quit', 'exit'):
                            break
                            
                        elif command == 'tools':
                            if available_tools:
                                print("Available MCP Tools:")
                                for i, name in enumerate(sorted(available_tools), 1):
                                    print(f"{i}. {name}")
                            else:
                                print("Could not retrieve tool list from server.")
                            continue
                        elif command == 'browser_automation':
                            try:
                                task_arg = command_arg.strip()
                                task = task_arg
                                headless = True
                                if "--headed" in task:
                                    headless = False
                                    task = task.replace("--headed", "").strip()
                                if "--headless" in task:
                                    headless = True
                                    task = task.replace("--headless", "").strip()
                                if not task:
                                    if not interactive_input:
                                        print("Usage: /browser_automation <task>")
                                        continue
                                    task = await prompt_session.prompt_async("Task: ")
                                if not task.strip():
                                    print("Aborted: empty task.")
                                    continue
                                if interactive_input and not task_arg:
                                    headless_ans = await prompt_session.prompt_async("Headless? (Y/n): ")
                                    headless = not (headless_ans.strip().lower() == 'n')
                                print("🚀 Starting browser automation...")
                                tool_name = "browser_automation" if "browser_automation" in available_tools else None
                                if not tool_name and "browse_web" in available_tools:
                                    # Best-effort compatibility path.
                                    tool_name = "browse_web"
                                    res = await session.call_tool(tool_name, {'action': 'navigate', 'url': task.strip(), 'headless': headless})
                                elif tool_name:
                                    res = await session.call_tool(tool_name, {'task': task, 'headless': headless})
                                else:
                                    print("Browser tool is not exposed by host. Enable MCP_EXPOSE_EXTRA_TOOLS=1.")
                                    continue
                                try:
                                    print(res.content[0].text)
                                except Exception:
                                    print(res)
                            except Exception as e:
                                print(f"Browser automation error: {e}")
                            continue

                        elif command == 'memory_status':
                            if "memory_storage_info" not in available_tools and available_tools:
                                print("memory_storage_info tool is unavailable on this host.")
                                continue
                            try:
                                result = await session.call_tool('memory_storage_info')
                                print(_extract_tool_text(result))
                            except Exception as e:
                                print(f"Memory status error: {e}")
                            continue

                        elif command == 'memory_migrate':
                            if "migrate_legacy_memories" not in available_tools and available_tools:
                                print("migrate_legacy_memories tool is unavailable on this host.")
                                continue
                            try:
                                preview = await session.call_tool('migrate_legacy_memories', {'dry_run': True})
                                print("\n--- Migration Preview ---")
                                print(_extract_tool_text(preview))
                                if not interactive_input:
                                    print("Non-interactive mode: preview only.")
                                else:
                                    apply_ans = await prompt_session.prompt_async("Apply migration now? (y/N): ")
                                    if apply_ans.strip().lower() == "y":
                                        applied = await session.call_tool('migrate_legacy_memories', {'dry_run': False})
                                        print("\n--- Migration Applied ---")
                                        print(_extract_tool_text(applied))
                            except Exception as e:
                                print(f"Memory migration error: {e}")
                            continue

                        elif command == 'memory_cleanup':
                            if "cleanup_legacy_memory_paths" not in available_tools and available_tools:
                                print("cleanup_legacy_memory_paths tool is unavailable on this host.")
                                continue
                            try:
                                preview = await session.call_tool('cleanup_legacy_memory_paths', {'confirm': False})
                                print("\n--- Cleanup Preview ---")
                                print(_extract_tool_text(preview))
                                if not interactive_input:
                                    print("Non-interactive mode: preview only.")
                                else:
                                    apply_ans = await prompt_session.prompt_async("Archive these paths now? (y/N): ")
                                    if apply_ans.strip().lower() == "y":
                                        applied = await session.call_tool('cleanup_legacy_memory_paths', {'confirm': True})
                                        print("\n--- Cleanup Applied ---")
                                        print(_extract_tool_text(applied))
                            except Exception as e:
                                print(f"Memory cleanup error: {e}")
                            continue
                            
                        elif command == 'memory':
                            print("🔍 Inspecting memory...")
                            mems = await session.call_tool('list_all_memories')
                            try:
                                raw = json.loads(mems.content[0].text)
                                all_memories = _normalize_memory_listing(raw)

                                print("\n--- AGENT'S CURRENT MEMORY ---")
                                if all_memories:
                                    for i, mem in enumerate(all_memories):
                                        mem_id = str(mem.get("id", "N/A"))
                                        scope = str(mem.get("scope", "unknown"))
                                        content = str(mem.get("content", "N/A"))
                                        print(f"{i+1}. [{scope}] (ID: {mem_id[:8]}) - {content}")

                                    # Interactive deletion
                                    if interactive_input:
                                        delete_input = await prompt_session.prompt_async(
                                            "\nEnter memory ID to delete (or press Enter to skip): "
                                        )

                                        if delete_input.strip():
                                            matching_mem = None
                                            for mem in all_memories:
                                                if str(mem.get('id', '')).startswith(delete_input.strip()):
                                                    matching_mem = mem
                                                    break

                                            if matching_mem:
                                                confirm = await prompt_session.prompt_async(
                                                    f"Delete '{matching_mem.get('content', '')[:50]}...'? (y/N): "
                                                )
                                                if confirm.lower() == 'y':
                                                    result = await session.call_tool('delete_memory', {'memory_id': matching_mem['id']})
                                                    print("Memory deleted successfully!")
                                            else:
                                                print("Memory ID not found.")
                                else:
                                    print("Memory is empty.")
                                print("---------------------------------")
                            except (json.JSONDecodeError, IndexError, TypeError) as e:
                                print(f"Could not parse memory response: {e}")
                            continue
                            
                        elif command == 'files':
                            print("📁 Available files:")
                            try:
                                files = await _list_documents(session)
                                if files:
                                    for i, file in enumerate(files, 1):
                                        print(f"{i}. {file}")
                                else:
                                    print("No files found.")
                            except Exception as e:
                                print(f"Error listing files: {e}")
                            continue

                        elif command == 'read':
                            target = _resolve_user_path(command_arg) if command_arg else ""
                            if not target:
                                print("Usage: /read <path>")
                                continue
                            try:
                                if "read_file" in available_tools:
                                    result = await session.call_tool("read_file", {"path": command_arg, "max_chars": 12000})
                                    payload = json.loads(_extract_tool_text(result))
                                    if payload.get("error"):
                                        print(f"Read error: {payload['error']}")
                                    else:
                                        print(payload.get("content", ""))
                                        if payload.get("truncated"):
                                            print("\n... [truncated]")
                                else:
                                    if not os.path.isfile(target):
                                        print(f"Not a file: {target}")
                                        continue
                                    with open(target, "r", encoding="utf-8", errors="replace") as f:
                                        data = f.read()
                                    limit = 12000
                                    if len(data) > limit:
                                        print(data[:limit])
                                        print(f"\n... [truncated {len(data) - limit} chars]")
                                    else:
                                        print(data)
                            except Exception as e:
                                print(f"Read error: {e}")
                            continue

                        elif command in ('mkmd', 'mkdocx'):
                            if not command_arg:
                                print(f"Usage: /{command} <path> <content>")
                                continue
                            path_part, content_part = _split_path_and_content(command_arg)
                            if not path_part:
                                print(f"Usage: /{command} <path> <content>")
                                continue
                            if not content_part:
                                if not interactive_input:
                                    print(f"Usage: /{command} <path> <content>")
                                    continue
                                content_part = await _prompt_multiline(prompt_session, f"Enter content for {path_part}")

                            tool_name = "create_markdown" if command == "mkmd" else "create_docx"
                            if tool_name not in available_tools and available_tools:
                                print(f"{tool_name} tool is unavailable on this host.")
                                continue

                            try:
                                if command == "mkmd":
                                    result = await session.call_tool(
                                        "create_markdown",
                                        {"path": path_part, "body": content_part, "overwrite": True},
                                    )
                                else:
                                    result = await session.call_tool(
                                        "create_docx",
                                        {"path": path_part, "body": content_part, "overwrite": True},
                                    )
                                payload = json.loads(_extract_tool_text(result))
                                if payload.get("error"):
                                    print(f"File operation error: {payload['error']}")
                                else:
                                    print(f"✅ File created: {payload.get('path', path_part)}")
                            except Exception as e:
                                print(f"File operation error: {e}")
                            continue

                        elif command == 'write':
                            if not command_arg:
                                print("Usage: /write <path> <content>")
                                continue
                            path_part, content_part = _split_path_and_content(command_arg)
                            if not path_part:
                                print("Usage: /write <path> <content>")
                                continue
                            target = _resolve_user_path(path_part)
                            if not _is_safe_write_target(target):
                                print("Write blocked. Allowed roots: repo root and /tmp.")
                                continue
                            if not content_part:
                                if not interactive_input:
                                    print("Usage: /write <path> <content>")
                                    continue
                                content_part = await _prompt_multiline(prompt_session, "Enter file content")
                            try:
                                if "write_file" in available_tools:
                                    result = await session.call_tool(
                                        "write_file",
                                        {"path": path_part, "content": content_part, "overwrite": True},
                                    )
                                    payload = json.loads(_extract_tool_text(result))
                                    if payload.get("error"):
                                        print(f"Write error: {payload['error']}")
                                    else:
                                        print(f"Wrote {len(content_part)} chars to {payload.get('path', target)}")
                                else:
                                    parent = os.path.dirname(target) or "."
                                    os.makedirs(parent, exist_ok=True)
                                    with open(target, "w", encoding="utf-8") as f:
                                        f.write(content_part)
                                    print(f"Wrote {len(content_part)} chars to {target}")
                            except Exception as e:
                                print(f"Write error: {e}")
                            continue

                        elif command == 'exec':
                            if "code_execute" not in available_tools:
                                print("code_execute tool is unavailable. Start with --profile online or MCP_EXPOSE_EXTRA_TOOLS=1.")
                                continue
                            code = command_arg.strip()
                            if not code:
                                if not interactive_input:
                                    print("Usage: /exec <python>")
                                    continue
                                code = await _prompt_multiline(prompt_session, "Enter Python code to execute")
                            if not code:
                                print("Aborted: empty code.")
                                continue
                            try:
                                result = await session.call_tool('code_execute', {'code': code, 'timeout_seconds': 30})
                                payload = json.loads(_extract_tool_text(result))
                                status = payload.get("status", "unknown")
                                print(f"code_execute status: {status}")
                                stdout = payload.get("stdout", "")
                                stderr = payload.get("stderr", "")
                                if stdout:
                                    print("\n--- stdout ---")
                                    print(stdout)
                                if stderr:
                                    print("\n--- stderr ---")
                                    print(stderr)
                                if payload.get("error"):
                                    print(f"\n--- error ---\n{payload.get('error')}")
                            except Exception as e:
                                print(f"Execution error: {e}")
                            continue

                        elif command == 'web':
                            query = command_arg.strip()
                            if not query:
                                if not interactive_input:
                                    print("Usage: /web <query>")
                                    continue
                                query = (await prompt_session.prompt_async("Web query: ")).strip()
                            if not query:
                                print("Aborted: empty query.")
                                continue
                            if "web_search" not in available_tools and available_tools:
                                print("web_search tool is unavailable.")
                                continue
                            try:
                                result = await session.call_tool('web_search', {'query': query})
                                payload = _parse_json_dict(_extract_tool_text(result))
                                if payload.get("error"):
                                    print(f"Web search error: {payload['error']}")
                                else:
                                    print(payload.get("results", "No results found."))
                            except Exception as e:
                                print(f"Web search error: {e}")
                            continue

                        elif command == 'crawl':
                            query = command_arg.strip()
                            if not query:
                                if not interactive_input:
                                    print("Usage: /crawl <query>")
                                    continue
                                query = (await prompt_session.prompt_async("Crawl query: ")).strip()
                            if not query:
                                print("Aborted: empty query.")
                                continue
                            if "crawl_web" not in available_tools and available_tools:
                                print("crawl_web tool is unavailable.")
                                continue
                            try:
                                print(f"🕸️ Crawling web pages for: '{query}'...")
                                crawl = await session.call_tool(
                                    "crawl_web",
                                    {"query": query, "max_pages": 5, "max_depth": 1, "same_domain_only": True},
                                )
                                crawl_payload = _parse_json_dict(_extract_tool_text(crawl))
                                if crawl_payload.get("error"):
                                    print(f"Crawl error: {crawl_payload['error']}")
                                    continue

                                files = crawl_payload.get("files", [])
                                if not isinstance(files, list):
                                    files = []
                                files = [str(x) for x in files if str(x).strip()]

                                print(f"Crawled pages: {crawl_payload.get('crawled_count', len(files))}")
                                if files:
                                    print("Indexed files:")
                                    for fp in files[:10]:
                                        print(f"- {fp}")

                                if files and ("query_knowledge_base" in available_tools or not available_tools):
                                    rag = await session.call_tool(
                                        "query_knowledge_base",
                                        {"query": query, "version": "v2", "files": files[:5]},
                                    )
                                    rag_payload = _parse_json_dict(_extract_tool_text(rag))
                                    if rag_payload.get("error"):
                                        print(f"RAG error: {rag_payload['error']}")
                                    else:
                                        print("\n--- Crawled RAG Result ---")
                                        print(rag_payload.get("result", "No result found."))
                            except Exception as e:
                                print(f"Crawl error: {e}")
                            continue

                        elif command == 'rag':
                            import re
                            # Extract quoted query: /rag "query text" [path] [version]
                            match = re.search(r'"([^"]+)"', user_input)
                            if not match:
                                print("Usage: /rag \"<query>\" [path] [version]")
                                continue
                            
                            query = match.group(1)
                            # Get remaining parts after quote
                            remainder = user_input[match.end():].strip()
                            parts = remainder.split()
                            
                            # Defaults
                            version = 'v2'
                            doc_path = ''
                            
                            # Parse: first part is path, second is version (or vice versa)
                            if len(parts) > 0:
                                if parts[0] in ['v1', 'v2', 'v3']:
                                    version = parts[0]
                                    if len(parts) > 1:
                                        doc_path = parts[1]
                                else:
                                    doc_path = parts[0]
                                    if len(parts) > 1 and parts[1] in ['v1', 'v2', 'v3']:
                                        version = parts[1]

                            print(f"📚 Querying knowledge base with RAG {version}...")

                            try:
                                params = {'query': query, 'version': version}
                                if doc_path:
                                    params['doc_path'] = doc_path

                                rag_result = await session.call_tool('query_knowledge_base', params)
                                knowledge = json.loads(rag_result.content[0].text)

                                if 'result' in knowledge:
                                    print("\n--- RAG Result ---")
                                    print(knowledge['result'])
                                    print("--------------------")
                                else:
                                    print(f"Error from RAG tool: {knowledge.get('error', 'Unknown error')}")

                            except (json.JSONDecodeError, IndexError, TypeError) as e:
                                print(f"Could not parse RAG response: {e}")
                            except Exception as e:
                                print(f"An error occurred during RAG query: {e}")
                            continue

                        else:
                            print(f"Unknown command: /{command}. Type /help for available commands.")
                            continue

                    # Save user turn into short-term memory
                    try:
                        stm.chat_memory.add_user_message(user_input)
                    except Exception:
                        pass

                    # --- Step 1: Process @ mentions for document references ---
                    document_context = ""
                    mentions = [word[1:] for word in user_input.split() if word.startswith("@")]
                    
                    if mentions:
                        print(f"Found document mentions: {mentions}")
                        try:
                            # Get list of available documents
                            doc_list = await _list_documents(session)
                            
                            for mention in mentions:
                                if mention in doc_list:
                                    print(f"Fetching content for: {mention}")
                                    content = await _read_document_content(session, mention)
                                    document_context += f'\n<document id="{mention}">\n{content}\n</document>\n'
                        except Exception as e:
                            print(f"Could not fetch document resources: {e}")

                    # --- Step 2: Search Internal Memory ---
                    print("🧠 Searching memories...")
                    memories = []
                    if "search_memories" in available_tools or not available_tools:
                        search_result = await session.call_tool('search_memories', {'query': user_input, 'top_k': 3})
                        try:
                            memories = json.loads(search_result.content[0].text)
                        except (json.JSONDecodeError, IndexError, TypeError):
                            memories = []
                    
                    memory_context = "--- Relevant Memories ---\n"
                    if memories and isinstance(memories, list):
                        valid_mems = [mem.get('content') for mem in memories if mem and 'content' in mem]
                        memory_context += "\n".join([f"- {mem}" for mem in valid_mems]) if valid_mems else "No relevant memories found."
                    else:
                        memory_context += "No relevant memories found."

                    # --- Step 3: Decide which tool to use (if any) ---
                    # Skip tool decision if we already have document context from @ mentions
                    external_context = ""
                    if not document_context:
                        print("🤔 Analyzing query for tool use...")
                        decision_prompt = f"""You are a tool-use decision engine. Analyze the user query and extract:
1. tool: "rag" (for documents/knowledge), "web_search" (for current info), or "none"
2. doc_path: Extract any path mentioned. Convert common names to full paths:
   - "Desktop" → "~/Desktop"
   - "Documents" → "~/Documents"
   - "Downloads" → "~/Downloads"
   - Relative paths like "tools" → "./tools"
   - Absolute paths stay as-is
   - Empty string if not mentioned
3. version: Extract RAG version if mentioned ("v1", "v2", "v3"). Default "v2" if not specified.

Examples:
- "search Desktop for Napoleon" → {{"tool": "rag", "doc_path": "~/Desktop", "version": "v2"}}
- "what's in tools folder using v3?" → {{"tool": "rag", "doc_path": "./tools", "version": "v3"}}
- "current weather" → {{"tool": "web_search", "doc_path": "", "version": "v2"}}

User Query: "{user_input}"

JSON Response:"""
                        
                        # Use Gemini or Ollama for tool decision
                        if LLM_MODEL == 'gemini':
                            response = genai_client.models.generate_content(
                                model='gemini-2.0-flash-exp',
                                contents=decision_prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type='application/json',
                                    response_schema={
                                        'type': 'object',
                                        'properties': {
                                            'tool': {'type': 'string', 'enum': ['rag', 'web_search', 'none']},
                                            'doc_path': {'type': 'string'},
                                            'version': {'type': 'string', 'enum': ['v1', 'v2', 'v3']}
                                        },
                                        'required': ['tool', 'doc_path', 'version']
                                    }
                                )
                            )
                            decision = json.loads(response.text)
                        else:
                            if ollama is None:
                                raise RuntimeError("Ollama backend not initialized.")
                            decision_response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': decision_prompt}], format='json')
                            content = decision_response['message']['content']
                            if not content.strip():
                                raise json.JSONDecodeError("Empty response", "", 0)
                            decision = json.loads(content)
                        
                        try:
                            tool_to_use = decision.get("tool", "none")
                            doc_path = decision.get("doc_path", "")
                            version = decision.get("version", "v2")
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Tool decision parsing error: {e}, defaulting to none")
                            tool_to_use = "none"
                            doc_path = ""
                            version = "v2"

                        try:
                            if tool_to_use == "rag":
                                print(f"📚 Querying knowledge base for: '{user_input}'...")
                                params = {'query': user_input, 'version': version}
                                if doc_path:
                                    params['doc_path'] = doc_path
                                rag_result = await session.call_tool('query_knowledge_base', params)
                                knowledge = json.loads(rag_result.content[0].text)
                                result_text = knowledge.get('result', 'No result found.')
                                
                                # Check if RAG found relevant info
                                if (
                                    ('No relevant' in result_text or 'error' in result_text.lower() or len(result_text) < 100)
                                    and ("web_search" in available_tools or not available_tools)
                                ):
                                    print("⚠️  Local knowledge insufficient, using web -> crawl -> RAG...")
                                    web_context, crawled_files = await _web_crawl_rag_context(
                                        session=session,
                                        available_tools=available_tools,
                                        query=user_input,
                                        version=version,
                                        max_pages=5,
                                        max_depth=1,
                                    )
                                    if crawled_files:
                                        print(f"🕸️ Crawled {len(crawled_files)} pages for retrieval.")
                                    external_context = web_context or f"\n--- External Knowledge (Web) ---\nNo results found."
                                else:
                                    external_context = f"\n--- External Knowledge (Books) ---\n{result_text}"
                            
                            elif tool_to_use == "web_search" and ("web_search" in available_tools or not available_tools):
                                print(f"🌐 Running web research pipeline for: '{user_input}'...")
                                web_context, crawled_files = await _web_crawl_rag_context(
                                    session=session,
                                    available_tools=available_tools,
                                    query=user_input,
                                    version=version,
                                    max_pages=5,
                                    max_depth=1,
                                )
                                if crawled_files:
                                    print(f"🕸️ Crawled {len(crawled_files)} pages for retrieval.")
                                external_context = web_context or "\n--- External Knowledge (Web) ---\nNo results found."

                        except Exception as e:
                            print(f"Tool execution error: {e}")
                    else:
                        print("Using document context from @ mentions, skipping tool selection.")

                    # --- Step 4: Synthesize and Respond ---
                    short_term_context = stm.buffer

                    final_prompt = f"""You are a hyper-intelligent assistant. Your single most important duty is to maintain factual accuracy.
You have access to your personal memory, external knowledge base, web search, and can reference specific documents.
You are running inside a local operator CLI that supports real actions via slash commands.
Never claim you cannot create/read files, run code, or use tools.
If asked to do those actions, provide exact command-oriented guidance using:
/write, /mkmd, /mkdocx, /read, /exec, /web, /crawl, /rag, /browser_automation, /memory_status, /memory_migrate, /memory_cleanup.

Your primary source of truth is your memory. If the user contradicts it, you MUST correct them.
Use the provided information sources to answer questions when appropriate.

--- CONVERSATION HISTORY ---
{short_term_context}
---

--- MEMORY ---
{memory_context}
---
{document_context if document_context else external_context}
---

Note: If the user's query contains references to documents like "@Napoleon.pdf", the "@" is only a way of mentioning the doc. 
The actual document content (if available) is provided above. Answer directly and concisely using the provided information.

User: {user_input}"""
                    
                    print("💡 Synthesizing final response...")
                    
                    # Prepare chat parameters
                    chat_params = {
                        'model': LLM_MODEL, 
                        'messages': [{'role': 'user', 'content': final_prompt}], 
                        'stream': True
                    }
                    
                    # Add thinking parameter if specified and model supports it
                    if think_level and LLM_MODEL.startswith('gpt-oss'):
                        chat_params['think'] = think_level
                        print("🧠 Juliette is thinking...")
                    
                    assistant_output = ""
                    thinking_output = ""
                    
                    if LLM_MODEL == 'gemini':
                        response = genai_client.models.generate_content(
                            model='gemini-2.0-flash-exp',
                            contents=final_prompt
                        )
                        answer = response.text.strip()
                        print(f"\nJuliette: {answer}\n")
                        assistant_output = answer
                    else:
                        if ollama is None:
                            raise RuntimeError("Ollama backend not initialized.")
                        stream = ollama.chat(**chat_params)
                        thinking_started = False
                        response_started = False
                        
                        for chunk in stream:
                            # Handle thinking output FIRST
                            if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
                                if not thinking_started and think_level:
                                    print(f"\n💭 Raw CoT Thinking:")
                                    print("=" * 50)
                                    thinking_started = True
                                
                                thinking_chunk = chunk.message.thinking
                                thinking_output += thinking_chunk
                                if think_level:  # Display raw thinking in real-time
                                    print(thinking_chunk, end="", flush=True)
                            
                            # Handle regular content AFTER thinking
                            elif chunk.message.content:
                                if thinking_started and not response_started and think_level:
                                    print("\n" + "=" * 50)
                                    print(f"\nJuliette: ", end="", flush=True)
                                    response_started = True
                                elif not response_started:
                                    print(f"\nJuliette: ", end="", flush=True)
                                    response_started = True
                                    
                                content = chunk.message.content
                                print(content, end="", flush=True)
                                assistant_output += content
                        
                        print()  # New line after streaming
                    
                    try:
                        stm.chat_memory.add_ai_message(assistant_output)
                    except Exception:
                        pass
                    

                except (EOFError, KeyboardInterrupt):
                    print("\nExiting...")
                    break
                except Exception as e:
                    msg = str(e)
                    if "Failed to connect to Ollama" in msg:
                        print("Ollama is not reachable. Start it with: ollama serve")
                        continue
                    print(f"\n--- An Error Occurred in the Loop ---", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Juliette - Intelligent Agent with Thinking")
    parser.add_argument(
        "--profile",
        choices=["online", "offline"],
        default=os.getenv("AGENT_PROFILE", "online"),
        help="Runtime profile: online enables extra tools, offline keeps local-only core tools",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini"],
        default="ollama",
        help="LLM backend to use (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (default: OLLAMA_MODEL env or qwen3:8b)",
    )
    parser.add_argument('--think', choices=['low', 'medium', 'high'], 
                        help='Enable thinking mode for supported models (gpt-oss)')
    args = parser.parse_args()
    
    try:
        if args.profile == "online":
            os.environ["MCP_EXPOSE_EXTRA_TOOLS"] = "1"
        else:
            os.environ["MCP_EXPOSE_EXTRA_TOOLS"] = "0"
        os.environ.setdefault("MCP_EXPOSE_RESOURCES", "1")
        initialize_llm(provider=args.provider, model=args.model)
        asyncio.run(main(think_level=args.think))
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        print("\n--- A FATAL ERROR OCCURRED ---")
        traceback.print_exc()
