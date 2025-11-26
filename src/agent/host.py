#!/usr/bin/env python
# coding: utf-8
import os
import sys
import warnings
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
import importlib
from typing import Dict, List
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from memory.core import Mem0
from memory.scoped_manager import ScopedMemoryManager
from memory.memory_config import MemoryScope
from tools.code_exec import run_code
from tools.vision import analyze_media_with_ollama
from tools.tts import text_to_speech
from tools.stt import speech_to_text
import glob

mcp = FastMCP("unified_knowledge_agent_host")
memory_database = None  # Legacy
scoped_memory = None

try:
    # Initialize scoped memory manager
    scoped_memory = ScopedMemoryManager(project_root=project_root)
    # Legacy memory system with Qdrant backend - use global path
    from pathlib import Path
    global_legacy_path = Path.home() / ".modular_kingdom" / "legacy_memories"
    global_legacy_path.mkdir(parents=True, exist_ok=True)
    memory_database = Mem0(storage_path=str(global_legacy_path))
except Exception as e:
    sys.stderr.write(f"[HOST] WARNING: Memory systems disabled: {e}\n")
    memory_database = None
    scoped_memory = None

@mcp.tool(
    name="save_fact",
    description="Save structured facts to the memory system with automatic processing and fact extraction from the provided content"
)
def save_fact(
    fact_data: Dict = Field(description="Dictionary containing 'content' key with the fact to save")
) -> str:
    if memory_database is None:
        return json.dumps({"error": "MemoryDB is not initialized on the host."})
    try:
        content_to_save = fact_data.get('content')
        if not content_to_save:
            return json.dumps({"error": "No 'content' field in fact JSON."})
        memory_database.add(content_to_save)
        return json.dumps({"status": "success", "message": f"Fact sent to processing: '{content_to_save}'"})
    except Exception as e:
        return json.dumps({"error": f"Error saving fact on host: {str(e)}"})

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
    """Delete a memory by its ID."""
    if memory_database is None:
        return json.dumps({"error": "MemoryDB is not initialized on the host."})
    try:
        memory_database.direct_delete(memory_id)
        return json.dumps({"status": "success", "message": f"Memory {memory_id} deleted successfully"})
    except Exception as e:
        return json.dumps({"error": f"Error deleting memory on host: {str(e)}"})

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
    doc_path: str = Field(default="", description="Optional path to a specific documents directory")
) -> str:
    """
    A tool to query the external knowledge base (RAG system) with selectable versions.
    """
    sys.stderr.write(f"[HOST] RAG tool called with query: '{query[:20]}...', version: {version}, path: '{doc_path}'\n")
    sys.stderr.flush()

    try:
        # Map version to module name
        module_map = {
            "v1": "rag.fetch",
            "v2": "rag.fetch_2",
            "v3": "rag.fetch_3"
        }

        module_name = module_map.get(version)
        if not module_name:
            return json.dumps({"error": f"Invalid RAG version '{version}'. Must be 'v1', 'v2', or 'v3'."})

        # Dynamically import the correct module
        rag_module = importlib.import_module(module_name)
        fetchExternalKnowledge = getattr(rag_module, 'fetchExternalKnowledge')

        sys.stderr.write(f"[HOST] Calling fetchExternalKnowledge from {module_name}\n")
        sys.stderr.flush()

        # Call the function with appropriate arguments
        if doc_path and isinstance(doc_path, str):
            results = fetchExternalKnowledge(query, doc_path=doc_path)
        else:
            results = fetchExternalKnowledge(query)

        sys.stderr.write(f"[HOST] RAG ({version}) returned {len(results)} chars\n")
        sys.stderr.flush()

        return json.dumps({"result": results})

    except (ModuleNotFoundError, AttributeError):
        error_msg = f"Could not find or load 'fetchExternalKnowledge' function in module for version '{version}'."
        sys.stderr.write(f"[HOST] RAG error: {error_msg}\n")
        sys.stderr.flush()
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Error querying knowledge base on host with version {version}: {str(e)}"
        sys.stderr.write(f"[HOST] RAG error: {error_msg}\n")
        sys.stderr.flush()
        return json.dumps({"error": error_msg})

@mcp.tool(
    name="list_all_memories",
    description="Retrieve and list all saved memories from the memory system for review and management"
)
def list_all_memories() -> str:
    """A tool to inspect the contents of the memory database."""
    if memory_database is None:
        return json.dumps([{"error": "MemoryDB not initialized on the host."}])
    try:
        all_mems = memory_database.get_all_memories()
        return json.dumps(all_mems)
    except Exception as e:
        return json.dumps([{"error": f"Error listing memories on host: {str(e)}"}])

@mcp.tool(
    name="code_execute",
    description="Execute Python code in a sandboxed subprocess and return stdout/stderr"
)
def code_execute(
    code: str = Field(description="Python code to execute"),
    timeout_seconds: int = Field(default=15, description="Execution timeout in seconds")
) -> str:
    try:
        return run_code(code=code, timeout_seconds=timeout_seconds)
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
        return analyze_media_with_ollama(model=model, paths=paths)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

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
    except Exception as e:
        return json.dumps([])

@mcp.resource("docs://documents/{doc_id}", mime_type="text/plain")
def get_document_content(doc_id: str) -> str:
    """Returns the content of a specific document."""
    try:
        files_dir = os.path.join(project_root, "rag", "files")
        file_path = os.path.join(files_dir, doc_id)
        
        if not os.path.exists(file_path):
            raise ValueError(f"Document {doc_id} not found")
        
        if doc_id.lower().endswith('.pdf'):
            import fitz
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        return f"Error: Could not retrieve content for {doc_id}: {str(e)}"

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
            
        return text_to_speech(text, **kwargs)
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
            
        return speech_to_text(**kwargs)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

mcp.run(transport='stdio')
