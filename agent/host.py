#!/usr/bin/env python
# coding: utf-8
import os
import sys
import warnings
import logging

# --- FIX for ModuleNotFoundError ---
# Add the parent directory ('A-Modular-Kingdom') to the system path.
# This allows us to import modules from sibling directories like 'memory'.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- DIAGNOSTIC PRINT ---
print(f"[HOST] Script started. Project root added to path: {project_root}")

logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from typing import Dict

# --- DIAGNOSTIC PRINT ---
print("[HOST] Importing local modules (mcp, memory.core, rag, tools)...")
from mcp.server.fastmcp import FastMCP
from memory.core import Mem0
from rag.fetch_2 import fetchExternalKnowledge
from tools.web_search import perform_web_search
print("[HOST] Local modules imported.")

# --- DIAGNOSTIC PRINT ---
print("[HOST] Initializing FastMCP...")
mcp = FastMCP("unified_knowledge_agent_host")
print("[HOST] FastMCP initialized.")
memory_database = None

try:
    # The DB path should be relative to the project root for consistency.
    chroma_path = os.path.join(project_root, "agent_chroma_db")
    # --- DIAGNOSTIC PRINT ---
    print(f"[HOST] Initializing Mem0 with path: {chroma_path}...")
    memory_database = Mem0(chroma_path=chroma_path)
    print("[HOST] Mem0 initialized successfully.")
except Exception as e:
    sys.stderr.write(f"[HOST] ❌ CRITICAL: Could not initialize MemoryDB: {e}\n")
    sys.exit(1)

@mcp.tool()
def save_fact(fact_data: Dict) -> str:
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

@mcp.tool()
def search_memories(query: str, top_k: int = 3) -> str:
    if memory_database is None:
        return json.dumps([{"error": "MemoryDB not initialized on the host."}] )
    try:
        results = memory_database.search(query, k=top_k)
        return json.dumps([{"content": res["content"]} for res in results])
    except Exception as e:
        return json.dumps([{"error": f"Error searching memories on host: {str(e)}"}])

@mcp.tool()
def query_knowledge_base(query: str) -> str:
    """A tool to query the external knowledge base (RAG system)."""
    try:
        # --- DIAGNOSTIC PRINT ---
        print(f"[HOST] Querying RAG with: '{query}'")
        results = fetchExternalKnowledge(query)
        print(f"[HOST] RAG results received.")
        return json.dumps({"result": results})
    except Exception as e:
        sys.stderr.write(f"[HOST] ❌ CRITICAL: RAG query failed: {e}\n")
        return json.dumps({"error": f"Error querying knowledge base on host: {str(e)}"})

@mcp.tool()
def list_all_memories() -> str:
    """A tool to inspect the contents of the memory database."""
    if memory_database is None:
        return json.dumps([{"error": "MemoryDB not initialized on the host."}])
    try:
        all_mems = memory_database.get_all_memories()
        return json.dumps(all_mems)
    except Exception as e:
        return json.dumps([{"error": f"Error listing memories on host: {str(e)}"}])

@mcp.tool()
def web_search(query: str) -> str:
    """A tool to perform a web search using the Google Search API."""
    try:
        print(f"[HOST] Performing web search for: '{query}'")
        results = perform_web_search(query)
        print(f"[HOST] Web search raw result: {results}") # <-- ADDED LOGGING
        print(f"[HOST] Web search results received.")
        return results # The result is already a JSON string
    except Exception as e:
        sys.stderr.write(f"[HOST] ❌ CRITICAL: Web search failed: {e}\n")
        return json.dumps({"error": f"Error performing web search on host: {str(e)}"})

# --- DIAGNOSTIC PRINT ---
print("[HOST] Tools defined. Starting MCP run loop...")
mcp.run(transport='stdio')
# --- This next line will not be printed, as mcp.run() is a blocking call ---
print("[HOST] MCP run loop exited.")
