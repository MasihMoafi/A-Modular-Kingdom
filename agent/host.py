#!/usr/bin/env python
# coding: utf-8
import os
import sys
import warnings
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json
from typing import Dict, List
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from memory.core import Mem0
from rag.fetch_3 import fetchExternalKnowledge
from tools.web_search import perform_web_search
import glob

mcp = FastMCP("unified_knowledge_agent_host")
memory_database = None

try:
    chroma_path = os.path.join(project_root, "agent_chroma_db")
    memory_database = Mem0(chroma_path=chroma_path)
except Exception as e:
    sys.stderr.write(f"[HOST] Could not initialize MemoryDB: {e}\n")
    sys.exit(1)

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
    description="Save content directly to the personal memory system for future reference and retrieval"
)
def save_direct_memory(
    content: str = Field(description="The text content to save to memory")
) -> str:
    """Save content directly to memory without fact extraction."""
    if memory_database is None:
        return json.dumps({"error": "MemoryDB is not initialized on the host."})
    try:
        client, collection = memory_database._get_client_and_collection()
        import uuid
        new_id = str(uuid.uuid4())
        collection.add(ids=[new_id], documents=[content])
        return json.dumps({"status": "success", "message": f"Content saved directly to memory: '{content[:50]}...'"})
    except Exception as e:
        return json.dumps({"error": f"Error saving direct memory on host: {str(e)}"})

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
        client, collection = memory_database._get_client_and_collection()
        collection.delete(ids=[memory_id])
        return json.dumps({"status": "success", "message": f"Memory {memory_id} deleted successfully"})
    except Exception as e:
        return json.dumps({"error": f"Error deleting memory on host: {str(e)}"})

@mcp.tool(
    name="search_memories",
    description="Search through saved memories using semantic similarity to find relevant stored information"
)
def search_memories(
    query: str = Field(description="The search query to find relevant memories"),
    top_k: int = Field(default=3, description="Number of top matching memories to return")
) -> str:
    if memory_database is None:
        return json.dumps([{"error": "MemoryDB not initialized on the host."}] )
    try:
        results = memory_database.search(query, k=top_k)
        return json.dumps([{"content": res["content"]} for res in results])
    except Exception as e:
        return json.dumps([{"error": f"Error searching memories on host: {str(e)}"}])

@mcp.tool(
    name="query_knowledge_base",
    description="Search external documents and knowledge base using the RAG system to find relevant information from stored documents"
)
def query_knowledge_base(
    query: str = Field(description="The search query to find relevant information in the knowledge base")
) -> str:
    """A tool to query the external knowledge base (RAG system)."""
    import sys
    sys.stderr.write(f"[HOST] RAG tool called with: {query[:20]}...\n")
    sys.stderr.flush()
    try:
        sys.stderr.write("[HOST] About to call fetchExternalKnowledge\n")
        sys.stderr.flush()
        results = fetchExternalKnowledge(query)
        sys.stderr.write(f"[HOST] RAG returned {len(results)} chars\n")
        sys.stderr.flush()
        return json.dumps({"result": results})
    except Exception as e:
        sys.stderr.write(f"[HOST] RAG error: {e}\n")
        sys.stderr.flush()
        return json.dumps({"error": f"Error querying knowledge base on host: {str(e)}"})

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
    name="web_search",
    description="Perform a web search to find current information that may not be available in local memory or knowledge base"
)
def web_search(
    query: str = Field(description="The search query to look up on the web")
) -> str:
    """A tool to perform a web search using the Google Search API."""
    try:
        results = perform_web_search(query)
        return results
    except Exception as e:
        return json.dumps({"error": f"Error performing web search on host: {str(e)}"})

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

mcp.run(transport='stdio')
