"""Memory MCP Server - RAG and Memory tools via Model Context Protocol.

This server exposes:
- RAG tools: query_knowledge_base for document search
- Memory tools: save, search, delete, list memories
"""

from typing import Any

from fastmcp import FastMCP

from memory_mcp.config import Settings, get_settings
from memory_mcp.memory import MemoryStore
from memory_mcp.rag import RAGPipeline


def create_server(
    settings: Settings | None = None,
    document_paths: list[str] | None = None,
) -> FastMCP:
    """Create and configure the MCP server.

    Args:
        settings: Optional settings override (uses get_settings() if not provided)
        document_paths: Optional list of paths to index for RAG

    Returns:
        Configured FastMCP server instance
    """
    if settings is None:
        settings = get_settings()

    mcp = FastMCP(settings.server_name)

    memory_store = MemoryStore(settings)

    _rag_pipelines: dict[str, RAGPipeline] = {}

    def get_rag_pipeline(doc_path: str | None = None) -> RAGPipeline:
        """Get or create RAG pipeline for given path."""
        paths = [doc_path] if doc_path else (document_paths or [])
        cache_key = "|".join(sorted(paths))

        if cache_key not in _rag_pipelines:
            pipeline = RAGPipeline(
                settings=settings,
                document_paths=paths,
            )
            pipeline.index()
            _rag_pipelines[cache_key] = pipeline

        return _rag_pipelines[cache_key]

    @mcp.tool()
    def query_knowledge_base(
        query: str,
        doc_path: str = "",
        top_k: int | None = None,
    ) -> str:
        """Search codebases and documents using RAG. 98% less context than reading files.

        IMPORTANT: For codebase exploration, you MUST provide doc_path pointing to the
        project directory. Without doc_path, searches default internal docs only.

        Use cases: 'how does X work?', 'what's in folder Y?', architectural questions.
        Works on: .py, .md, .ipynb, .js, .ts, .json, .pdf, etc.

        Args:
            query: The search query (e.g., 'how does authentication work?')
            doc_path: Path to codebase/documents directory. REQUIRED for codebase exploration.
            top_k: Number of results to return (default: 5)

        Returns:
            Formatted search results from relevant documents
        """
        try:
            pipeline = get_rag_pipeline(doc_path if doc_path else None)
            return pipeline.search(query, top_k=top_k)
        except Exception as e:
            return f"RAG search failed: {str(e)}"

    @mcp.tool()
    def save_memory(content: str) -> dict[str, Any]:
        """Save content to the memory system.

        Stores the provided content as a searchable memory that can be
        retrieved later using search_memories.

        Args:
            content: The text content to save to memory

        Returns:
            Dictionary with the memory ID and status
        """
        try:
            memory_id = memory_store.add(content)
            return {
                "status": "success",
                "memory_id": memory_id,
                "message": f"Memory saved with ID: {memory_id[:8]}...",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    def save_fact(fact_data: dict[str, Any]) -> dict[str, Any]:
        """Save structured fact data to memory.

        This tool accepts a dictionary containing fact information and
        stores it with metadata for later retrieval.

        Args:
            fact_data: Dictionary containing 'content' key with the fact to save.
                       Can also include additional metadata fields.

        Returns:
            Dictionary with the memory ID and status
        """
        try:
            content = fact_data.get("content", "")
            if not content:
                return {"status": "error", "message": "No content provided"}

            metadata = {k: v for k, v in fact_data.items() if k != "content"}
            memory_id = memory_store.add(content, metadata=metadata if metadata else None)

            return {
                "status": "success",
                "memory_id": memory_id,
                "message": f"Fact saved with ID: {memory_id[:8]}...",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    def search_memories(query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search through saved memories.

        Performs semantic and keyword search to find relevant memories
        matching the query.

        Args:
            query: The search query to find relevant memories
            top_k: Number of top matching memories to return

        Returns:
            List of matching memories with id, content, and metadata
        """
        try:
            return memory_store.search(query, k=top_k)
        except Exception as e:
            return [{"error": str(e)}]

    @mcp.tool()
    def delete_memory(memory_id: str) -> dict[str, Any]:
        """Delete a specific memory by ID.

        Permanently removes the memory with the given ID from storage.

        Args:
            memory_id: The unique ID of the memory to delete

        Returns:
            Dictionary with deletion status
        """
        try:
            success = memory_store.delete(memory_id)
            if success:
                return {"status": "success", "message": f"Memory {memory_id[:8]}... deleted"}
            return {"status": "error", "message": "Memory not found or deletion failed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    def list_all_memories() -> list[dict[str, Any]]:
        """Retrieve all saved memories.

        Returns a complete list of all memories stored in the system
        for review and management.

        Returns:
            List of all memories with id, content, and metadata
        """
        try:
            return memory_store.get_all()
        except Exception as e:
            return [{"error": str(e)}]

    return mcp
