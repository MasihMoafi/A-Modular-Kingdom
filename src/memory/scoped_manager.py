"""
Scoped memory manager for hierarchical memory organization.
Provides unified interface to global and project-scoped memories.
"""
import os
import uuid
from typing import List, Dict, Optional
from memory.core import Mem0
from memory.markdown_store import MarkdownMemoryStore
from memory.memory_config import MemoryConfig, MemoryScope


class ScopedMemoryManager:
    """Manager for multiple scoped memory instances."""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize scoped memory manager.
        
        Args:
            project_root: Root directory of current project
        """
        self.config = MemoryConfig(project_root)
        self._instances: Dict[MemoryScope, Mem0] = {}
        self.markdown_store = MarkdownMemoryStore(self.config)
        self.use_qdrant = os.getenv("AMK_MEMORY_VECTOR_BACKEND", "").lower() == "qdrant"
        
    def _get_instance(self, scope: MemoryScope) -> Mem0:
        """Get or create Mem0 instance for a scope."""
        if scope not in self._instances:
            storage_path = str(self.config.get_storage_path(scope))
            collection_name = self.config.get_collection_name(scope)
            self._instances[scope] = Mem0(
                storage_path=storage_path,
                collection_name=collection_name
            )
        return self._instances[scope]
    
    def save(self, content: str, scope: Optional[MemoryScope] = None) -> str:
        """
        Save memory to specified scope (or infer scope).
        
        Args:
            content: Content to save
            scope: Target scope (inferred if None)
            
        Returns:
            Memory ID
        """
        if scope is None:
            scope = self.config.infer_scope_from_content(content)
        
        memory_id = str(uuid.uuid4())
        self.markdown_store.append(memory_id, content, scope)

        if self.use_qdrant:
            try:
                instance = self._get_instance(scope)
                instance.direct_add(content, metadata={"scope": scope.value, "markdown_id": memory_id})
            except Exception:
                pass
        return memory_id
    
    def search(
        self, 
        query: str, 
        k: int = 3, 
        scopes: Optional[List[MemoryScope]] = None
    ) -> List[Dict]:
        """
        Search across scopes with priority ordering.
        
        Args:
            query: Search query
            k: Number of results per scope
            scopes: Scopes to search (uses priority order if None)
            
        Returns:
            Combined search results with scope metadata
        """
        if scopes is None:
            scopes = self.config.get_search_priority()
        
        all_results = []
        for scope in scopes:
            try:
                results = self.markdown_store.search(query, k=k, scopes=[scope])

                if self.use_qdrant:
                    try:
                        instance = self._get_instance(scope)
                        qdrant_results = instance.search(query, k=k)
                    except Exception:
                        qdrant_results = []
                    by_id = {result["id"]: result for result in results}
                    for result in qdrant_results:
                        by_id.setdefault(result.get("id"), result)
                    results = list(by_id.values())
                
                # Add scope to metadata
                for result in results:
                    if not result.get("metadata"):
                        result["metadata"] = {}
                    result["metadata"]["scope"] = scope.value
                    
                all_results.extend(results)
            except Exception:
                continue  # Skip non-existent scopes
        
        return all_results
    
    def delete(self, memory_id: str, scope: MemoryScope) -> None:
        """Delete memory from specific scope."""
        deleted_markdown = self.markdown_store.delete(memory_id)
        deleted_qdrant = False
        if self.use_qdrant:
            try:
                instance = self._get_instance(scope)
                deleted_qdrant = instance.direct_delete(memory_id)
            except Exception:
                deleted_qdrant = False
        if not (deleted_markdown or deleted_qdrant):
            raise KeyError(memory_id)
    
    def list_all(self, scope: MemoryScope) -> List[Dict]:
        """List all memories in a scope."""
        items = {item["id"]: item for item in self.markdown_store.list_all(scope)}
        if self.use_qdrant:
            try:
                instance = self._get_instance(scope)
                for item in instance.get_all_memories():
                    items.setdefault(item["id"], item)
            except Exception:
                pass
        return list(items.values())
