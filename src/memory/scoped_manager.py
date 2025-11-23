"""
Scoped memory manager for hierarchical memory organization.
Provides unified interface to global and project-scoped memories.
"""
from typing import List, Dict, Optional
from memory.core import Mem0
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
        
    def _get_instance(self, scope: MemoryScope) -> Mem0:
        """Get or create Mem0 instance for a scope."""
        if scope not in self._instances:
            storage_path = str(self.config.get_storage_path(scope))
            collection_name = self.config.get_collection_name(scope)
            self._instances[scope] = Mem0(
                chroma_path=storage_path,
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
        
        instance = self._get_instance(scope)
        memory_id = instance.direct_add(content, metadata={"scope": scope.value})
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
                instance = self._get_instance(scope)
                results = instance.search(query, k=k)
                
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
        instance = self._get_instance(scope)
        instance.direct_delete(memory_id)
    
    def list_all(self, scope: MemoryScope) -> List[Dict]:
        """List all memories in a scope."""
        instance = self._get_instance(scope)
        return instance.get_all_memories()
