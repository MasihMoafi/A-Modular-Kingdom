"""
Memory configuration for hierarchical scoped memory system.
Defines memory scopes, storage paths, and search priorities.
"""
import os
import hashlib
from enum import Enum
from typing import Optional
from pathlib import Path


class MemoryScope(Enum):
    """Memory scope enumeration for hierarchical organization."""
    GLOBAL_RULES = "global_rules"
    GLOBAL_PREFERENCES = "global_preferences"
    GLOBAL_PERSONAS = "global_personas"
    PROJECT_CONTEXT = "project_context"
    PROJECT_SESSIONS = "project_sessions"


class MemoryConfig:
    """Configuration manager for scoped memory system."""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize memory configuration.
        
        Args:
            project_root: Root directory of current project. If None, uses cwd.
        """
        self.project_root = project_root or os.getcwd()
        self.global_memory_base = Path.home() / ".gemini" / "memories"
        self.project_hash = self._compute_project_hash(self.project_root)
        
    def _compute_project_hash(self, path: str) -> str:
        """Generate short hash from project path for directory naming."""
        return hashlib.md5(path.encode()).hexdigest()[:12]
    
    def get_storage_path(self, scope: MemoryScope) -> Path:
        """
        Get ChromaDB storage path for a given scope.
        
        Args:
            scope: Memory scope to get path for
            
        Returns:
            Path object for ChromaDB storage
        """
        if scope.name.startswith("GLOBAL_"):
            # Global memories stored in ~/.gemini/memories/global/
            scope_name = scope.value.replace("global_", "")
            return self.global_memory_base / "global" / scope_name
        else:
            # Project memories stored in ~/.gemini/memories/projects/{hash}/
            scope_name = scope.value.replace("project_", "")
            return self.global_memory_base / "projects" / self.project_hash / scope_name
    
    def get_collection_name(self, scope: MemoryScope) -> str:
        """Get ChromaDB collection name for a scope."""
        return scope.value
    
    @staticmethod
    def get_search_priority() -> list[MemoryScope]:
        """
        Define search priority order (global first, then project).
        
        Returns:
            Ordered list of scopes to search
        """
        return [
            MemoryScope.GLOBAL_RULES,
            MemoryScope.GLOBAL_PREFERENCES,
            MemoryScope.GLOBAL_PERSONAS,
            MemoryScope.PROJECT_CONTEXT,
            MemoryScope.PROJECT_SESSIONS,
        ]
    
    def parse_scope_prefix(self, text: str) -> tuple[Optional[MemoryScope], str]:
        """
        Parse scope prefix from text like '#global:rule content here'.
        
        Args:
            text: Input text potentially containing scope prefix
            
        Returns:
            Tuple of (scope, cleaned_content). Scope is None if no prefix found.
        """
        if not text.startswith("#"):
            return None, text
            
        # Extract prefix
        if ":" not in text:
            return None, text
            
        prefix_part = text[1:text.index(":")]
        content_after_first_colon = text[text.index(":") + 1:]
        
        # Check if there's a second colon for format like "#global:rule"
        if ":" in content_after_first_colon:
            second_part = content_after_first_colon[:content_after_first_colon.index(":")]
            full_prefix = f"{prefix_part}:{second_part}".lower()
            content = content_after_first_colon[content_after_first_colon.index(":") + 1:].strip()
        else:
            full_prefix = prefix_part.lower()
            content = content_after_first_colon.strip()
        
        # Map prefix to scope
        prefix_map = {
            "global:rule": MemoryScope.GLOBAL_RULES,
            "global:pref": MemoryScope.GLOBAL_PREFERENCES,
            "global:preference": MemoryScope.GLOBAL_PREFERENCES,
            "persona": MemoryScope.GLOBAL_PERSONAS,
            "project:context": MemoryScope.PROJECT_CONTEXT,
            "project:session": MemoryScope.PROJECT_SESSIONS,
        }
        
        scope = prefix_map.get(full_prefix)
        return scope, content
    
    def infer_scope_from_content(self, content: str) -> MemoryScope:
        """
        Infer memory scope from content when no explicit prefix given.
        
        Args:
            content: Memory content to analyze
            
        Returns:
            Best-guess scope for the content
        """
        lower = content.lower()
        
        # Check for rule indicators (strong keywords)
        if any(word in lower for word in ["always ", "never ", "must ", " rule", "should "]):
            return MemoryScope.GLOBAL_RULES
        
        # Check for persona indicators (highest specificity)
        if any(word in lower for word in ["role:", "persona:", "act as", "you are"]):
            return MemoryScope.GLOBAL_PERSONAS
        
        # Check for preference indicators (avoid generic "use" which is too broad)
        if any(word in lower for word in ["prefer", " like", "favorite", " is a "]):
            return MemoryScope.GLOBAL_PREFERENCES
        
        # Default to project context
        return MemoryScope.PROJECT_CONTEXT
