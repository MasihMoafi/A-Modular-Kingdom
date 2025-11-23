"""
Tests for scoped memory system.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from memory.scoped_manager import ScopedMemoryManager
from memory.memory_config import MemoryScope, MemoryConfig


class TestMemoryConfig:
    """Test memory configuration and scope parsing."""
    
    def test_scope_prefix_parsing(self):
        """Test parsing scope prefixes from content."""
        config = MemoryConfig()
        
        # Test global rule
        scope, content = config.parse_scope_prefix("#global:rule Always use types")
        assert scope == MemoryScope.GLOBAL_RULES
        assert content == "Always use types"
        
        # Test global preference  
        scope, content = config.parse_scope_prefix("#global:pref Dark mode")
        assert scope == MemoryScope.GLOBAL_PREFERENCES
        assert content == "Dark mode"
        
        # Test persona
        scope, content = config.parse_scope_prefix("#persona You are a teacher")
        assert scope == MemoryScope.GLOBAL_PERSONAS
        assert content == "You are a teacher"
        
        # Test no prefix
        scope, content = config.parse_scope_prefix("Regular content")
        assert scope is None
        assert content == "Regular content"
    
    def test_scope_inference(self):
        """Test inferring scope from content."""
        config = MemoryConfig()
        
        # Rules
        assert config.infer_scope_from_content("Always commit often") == MemoryScope.GLOBAL_RULES
        assert config.infer_scope_from_content("Never use placeholders") == MemoryScope.GLOBAL_RULES
        
        # Preferences
        assert config.infer_scope_from_content("I prefer Python") == MemoryScope.GLOBAL_PREFERENCES
        assert config.infer_scope_from_content("User likes concise messages") == MemoryScope.GLOBAL_PREFERENCES
        
        # Personas
        assert config.infer_scope_from_content("role: expert coder") == MemoryScope.GLOBAL_PERSONAS
        
        # Default to project
        assert config.infer_scope_from_content("Uses ChromaDB") == MemoryScope.PROJECT_CONTEXT


class TestScopedMemoryManager:
    """Test scoped memory manager."""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def manager(self, temp_project):
        """Create scoped memory manager."""
        return ScopedMemoryManager(project_root=temp_project)
    
    def test_save_to_global_rules(self, manager):
        """Test saving to global rules scope."""
        memory_id = manager.save(
            "Always use type hints",
            scope=MemoryScope.GLOBAL_RULES
        )
        assert memory_id is not None
        
    def test_save_to_project_context(self, manager):
        """Test saving to project context scope."""
        memory_id = manager.save(
            "Uses FastMCP for server",
            scope=MemoryScope.PROJECT_CONTEXT
        )
        assert memory_id is not None
    
    def test_search_with_priority(self, manager):
        """Test search respects scope priority."""
        # Save to different scopes
        manager.save("Global coding rule", scope=MemoryScope.GLOBAL_RULES)
        manager.save("Project uses Django", scope=MemoryScope.PROJECT_CONTEXT)
        
        # Search should return results with scope metadata
        results = manager.search("coding", k=5)
        assert len(results) > 0
        assert any("scope" in r.get("metadata", {}) for r in results)
    
    def test_scope_isolation(self, manager):
        """Test that scopes are isolated."""
        # Save to different scopes
        rule_id = manager.save("Rule content", scope=MemoryScope.GLOBAL_RULES)
        context_id = manager.save("Context content", scope=MemoryScope.PROJECT_CONTEXT)
        
        # IDs should be different
        assert rule_id != context_id
        
        # List from each scope
        rules = manager.list_all(MemoryScope.GLOBAL_RULES)
        contexts = manager.list_all(MemoryScope.PROJECT_CONTEXT)
        
        rule_contents = [r["content"] for r in rules]
        context_contents = [c["content"] for c in contexts]
        
        assert "Rule content" in rule_contents
        assert "Rule content" not in context_contents
        assert "Context content" in context_contents
        assert "Context content" not in rule_contents
    
    def test_delete_from_scope(self, manager):
        """Test deleting memory from specific scope."""
        memory_id = manager.save("To delete", scope=MemoryScope.GLOBAL_PREFERENCES)
        
        # Verify it exists
        all_prefs = manager.list_all(MemoryScope.GLOBAL_PREFERENCES)
        assert any(m["id"] == memory_id for m in all_prefs)
        
        # Delete it
        manager.delete(memory_id, scope=MemoryScope.GLOBAL_PREFERENCES)
        
        # Verify it's gone
        all_prefs = manager.list_all(MemoryScope.GLOBAL_PREFERENCES)
        assert not any(m["id"] == memory_id for m in all_prefs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
