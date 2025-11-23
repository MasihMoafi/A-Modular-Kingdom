# Scoped Memory System - Usage Guide

## Overview

Hierarchical memory system with global and project-specific scopes:
- **Global memories**: Persist across all projects (rules, preferences, personas)
- **Project memories**: Isolated per project (context, sessions)

## Quick Start

### Saving Memories

```python
# Via MCP tool - with scope prefix
save_memory(content="#global:rule:Always use type hints")
save_memory(content="#global:pref:Prefer dark mode")
save_memory(content="#persona:You are an expert Python developer")
save_memory(content="#project:context:Uses FastMCP for server")

# Via MCP tool - explicit scope
save_memory(content="Keep messages concise", scope="global_rules")

# Convenience tool for global rules
set_global_rule(rule="Never use placeholders in code")

# Auto-inferred scope (based on content keywords)
save_memory(content="Always commit frequently")  # → global_rules
save_memory(content="User prefers Python")  # → global_preferences
save_memory(content="Uses ChromaDB")  # → project_context
```

### Searching Memories

```python
# Search with priority: global rules → preferences → personas → project
search_memories(query="coding standards", top_k=5)
# Returns: [{\"content\": \"...\", \"scope\": \"global_rules\", \"id\": \"...\"}, ...]

# Search specific scope only
search_memories(query="preferences", top_k=3, scope_filter="global_preferences")
```

## Storage Locations

```
~/.gemini/memories/
├── global/
│   ├── rules/          # Global coding rules
│   ├── preferences/    # User preferences  
│   └── personas/       # Reusable personalities
└── projects/
    └── {project_hash}/
        ├── context/    # Project-specific facts
        └── sessions/   # Temporary session data
```

## Memory Scopes

| Scope | Value | Use Case |
|-------|-------|----------|
| GLOBAL_RULES | `global_rules` | Coding standards, instructions |
| GLOBAL_PREFERENCES | `global_preferences` | Personal preferences |
| GLOBAL_PERSONAS | `global_personas` | Reusable personalities/roles |
| PROJECT_CONTEXT | `project_context` | Project architecture, dependencies |
| PROJECT_SESSIONS | `project_sessions` | Current task, recent changes |

## Scope Inference

Content keywords trigger automatic scope classification:

- **Rules**: `always`, `never`, `must`, `should`, `rule`
- **Preferences**: `prefer`, `like`, `favorite`, `is a`
- **Personas**: `role:`, `persona:`, `act as`, `you are`
- **Default**: Project context

## Advanced Usage

### Direct Python API

```python
from memory.scoped_manager import ScopedMemoryManager
from memory.memory_config import MemoryScope

manager = ScopedMemoryManager(project_root="/path/to/project")

# Save to specific scope
memory_id = manager.save("Content", scope=MemoryScope.GLOBAL_RULES)

# Search across all scopes
results = manager.search("query", k=5)

# List all in scope
all_rules = manager.list_all(MemoryScope.GLOBAL_RULES)

# Delete from scope
manager.delete(memory_id, scope=MemoryScope.GLOBAL_RULES)
```

### Changing System Context (Gemini)

To modify global rules that apply to all sessions:

```bash
# In any Gemini chat session
#global:rule:Keep responses under 100 words
#global:rule:Always ask for clarification when unsure
#global:pref:Masih prefers concise technical explanations
```

These persist across projects and sessions automatically.

## Testing

Run comprehensive tests:

```bash
cd /home/masih/Desktop/projects/A-Modular-Kingdom
uv run python -m pytest src/memory/test_scoped_memory.py -v
```

Tests verify:
- ✅ Scope isolation (global != project)
- ✅ Search prioritization (global first)
- ✅ Prefix parsing (`#global:rule:content`)
- ✅ Scope inference from keywords
- ✅ CRUD operations per scope

## Migration from Legacy

See `implementation_plan.md` section on migration for converting existing flat `agent_memories` to scoped structure.
