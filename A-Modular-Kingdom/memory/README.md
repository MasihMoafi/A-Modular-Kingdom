# Memory System

## Two Collections
1. **system_prompts** - Your curated instructions (manual, high priority)
2. **agent_memories** - Conversation facts (auto-saved, can be cleared)

## Usage
```python
# Add system prompt (permanent instruction)
add_system_prompt("Always use TypeScript")

# Manual save to conversation memory
save_memory("User prefers dark mode")

# Clear conversation memory (keeps system prompts)
clear_conversation_memory()
```

## Search Priority
1. System prompts (searched first)
2. Conversation memory (searched second)

## Embedding: embeddinggemma (Ollama)
