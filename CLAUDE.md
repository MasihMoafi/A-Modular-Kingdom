# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A-Modular-Kingdom is an AI infrastructure for building multi-agent systems. It provides unified access to long-term memory, RAG, and tools through the Model Context Protocol (MCP).

## Common Commands

### Installation
```bash
uv sync                    # Install dependencies (preferred)
pip install -e .           # Alternative with pip
```

### Running the System
```bash
# Start MCP server (foundation layer)
python src/agent/host.py

# Start interactive client (separate terminal)
python src/agent/main.py
```

### memory-mcp Package (packages/memory-mcp/)
```bash
cd packages/memory-mcp

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Format
black src/

# Type check
mypy src/memory_mcp --ignore-missing-imports
```

### Multi-Agent Systems

**Council Chamber (hierarchical):**
```bash
python multiagents/council_chamber/code_agent_server.py        # Port 8000
python multiagents/council_chamber/enhanced_sexy_teacher_server.py  # Port 8001
python multiagents/council_chamber/queen_juliette.py           # Coordinator
```

**Gym (sequential/CrewAI):**
```bash
cd multiagents/gym
python setup.py && python main.py  # Web UI at localhost:8000
```

## Architecture

```
┌─────────────────────────────────────┐
│       Multi-Agent Layer             │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Council   │  │     Gym     │   │
│  │   Chamber   │  │  (CrewAI)   │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
           │                │
           ▼                ▼
┌─────────────────────────────────────┐
│        Foundation Layer             │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │   RAG   │ │ Memory  │ │ Tools  │ │
│  └─────────┘ └─────────┘ └────────┘ │
│              host.py                │
└─────────────────────────────────────┘
```

### Key Entry Points
- `src/agent/host.py` - MCP server exposing all tools (memory, RAG, code exec, vision, TTS/STT, browser)
- `src/agent/main.py` - Interactive chat client with auto-completion
- `packages/memory-mcp/` - Standalone MCP package (published as `rag-mem` on PyPI)

### RAG System (src/rag/)
Three implementations with different strategies:
- **V1** (`core.py`) - Chroma + BM25 ensemble
- **V2** (`core_2.py`) - Qdrant + BM25 + CrossEncoder reranking
- **V3** (`core_3.py`) - Qdrant + BM25 + RRF fusion + LLM reranking

### Memory System (src/memory/)
Mem0-based persistent memory with ChromaDB, automatic fact extraction, and semantic search.

### Tools (src/tools/)
- `web_search.py` - DuckDuckGo integration
- `code_exec.py` - Safe Python sandbox
- `vision.py` - Image/video analysis via Ollama
- `tts.py` / `stt.py` - Voice capabilities (Kokoro, Whisper)
- `browser_agent_playwright.py` - Web automation

## Code Style

- Line length: 100 characters
- Linter: ruff (select: E, F, I, W)
- Formatter: black
- Type checker: mypy
- Target Python: 3.10+

## Development Guidelines

- Use `uv` for Python execution and package management
- Commit frequently with descriptive messages
- Run lint/typecheck after code changes
- Only read headers of CSV files (never full content)
- Keep interactions concise; ask for clarity when needed
