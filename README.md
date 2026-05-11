# 🏰 A-Modular-Kingdom

> **Production-ready MCP server with RAG, memory, and tools**

Stop rebuilding the same infrastructure. Connect any AI agent to long-term memory, document retrieval, and 8+ powerful tools through the Model Context Protocol.

## The Problem

Building AI agents? You keep reinventing:
- **Long-term memory** that persists across sessions
- **Document retrieval (RAG)** for knowledge access
- **Tool integration** (web search, vision, code execution, browser automation)

Every project starts from scratch. Every agent rebuilds the wheel.

## The Solution

**A-Modular-Kingdom** is the infrastructure layer you're missing:

```bash
# Start the MCP server
python src/agent/host.py
```

Now any agent (Claude Desktop, custom chatbots, multi-agent systems) gets instant access to:
- ✅ Hierarchical memory (global rules, project context)
- ✅ 3 RAG implementations (v1/v2/v3) for document search
- ✅ 8 production-ready tools via MCP protocol

**One foundation. Infinite applications.**

---

## 🏗️ Architecture

<img width="2275" alt="architecture" src="https://github.com/user-attachments/assets/6e4eaca7-0cae-43b8-a60d-fc8bdfe8c77e" />

---

## 📑 Table of Contents

- [✨ Core Features](#-core-features)
- [🚀 Quick Start](#-quick-start)
- [🛠️ Available Tools](#️-available-tools)
- [📚 RAG System](#-rag-system)
- [🧠 Memory System](#-memory-system)
- [📦 Package Installation](#-package-installation)
- [🎯 Integration Examples](#-integration-examples)
- [🤖 Example Applications](#-example-applications)
- [🤝 Contributing](#-contributing)

---

## ✨ Core Features

- **MCP Protocol** - Standard interface for AI tool access
- **3 RAG Versions** - Choose your retrieval strategy (FAISS, Qdrant, custom)
- **Scoped Memory** - Global rules, preferences, project-specific context
- **8+ Tools** - Vision, code exec, browser, web search, TTS/STT, and more
- **No Vendor Lock-in** - Local Ollama models, open-source stack
- **Production Ready** - Smart reindexing, Unicode support, error handling

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
Python 3.10+
Ollama (for embeddings: ollama pull embeddinggemma)

# Optional
UV package manager (faster than pip)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MasihMoafi/A-Modular-Kingdom.git
cd A-Modular-Kingdom

# Install dependencies
uv sync
# or: pip install -e .

# Pull required Ollama model
ollama pull embeddinggemma
```

### Start the MCP Server

```bash
# Start host.py MCP server
python src/agent/host.py
```

### Connect Your Agent

**Option 1: Claude Desktop**
```json
// Add to claude_desktop_config.json
{
  "mcpServers": {
    "a-modular-kingdom": {
      "command": "python",
      "args": ["/full/path/to/A-Modular-Kingdom/src/agent/host.py"]
    }
  }
}
```

**Option 2: Interactive Client**
```bash
# Use the included chat interface
python src/agent/main.py
```

**Option 3: Custom Integration**
```python
# Connect via MCP in your own agent
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command="python",
    args=["/path/to/host.py"]
)
# Use with ToolCollection.from_mcp(server_params)
```

---

## 🛠️ Available Tools

The MCP server exposes these tools:

| Tool | Description | Use Case |
|------|-------------|----------|
| `query_knowledge_base` | RAG search (v1/v2/v3) | "How does auth work in this codebase?" |
| `save_memory` | Scoped memory storage | Save global rules or project context |
| `search_memories` | Semantic memory search | Retrieve past decisions/preferences |
| `web_search` | DuckDuckGo search | Current events, latest docs |
| `browser_automation` | Playwright web scraping | Extract text/screenshot from URLs |
| `code_execute` | Safe Python sandbox | Run code in isolated environment |
| `analyze_media` | Vision with Ollama | Analyze images/videos |
| `text_to_speech` | TTS (pyttsx3/kokoro) | Generate audio from text |
| `speech_to_text` | Whisper STT | Transcribe audio files |

---

## 📚 RAG System

Three implementations with different trade-offs:

### V1 - Simple & Fast
- **Stack:** FAISS + BM25
- **Speed:** <1s
- **Use Case:** Small projects, quick prototypes

### V2 - Production (Recommended)
- **Stack:** Qdrant + BM25 + CrossEncoder reranking
- **Speed:** <1s with smart caching
- **Use Case:** Production apps, large codebases
- **Features:** Smart reindexing, cloud-ready

### V3 - Advanced (Highest Accuracy)
- **Stack:** Qdrant + BM25 + RRF fusion + CrossEncoder reranking
- **Speed:** <1s (cached), 6.7s (cold start)
- **Use Case:** Maximum accuracy, complex queries
- **Features:** Contextual retrieval, hybrid fusion

### Benchmark Results (LLM-as-Judge)

| Metric | V2 | V3 |
|--------|-----|-----|
| **Groundedness** | 100% | 100% |
| **Relevance** | 80-98% | 78-88% |
| **Completeness** | 75-95% | 75-98% |
| **Average** | **84-98%** | **84-88%** |

*Evaluated with curated queries on Napoleon.pdf and RAG documentation. Judge: Gemini 2.5 Flash. Results vary based on indexed content.*

**Usage:**
```python
# Via MCP tool
query_knowledge_base(
    query="How does authentication work?",
    version="v2",  # or "v1", "v3"
    doc_path="./src"  # optional
)
```

**Supported Files:** `.py`, `.md`, `.txt`, `.pdf`, `.ipynb`, `.js`, `.ts`

---

## 🧠 Memory System

Hierarchical scoped memory with automatic categorization:

### Memory Scopes

| Scope | Persistence | Use Case |
|-------|-------------|----------|
| **Global Rules** | Forever, all projects | "Always use type hints" |
| **Global Preferences** | Forever, all projects | "Prefer dark mode" |
| **Global Personas** | Forever, all projects | Reusable agent personalities |
| **Project Context** | Current project | Architecture decisions, tech stack |
| **Project Sessions** | Temporary | Current task, recent changes |

### Usage

```python
# Save with explicit scope
save_memory(content="Always validate user input", scope="global_rules")

# Or use prefix shortcuts
save_memory(content="#global:rule:Never use eval()")
save_memory(content="#project:context:Uses FastAPI backend")

# Auto-inference from keywords
save_memory(content="User prefers Python 3.12")  # → global_preferences

# Search with priority (global → project)
search_memories(query="coding standards", top_k=5)
```

**Storage:** `~/.modular_kingdom/memories/` (global) + project-specific folders

---

## 📦 Package Installation

> **Coming soon:** `pip install rag-mem` (PyPI release in progress)

Install from source in the meantime:

```bash
cd packages/memory-mcp
pip install -e .
```

---

## 🎯 Integration Examples

### Claude Desktop

Already using Claude Code? Add A-Modular-Kingdom tools:

```json
{
  "mcpServers": {
    "a-modular-kingdom": {
      "command": "python",
      "args": ["/path/to/src/agent/host.py"]
    }
  }
}
```

Now Claude has access to your codebase RAG, persistent memory, and all tools.

### Gemini CLI

```json
// gemini-extension.json
{
  "mcpServers": {
    "unified_knowledge_agent": {
      "command": "python",
      "args": ["/path/to/src/agent/host.py"]
    }
  }
}
```

### Custom Agent

```python
from smolagents import ToolCallingAgent, ToolCollection
from mcp import StdioServerParameters

# Connect to MCP server
params = StdioServerParameters(
    command="python",
    args=["/path/to/host.py"]
)

with ToolCollection.from_mcp(params) as tools:
    agent = ToolCallingAgent(tools=list(tools.tools))
    result = agent.run("Search the codebase for auth logic")
```

---

## 🤖 Example Applications

This repository includes example multi-agent systems built on the foundation:

### Council Chamber (Hierarchical)
- 3-tier agent hierarchy (Queen → Teacher → Code Agent)
- Validation loops and task delegation
- Uses ACP SDK + smolagents
- **Location:** `multiagents/council_chamber/`

### Gym (Sequential)
- Fitness planning workflow (Interview → Plan → Nutrition)
- CrewAI-powered coordination
- Web interface included
- **Location:** `multiagents/gym/`

**Note:** These are demonstration applications, not the core product. The foundation (`host.py`) is the main offering.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     Your AI Application             │
│  (Agents, Chatbots, Workflows)      │
└────────────┬────────────────────────┘
             │ MCP Protocol
┌────────────▼────────────────────────┐
│      A-Modular-Kingdom              │
│  ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │   RAG   │ │ Memory  │ │ Tools  ││
│  │ V1/V2/V3│ │ Scoped  │ │ 8+     ││
│  └─────────┘ └─────────┘ └────────┘│
│           host.py (MCP Server)      │
└─────────────────────────────────────┘
```

---

## 🧪 Testing & Performance

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_rag_v2.py -v
pytest tests/test_rag_v3.py -v
pytest tests/test_memory_global.py -v

# Run benchmarks
python tests/benchmark_rag.py
```

### Performance

**Benchmark Results (GPU/CUDA):**

| Version | Docs | Cold Start | Warm Query |
|---------|------|------------|------------|
| V2 | 100 | 26.8s | **0.31s** |
| V3 | 100 | 13.9s | **0.02s** (15x faster!) |

**Key Features:**
- ✅ GPU acceleration (CUDA) for embeddings and reranking
- ✅ Smart caching (warm queries <0.5s)
- ✅ Tested with .py, .md, .txt, .ipynb files
- ✅ Global memory access from any directory

**See detailed benchmarks:** [docs/RAG_PERFORMANCE.md](docs/RAG_PERFORMANCE.md)

### Docker Testing

Package verified to work in isolation:

```bash
docker build -f Dockerfile.test -t rag-mem-test .
docker run --rm rag-mem-test
```

---

## 🤝 Contributing

Contributions welcome! Focus areas:

1. **Additional RAG strategies** - New retrieval techniques
2. **New tool integrations** - Expand MCP tool offerings
3. **Performance optimizations** - Speed improvements
4. **Documentation improvements** - Tutorials, examples

### Development Setup

```bash
# Fork and clone
git clone https://github.com/MasihMoafi/A-Modular-Kingdom.git
cd A-Modular-Kingdom

# Create branch
git checkout -b feature/your-feature

# Install dev dependencies
uv sync

# Make changes and test
pytest tests/

# Commit with descriptive message
git commit -m "feat: add new tool"

# Push and create PR
git push origin feature/your-feature
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## Links

- **Medium Article:** https://medium.com/@masihmoafi12/a-modular-kingdom-fcaa69a6c1f0
- **Demo Video:** https://www.youtube.com/watch?v=hWoQnAr6R_E
- **PyPI Package:** [rag-mem](https://pypi.org/project/rag-mem/)

---

*A-Modular-Kingdom: The infrastructure layer AI agents deserve* 🏰
