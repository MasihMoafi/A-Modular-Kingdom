# 🏰 A-Modular-Kingdom

**Stop rebuilding RAG, memory, and tools for every AI project**

A production-ready AI infrastructure that gives you everything you need to build intelligent agents: RAG with smart reindexing, persistent memory across conversations, browser automation, web search, vision, and code execution—all accessible through the Model Context Protocol (MCP).

## 🎯 Why This Exists

Building AI agents means solving the same problems repeatedly:
- 📚 **RAG**: Indexing documents, handling updates, semantic search
- 🧠 **Memory**: Persisting context across sessions, semantic retrieval
- 🛠️ **Tools**: Browser automation, web search, vision, code execution
- 🔌 **Integration**: Connecting everything reliably

**A-Modular-Kingdom solves this once.** Use it as a foundation for any AI project.

## ✨ What You Get

- **📚 Production RAG**: 3 versions (FAISS+BM25, custom indexes, LLM reranking), smart file-change detection, supports .py/.md/.txt/.pdf
- **🧠 Persistent Memory**: Two-tier system (curated prompts + conversation memory), semantic search, ChromaDB + Ollama embeddings
- **🌐 Browser Automation**: Clean Playwright integration, proper Unicode handling, 2-5s execution, no LLM dependency
- **🔍 Web Search**: DuckDuckGo integration, structured results
- **👁️ Vision**: Ollama multimodal analysis for images/videos
- **⚡ Code Execution**: Sandboxed Python execution with timeout
- **🎤 Voice**: Text-to-speech (pyttsx3/gtts/kokoro) and speech-to-text (Whisper)
- **🔌 MCP Protocol**: Standard interface for AI tools and agents

## 🎬 Quick Demo

```python
# Use RAG to search your codebase
from mcp import query_knowledge_base

result = query_knowledge_base(
    query="how does browser automation work",
    doc_path="./tools",
    version=2  # FAISS + BM25 + CrossEncoder
)

# Browser automation with clean output
from mcp import browser_automation

result = browser_automation(
    task="Go to https://github.com/trending and get top 3 repos",
    headless=True
)

# Persistent memory across sessions
from mcp import save_memory, search_memories

save_memory("User prefers concise responses")
results = search_memories("user preferences", top_k=3)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│       Your AI Application           │
│   (Agents, Chatbots, Workflows)     │
└─────────────────────────────────────┘
                 │
                 │ MCP Protocol
                 ▼
┌─────────────────────────────────────┐
│         A-Modular-Kingdom           │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │   RAG   │ │ Memory  │ │ Tools  │ │
│  │  V1/V2  │ │ ChromaDB│ │Browser │ │
│  │  /V3    │ │ +Ollama │ │Vision  │ │
│  └─────────┘ └─────────┘ └────────┘ │
│              host.py (MCP Server)   │
└─────────────────────────────────────┘
```

**Key Features:**
- **MCP Standard**: Works with any MCP-compatible client (Claude Desktop, custom agents)
- **Modular**: Use only what you need (RAG only, memory only, or full stack)
- **Production-Ready**: Smart reindexing, error handling, proper Unicode support
- **No Vendor Lock-in**: Local Ollama models, open-source tools

## 🚀 Quick Start

### Prerequisites
```bash
# Required
Python 3.11+
Ollama (for embeddings: ollama pull embeddinggemma)

# Recommended
uv (Python package manager)
```

### Installation

```bash
# Clone and install
git clone https://github.com/MasihMoafi/A-Modular-Kingdom.git
cd A-Modular-Kingdom/A-Modular-Kingdom

# Install dependencies
uv sync  # or: pip install -r requirements.txt

# Start MCP server
python agent/host.py
```

### Usage Options

**1. As MCP Server (Recommended)**
```json
// Add to your MCP client config (e.g., Claude Desktop)
{
  "mcpServers": {
    "a-modular-kingdom": {
      "command": "python",
      "args": ["/path/to/A-Modular-Kingdom/agent/host.py"]
    }
  }
}
```

**2. As Python Library**
```python
from rag.fetch_2 import fetchExternalKnowledge
from memory.core import Mem0

# Use RAG
results = fetchExternalKnowledge("your query", doc_path="./docs")

# Use Memory
memory = Mem0()
memory.add("Important fact to remember")
```

**3. Interactive CLI**
```bash
python agent/main.py  # Chat interface with all tools
```

## 📚 Core Components

### RAG System (3 Versions)

**V2 (Recommended)**: FAISS + BM25 + CrossEncoder
- Smart file-change detection (only reindex when needed)
- Code-aware search (boosts .py files for code queries)
- Supports: .py, .md, .txt, .pdf
- Fast: <1s search time

**V1**: Basic Chroma + BM25
**V3**: Custom indexes + RRF fusion + LLM reranking

```python
# Automatic reindexing when files change
pipeline = get_rag_pipeline(doc_path="./my-project")
results = pipeline.search("how does authentication work")
```

### Memory System

Two-tier architecture:
- **System Prompts**: Curated, high-priority instructions (never auto-deleted)
- **Conversation Memory**: Auto-saved facts from conversations (can be cleared)

```python
memory = Mem0(chroma_path="./agent_chroma_db")

# Add permanent instruction
memory.add_system_prompt("User prefers minimal explanations")

# Search across both tiers
results = memory.search("user preferences", k=3)
```

### Browser Automation

Fast Playwright integration with clean text extraction:
- Removes navigation clutter
- Proper Unicode/emoji support
- 2-5s execution time
- No LLM dependency

```python
result = await browse_web_playwright(
    "Go to https://github.com/trending",
    headless=True
)
# Returns: clean text + screenshot
```

### Tools

| Tool | Description | Tech |
|------|-------------|------|
| `query_knowledge_base` | RAG search | FAISS, BM25, CrossEncoder |
| `search_memories` | Semantic memory search | ChromaDB, Ollama |
| `browser_automation` | Web scraping | Playwright |
| `web_search` | Internet search | DuckDuckGo |
| `analyze_media` | Vision analysis | Ollama multimodal |
| `code_execute` | Python sandbox | subprocess |
| `text_to_speech` | TTS | pyttsx3/gtts/kokoro |
| `speech_to_text` | STT | Whisper |

## 🎯 Use Cases

**1. AI Coding Assistant**
- RAG searches your codebase for relevant functions
- Memory remembers your coding style and preferences
- Browser automation fetches documentation
- Code execution tests snippets

**2. Research Assistant**
- Web search finds papers and articles
- Browser automation extracts content from websites
- RAG indexes and searches your research library
- Memory tracks research threads across sessions

**3. Multi-Agent Systems**
- Shared memory across multiple agents
- Common tool access (no rebuilding infrastructure)
- MCP protocol for reliable communication
- Examples: Council Chamber (hierarchical), Gym (sequential)

## 📊 Performance

| Component | Speed | Notes |
|-----------|-------|-------|
| RAG Search | <1s | FAISS + BM25, reranking |
| Memory Search | <100ms | ChromaDB semantic search |
| Browser Automation | 2-5s | Playwright, clean output |
| Code Execution | <15s | Configurable timeout |

## 🛠️ Development

```bash
# Run tests
python test_tools.py

# Check tool status
cat TOOLS_STATUS.md

# View architecture
cat A_Modular_Kingdom_Report.html
```

## 📖 Documentation

- **[RAG Versions](rag/)**: V1, V2, V3 implementations and benchmarks
- **[Memory System](memory/)**: Two-tier architecture details
- **[Tools](tools/)**: Individual tool documentation
- **[MCP Integration](agent/host.py)**: Server implementation

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional RAG strategies
- New tool integrations
- Performance optimizations
- Documentation improvements

## 📜 License

[Add your license here]

## 🔗 Related Projects

- **[Voice-commander](https://github.com/MasihMoafi/Voice-commander)**: Local voice transcription with AI refinement
- **[Eyes-Wide-Shut](https://github.com/MasihMoafi/Eyes-Wide-Shut)**: LLM security research and red-teaming

---

**Built with focus on production-readiness, modularity, and developer experience** 🏰

