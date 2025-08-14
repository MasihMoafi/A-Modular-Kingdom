# 🏰 A-Modular-Kingdom

**The Foundation for AI-Powered Multi-Agent Systems**

A-Modular-Kingdom is a comprehensive AI infrastructure that provides the building blocks for sophisticated multi-agent workflows. Built with modularity and standardization at its core, it seamlessly connects different multi-agent architectures through a unified foundation.

![Architecture](../architecture.png)

## 🎯 What Makes This Special

This isn't just another AI project - it's a **foundation** that enables:

- **🔗 Seamless Integration**: Multi-agent systems connect to `host.py` for instant access to long-term memory, RAG, and powerful tools
- **🏗️ Modular Architecture**: Build hierarchical (Council Chamber) or sequential (Gym) workflows on the same foundation  
- **🛠️ Rich Toolset**: Vision, code execution, browser automation, web search, and more - all standardized and ready to use
- **📚 Smart Memory**: Persistent memory and RAG systems that work across all your agents
- **🌐 ACP Communication**: Agents communicate through ACP servers for reliable, structured interactions

## 🏛️ Multi-Agent Systems

### 👑 Council Chamber (Hierarchical)
A sophisticated royal court where agents have defined roles and hierarchy:

```
👑 King (User) → 👸 Queen Juliette → 🔥 Sexy Teacher → 🤖 Code Agent
```

**Features:**
- **Hierarchical validation**: Each level validates the work of subordinates
- **Smart delegation**: Intelligent routing based on task complexity  
- **MCP tool integration**: Sexy Teacher uses all foundation tools
- **Code-first approach**: Code Agent writes solutions as executable code using smolagents

[**📖 Learn more about Council Chamber →**](../multiagents/council_chamber/)

### 🏋️ Gym (Sequential) 
A fitness-focused multi-agent system with specialized roles:

```
Interviewer → Plan Generator → Progress Tracker → Motivator → Nutrition Agent
```

**Features:**
- **CrewAI powered**: Built on the CrewAI framework for sequential workflows
- **Specialized agents**: Each agent has a specific fitness domain expertise
- **Web interface**: Modern chat interface for user interaction
- **Flexible LLM support**: Works with local Ollama or cloud providers

[**📖 Learn more about Gym →**](../multiagents/gym/)

## 🧠 Core Infrastructure

### 🖥️ Host.py - The Central Hub
The heart of A-Modular-Kingdom, providing MCP (Model Context Protocol) access to:

- **📚 RAG System**: Advanced document retrieval with multiple strategies (V1, V2, V3)
- **🧠 Memory Core**: Persistent conversation and context memory
- **👁️ Vision Tools**: Image analysis and processing capabilities  
- **⚡ Code Execution**: Safe Python code execution environment
- **🌐 Browser Automation**: Web interaction through Playwright
- **🔍 Web Search**: Intelligent web search capabilities

### 🔧 Tool Ecosystem

| Tool | Purpose | Status |
|------|---------|--------|
| **RAG** | Document retrieval & knowledge | ✅ Multiple versions |
| **Memory** | Long-term conversation storage | ✅ Fully integrated |  
| **Vision** | Image analysis | ✅ Ready to use |
| **Code Exec** | Safe Python execution | ✅ Sandboxed |
| **Browser** | Web automation | ✅ Playwright powered |
| **Web Search** | Information retrieval | ✅ Integrated |
| **Structured Output** | Formatted responses | 🔄 Coming soon |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local LLM)
- UV package manager (recommended)

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd A-Modular-Kingdom

# Install dependencies
uv sync

# Start the foundation
python agent/host.py
```

### Launch Multi-Agent Systems

**Council Chamber:**
```bash
# Terminal 1: Start Code Agent
python council_chamber/code_agent_server.py

# Terminal 2: Start Sexy Teacher  
python council_chamber/enhanced_sexy_teacher_server.py

# Terminal 3: Start Queen Juliette
python council_chamber/queen_juliette.py
```

**Gym:**
```bash
cd gym/
python setup.py
python main.py
```

## 🏗️ Architecture

A-Modular-Kingdom follows a **modular foundation** approach:

![Architecture Diagram](../architecture.png)

```
┌─────────────────────────────────────┐
│          Multi-Agent Layer          │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Council   │  │     Gym     │   │
│  │   Chamber   │  │             │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
           │                │
           ▼                ▼
┌─────────────────────────────────────┐
│         Foundation Layer            │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │   RAG   │ │ Memory  │ │ Tools  │ │
│  └─────────┘ └─────────┘ └────────┘ │
│              host.py                │
└─────────────────────────────────────┘
```

**Key Principles:**
- **Standard Interface**: All multi-agent systems use the same foundation
- **ACP Communication**: Reliable agent-to-agent communication  
- **Tool Sharing**: Common tools available to all agents
- **Memory Persistence**: Shared memory across sessions

## 📖 Documentation

- **[ACP Tutorial](temp/ACP/)**: Learn about Agent Communication Protocol
- **[RAG Documentation](rag/)**: Multiple RAG implementations and evaluations
- **[Memory System](memory/)**: Conversation and context persistence
- **[Tool Documentation](tools/)**: Individual tool guides

## 🤝 Contributing

A-Modular-Kingdom grows through experimentation and iteration. Each multi-agent system teaches us more about effective AI coordination.

## 🔗 External Resources

- **[ACP Tutorial]([])**  
- **[Smolagents Paper]([])**  
- **[MCP Documentation](https://modelcontextprotocol.io/)**

---

*A-Modular-Kingdom: Where AI agents come together in harmony* 🏰✨

