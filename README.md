# 🏰 A-Modular-Kingdom

**The Ultimate AI Multi-Agent Foundation**

A comprehensive monorepo that provides the infrastructure for sophisticated multi-agent AI systems. Built with modularity at its core, A-Modular-Kingdom enables both hierarchical and sequential agent workflows through a unified foundation.

![Architecture](A-Modular-Kingdom/architecture.png)

## 🎯 What Makes This Special

- **🏗️ Unified Foundation**: Single MCP host provides RAG, memory, vision, code execution, browser automation, and web search to all agents
- **🔄 Multiple Paradigms**: Support both hierarchical (Council Chamber) and sequential (Gym) multi-agent workflows  
- **🛠️ Rich Toolset**: Production-ready tools that agents can seamlessly access
- **📚 Persistent Memory**: Cross-session memory and knowledge retention
- **🌐 ACP Communication**: Reliable agent-to-agent communication protocol

---

## 🏛️ Multi-Agent Systems

### 👑 Council Chamber - Hierarchical Intelligence

A sophisticated royal court where agents have defined roles and hierarchy, each validating and enhancing the work of their subordinates.

![Council Chamber](multiagents/council_chamber/2.png)

**Hierarchy:** King (User) → Queen Juliette → Sexy Teacher → Code Agent

**Key Features:**
- **Smart Delegation**: Intelligent task routing based on complexity
- **Validation Loops**: Each level validates subordinate work
- **MCP Integration**: Sexy Teacher accesses all foundation tools
- **Code-First Solutions**: Code Agent writes executable solutions using smolagents

**Location:** `multiagents/council_chamber/`

### 🏋️ Gym - Sequential Specialization  

A fitness-focused multi-agent system where specialized agents work in sequence to provide comprehensive health and fitness guidance.

![Gym](multiagents/gym/3.png)

**Flow:** Interviewer → Plan Generator → Progress Tracker → Motivator → Nutrition Agent

**Key Features:**
- **CrewAI Powered**: Built on the CrewAI framework
- **Domain Expertise**: Each agent specializes in specific fitness areas
- **Web Interface**: Modern, responsive chat interface
- **Flexible LLM**: Works with local Ollama or cloud providers

**Location:** `multiagents/gym/`

---

## 🧠 Foundation - The Core Infrastructure

The heart of A-Modular-Kingdom lives in `A-Modular-Kingdom/` and provides:

### 🖥️ MCP Host (`agent/host.py`)
Central hub exposing all tools through Model Context Protocol:

| Tool | Purpose | Status |
|------|---------|--------|
| **RAG** | Document retrieval & knowledge base | ✅ V3 Ready |
| **Memory** | Persistent conversation storage | ✅ ChromaDB |  
| **Vision** | Image analysis & processing | ✅ Multimodal |
| **Code Exec** | Safe Python execution | ✅ Sandboxed |
| **Browser** | Web automation via Playwright | ✅ Full Control |
| **Web Search** | Intelligent information retrieval | ✅ Integrated |

### 📚 Knowledge Systems
- **RAG V3**: Advanced retrieval with multiple strategies
- **Memory Core**: Long-term conversation and context storage
- **GLOBAL_RULES**: Standardized agent behavior guidelines

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required
Python 3.8+
Ollama (for local LLM)
uv (package manager)
```

### 1. Foundation Setup
```bash
# Start the MCP host (provides tools to all agents)
python A-Modular-Kingdom/agent/host.py
```

### 2. Launch Council Chamber
```bash
# Terminal 1: Code Agent (port 8000)
python multiagents/council_chamber/code_agent_server.py

# Terminal 2: Sexy Teacher (port 8001)  
python multiagents/council_chamber/enhanced_sexy_teacher_server.py

# Terminal 3: Queen Juliette (main interface)
python multiagents/council_chamber/queen_juliette.py
```

### 3. Launch Gym
```bash
cd multiagents/gym/
python setup.py  # First time only
python main.py   # Visit http://localhost:8000
```

---

## 🏗️ Architecture Philosophy

A-Modular-Kingdom follows a **modular foundation** approach:

```
┌─────────────────────────────────────────────┐
│              Multi-Agent Layer              │
│  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Council Chamber │  │      Gym        │   │
│  │  (Hierarchical) │  │  (Sequential)   │   │
│  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              Foundation Layer               │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐   │
│  │   RAG   │ │ Memory  │ │    Tools    │   │
│  └─────────┘ └─────────┘ └─────────────┘   │
│                host.py                      │
└─────────────────────────────────────────────┘
```

**Core Principles:**
- **Standardized Interface**: All systems use the same foundation
- **Tool Sharing**: Common capabilities across all agents
- **Memory Persistence**: Shared context and knowledge
- **Communication Protocol**: ACP for reliable agent interaction

---

## 📖 Documentation

- **[Council Chamber Guide](multiagents/council_chamber/)**: Hierarchical multi-agent setup
- **[Gym Guide](multiagents/gym/)**: Sequential fitness agent system  
- **[RAG Documentation](A-Modular-Kingdom/rag/)**: Knowledge retrieval systems
- **[Memory System](A-Modular-Kingdom/memory/)**: Conversation persistence
- **[Tools Overview](A-Modular-Kingdom/tools/)**: Available agent capabilities

---

## 🤝 Contributing

A-Modular-Kingdom grows through experimentation and real-world testing. Each multi-agent system teaches us more about effective AI coordination and collaboration patterns.

---

*A-Modular-Kingdom: Where AI agents unite under one foundation* 🏰✨
