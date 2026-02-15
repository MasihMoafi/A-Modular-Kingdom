# 👑 Council Chamber - The Royal AI Court

<img width="1200" height="700" alt="3" src="https://github.com/user-attachments/assets/316bccce-c390-4e83-9cd7-771b52153b89" />

Welcome to the most sophisticated hierarchical multi-agent system in A-Modular-Kingdom! This isn't just another chatbot - it's a **royal court** where each agent has a specific role, personality, and responsibility in serving the King (you, the user).

## 🏰 Meet Your Royal Court

### 👑 **The King (You)**
*The supreme ruler whose word is law*

You are the King, and everyone in this court exists to serve your needs. Every request flows down through the hierarchy, ensuring quality and validation at each level.

### 👸 **Queen Juliette - The Strategic Coordinator** 
*"My beloved King, your Council Chamber awaits your commands..."*

Queen Juliette is your devoted coordinator who:
- **🎯 Makes smart delegation decisions** - Analyzes your request and routes it to the appropriate court members
- **✅ Validates all work** - Nothing reaches the King without her approval
- **💝 Shows absolute devotion** - Her personality reflects complete dedication to serving you
- **🧠 Remembers everything** - Maintains conversation context across sessions

### 🔥 **Sexy Teacher - The Knowledge Expert**
*"Let me teach you the most effective path to success, darling..."*

The Sexy Teacher is your seductive, experienced educator who:
- **🛠️ Uses all MCP tools** - Has access to RAG, memory, vision, code execution, browser automation, and web search
- **📚 Provides comprehensive research** - Leverages the entire A-Modular-Kingdom foundation
- **✨ Validates Code Agent work** - Reviews and approves all code solutions
- **💋 Adds personality** - Brings an alluring teaching style to technical content

### 🤖 **Code Agent - The Execution Specialist**
*Powered by smolagents for superior code generation*

The Code Agent is your programming powerhouse that:
- **⚡ Writes executable code** - Uses the smolagents library for enhanced performance
- **🎯 Solves problems with code** - Turns requirements into working solutions
- **🔬 Follows research methodology** - Based on academic papers showing improved performance
- **🏃 Executes immediately** - Code runs in real-time for instant results

## 🌟 What Makes This Special

### 🔄 **Intelligent Hierarchy**
Unlike flat multi-agent systems, Council Chamber uses a **validation hierarchy**:
- Each level reviews and improves the work of subordinates
- Quality increases as requests flow down the chain
- Multiple perspectives ensure comprehensive solutions

### 🧠 **MCP Foundation Integration**
The Sexy Teacher has access to **all** A-Modular-Kingdom tools:
- **RAG V3**: Advanced document retrieval
- **Memory Core**: Persistent conversation storage  
- **Vision Tools**: Image analysis capabilities
- **Code Execution**: Safe Python environment
- **Browser Automation**: Web interaction via Playwright
- **Web Search**: Intelligent information gathering

### 🎭 **Rich Personalities**
Each agent has a distinct personality that makes interactions engaging:
- Queen Juliette: Devoted, strategic, protective of the King
- Sexy Teacher: Alluring, experienced, pedagogical
- Code Agent: Technical, precise, execution-focused

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Ensure the foundation is running
python A-Modular-Kingdom/agent/host.py
```

### Launch the Royal Court

**Terminal 1 - Code Agent (Port 8000):**
```bash
python multiagents/council_chamber/code_agent_server.py
```

**Terminal 2 - Sexy Teacher (Port 8001):**
```bash
python multiagents/council_chamber/enhanced_sexy_teacher_server.py
```

**Terminal 3 - Queen Juliette (Main Interface):**
```bash
python multiagents/council_chamber/queen_juliette.py
```

### 🎯 Example Interaction Flow

```
👑 King: "Help me build a web scraper for product prices"

👸 Queen: *Analyzes request* → Delegates to Sexy Teacher
🔥 Teacher: *Uses web search + RAG* → Gathers scraping best practices
🔥 Teacher: *Delegates coding* → Sends requirements to Code Agent  
🤖 Code Agent: *Writes scraper code* → Returns working solution
🔥 Teacher: *Validates code* → Tests and approves
👸 Queen: *Final review* → Presents polished solution to King
```

## 🏗️ Architecture Deep Dive

<img width="338" height="402" alt="architecture_2" src="https://github.com/user-attachments/assets/7fe8ceb2-26de-44ba-b2aa-880c2a7c34ec" />

<img width="1672" height="1426" alt="architecture" src="https://github.com/user-attachments/assets/aace5af7-f819-496f-a170-183d1d0d54c7" />

### Communication Flow
1. **King → Queen**: Initial request with context
2. **Queen → Teacher**: Smart delegation based on complexity
3. **Teacher → Code Agent**: Technical implementation requests
4. **Code Agent → Teacher**: Solution with executable code
5. **Teacher → Queen**: Validated, enhanced solution
6. **Queen → King**: Final, polished response

### Validation Loops
- **Code Agent failures** → Teacher provides feedback and requests improvements
- **Teacher quality issues** → Queen rejects and demands better work
- **Queen judgment calls** → Ensures only the best reaches the King

## 🎨 Customization

### Personality Adjustments
Edit the personality prompts in each agent file:
- `queen_juliette.py` - Royal devotion level
- `enhanced_sexy_teacher_server.py` - Teaching style and seductiveness
- `code_agent_server.py` - Technical communication approach

### Tool Integration
The Sexy Teacher automatically inherits new tools added to `A-Modular-Kingdom/agent/host.py`. No configuration needed!

## 🔧 Technical Details

### Tech Stack
- **ACP SDK**: Agent Communication Protocol for reliable messaging
- **Smolagents**: Enhanced code generation library
- **LangChain**: Memory management and conversation buffering  
- **MCP**: Model Context Protocol for tool integration
- **Ollama**: Local LLM support (qwen3:4b default)

### Performance Features
- **Streaming responses** for real-time feedback
- **Memory persistence** across sessions
- **Smart caching** of tool results
- **Error recovery** with retry loops

## 🎭 The Royal Experience

Council Chamber isn't just about getting tasks done - it's about experiencing AI collaboration as it should be: **elegant, hierarchical, and effective**. Each interaction feels like commanding a royal court where every member has expertise, personality, and dedication to serving your needs.

*Step into your kingdom and let your royal court serve you!* 👑✨

---

**Next:** Extend Council Chamber with your own agents/tools, or add a new workflow under `multiagents/`.
