# Building a Two-Brained Local AI: How We Achieved 93% RAG Accuracy and Banished "Terminal Bloat"

When building an autonomous coding agent that works alongside you in the terminal, you quickly run into two massive roadblocks: **The Knowledge Gap** (how does it learn massive, fragmented codebases?) and **The Context Limit** (how does it not immediately drown in 50,000 lines of its own terminal output?).

We set out to solve this by explicitly decoupling the architecture into two distinct systems, effectively giving the AI two brains: **A Modular Kingdom (AMK)** for long-term knowledge retrieval, and **Elpis**, a Rust-based working memory manager that acts as the active prefrontal cortex.

Here is how we unified hybrid search and aggressive context pruning to build a lightning-fast local agent.

---

## The Long-Term Memory: A Unified Hybrid Architecture

Most local RAG setups force you to choose between dense Vector Search (great for semantic vibes) and BM25 (great for exact keyword matching). We didn't want to choose.

Inside our Python backend (AMK), we built a Unified Hybrid Architecture that fuses Vector and BM25 search natively using **Reciprocal Rank Fusion (RRF)**. We optimized it heavily for our local `qwen3-embedding:8b` model running on Qdrant.

But the real magic wasn't the algorithm; it was the chunk sizing. 
In our ablation testing, pushing 4000-character chunks crashed our local evaluator with context overflow, while 300-character chunks destroyed semantic meaning and tanked retrieval success to ~31%. When we hit the golden ratio of **1000 characters** with a 150-character overlap, our success rate shot to an astonishing **93.3%** on zero-shot "needle-in-a-haystack" queries across massive documents.

We also designed the engine to dynamically swap "Judges". Instead of forcing heavy CrossEncoders for reranking, the architecture natively allows passing the top retrieved chunks through an LLM to actively reason over their relevance before returning them.

---

## The Short-Term Memory: Elpis (The Rust TUI)

If the RAG backend is a library, the Terminal UI (Elpis) is the active workspace. And a workspace gets messy. 

If you feed every single stdout line, every failed bash command, and every repetitive JSON payload back into the LLM, the context window implodes. Worse, the LLM gets distracted. To solve this, we drew heavy inspiration from OpenAI's Codex CLI and the OpenClaw framework.

### 1. Tail-Context Preservation (The Sliding Window)
Memory isn't the context killer; repetitive agent chatter and terminal outputs are. Elpis aggressively scrubs the active context. If an API call dumps a massive payload, Elpis physically truncates the oldest parts and replaces them with `...<truncated>`. It creates a sliding window that ensures the LLM is only ever reacting to the freshest, most relevant state of the terminal.

### 2. Signal Extraction
Users are chaotic. We write typos, give contradictory instructions, and dump massive stack traces into the prompt. Before Elpis triggers any heavy agent workflow, it passes the raw prompt through a "Signal Extractor"—a cheap, fast LLM pass whose only job is to filter the noise and extract the deterministic action intent. 

### 3. Execution Policies
Running local agents is dangerous. Inspired by Codex, Elpis has built-in execution policies. The agent can freely read files and search the web, but any structural bash commands or system modifications require an interactive `[Y/n]` approval directly in the TUI.

---

## The Ultimate Sophistication

By treating RAG and Working Memory as entirely separate engineering challenges, we achieved something remarkable. The system can pull impossibly specific knowledge out of massive codebases at lightning speed (93% accuracy), without ever drowning the active LLM in context bloat. 

We didn't need 30+ overly complicated tools. We just needed a well-tuned algorithm, an aggressive garbage collector for terminal logs, and a clean, fast Rust interface. Simplicity, as it turns out, really is the ultimate sophistication.
