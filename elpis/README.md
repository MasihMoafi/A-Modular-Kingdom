# Elpis (The Frontend Terminal Dashboard)

## 📖 Definition
**Elpis** is the active "Brain" and interactive terminal dashboard of the A-Modular-Kingdom (AMK) ecosystem. While the AMK Python backend handles long-term storage, indexing, and deep RAG, Elpis handles the **Short-Term Working Memory** and user interaction.

Built in Rust (originally starting as `amk-tui`), Elpis is a lightning-fast, terminal-native environment designed to prevent "terminal bloat" and keep LLM context pristine.

## 🌟 Core Principles & Features

### 1. Dynamic Context Management (Sliding Window)
Inspired by OpenClaw's `KEEP_PRUNED_CONNECTIONS`, Elpis aggressively scrubs useless context. Every API call is a fresh context window. Instead of dumping endless conversation logs into the LLM, Elpis uses Tail-Context Preservation, actively truncating massive payloads and old history with `...<truncated>` so the agent stays fast and focused.

### 2. Signal Extraction
Elpis features a pre-processing step where a lightweight, fast LLM strips the pure signal and action items from a noisy or distracted user input. It filters the "chatter" so the heavy-lifting agent only sees deterministic, pure action intents.

### 3. Execution Policies & Slash Commands
Inspired by OpenAI's Codex CLI:
- **Execution Policy `[Y/n]`:** Elpis refuses to run dangerous or permanent shell commands without interactive user approval in the terminal.
- **Slash Commands:** Seamless UI macros (e.g., `/plan`, `/goal`, `/auth`) that allow the user to shift agent states or switch backend models (from local Ollama to OpenRouter) instantly.

### 4. Simplicity is the Ultimate Sophistication
Do not overload agents with 30+ tools. Elpis limits tool access contextually based on the user's workspace, keeping the AI's operating system minimal while allowing the LLM its natural verbosity for robust reasoning.

---

*This directory serves as the conceptual home for Elpis design docs. The actual Rust implementation lives in the `amk-tui` repository.*
