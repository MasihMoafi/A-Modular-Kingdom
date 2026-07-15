# Project Context & Conversation Status

This file tracks the current implementation status and project structure of the Elpis Rust client and its provider integrations.

---

## 🏗️ Project Architecture & Locations

- **TUI Frontend Client:** `/home/masih/Desktop/f/p/Elpis/tui`
  - **Source Code:** `src/main.rs` (Crossterm events + Ratatui UI rendering loop)
  - **Compile Command:** `cargo build --release`
  - **Run Command:** `cargo run --release`
- **Backend MCP Host:** `/home/masih/Desktop/f/p/Elpis`
  - **Agent Script:** `src/agent/main.py` (Ollama/Gemini/OpenRouter handler)

---

## 🛠️ Implemented TUI Features (UI/UX Dept)

1. **Checklist Sidebar (Yazi-Style context selection):**
   - Press **`Tab`** or **`Esc`** to switch active pane focus between the **Input Box** and the **Checklist panel** (active pane has a **Yellow** border, inactive has a **Blue** / **Dark Gray** border).
   - Navigate checklist files using **`j`** / **`k`** or **`Up`** / **`Down`** arrows.
   - Press **`Space`** or **`Enter`** to toggle selection (`[x]` vs `[ ]`). Context files are loaded persistently for *all* subsequent queries in the session.
   - Inspect highlighted file contents directly inside the Chat History log by pressing **`o`** or **`v`**.
   - Removed the `" --- CONTEXT FILES (yazi) ---"` label and `"JULIETTE_RULES.md"` from the right pane checklist to delete redundancies.
   - Removed deprecated `CONTEXT.md` and `README.md` from the list of checklist files.
   - Removed the obsolete `/t` toggle command.

2. **Input Box & System Prompt Editing, Copying & Mouse Selection:**
   - **Blinking Cursor:** Displays a blinking block cursor. If the message input spans multiple lines, the cursor wrapped position (2D offset) is dynamically calculated.
   - **Multi-line Wrapping & Sizing:** The message input box enables text wrapping (`.wrap(Wrap { trim: false })`) and dynamically scales its height (grows up to 10 rows vertically) as your text wraps. The chat history pane dynamically shrinks to accommodate the input box.
   - **Ctrl+U / Ctrl+Z:** Deletes/clears input in both the bottom input area and the System Prompt sidebar editor. Pressing **`Ctrl+Z`** undo-restores the cleared content.
   - **Terminal-Native Mouse Selection:** Explicitly disables mouse capture on startup using `DisableMouseCapture`, ensuring click-and-drag native terminal selection behaves cleanly without application interference.
   - **Ctrl+C clipboard copy:** Pressing **`Ctrl+C`** at any time copies the last agent response directly to the system clipboard using `xclip`, showing a system feedback message. Raw exit handler is removed (use **`:qa`** or double **`Ctrl+D`** to quit).
   - **Real System Prompts:** Rather than injecting system instructions as a user prompt prefix, the backend (`main.py`) dynamically reads `system_prompt.md` on every request and configures it as a native system instruction for the LLM backend (Ollama system role, Gemini system instruction config, OpenRouter system message).
   - **Dynamic Prompt Reloading:** The TUI client reloads the local `system_prompt.md` file from disk as soon as the agent finishes responding, ensuring prompts written by the agent using file-writing tools show up instantly.

3. **Autocomplete completions Dropdown:**
   - Typing **`/`** triggers command completions dropdown showing only `/auth`.
   - **Interactive `/auth` Menu:** Typing `/auth` or pressing `Tab`/`Enter` on it triggers the dropdown showing provider choices (`ollama`, `openrouter`). Selecting a provider automatically updates the input and opens the next dropdown listing the provider's models.
   - Typing **`@`** triggers context files autocomplete dropdown.
   - **Tab Auto-fill:** Pressing **`Tab`** or **`Enter`** autocompletes and fills the selected completion item immediately (instead of just cycling).
   - **Navigate & Cycle:** Use **`Down`** or **`Up`** arrow keys to cycle highlight options.
   - **Auto-close:** The completions menu automatically closes as soon as a final model is selected (or `@` file mention is chosen), allowing you to press **`Enter`** directly to execute the `/auth` command or send your query. Press **`Esc`** to close manually at any time.

---

## 🔌 Authentication & Provider Switching (`/auth`)

- Type `/auth <provider> [model]` (e.g., `/auth open-router google/gemini-3-flash-preview` or `/auth ollama qwen3:8b`) to switch active provider and model. Autocompletion of providers (`ollama`, `openrouter`) is triggered after `/auth `.
- **Models Selection:**
  - **OpenRouter:** Supports `google/gemini-3-flash-preview` (default), `moonshotai/kimi-k2.6`, `deepseek/deepseek-chat`, and `deepseek/deepseek-r1`.
  - **Ollama:** Supports local models `qwen3:8b` and `llama3.2:3b`.
- **TUI Header Info Display:** The top border/header dynamically shows the active provider, model, and workspace (e.g. `Provider: ollama | Model: qwen3:8b | Workspace: /home/masih/...`).
- **OpenRouter Backend Integration:**
  - Standard HTTP requests chat stream with SSE token processing in `src/agent/main.py`.
  - Authenticates via your local key file `/home/masih/.openrouter_api_key`.
- **Context Status Limit:** `/status` command outputs context window limit, estimated used tokens (conversation history + prompt approximation), and remaining tokens.
