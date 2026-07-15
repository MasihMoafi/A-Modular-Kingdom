# Rust TUI Client Context (`elpis` CLI)

This directory contains the source code for the Elpis Rust Terminal User Interface (TUI). It acts as a unified frontend dashboard for the Elpis Python host, Codex, and Kiro.

---

## 🛠️ Key UI/UX Features & Keybindings

### 1. Focus Navigation & Panes
- **Switch Focus:** Press **`Tab`** or **`Esc`** to cycle focus between the **Message Input Box** and the **Context Files Checklist** sidebar.
- **Visual Cues:** The active pane gets a **Yellow** border, and the inactive pane gets a **Blue** or **Dark Gray** border.

### 2. Context Files Checklist Panel (Yazi-Style)
- **Navigate:** Use **`j`** / **`k`** or the **`Up`** / **`Down`** arrow keys to highlight files when the Checklist panel is focused.
- **Toggle Selection:** Press **`Space`** or **`Enter`** to select or deselect a file (`[x]` vs `[ ]`).
- **Persistent Session Loading:** Toggled files remain loaded *persistently* for all subsequent queries in the active TUI session (injected as prepended system context) until manually unchecked.
- **File Inspection (Yazi Open/View):** Press **`o`** or **`v`** on any highlighted file to inspect and render its full contents directly inside the Chat History logs.

### 3. Message Input Editing Controls
- **Blinking Cursor:** Displays a blinking block cursor inside the active input area.
- **Ctrl+U:** Clears the entire message input line.
- **Ctrl+Z:** Undo-restores the last message cleared by `Ctrl+U`.
- **Ctrl+C Copying:** The default `Ctrl+C` exit hook has been removed from the application event loop, allowing you to select chat lines with your mouse and use the terminal's native copy shortcuts seamlessly.
- **Quitting:** Exit the client cleanly by typing **`:qa`** in the message input and pressing Enter, or by pressing **`Ctrl+D` twice**.

### 4. Dropdown Autocomplete Completions Menu
- **Slash Commands Autocomplete:** Typing **`/`** triggers a completion menu showing all valid commands (`/skills`, `/agent`, `/sys`, `/setsys`, `/auth`).
- **Mentions Autocomplete:** Typing **`@`** triggers a list of available context files (`AGENTS.md`, `progress.md`, etc.).
- **Navigation:** Use **`Tab`** / **`Down`** (next choice) or **`Up`** (previous choice) to cycle through completion items.
- **Selection:** Press **`Enter`** to autocomplete and fill the text box. Press **`Esc`** to close/discard completions.

---

## 🔌 Provider Authentication & Active Switching (`/auth`)

The Elpis agent defaults to using local Ollama (`qwen3:8b`), but supports active switching to **OpenRouter** APIs:
- **Command:** Type `/auth open-router` to switch providers, or `/auth ollama` to switch back.
- **Authentication:** OpenRouter pulls its key directly from `~/.openrouter_api_key` or your active shell environment (`OPENROUTER_API_KEY`).
- **Default OpenRouter Model:** `google/gemini-2.5-flash` (customizable in arguments).
- **Context Status:** Typing `/status` prints context window limit metrics, estimated context tokens used, and remaining context window space (character approximation `chars / 4` logic in `main.py`).
- **Live Context Indicator:** The header bar always shows `Context: NN% left`, computed client-side in the Rust TUI (same chars/4 heuristic and per-model limits as `/status`), so you don't need to type a command to check it.

---

## 🏗️ Architecture & Commands

- **Path:** `/home/masih/Desktop/f/p/Elpis/tui`
- **TUI Source:** `src/main.rs` (Crossterm event processing + Ratatui UI drawing)
- **Compile command:** `cargo build --release`
- **Run command:** `cargo run --release`
