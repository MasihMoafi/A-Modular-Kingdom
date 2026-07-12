use std::{
    io::{self, stdout, Write},
    process::{Command, Stdio},
    sync::mpsc::{self, Sender as MpscSender},
    thread,
    time::Duration,
};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, BorderType, Paragraph, Wrap, Clear},
    Terminal, Frame,
};
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentType {
    Amk,
    Codex,
    Kiro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Input,
    Context,
    SysPrompt,
    Approval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageSender {
    User,
    Agent,
    System,
}

#[derive(Debug, Clone)]
enum TuiEvent {
    StdoutLine(String),
    StderrLine(String),
    ProcessExited(i32),
}

struct ChatMessage {
    sender: MessageSender,
    text: String,
}

#[derive(Debug, Clone)]
struct ApprovalRequest {
    req_type: String, // "write_file" or "execute_command"
    path: Option<String>,
    content: Option<String>,
    command: Option<String>,
}

#[derive(Debug, Clone)]
enum DiffLine {
    Unchanged(String),
    Added(String),
    Deleted(String),
}

fn compute_line_diff(old: &str, new: &str) -> Vec<DiffLine> {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    
    let mut diff = Vec::new();
    let mut i = 0;
    let mut j = 0;
    
    while i < old_lines.len() || j < new_lines.len() {
        if i < old_lines.len() && j < new_lines.len() {
            if old_lines[i] == new_lines[j] {
                diff.push(DiffLine::Unchanged(old_lines[i].to_string()));
                i += 1;
                j += 1;
            } else {
                let mut found_match = false;
                for lookahead in 1..5 {
                    if j + lookahead < new_lines.len() && old_lines[i] == new_lines[j + lookahead] {
                        for k in 0..lookahead {
                            diff.push(DiffLine::Added(new_lines[j + k].to_string()));
                        }
                        j += lookahead;
                        found_match = true;
                        break;
                    }
                    if i + lookahead < old_lines.len() && old_lines[i + lookahead] == new_lines[j] {
                        for k in 0..lookahead {
                            diff.push(DiffLine::Deleted(old_lines[i + k].to_string()));
                        }
                        i += lookahead;
                        found_match = true;
                        break;
                    }
                }
                
                if !found_match {
                    diff.push(DiffLine::Deleted(old_lines[i].to_string()));
                    diff.push(DiffLine::Added(new_lines[j].to_string()));
                    i += 1;
                    j += 1;
                }
            }
        } else if i < old_lines.len() {
            diff.push(DiffLine::Deleted(old_lines[i].to_string()));
            i += 1;
        } else {
            diff.push(DiffLine::Added(new_lines[j].to_string()));
            j += 1;
        }
    }
    
    diff
}

struct App {
    messages: Vec<ChatMessage>,
    input: String,
    model: String,
    workspace: String,
    stdin_tx: Option<MpscSender<String>>,
    in_reply: bool,
    current_reply: String,
    backend_connected: bool,
    backend_error: Option<String>,
    system_prompt: String,
    is_first_message: bool,
    ctrl_d_count: u8,
    context_files: Vec<(String, bool)>,
    agent_type: AgentType,
    focus: Focus,
    context_cursor: usize,
    completions: Vec<String>,
    completion_selected: usize,
    show_completions: bool,
    provider: String,
    input_backup: Option<String>,
    is_command_reply: bool,
    pending_approval: Option<ApprovalRequest>,
    yolo_mode: bool,
}

impl App {
    fn new(stdin_tx: MpscSender<String>, system_prompt: String) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            model: "Detecting...".to_string(),
            workspace: "Detecting...".to_string(),
            stdin_tx: Some(stdin_tx),
            in_reply: false,
            current_reply: String::new(),
            backend_connected: false,
            backend_error: None,
            system_prompt,
            is_first_message: true,
            ctrl_d_count: 0,
            context_files: vec![
                ("AGENTS.md".to_string(), false),
                ("CODEX_CODING_GUIDELINES.md".to_string(), false),
                ("ARTIFACT_RULES.md".to_string(), false),
                ("TERMINAL_AND_GIT_RULES.md".to_string(), false),
                ("readme.md".to_string(), false),
                ("progress.md".to_string(), false),
            ],
            agent_type: AgentType::Amk,
            focus: Focus::Input,
            context_cursor: 0,
            completions: Vec::new(),
            completion_selected: 0,
            show_completions: false,
            provider: "ollama".to_string(),
            input_backup: None,
            is_command_reply: false,
            pending_approval: None,
            yolo_mode: false,
        }
    }

    fn add_message(&mut self, sender: MessageSender, text: String) {
        let pruned_text = if text.len() > 8000 {
            format!("{}...\n<truncated to save context>", &text[..8000])
        } else {
            text
        };

        if sender == MessageSender::System {
            if let Some(last_msg) = self.messages.last_mut() {
                if last_msg.sender == MessageSender::System {
                    let last_base = if let Some(idx) = last_msg.text.find(" (x") {
                        &last_msg.text[..idx]
                    } else {
                        &last_msg.text
                    };
                    if last_base == pruned_text {
                        let count = if let Some(idx) = last_msg.text.find(" (x") {
                            last_msg.text[idx + 3..last_msg.text.len() - 1].parse::<u32>().unwrap_or(1) + 1
                        } else {
                            2
                        };
                        last_msg.text = format!("{} (x{})", last_base, count);
                        return;
                    }
                }
            }
        }

        self.messages.push(ChatMessage {
            sender,
            text: pruned_text,
        });

        self.prune_context();
    }

    fn prune_context(&mut self) {
        let limit = self.context_window_limit();
        let target_tokens = (limit as f64 * 0.7) as i64;

        loop {
            let mut total_chars = self.system_prompt.len() + self.current_reply.len();
            for msg in &self.messages {
                total_chars += msg.text.len();
            }
            let est_tokens = (total_chars / 4) as i64;

            if est_tokens <= target_tokens || self.messages.is_empty() {
                break;
            }

            if let Some(pos) = self.messages.iter().position(|m| m.sender == MessageSender::User || m.sender == MessageSender::Agent) {
                self.messages.remove(pos);
            } else {
                break;
            }
        }
    }

    fn update_completions(&mut self) {
        self.completions.clear();
        self.show_completions = false;

        let input_trimmed = self.input.trim_start();
        if input_trimmed.starts_with('/') {
            let parts: Vec<&str> = input_trimmed.split_whitespace().collect();

            // Suggest main commands (only /auth and /model)
            let is_typing_cmd = parts.len() <= 1 && !self.input.ends_with(' ');
            if is_typing_cmd {
                let cmd_typed = parts.first().copied().unwrap_or("/");
                let commands = vec!["/auth", "/model", "/rag", "/yolo"];
                for cmd in commands {
                    if cmd.starts_with(cmd_typed) {
                        self.completions.push(cmd.to_string());
                    }
                }
            } else if parts.first() == Some(&"/auth") {
                // Interactive /auth selection menu:
                let provs = vec!["ollama", "gemini", "openrouter"];
                let selecting_provider = parts.len() == 1 || (parts.len() == 2 && !self.input.ends_with(' '));

                if selecting_provider {
                    let typed_prov = if parts.len() >= 2 { parts[1] } else { "" };
                    for prov in provs {
                        if prov.starts_with(typed_prov) {
                            self.completions.push(prov.to_string());
                        }
                    }
                }
            } else if parts.first() == Some(&"/model") {
                // Suggest models based on the active provider
                let typed_model = if parts.len() >= 2 {
                    parts[1..].join(" ")
                } else {
                    "".to_string()
                };

                let models = match self.provider.as_str() {
                    "ollama" => vec!["qwen3:8b", "llama3.2:3b"],
                    "gemini" => vec![
                        "gemini-3.5-flash",
                        "gemini-3.5-pro",
                    ],
                    "openrouter" | "open-router" => vec![
                        "google/gemini-3.5-flash",
                        "google/gemini-3.5-pro",
                        "moonshotai/kimi-k2.6",
                        "deepseek/deepseek-v4-pro",
                        "deepseek/deepseek-v4-flash",
                    ],
                    _ => vec![],
                };

                // If a model is fully selected or typed, clear completions
                let is_fully_specified = parts.len() >= 2 && models.contains(&typed_model.as_str());
                if is_fully_specified {
                    self.show_completions = false;
                    self.completions.clear();
                } else {
                    for model in models {
                        if model.to_lowercase().starts_with(&typed_model.to_lowercase()) {
                            self.completions.push(model.to_string());
                        }
                    }
                }
            }
        } else if let Some(last_at_idx) = self.input.rfind('@') {
            let query = &self.input[last_at_idx + 1..];
            if !query.contains(' ') {
                for (name, _) in &self.context_files {
                    if name.to_lowercase().contains(&query.to_lowercase()) {
                        self.completions.push(format!("@{}", name));
                    }
                }
            }
        }

        if !self.completions.is_empty() {
            self.show_completions = true;
            if self.completion_selected >= self.completions.len() {
                self.completion_selected = 0;
            }
        }
    }

    fn select_completion(&mut self) {
        if !self.show_completions || self.completions.is_empty() {
            return;
        }

        let completed = &self.completions[self.completion_selected];
        if completed.starts_with('/') {
            // Auto-completed a command, e.g. "/auth" or "/model". Append a space.
            self.input = format!("{} ", completed);
        } else if completed.starts_with('@') {
            if let Some(last_at_idx) = self.input.rfind('@') {
                self.input = format!("{}{}", &self.input[..last_at_idx], completed);
            }
            self.show_completions = false;
            self.completions.clear();
            self.completion_selected = 0;
            return;
        } else {
            let parts: Vec<&str> = self.input.split_whitespace().collect();
            if parts.first() == Some(&"/auth") {
                // Completed the provider, close the completions
                self.input = format!("/auth {}", completed);
                self.show_completions = false;
                self.completions.clear();
                self.completion_selected = 0;
                return;
            } else if parts.first() == Some(&"/model") {
                // Completed the model, close the completions
                self.input = format!("/model {}", completed);
                self.show_completions = false;
                self.completions.clear();
                self.completion_selected = 0;
                return;
            } else {
                if let Some(last_space_idx) = self.input.rfind(' ') {
                    self.input = format!("{} {} ", &self.input[..last_space_idx], completed);
                } else {
                    self.input = format!("{} ", completed);
                }
            }
        }

        // Re-evaluate completions to automatically keep the menu open if there are next steps
        self.update_completions();
        if self.completions.is_empty() {
            self.show_completions = false;
            self.completion_selected = 0;
        } else {
            self.show_completions = true;
            self.completion_selected = 0;
        }
    }

    fn handle_stdout_line(&mut self, line: String) {
        debug_log(&format!("[STDOUT] {}", line));
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }

        // Intercept ELPIS_REQUEST_APPROVAL
        if trimmed.starts_with("ELPIS_REQUEST_APPROVAL ") {
            let json_str = trimmed.replacen("ELPIS_REQUEST_APPROVAL ", "", 1);
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                let req_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let path = parsed.get("path").and_then(|v| v.as_str()).map(|s| s.to_string());
                let content = parsed.get("content").and_then(|v| v.as_str()).map(|s| s.to_string());
                let command = parsed.get("command").and_then(|v| v.as_str()).map(|s| s.to_string());

                let req = ApprovalRequest {
                    req_type,
                    path,
                    content,
                    command,
                };

                // If YOLO mode is on, auto-approve immediately
                if self.yolo_mode {
                    if let Some(ref tx) = self.stdin_tx {
                        let _ = tx.send("{\"status\": \"accept\"}\n".to_string());
                    }
                    return;
                }

                self.pending_approval = Some(req);
                self.focus = Focus::Approval;
                return;
            }
        }

        if self.agent_type == AgentType::Amk {
            // Match model and workspace info for AMK
            if trimmed.starts_with("Workspace:") {
                self.workspace = trimmed.replacen("Workspace:", "", 1).trim().to_string();
                self.backend_connected = true;
            } else if trimmed.starts_with("Using Ollama") {
                self.provider = "ollama".to_string();
                self.model = trimmed.replacen("Using Ollama", "", 1).trim().to_string();
            } else if trimmed.starts_with("Using Gemini API") {
                self.provider = "gemini".to_string();
                let model_part = trimmed.replacen("Using Gemini API", "", 1).trim().to_string();
                if model_part.starts_with('(') && model_part.ends_with(')') {
                    self.model = model_part[1..model_part.len()-1].trim().to_string();
                } else if !model_part.is_empty() {
                    self.model = model_part;
                } else {
                    self.model = "gemini-3.5-flash".to_string();
                }
            } else if trimmed.starts_with("Using OpenRouter model:") {
                self.provider = "openrouter".to_string();
                self.model = trimmed.replacen("Using OpenRouter model:", "", 1).trim().to_string();
            }

            // Handle streaming response capture for AMK
            if trimmed.starts_with("Juliette:") {
                self.in_reply = true;
                let content = trimmed.replacen("Juliette:", "", 1).trim_start().to_string();
                self.current_reply = content;
            } else if self.in_reply {
                // Check if the agent finished responding and prompt is presented again
                if trimmed.starts_with('>') || trimmed == ">" {
                    self.in_reply = false;
                    let finished = std::mem::take(&mut self.current_reply);
                    if !finished.trim().is_empty() {
                        let sender = if self.is_command_reply {
                            MessageSender::System
                        } else {
                            MessageSender::Agent
                        };
                        self.add_message(sender, finished);
                    }
                    self.is_command_reply = false;
                    // Reload system prompt if modified by agent tools
                    let sys_prompt_file = "/home/masih/Desktop/f/p/Elpis/tui/system_prompt.md";
                    if let Ok(content) = std::fs::read_to_string(sys_prompt_file) {
                        self.system_prompt = content;
                    }
                } else {
                    if !self.current_reply.is_empty() {
                        self.current_reply.push('\n');
                    }
                    self.current_reply.push_str(&line);
                }
            }
        } else {
            // For Codex / Kiro, stream the full stdout as response text
            self.in_reply = true;
            if !self.current_reply.is_empty() {
                self.current_reply.push('\n');
            }
            self.current_reply.push_str(&line);
        }
    }

    fn read_context_file(&self, filename: &str) -> Option<String> {
        let path = match filename {
            "AGENTS.md" => Some("/home/masih/.codex/AGENTS.md"),
            "CODEX_CODING_GUIDELINES.md" => Some("/home/masih/.codex/CODEX_CODING_GUIDELINES.md"),
            "ARTIFACT_RULES.md" => Some("/home/masih/.codex/ARTIFACT_RULES.md"),
            "JULIETTE_RULES.md" => Some("/home/masih/.codex/JULIETTE_RULES.md"),
            "TERMINAL_AND_GIT_RULES.md" => Some("/home/masih/.codex/TERMINAL_AND_GIT_RULES.md"),
            "readme.md" => Some("/home/masih/Desktop/f/p/Elpis/readme.md"),
            "progress.md" => Some("/home/masih/Desktop/f/p/Elpis/progress.md"),
            _ => None,
        };
        path.and_then(|p| std::fs::read_to_string(p).ok())
    }

    /// Approximate token context window for the active model. Mirrors the limits used
    /// by the Python backend's `/status` command so both stay in agreement.
    fn context_window_limit(&self) -> i64 {
        let model_lower = self.model.to_lowercase();
        if model_lower.contains("gemini") {
            1_048_576
        } else if model_lower.contains("deepseek") {
            64_000
        } else {
            32_768 // Default Ollama/Qwen limit
        }
    }

    /// Estimates the percentage of context window remaining using the same chars/4
    /// heuristic the Python backend uses for `/status`, so the live header figure and
    /// `/status` output stay consistent. Recomputed live from system prompt + full
    /// chat history + any in-flight streaming reply.
    fn context_percent_left(&self) -> i64 {
        let mut total_chars = self.system_prompt.len();
        for msg in &self.messages {
            total_chars += msg.text.len();
        }
        total_chars += self.current_reply.len();

        let est_tokens = (total_chars / 4) as i64;
        let limit = self.context_window_limit();
        let remaining = (limit - est_tokens).max(0);
        ((remaining as f64 / limit as f64) * 100.0)
            .clamp(0.0, 100.0)
            .round() as i64
    }
}

fn debug_log(msg: &str) {
    if std::env::var_os("AMK_DEBUG").is_none() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/amk_tui_stdout.log") {
        let _ = writeln!(file, "{}", msg);
    }
}

fn start_agent(
    agent_type: AgentType,
    provider: &str,
    model: &str,
    caller_cwd: std::path::PathBuf,
    event_tx: mpsc::Sender<TuiEvent>,
) -> Option<(std::process::Child, mpsc::Sender<String>)> {
    let python_path = "/home/masih/Desktop/f/p/Elpis/.venv/bin/python";
    let script_path = "src/agent/main.py";
    let working_dir = "/home/masih/Desktop/f/p/Elpis";

    debug_log(&format!("[START_AGENT] Spawning agent: {:?}, provider={}, model={}, working_dir={}", agent_type, provider, model, working_dir));

    let (cmd_name, args, dir, is_amk) = match agent_type {
        AgentType::Amk => {
            let mut amk_args = vec!["-u".to_string(), script_path.to_string()];
            amk_args.push("--provider".to_string());
            amk_args.push(provider.to_string());
            if !model.is_empty() && model != "Detecting..." {
                amk_args.push("--model".to_string());
                amk_args.push(model.to_string());
            }
            (
                python_path.to_string(),
                amk_args,
                std::path::PathBuf::from(working_dir),
                true,
            )
        }
        AgentType::Codex => (
            "codex".to_string(),
            vec!["exec".to_string()],
            caller_cwd.clone(),
            false,
        ),
        AgentType::Kiro => (
            "kiro-cli".to_string(),
            vec!["chat".to_string()],
            caller_cwd.clone(),
            false,
        ),
    };

    let mut cmd = Command::new(cmd_name);
    cmd.envs(std::env::vars());
    cmd.args(&args);
    cmd.current_dir(dir);
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if is_amk {
        cmd.env("AMK_WORKSPACE_ROOT", caller_cwd);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            debug_log(&format!("[SPAWN_ERROR] Failed to spawn child process: {:?}", e));
            return None;
        }
    };
    let stdin_writer = match child.stdin.take() {
        Some(w) => w,
        None => {
            debug_log("[SPAWN_ERROR] Failed to take stdin");
            return None;
        }
    };
    let stdout_reader = match child.stdout.take() {
        Some(r) => r,
        None => {
            debug_log("[SPAWN_ERROR] Failed to take stdout");
            return None;
        }
    };
    let stderr_reader = match child.stderr.take() {
        Some(r) => r,
        None => {
            debug_log("[SPAWN_ERROR] Failed to take stderr");
            return None;
        }
    };

    let (stdin_tx, stdin_rx) = mpsc::channel::<String>();

    // Spawn stdin writer thread
    thread::spawn(move || {
        let mut writer = stdin_writer;
        while let Ok(msg) = stdin_rx.recv() {
            if writeln!(writer, "{}", msg).is_err() || writer.flush().is_err() {
                break;
            }
        }
    });

    // Spawn stdout reader thread
    let tx_out = event_tx.clone();
    thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(stdout_reader);
        for line in reader.lines() {
            if let Ok(l) = line {
                debug_log(&format!("[STDOUT_RAW] {}", l));
                let _ = tx_out.send(TuiEvent::StdoutLine(l));
            }
        }
    });

    // Spawn stderr reader thread
    let tx_err = event_tx.clone();
    thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(stderr_reader);
        for line in reader.lines() {
            if let Ok(l) = line {
                debug_log(&format!("[STDERR] {}", l));
                let _ = tx_err.send(TuiEvent::StderrLine(l));
            }
        }
    });

    Some((child, stdin_tx))
}

fn main() -> io::Result<()> {
    // Check/create system prompt file in global folder using .md
    let sys_prompt_file = "/home/masih/Desktop/f/p/Elpis/tui/system_prompt.md";
    let system_prompt = if std::path::Path::new(sys_prompt_file).exists() {
        std::fs::read_to_string(sys_prompt_file).unwrap_or_default()
    } else {
        let default_prompt = "You are a helpful AI assistant. Answer concisely.";
        let _ = std::fs::write(sys_prompt_file, default_prompt);
        default_prompt.to_string()
    };

    // Get caller's current working directory to pass as workspace root
    let caller_cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

    // Setup communication channels
    let (event_tx, event_rx) = mpsc::channel::<TuiEvent>();

    // Spawn default AMK agent
    let (mut active_child, stdin_tx) = match start_agent(AgentType::Amk, "ollama", "qwen3:8b", caller_cwd.clone(), event_tx.clone()) {
        Some((child, tx)) => (Some(child), Some(tx)),
        None => {
            println!("Error launching background Python agent process.");
            return Ok(());
        }
    };

    // Setup Terminal Raw Mode
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let _ = io::stdout().execute(crossterm::event::DisableMouseCapture);
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut app = App::new(stdin_tx.expect("Failed to get stdin channel"), system_prompt);
    app.backend_connected = true;

    loop {
        // Draw TUI UI
        terminal.draw(|f| ui(f, &app))?;

        // Poll child process exit
        if app.backend_connected {
            if let Some(ref mut child) = active_child {
                if let Ok(Some(status)) = child.try_wait() {
                    let code = status.code().unwrap_or(0);
                    let _ = event_tx.send(TuiEvent::ProcessExited(code));
                }
            }
        }

        // Handle events from backend channel (drain all pending events)
        while let Ok(event) = event_rx.try_recv() {
            match event {
                TuiEvent::StdoutLine(line) => {
                    app.handle_stdout_line(line);
                }
                TuiEvent::StderrLine(line) => {
                    let err = line.trim().to_string();
                    if !err.is_empty() && !err.contains("Warning") {
                        app.backend_error = Some(err);
                    }
                }
                TuiEvent::ProcessExited(code) => {
                    app.backend_connected = false;
                    let err_msg = if let Some(ref err) = app.backend_error {
                        format!("❌ Agent process exited (code {}): {}", code, err)
                    } else {
                        format!("❌ Agent process exited with code {}", code)
                    };
                    app.add_message(MessageSender::System, err_msg.clone());
                    
                    // For non-AMK runtimes (Codex), exiting concludes the prompt output
                    if app.agent_type != AgentType::Amk {
                        app.in_reply = false;
                        let finished = std::mem::take(&mut app.current_reply);
                        if !finished.trim().is_empty() {
                            app.add_message(MessageSender::Agent, finished);
                        }
                    } else {
                        app.backend_error = Some(err_msg);
                    }
                }
            }
        }

        // Avoid spinning CPU at 100% since try_recv is non-blocking
        thread::sleep(Duration::from_millis(10));

        // Handle Crossterm user keyboard events
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                // Exit on double Ctrl+D
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('d') {
                    app.ctrl_d_count += 1;
                    if app.ctrl_d_count >= 2 {
                        break;
                    }
                    continue;
                } else {
                    app.ctrl_d_count = 0;
                }

                // Copy last agent response to clipboard on Ctrl+C
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    let last_agent_reply = app.messages.iter().rev()
                        .find(|m| m.sender == MessageSender::Agent)
                        .map(|m| m.text.clone());

                    if let Some(text) = last_agent_reply {
                        if let Ok(mut child) = Command::new("xclip")
                            .args(&["-selection", "clipboard"])
                            .stdin(Stdio::piped())
                            .spawn()
                        {
                            if let Some(mut stdin) = child.stdin.take() {
                                let _ = stdin.write_all(text.as_bytes());
                            }
                            thread::spawn(move || {
                                let _ = child.wait();
                            });
                            app.messages.push(ChatMessage {
                                sender: MessageSender::System,
                                text: "📋 Copied last agent response to clipboard.".to_string(),
                            });
                        } else {
                            app.messages.push(ChatMessage {
                                sender: MessageSender::System,
                                text: "❌ Failed to copy to clipboard (xclip error).".to_string(),
                            });
                        }
                    } else {
                        app.messages.push(ChatMessage {
                            sender: MessageSender::System,
                            text: "❌ No agent response found to copy.".to_string(),
                        });
                    }
                    continue;
                }

                match app.focus {
                    Focus::Input => {
                        // Handle Ctrl+U and Ctrl+Z
                        let is_ctrl_u = (key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('u') || key.code == KeyCode::Char('U')))
                            || key.code == KeyCode::Char('\u{15}');
                        let is_ctrl_z = (key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('z') || key.code == KeyCode::Char('Z')))
                            || key.code == KeyCode::Char('\u{1a}');

                        if is_ctrl_u {
                            app.input_backup = Some(app.input.clone());
                            app.input.clear();
                            app.update_completions();
                            continue;
                        }
                        if is_ctrl_z {
                            if let Some(ref backup) = app.input_backup {
                                app.input = backup.clone();
                                app.update_completions();
                            }
                            continue;
                        }

                        // If completions are active, intercept navigation/selection keys first
                        if app.show_completions && !app.completions.is_empty() {
                            match key.code {
                                KeyCode::Up => {
                                    app.completion_selected = (app.completion_selected + app.completions.len() - 1) % app.completions.len();
                                    continue;
                                }
                                KeyCode::Down => {
                                    app.completion_selected = (app.completion_selected + 1) % app.completions.len();
                                    continue;
                                }
                                KeyCode::Tab | KeyCode::Enter => {
                                    app.select_completion();
                                    continue;
                                }
                                KeyCode::Esc => {
                                    app.show_completions = false;
                                    app.completions.clear();
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        // Handle Shift+Enter, Alt+Enter, Ctrl+Enter to insert newlines
                        if key.code == KeyCode::Enter && (key.modifiers.contains(KeyModifiers::SHIFT) || key.modifiers.contains(KeyModifiers::ALT) || key.modifiers.contains(KeyModifiers::CONTROL)) {
                            app.input.push('\n');
                            app.update_completions();
                            continue;
                        }

                        match key.code {
                            KeyCode::Tab => {
                                if app.input.trim_start().starts_with('/') || app.input.contains('@') {
                                    app.update_completions();
                                    if !app.completions.is_empty() {
                                        app.show_completions = true;
                                        app.select_completion();
                                        continue;
                                    }
                                }
                                app.focus = Focus::SysPrompt;
                            }
                            KeyCode::Enter => {
                                let trimmed = app.input.trim();
                                if trimmed == "/auth" {
                                    app.update_completions();
                                    continue;
                                }
                                let msg = std::mem::take(&mut app.input);
                            let trimmed = msg.trim();
                            if trimmed == ":qa" {
                                break;
                            }

                            if !trimmed.is_empty() {
                                // Intercept /agent <amk|codex|kiro>
                                if trimmed.starts_with("/agent ") {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    if parts.len() >= 2 {
                                        let new_agent = match parts[1].to_lowercase().as_str() {
                                            "amk" => Some(AgentType::Amk),
                                            "codex" => Some(AgentType::Codex),
                                            "kiro" => Some(AgentType::Kiro),
                                            _ => None,
                                        };

                                        if let Some(agent) = new_agent {
                                            app.agent_type = agent;
                                            app.model = match agent {
                                                AgentType::Amk => "Detecting...".to_string(),
                                                AgentType::Codex => "gpt-5.5 (OpenAI)".to_string(),
                                                AgentType::Kiro => "Kiro Agent (AWS)".to_string(),
                                            };
                                            app.add_message(MessageSender::User, msg.clone());
                                            app.add_message(MessageSender::System, format!("🔄 Switched active agent to: {:?}", agent));

                                            // Kill current child process
                                            if let Some(mut child) = active_child.take() {
                                                let _ = child.kill();
                                                let _ = child.wait();
                                            }
                                            app.backend_connected = false;
                                            app.is_first_message = true;

                                            // Spawn new agent
                                            if let Some((c, tx)) = start_agent(agent, &app.provider, &app.model, caller_cwd.clone(), event_tx.clone()) {
                                                active_child = Some(c);
                                                app.stdin_tx = Some(tx);
                                                app.backend_connected = true;
                                            } else {
                                                app.add_message(MessageSender::System, format!("❌ Failed to launch {:?} agent. Is it installed and on PATH?", agent));
                                            }
                                        } else {
                                            app.add_message(MessageSender::System, format!("❌ Unknown agent: {}. Choose amk, codex, or kiro.", parts[1]));
                                        }
                                    }
                                    continue;
                                }

                                // Intercept /sys <content>
                                if trimmed.starts_with("/sys ") {
                                    let content = trimmed.replacen("/sys ", "", 1).trim().to_string();
                                    app.system_prompt = content.clone();
                                    let _ = std::fs::write(sys_prompt_file, &content);
                                    app.add_message(MessageSender::User, msg.clone());
                                    app.add_message(MessageSender::System, "⚙️ System prompt updated and saved to system_prompt.md.".to_string());
                                    continue;
                                }

                                // Intercept /setsys
                                if trimmed == "/setsys" {
                                    let last_reply = app.messages.iter().rev()
                                        .find(|m| m.sender == MessageSender::Agent)
                                        .map(|m| m.text.clone());

                                    if let Some(reply) = last_reply {
                                        app.system_prompt = reply.clone();
                                        let _ = std::fs::write(sys_prompt_file, &reply);
                                        app.add_message(MessageSender::User, "/setsys".to_string());
                                        app.add_message(MessageSender::System, "⚙️ Last response set as active system prompt and saved to system_prompt.md.".to_string());
                                    } else {
                                        app.add_message(MessageSender::System, "❌ No response found from agent to use.".to_string());
                                    }
                                    continue;
                                }

                                // Intercept /yolo
                                if trimmed == "/yolo" {
                                    app.yolo_mode = !app.yolo_mode;
                                    app.add_message(MessageSender::User, "/yolo".to_string());
                                    let status = if app.yolo_mode { "ENABLED (No approval prompts)" } else { "DISABLED (Prompts for all changes)" };
                                    app.add_message(MessageSender::System, format!("⚡ YOLO Mode is now {}.", status));
                                    continue;
                                }

                                // Intercept /auth <provider>
                                if trimmed.starts_with("/auth ") || trimmed == "/auth" {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    if parts.len() < 2 {
                                        app.add_message(MessageSender::System, "❌ Usage: /auth <ollama|gemini|openrouter>".to_string());
                                        continue;
                                    }
                                    let new_provider = match parts[1].to_lowercase().as_str() {
                                        "ollama" => Some("ollama"),
                                        "gemini" | "gemini-api" => Some("gemini"),
                                        "openrouter" | "open-router" => Some("openrouter"),
                                        _ => None,
                                    };

                                    if let Some(prov) = new_provider {
                                        app.provider = prov.to_string();
                                        app.model = match prov {
                                            "openrouter" => "moonshotai/kimi-k2.6".to_string(),
                                            "gemini" => "gemini-3.5-flash".to_string(),
                                            _ => "qwen3:8b".to_string(),
                                        };

                                        app.add_message(MessageSender::User, msg.clone());
                                        app.add_message(MessageSender::System, format!("🔄 Switched active provider to: {} (Default Model: {})", prov, app.model));

                                        // Kill current child process
                                        if let Some(mut child) = active_child.take() {
                                            let _ = child.kill();
                                            let _ = child.wait();
                                        }
                                        app.backend_connected = false;
                                        app.is_first_message = true;

                                        // Restart agent
                                        if let Some((c, tx)) = start_agent(app.agent_type, &app.provider, &app.model, caller_cwd.clone(), event_tx.clone()) {
                                            active_child = Some(c);
                                            app.stdin_tx = Some(tx);
                                            app.backend_connected = true;
                                        } else {
                                            app.add_message(MessageSender::System, "❌ Failed to restart agent with the new provider.".to_string());
                                        }
                                    } else {
                                        app.add_message(MessageSender::System, format!("❌ Unknown provider: {}. Choose ollama, gemini, or openrouter.", parts[1]));
                                    }
                                    continue;
                                }

                                // Intercept /model <model_name>
                                if trimmed.starts_with("/model ") || trimmed == "/model" {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    if parts.len() < 2 {
                                        app.add_message(MessageSender::System, "❌ Usage: /model <model_name>".to_string());
                                        continue;
                                    }
                                    let new_model = parts[1..].join(" ");
                                    app.model = new_model.clone();

                                    app.add_message(MessageSender::User, msg.clone());
                                    app.add_message(MessageSender::System, format!("🔄 Switched active model to: {}", app.model));

                                    // Kill current child process
                                    if let Some(mut child) = active_child.take() {
                                        let _ = child.kill();
                                        let _ = child.wait();
                                    }
                                    app.backend_connected = false;
                                    app.is_first_message = true;

                                    // Restart agent
                                    if let Some((c, tx)) = start_agent(app.agent_type, &app.provider, &app.model, caller_cwd.clone(), event_tx.clone()) {
                                        active_child = Some(c);
                                        app.stdin_tx = Some(tx);
                                        app.backend_connected = true;
                                    } else {
                                        app.add_message(MessageSender::System, "❌ Failed to restart agent with the new model.".to_string());
                                    }
                                    continue;
                                }

                                // Intercept /status: forward as-is, capture backend's printed reply as a System message
                                if trimmed == "/status" {
                                    app.add_message(MessageSender::User, msg.clone());
                                    app.in_reply = true;
                                    app.is_command_reply = true;
                                    app.current_reply.clear();
                                    if let Some(ref tx) = app.stdin_tx {
                                        let _ = tx.send(trimmed.to_string());
                                    }
                                    continue;
                                }

                                // If child process exited (e.g. Codex single-run), spawn it again before message
                                if !app.backend_connected {
                                    if let Some((c, tx)) = start_agent(app.agent_type, &app.provider, &app.model, caller_cwd.clone(), event_tx.clone()) {
                                        active_child = Some(c);
                                        app.stdin_tx = Some(tx);
                                        app.backend_connected = true;
                                    } else {
                                        app.add_message(MessageSender::System, "Failed to relaunch agent process.".to_string());
                                    }
                                }

                                // Standard message sending
                                let mut context_str = String::new();
                                let mut active_names = Vec::new();

                                // Scan for @ resource mentions in the message
                                for (name, _) in &app.context_files {
                                    let mention = format!("@{}", name);
                                    if msg.contains(&mention) {
                                        if let Some(content) = app.read_context_file(name) {
                                            context_str.push_str(&format!("[Context File: {}]\n{}\n\n", name, content));
                                            active_names.push(name.clone());
                                        }
                                    }
                                }

                                // Add checklist selected files (if not already added via @ mention)
                                let selected_files: Vec<String> = app.context_files.iter()
                                    .filter(|(_, selected)| *selected)
                                    .map(|(name, _)| name.clone())
                                    .collect();

                                for name in selected_files {
                                    if !active_names.contains(&name) {
                                        if let Some(content) = app.read_context_file(&name) {
                                            context_str.push_str(&format!("[Context File: {}]\n{}\n\n", name, content));
                                            active_names.push(name.clone());
                                        }
                                    }
                                }

                                let sent_msg = if !context_str.is_empty() {
                                    format!("{}{}", context_str, msg)
                                } else {
                                    msg.clone()
                                };

                                if app.is_first_message {
                                    app.is_first_message = false;
                                }

                                app.add_message(MessageSender::User, msg.clone());

                                if !active_names.is_empty() {
                                    app.add_message(MessageSender::System, format!("📎 Injected context from: {}", active_names.join(", ")));
                                }

                                if let Some(ref tx) = app.stdin_tx {
                                    let _ = tx.send(sent_msg);
                                }
                            }
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                            app.update_completions();
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                            app.update_completions();
                        }
                        _ => {}
                    }
                },
                    Focus::Context => match key.code {
                        KeyCode::Tab | KeyCode::Esc => {
                            app.focus = Focus::Input;
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            if app.context_cursor > 0 {
                                app.context_cursor -= 1;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if app.context_cursor + 1 < app.context_files.len() {
                                app.context_cursor += 1;
                            }
                        }
                        KeyCode::Char(' ') | KeyCode::Enter => {
                            let real_idx = app.context_cursor;
                            app.context_files[real_idx].1 = !app.context_files[real_idx].1;

                            let (name, is_selected) = &app.context_files[real_idx];
                            let status_str = if *is_selected { "selected for next query" } else { "deselected" };

                            app.messages.push(ChatMessage {
                                sender: MessageSender::System,
                                text: format!("📎 Context file '{}' is now {}.", name, status_str),
                            });
                        }
                        // Open and View file inside TUI like Yazi
                        KeyCode::Char('o') | KeyCode::Char('v') => {
                            let name = &app.context_files[app.context_cursor].0;
                            let text = match app.read_context_file(name) {
                                Some(content) => format!("📖 View {}:\n{}", name, content),
                                None => format!("❌ Could not read '{}' (file not found).", name),
                            };
                            app.messages.push(ChatMessage {
                                sender: MessageSender::System,
                                text,
                            });
                        }
                        // Any other printable character: jump to the input box and start typing it there
                        // instead of silently discarding the keystroke.
                        KeyCode::Char(c) => {
                            app.focus = Focus::Input;
                            app.input.push(c);
                            app.update_completions();
                        }
                        _ => {}
                    },
                    Focus::SysPrompt => {
                        let is_ctrl_u = (key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('u') || key.code == KeyCode::Char('U')))
                            || key.code == KeyCode::Char('\u{15}');
                        let is_ctrl_z = (key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('z') || key.code == KeyCode::Char('Z')))
                            || key.code == KeyCode::Char('\u{1a}');

                        if is_ctrl_u {
                            app.input_backup = Some(app.system_prompt.clone());
                            app.system_prompt.clear();
                            continue;
                        }
                        if is_ctrl_z {
                            if let Some(ref backup) = app.input_backup {
                                app.system_prompt = backup.clone();
                            }
                            continue;
                        }

                        // Handle Shift+Enter, Alt+Enter, Ctrl+Enter to insert newlines
                        if key.code == KeyCode::Enter && (key.modifiers.contains(KeyModifiers::SHIFT) || key.modifiers.contains(KeyModifiers::ALT) || key.modifiers.contains(KeyModifiers::CONTROL)) {
                            app.system_prompt.push('\n');
                            continue;
                        }

                        match key.code {
                            KeyCode::Tab => {
                                app.focus = Focus::Context;
                            }
                            KeyCode::Esc => {
                                app.focus = Focus::Input;
                            }
                            KeyCode::Char(c) => {
                                app.system_prompt.push(c);
                            }
                            KeyCode::Backspace => {
                                app.system_prompt.pop();
                            }
                            KeyCode::Enter => {
                                let sys_prompt_file = "/home/masih/Desktop/f/p/Elpis/tui/system_prompt.md";
                                let _ = std::fs::write(sys_prompt_file, &app.system_prompt);
                                app.messages.push(ChatMessage {
                                    sender: MessageSender::System,
                                    text: "⚙️ System prompt updated and saved to system_prompt.md.".to_string(),
                                });
                                app.focus = Focus::Input;
                            }
                            _ => {}
                        }
                    }
                    Focus::Approval => match key.code {
                        KeyCode::Char('y') | KeyCode::Char('a') | KeyCode::Enter => {
                            if let Some(ref tx) = app.stdin_tx {
                                let _ = tx.send("{\"status\": \"accept\"}\n".to_string());
                            }
                            app.pending_approval = None;
                            app.focus = Focus::Input;
                        }
                        KeyCode::Char('n') | KeyCode::Char('r') | KeyCode::Esc => {
                            if let Some(ref tx) = app.stdin_tx {
                                let _ = tx.send("{\"status\": \"reject\"}\n".to_string());
                            }
                            app.pending_approval = None;
                            app.focus = Focus::Input;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Kill active background processes on quit
    if let Some(mut child) = active_child {
        let _ = child.kill();
    }

    // Restore terminal configuration on exit
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn count_wrapped_lines(text: &str, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let mut total_lines = 0;
    let lines: Vec<&str> = text.split('\n').collect();
    for line in lines {
        let line_width = UnicodeWidthStr::width(line);
        if line_width == 0 {
            total_lines += 1;
        } else {
            total_lines += (line_width + width - 1) / width;
        }
    }
    total_lines
}

fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Check if user is typing a command to dynamically size the input box height
    let is_typing_cmd = app.input.starts_with('/');
    let text_width = size.width.saturating_sub(2) as usize;
    let input_lines = if text_width > 0 {
        count_wrapped_lines(&app.input, text_width)
    } else {
        1
    };

    let has_completions = app.show_completions && !app.completions.is_empty();
    let comp_height = if has_completions {
        (app.completions.len() + 2).min(7) as u16
    } else {
        0
    };

    // Dynamically calculate input box height: input_lines + helper/borders, capped by available terminal height to preserve at least 10 lines of chat history
    let input_content_lines = if is_typing_cmd {
        input_lines + 2
    } else {
        input_lines
    };
    let max_allowed_input = size.height.saturating_sub(3 + 10 + comp_height);
    let max_allowed_input = max_allowed_input.max(3); // Ensure at least 3 lines
    let input_height = (input_content_lines as u16 + 2).min(max_allowed_input);

    let constraints = if has_completions {
        vec![
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(comp_height),
            Constraint::Length(input_height),
        ]
    } else {
        vec![
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(input_height),
        ]
    };

    // Divide layout into Header (3 lines), Main Area (Min 10), Completions (if any), and Input Block (variable height)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(size);

    // 1. Render Header
    let header_text = format!(
        " 🏰 A-MODULAR-KINGDOM AGENT CLIENT | Context: {}% left | Provider: {} | Model: {} | Workspace: {} ",
        app.context_percent_left(), app.provider, app.model, app.workspace
    );
    let header = Paragraph::new(header_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD));
    f.render_widget(header, chunks[0]);

    // 2. Main Area: Horizontal split into Left (70% chat log) and Right (30% checklist & system info)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(chunks[1]);

    // Left Pane: Chat history log
    let mut chat_lines = Vec::new();
    for msg in &app.messages {
        match msg.sender {
            MessageSender::User => {
                chat_lines.push(Line::from(vec![
                    Span::styled("👤 You: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::styled(&msg.text, Style::default().fg(Color::Cyan)),
                ]));
            }
            MessageSender::Agent => {
                let label = match app.agent_type {
                    AgentType::Amk => "🤖 Juliette: ",
                    AgentType::Codex => "⚙️ Codex: ",
                    AgentType::Kiro => "🧠 Kiro: ",
                };
                chat_lines.push(Line::from(vec![
                    Span::styled(label, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::styled(&msg.text, Style::default().fg(Color::White)),
                ]));
            }
            MessageSender::System => {
                chat_lines.push(Line::from(vec![
                    Span::styled("⚙️ System: ", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                    Span::styled(&msg.text, Style::default().fg(Color::DarkGray)),
                ]));
            }
        }
        chat_lines.push(Line::from("")); // Add spacer
    }

    // If streaming in real-time, show current reply chunk
    if app.in_reply && !app.current_reply.is_empty() {
        let label = match app.agent_type {
            AgentType::Amk => "🤖 Juliette (streaming): ",
            AgentType::Codex => "⚙️ Codex (streaming): ",
            AgentType::Kiro => "🧠 Kiro (streaming): ",
        };
        chat_lines.push(Line::from(vec![
            Span::styled(label, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(&app.current_reply, Style::default().fg(Color::White)),
        ]));
    }

    let chat_width = (main_chunks[0].width.saturating_sub(2) as usize).max(1);
    let mut total_chat_rendered_lines = 0;
    for msg in &app.messages {
        let prefix = match msg.sender {
            MessageSender::User => "👤 You: ",
            MessageSender::Agent => match app.agent_type {
                AgentType::Amk => "🤖 Juliette: ",
                AgentType::Codex => "⚙️ Codex: ",
                AgentType::Kiro => "🧠 Kiro: ",
            },
            MessageSender::System => "⚙️ System: ",
        };

        let msg_lines: Vec<&str> = msg.text.split('\n').collect();
        for (i, line) in msg_lines.iter().enumerate() {
            let line_len = if i == 0 {
                UnicodeWidthStr::width(prefix) + UnicodeWidthStr::width(*line)
            } else {
                UnicodeWidthStr::width(*line)
            };
            if line_len == 0 {
                total_chat_rendered_lines += 1;
            } else {
                total_chat_rendered_lines += (line_len + chat_width - 1) / chat_width;
            }
        }
        total_chat_rendered_lines += 1; // spacer line
    }

    if app.in_reply && !app.current_reply.is_empty() {
        let prefix = match app.agent_type {
            AgentType::Amk => "🤖 Juliette (streaming): ",
            AgentType::Codex => "⚙️ Codex (streaming): ",
            AgentType::Kiro => "🧠 Kiro (streaming): ",
        };
        let msg_lines: Vec<&str> = app.current_reply.split('\n').collect();
        for (i, line) in msg_lines.iter().enumerate() {
            let line_len = if i == 0 {
                UnicodeWidthStr::width(prefix) + UnicodeWidthStr::width(*line)
            } else {
                UnicodeWidthStr::width(*line)
            };
            if line_len == 0 {
                total_chat_rendered_lines += 1;
            } else {
                total_chat_rendered_lines += (line_len + chat_width - 1) / chat_width;
            }
        }
    }

    let visible_height = main_chunks[0].height.saturating_sub(2);
    let max_scroll = (total_chat_rendered_lines as u16).saturating_sub(visible_height);

    let chat_block = Paragraph::new(chat_lines)
        .block(
            Block::default()
                .title(" Chat History ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .wrap(Wrap { trim: false })
        .scroll((max_scroll, 0));
    f.render_widget(chat_block, main_chunks[0]);

    // Right Pane: Yazi-style context checklist with active system prompt preview
    let agent_label = format!(" [Agent: {:?}] ", app.agent_type);
    let mut info_text = Vec::new();

    info_text.push(Line::from(vec![
        Span::styled(" ⚙️ SYSTEM PROMPT: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
    ]));

    let cleaned_sys_prompt = app.system_prompt.replace('\n', " ");
    let display_text = if app.focus == Focus::SysPrompt {
        // Show actual prompt. If it is long, show the last 20 characters to fit single-line editor
        let count = cleaned_sys_prompt.chars().count();
        if count > 20 {
            let suffix: String = cleaned_sys_prompt.chars().skip(count - 20).collect();
            format!("...{}", suffix)
        } else {
            cleaned_sys_prompt.clone()
        }
    } else {
        // Truncated preview
        let count = cleaned_sys_prompt.chars().count();
        if count > 20 {
            let prefix: String = cleaned_sys_prompt.chars().take(17).collect();
            format!("\"{}...\"", prefix)
        } else {
            format!("\"{}\"", cleaned_sys_prompt)
        }
    };

    info_text.push(Line::from(Span::styled(display_text.clone(), Style::default().fg(Color::Gray))));
    info_text.push(Line::from(""));
    info_text.push(Line::from(vec![
        Span::styled(" 📎 CONTEXT FILES: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
    ]));

    for (i, (name, selected)) in app.context_files.iter().enumerate() {
        let is_highlighted = app.focus == Focus::Context && app.context_cursor == i;
        let checkbox = if *selected { "[x] " } else { "[ ] " };
        let color = if *selected { Color::Green } else { Color::DarkGray };

        let mut line_spans = Vec::new();
        if is_highlighted {
            line_spans.push(Span::styled(" ❯ ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
        } else {
            line_spans.push(Span::from("   "));
        }

        line_spans.push(Span::styled(checkbox, Style::default().fg(color).add_modifier(Modifier::BOLD)));

        if is_highlighted {
            line_spans.push(Span::styled(name, Style::default().fg(Color::Black).bg(Color::Yellow)));
        } else {
            line_spans.push(Span::styled(name, Style::default().fg(Color::White)));
        }

        info_text.push(Line::from(line_spans));
    }

    info_text.push(Line::from(""));
    info_text.push(Line::from(vec![
        Span::styled("  [Tab] Switch Focus", Style::default().fg(Color::DarkGray)),
    ]));
    info_text.push(Line::from(vec![
        Span::styled("  [Space] Toggle | [o/v] View", Style::default().fg(Color::DarkGray)),
    ]));
    info_text.push(Line::from(vec![
        Span::styled("  :qa / Ctrl+D x2 Quit", Style::default().fg(Color::DarkGray)),
    ]));

    let context_border_color = match app.focus {
        Focus::Context | Focus::SysPrompt => Color::Yellow,
        Focus::Input | Focus::Approval => Color::Blue,
    };

    let info_block = Paragraph::new(info_text).block(
        Block::default()
            .title(agent_label)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(context_border_color)),
    );
    f.render_widget(info_block, main_chunks[1]);

    // 3. Render Completions Dropdown if active
    let has_completions = app.show_completions && !app.completions.is_empty();
    if has_completions {
        let comp_lines: Vec<Line> = app.completions.iter().enumerate().map(|(i, item)| {
            if app.completion_selected == i {
                Line::from(vec![
                    Span::styled(" ❯ ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::styled(item, Style::default().fg(Color::Black).bg(Color::Yellow)),
                ])
            } else {
                Line::from(vec![
                    Span::from("   "),
                    Span::styled(item, Style::default().fg(Color::White)),
                ])
            }
        }).collect();

        let comp_block = Paragraph::new(comp_lines)
            .block(
                Block::default()
                    .title(" Completions (Tab/Arrows to navigate, Enter to select) ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Yellow)),
            );
        f.render_widget(comp_block, chunks[2]);
    }

    let input_chunk_idx = if has_completions { 3 } else { 2 };
    let visible_input_height = chunks[input_chunk_idx].height.saturating_sub(2);
    let input_scroll = if is_typing_cmd {
        (input_lines as u16 + 2).saturating_sub(visible_input_height)
    } else {
        (input_lines as u16).saturating_sub(visible_input_height)
    };

    // 4. Render Input Block (with dynamic command helper list if user typed '/')
    let input_block = if is_typing_cmd {
        let text_lines = vec![
            Line::from(vec![
                Span::styled(" ❯ Commands: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled("/auth <provider> [model]", Style::default().fg(Color::Cyan)),
                Span::styled(" Switch LLM", Style::default().fg(Color::Gray)),
            ]),
            Line::from("─".repeat(size.width.saturating_sub(4) as usize).fg(Color::DarkGray)),
            Line::from(app.input.as_str()),
        ];
        Paragraph::new(text_lines).wrap(Wrap { trim: false }).scroll((input_scroll, 0))
    } else {
        Paragraph::new(app.input.as_str()).wrap(Wrap { trim: false }).scroll((input_scroll, 0))
    };

    let input_border_color = match app.focus {
        Focus::Input => Color::Yellow,
        Focus::Context | Focus::SysPrompt | Focus::Approval => Color::DarkGray,
    };

    let decorated_block = input_block.block(
        Block::default()
            .title(" Message ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(input_border_color)),
    )
    .style(Style::default().fg(Color::White));

    f.render_widget(decorated_block, chunks[input_chunk_idx]);

    // 5. Render blinking cursor position based on active focus
    if app.focus == Focus::SysPrompt {
        // Calculate coordinates for the system prompt editor line in the right sidebar
        let cleaned_sys_prompt = app.system_prompt.replace('\n', " ");
        let text_len = if cleaned_sys_prompt.chars().count() > 20 { 23 } else { cleaned_sys_prompt.chars().count() };
        let cursor_x = main_chunks[1].x + 1 + text_len as u16;
        let cursor_y = main_chunks[1].y + 2;
        f.set_cursor_position((cursor_x, cursor_y));
    } else if app.focus == Focus::Approval {
        // Hide cursor during approval modal by placing it out of view
    } else {
        let text_width = chunks[input_chunk_idx].width.saturating_sub(2) as usize;
        let mut cursor_row = 0;
        let mut cursor_col = 0;
        if text_width > 0 {
            let lines: Vec<&str> = app.input.split('\n').collect();
            for (i, line) in lines.iter().enumerate() {
                let char_count = UnicodeWidthStr::width(*line);
                if i < lines.len() - 1 {
                    if char_count == 0 {
                        cursor_row += 1;
                    } else {
                        cursor_row += (char_count + text_width - 1) / text_width;
                    }
                } else {
                    if char_count == 0 {
                        cursor_col = 0;
                    } else {
                        cursor_row += char_count / text_width;
                        cursor_col = char_count % text_width;
                    }
                }
            }
        }
        let paragraph_cursor_row = if is_typing_cmd {
            cursor_row + 2
        } else {
            cursor_row
        };
        let scrolled_cursor_row = paragraph_cursor_row.saturating_sub(input_scroll as usize);
        let cursor_x = chunks[input_chunk_idx].x + 1 + cursor_col as u16;
        let cursor_y = chunks[input_chunk_idx].y + 1 + scrolled_cursor_row as u16;
        // Clamp cursor_y to the boundaries of the input block to avoid drawing it outside
        let max_y = chunks[input_chunk_idx].y + chunks[input_chunk_idx].height.saturating_sub(2);
        let cursor_y = cursor_y.min(max_y);
        f.set_cursor_position((cursor_x, cursor_y));
    }

    // 6. Render Approval Modal popup over the layout
    if let Some(ref req) = app.pending_approval {
        let popup_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(size);

        let area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(15),
                Constraint::Percentage(70),
                Constraint::Percentage(15),
            ])
            .split(popup_layout[1])[1];

        // Clear the popup area
        f.render_widget(Clear, area);

        let block = Block::default()
            .title(format!(" ⚠️ APPROVAL REQUIRED: {} ", req.req_type.to_uppercase()))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

        let mut text = Vec::new();
        text.push(Line::from(""));

        match req.req_type.as_str() {
            "write_file" => {
                let path_str = req.path.as_deref().unwrap_or("unknown");
                text.push(Line::from(vec![
                    Span::styled("📝 Action: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::from("Write file to "),
                    Span::styled(path_str, Style::default().fg(Color::Cyan)),
                ]));
                text.push(Line::from(""));

                // Read old file content to show diff
                let old_content = std::fs::read_to_string(path_str).unwrap_or_default();
                let new_content = req.content.as_deref().unwrap_or("");
                let diff = compute_line_diff(&old_content, new_content);

                text.push(Line::from(Span::styled("--- PROPOSED EDITS (DIFF) ---", Style::default().fg(Color::DarkGray))));
                for d_line in diff {
                    match d_line {
                        DiffLine::Unchanged(l) => {
                            text.push(Line::from(Span::styled(format!("  {}", l), Style::default().fg(Color::Gray))));
                        }
                        DiffLine::Added(l) => {
                            text.push(Line::from(Span::styled(format!("+ {}", l), Style::default().fg(Color::Green))));
                        }
                        DiffLine::Deleted(l) => {
                            text.push(Line::from(Span::styled(format!("- {}", l), Style::default().fg(Color::Red))));
                        }
                    }
                }
            }
            "execute_command" => {
                let cmd_str = req.command.as_deref().unwrap_or("unknown");
                text.push(Line::from(vec![
                    Span::styled("🚀 Action: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::from("Execute local command"),
                ]));
                text.push(Line::from(""));
                text.push(Line::from(Span::styled("--- COMMAND ---", Style::default().fg(Color::DarkGray))));
                text.push(Line::from(Span::styled(format!("  {}", cmd_str), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))));
            }
            _ => {
                text.push(Line::from("Unknown action type."));
            }
        }

        text.push(Line::from(""));
        text.push(Line::from(vec![
            Span::styled("👉 Choices: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled("[a] Accept / [Enter] Accept  |  [r] Reject / [Esc] Reject", Style::default().fg(Color::White)),
        ]));

        let paragraph = Paragraph::new(text)
            .block(block)
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_message_coalescing() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx, "System Prompt".to_string());
        
        app.add_message(MessageSender::System, "Database initialized".to_string());
        app.add_message(MessageSender::System, "Database initialized".to_string());
        app.add_message(MessageSender::System, "Database initialized".to_string());
        
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].text, "Database initialized (x3)");
        
        app.add_message(MessageSender::System, "Connection lost".to_string());
        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[1].text, "Connection lost");
    }

    #[test]
    fn test_large_message_truncation() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx, "System Prompt".to_string());
        
        let huge_text = "A".repeat(10000);
        app.add_message(MessageSender::User, huge_text);
        
        assert_eq!(app.messages.len(), 1);
        assert!(app.messages[0].text.contains("<truncated to save context>"));
        assert_eq!(app.messages[0].text.len(), 8000 + "...\n<truncated to save context>".len());
    }

    #[test]
    fn test_sliding_window_pruning() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut app = App::new(tx, "System Prompt".to_string());
        app.model = "qwen3:8b".to_string(); // Limit is 32768
        
        // Populate messages to exceed 32768 tokens (approx 131,000 characters)
        // Each message will be 7000 characters (approx 1750 tokens)
        for i in 0..25 {
            app.add_message(MessageSender::User, format!("User message {} with repeating text: {}", i, "X".repeat(7000)));
            app.add_message(MessageSender::Agent, format!("Agent reply {} with repeating text: {}", i, "Y".repeat(7000)));
        }

        let total_chars: usize = app.system_prompt.len() + app.messages.iter().map(|m| m.text.len()).sum::<usize>();
        let est_tokens = (total_chars / 4) as i64;
        let limit = app.context_window_limit();
        let target_tokens = (limit as f64 * 0.7) as i64;
        
        assert!(est_tokens <= target_tokens, "est_tokens: {}, target_tokens: {}", est_tokens, target_tokens);
    }

    #[test]
    fn test_compute_line_diff() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nline2_modified\nline3\nline4\n";
        let diff = compute_line_diff(old, new);
        
        let mut added = Vec::new();
        let mut deleted = Vec::new();
        let mut unchanged = Vec::new();
        
        for d in diff {
            match d {
                DiffLine::Unchanged(l) => unchanged.push(l),
                DiffLine::Added(l) => added.push(l),
                DiffLine::Deleted(l) => deleted.push(l),
            }
        }
        
        assert_eq!(unchanged, vec!["line1", "line3"]);
        assert_eq!(deleted, vec!["line2"]);
        assert_eq!(added, vec!["line2_modified", "line4"]);
    }
}
