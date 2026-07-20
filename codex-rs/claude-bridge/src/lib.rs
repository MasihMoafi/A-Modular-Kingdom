//! Minimal subprocess bridge to the Claude Code CLI's non-interactive (`--print`) mode.
//!
//! Spawns `claude -p <prompt> --output-format stream-json` and parses its stdout into
//! typed events, or (via [`run_print_turn`]) aggregates a whole turn into one
//! [`TurnOutcome`]. Wired into the TUI's `ActiveRuntime::ClaudeCode` path
//! (`codex-rs/tui/src/chatwidget/claude_code_turn.rs`) as whole-message text-in/text-out
//! only — no incremental streaming render, and `tool_use`/`tool_result` content blocks
//! are not inspected or bridged to Elpis's approval/diff surfaces yet.
//!
//! The event schema below was captured empirically from a real `claude -p ... --verbose
//! --output-format stream-json` run (Claude Code 2.1.214), not assumed from docs. Field
//! casing is genuinely mixed in the real output (mostly snake_case, with `permissionMode`
//! and `apiKeySource` in camelCase) — modeled as observed, not normalized. Only `thinking`
//! and `text` assistant content-block types were empirically observed in this probe;
//! `tool_use`/`tool_result` block shapes are not yet verified, so message content is kept
//! as raw [`serde_json::Value`] rather than guessed at.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::Command;
use tokio::sync::mpsc;

#[derive(Debug, thiserror::Error)]
pub enum ClaudeBridgeError {
    #[error("failed to spawn claude process: {0}")]
    Spawn(#[source] std::io::Error),
    #[error("claude stdout was not piped")]
    NoStdout,
}

#[derive(Debug, thiserror::Error)]
#[error("failed to parse claude stream-json line: {source}\nline: {line}")]
pub struct ParseError {
    #[source]
    source: serde_json::Error,
    line: String,
}

/// Options for one non-interactive `claude -p` invocation.
#[derive(Debug, Clone, Default)]
pub struct ClaudePrintOptions {
    pub prompt: String,
    /// Passed as `--resume <id>` to continue a prior Claude Code session.
    pub resume_session_id: Option<String>,
    pub cwd: Option<PathBuf>,
    /// Passed as `--model <name>` (e.g. `haiku` for cheap distillation calls).
    pub model: Option<String>,
    /// Passed as `--append-system-prompt <text>` — how Elpis injects its composed
    /// working set (outcome records) into an otherwise fresh session.
    pub append_system_prompt: Option<String>,
}

/// One parsed line of `claude ... --output-format stream-json` output.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeStreamEvent {
    #[serde(rename = "system")]
    System(SystemInit),
    #[serde(rename = "assistant")]
    Assistant(MessageEvent),
    #[serde(rename = "user")]
    User(MessageEvent),
    #[serde(rename = "rate_limit_event")]
    RateLimitEvent(RateLimitEvent),
    #[serde(rename = "result")]
    Result(ResultEvent),
    /// Any event type not modeled above. Forward-compatible by design: an unrecognized
    /// `type` never fails parsing, it just carries the raw JSON through unexamined.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemInit {
    pub session_id: String,
    pub cwd: String,
    pub model: String,
    #[serde(rename = "permissionMode")]
    pub permission_mode: String,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageEvent {
    /// Kept as raw JSON: content blocks include at least `thinking` and `text`
    /// (both empirically observed); `tool_use`/`tool_result` shapes are unverified.
    pub message: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitEvent {
    pub session_id: String,
    pub rate_limit_info: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResultEvent {
    pub session_id: String,
    pub is_error: bool,
    pub result: Option<String>,
    pub total_cost_usd: Option<f64>,
    pub num_turns: Option<u32>,
}

/// Parse one line of stdout. Unknown `type` values succeed with
/// [`ClaudeStreamEvent::Unknown`] rather than erroring, so a future CLI version adding
/// event types doesn't break this parser.
pub fn parse_event_line(line: &str) -> Result<ClaudeStreamEvent, ParseError> {
    serde_json::from_str(line).map_err(|source| ParseError {
        source,
        line: line.to_string(),
    })
}

/// Overrides the `claude` binary invoked, for tests only (points at a fake script instead
/// of the real CLI, so routing logic can be exercised without a live authenticated session).
pub const BINARY_OVERRIDE_ENV: &str = "CLAUDE_BRIDGE_BINARY";

/// The `claude` binary to invoke, honoring [`BINARY_OVERRIDE_ENV`]. Shared with the
/// TUI's takeover mode so tests can substitute a fake CLI there too.
pub fn resolve_binary() -> String {
    std::env::var(BINARY_OVERRIDE_ENV).unwrap_or_else(|_| "claude".to_string())
}

fn build_command(options: &ClaudePrintOptions) -> Command {
    let mut cmd = Command::new(resolve_binary());
    cmd.arg("-p")
        .arg(&options.prompt)
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose");
    if let Some(id) = &options.resume_session_id {
        cmd.arg("--resume").arg(id);
    }
    if let Some(model) = &options.model {
        cmd.arg("--model").arg(model);
    }
    if let Some(text) = &options.append_system_prompt {
        cmd.arg("--append-system-prompt").arg(text);
    }
    if let Some(cwd) = &options.cwd {
        cmd.current_dir(cwd);
    }
    // stderr is piped (not inherited): the caller runs inside a raw-mode/alt-screen TUI,
    // and a subprocess writing straight to the terminal would corrupt the frame instead
    // of surfacing as a readable error.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Spawn `claude -p` and stream parsed events back over an unbounded channel. The
/// channel closes when the process's stdout closes (i.e. the turn ends). The caller
/// owns the returned [`Child`] and is responsible for eventually awaiting/killing it.
///
/// Not exercised against a real `claude` process in CI: GitHub Actions runners have no
/// authenticated Claude Code session, so this path is unit-testable only via
/// `parse_event_line` against captured fixtures, not end-to-end here.
pub async fn spawn_and_stream_events(
    options: ClaudePrintOptions,
) -> Result<
    (
        Child,
        mpsc::UnboundedReceiver<Result<ClaudeStreamEvent, ParseError>>,
    ),
    ClaudeBridgeError,
> {
    let mut child = build_command(&options)
        .spawn()
        .map_err(ClaudeBridgeError::Spawn)?;
    let stdout = child.stdout.take().ok_or(ClaudeBridgeError::NoStdout)?;
    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if tx.send(parse_event_line(&line)).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    });

    Ok((child, rx))
}

/// Outcome of one whole `claude -p` turn, aggregated from the event stream. Whole-message,
/// not incremental — callers wanting live token-by-token updates need `spawn_and_stream_events`
/// directly. `tool_use`/`tool_result` content blocks are not inspected here; only `text`
/// blocks are aggregated, since those block shapes are the only ones this crate has verified
/// (see module doc comment).
#[derive(Debug, Clone, Default)]
pub struct TurnOutcome {
    pub text: Option<String>,
    pub session_id: Option<String>,
    pub is_error: bool,
    pub error_message: Option<String>,
}

/// Ceiling on one `claude -p` turn (including the cheap haiku distillation call). Plain
/// text-in/text-out turns are normally seconds; this is a backstop against a hung
/// subprocess (e.g. an interactive auth prompt with no TTY to answer it) leaving the
/// turn silently unresolved forever.
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(180);

/// Runs one `claude -p` turn to completion and aggregates it into a [`TurnOutcome`].
/// Convenience wrapper over [`spawn_and_stream_events`] for callers that don't need
/// incremental events. Waits for the child process to exit. Not cancellable — used
/// for the internal haiku distillation call, which must not be interruptible by the
/// user. See [`run_cancellable_print_turn`] for the user-facing, cancellable turn.
pub async fn run_print_turn(options: ClaudePrintOptions) -> Result<TurnOutcome, ClaudeBridgeError> {
    run_print_turn_with_timeout(options, DEFAULT_TURN_TIMEOUT, None).await
}

/// Like [`run_print_turn`], but cancellable: if `cancel` fires before the turn
/// completes or times out, the child process is killed and the returned
/// [`TurnOutcome`] has `is_error: true` with `error_message: Some("cancelled")`
/// (that exact sentinel string — callers match on it to show a clearer message than
/// a generic failure).
pub async fn run_cancellable_print_turn(
    options: ClaudePrintOptions,
    cancel: tokio::sync::oneshot::Receiver<()>,
) -> Result<TurnOutcome, ClaudeBridgeError> {
    run_print_turn_with_timeout(options, DEFAULT_TURN_TIMEOUT, Some(cancel)).await
}

/// Why the aggregation loop below stopped.
enum StopReason {
    Completed,
    TimedOut,
    Cancelled,
}

async fn run_print_turn_with_timeout(
    options: ClaudePrintOptions,
    timeout: Duration,
    cancel: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<TurnOutcome, ClaudeBridgeError> {
    let (mut child, mut rx) = spawn_and_stream_events(options).await?;
    // Drained concurrently, not after the fact: stderr is piped now (see `build_command`),
    // and letting it sit unread risks the child blocking on a full pipe if it writes a lot.
    let stderr_task = child.stderr.take().map(|stderr| {
        tokio::spawn(async move {
            let mut buf = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut buf).await;
            buf
        })
    });

    let mut text = String::new();
    let mut outcome = TurnOutcome::default();

    let aggregate = async {
        while let Some(parsed) = rx.recv().await {
            match parsed {
                Ok(ClaudeStreamEvent::Assistant(message)) => {
                    if let Some(blocks) = message.message.get("content").and_then(Value::as_array) {
                        for block in blocks {
                            if block.get("type").and_then(Value::as_str) == Some("text")
                                && let Some(chunk) = block.get("text").and_then(Value::as_str)
                            {
                                text.push_str(chunk);
                            }
                        }
                    }
                }
                Ok(ClaudeStreamEvent::Result(result)) => {
                    outcome.session_id = Some(result.session_id);
                    outcome.is_error = result.is_error;
                    if result.is_error {
                        outcome.error_message = result.result.clone();
                    }
                }
                Ok(_) => {}
                Err(_) => {
                    // Malformed line from the CLI itself (not our concern to fail the whole
                    // turn over) — skip it and keep aggregating.
                }
            }
        }
    };

    let stop_reason = match cancel {
        Some(cancel_rx) => {
            tokio::select! {
                _ = aggregate => StopReason::Completed,
                _ = tokio::time::sleep(timeout) => StopReason::TimedOut,
                _ = cancel_rx => StopReason::Cancelled,
            }
        }
        None => {
            if tokio::time::timeout(timeout, aggregate).await.is_err() {
                StopReason::TimedOut
            } else {
                StopReason::Completed
            }
        }
    };

    if !matches!(stop_reason, StopReason::Completed) {
        let _ = child.start_kill();
    }
    let exit_status = child.wait().await.ok();
    let stderr_text = match stderr_task {
        Some(task) => task.await.unwrap_or_default(),
        None => String::new(),
    };

    outcome.text = if text.is_empty() { None } else { Some(text) };

    match stop_reason {
        StopReason::Cancelled => {
            outcome.is_error = true;
            outcome.error_message = Some("cancelled".to_string());
        }
        StopReason::TimedOut => {
            outcome.is_error = true;
            outcome.error_message = Some(format!(
                "claude did not respond within {}s{}",
                timeout.as_secs(),
                stderr_excerpt(&stderr_text)
            ));
        }
        StopReason::Completed => {
            if outcome.error_message.is_none()
                && (outcome.is_error
                    || (outcome.text.is_none()
                        && !exit_status.is_some_and(|status| status.success())))
            {
                outcome.is_error = true;
                outcome.error_message = Some(format!(
                    "claude exited with {}{}",
                    exit_status
                        .map_or_else(|| "unknown status".to_string(), |status| status.to_string()),
                    stderr_excerpt(&stderr_text)
                ));
            }
        }
    }

    Ok(outcome)
}

/// Formats captured stderr as an error-message suffix, capped so one runaway CLI dump
/// can't blow up the transcript.
fn stderr_excerpt(stderr: &str) -> String {
    const MAX_CHARS: usize = 500;
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let excerpt: String = trimmed.chars().take(MAX_CHARS).collect();
    format!(": {excerpt}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured verbatim (trimmed) from a real
    // `claude -p "reply with exactly one word: pong" --output-format stream-json --verbose`
    // run on Claude Code 2.1.214, 2026-07-18.
    const REAL_SYSTEM_INIT: &str = r#"{"type":"system","subtype":"init","cwd":"/tmp/claude-bridge-probe","session_id":"1cbc778a-c5ba-4451-9809-9cdc22d2abbf","tools":["Task","Bash","Read","Write"],"mcp_servers":[],"model":"claude-fable-5","permissionMode":"default","slash_commands":["clear","compact"],"apiKeySource":"none","claude_code_version":"2.1.214","output_style":"default","agents":["claude"],"skills":[],"plugins":[],"capabilities":["interrupt_receipt_v1"],"analytics_disabled":false,"product_feedback_disabled":false,"uuid":"aa3377f1-5148-4808-97e9-8133f198daaf","memory_paths":{"auto":"/x/memory/"},"fast_mode_state":"off"}"#;

    const REAL_RATE_LIMIT: &str = r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","resetsAt":1784389800,"rateLimitType":"five_hour"},"uuid":"5aa2025d-36a8-4b16-b323-4f0274277ca2","session_id":"1cbc778a-c5ba-4451-9809-9cdc22d2abbf"}"#;

    const REAL_ASSISTANT_TEXT: &str = r#"{"type":"assistant","message":{"model":"claude-fable-5","id":"msg_011Cd9eqn2HBuTiHsNk5Jey6","type":"message","role":"assistant","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn"}}"#;

    const REAL_RESULT: &str = r#"{"type":"result","subtype":"success","is_error":false,"duration_ms":2100,"num_turns":1,"result":"pong","session_id":"1cbc778a-c5ba-4451-9809-9cdc22d2abbf","total_cost_usd":0.001,"uuid":"c0ffee"}"#;

    #[test]
    fn parses_real_system_init() {
        let event = parse_event_line(REAL_SYSTEM_INIT).expect("should parse");
        match event {
            ClaudeStreamEvent::System(s) => {
                assert_eq!(s.session_id, "1cbc778a-c5ba-4451-9809-9cdc22d2abbf");
                assert_eq!(s.model, "claude-fable-5");
                assert_eq!(s.permission_mode, "default");
            }
            other => panic!("expected System, got {other:?}"),
        }
    }

    #[test]
    fn parses_real_rate_limit_event() {
        let event = parse_event_line(REAL_RATE_LIMIT).expect("should parse");
        assert!(matches!(event, ClaudeStreamEvent::RateLimitEvent(_)));
    }

    #[test]
    fn parses_real_assistant_text_message() {
        let event = parse_event_line(REAL_ASSISTANT_TEXT).expect("should parse");
        match event {
            ClaudeStreamEvent::Assistant(m) => {
                let content = m.message["content"][0]["text"].as_str().unwrap();
                assert_eq!(content, "pong");
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn parses_real_result_event() {
        let event = parse_event_line(REAL_RESULT).expect("should parse");
        match event {
            ClaudeStreamEvent::Result(r) => {
                assert!(!r.is_error);
                assert_eq!(r.result.as_deref(), Some("pong"));
                assert_eq!(r.num_turns, Some(1));
            }
            other => panic!("expected Result, got {other:?}"),
        }
    }

    #[test]
    fn unknown_event_type_does_not_fail_parsing() {
        let event = parse_event_line(r#"{"type":"some_future_event","data":123}"#)
            .expect("unknown types must not error");
        assert!(matches!(event, ClaudeStreamEvent::Unknown));
    }

    #[test]
    fn build_command_includes_resume_flag_when_set() {
        let options = ClaudePrintOptions {
            prompt: "hi".to_string(),
            resume_session_id: Some("abc-123".to_string()),
            ..Default::default()
        };
        let cmd = build_command(&options);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"--resume".to_string()));
        assert!(args.contains(&"abc-123".to_string()));
    }

    #[test]
    fn build_command_includes_model_and_append_system_prompt_when_set() {
        let options = ClaudePrintOptions {
            prompt: "hi".to_string(),
            model: Some("haiku".to_string()),
            append_system_prompt: Some("prior outcome records".to_string()),
            ..Default::default()
        };
        let cmd = build_command(&options);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"haiku".to_string()));
        assert!(args.contains(&"--append-system-prompt".to_string()));
        assert!(args.contains(&"prior outcome records".to_string()));
    }

    /// Writes a fake `claude`-like script that ignores its arguments and prints canned
    /// stream-json lines matching the real observed schema. Real subprocess, fake binary —
    /// exercises the actual spawn/parse/aggregate path without needing an authenticated
    /// `claude` session, which CI runners don't have.
    #[cfg(unix)]
    fn write_fake_claude_script(dir: &tempfile::TempDir, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.path().join("fake-claude.sh");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write fake script");
        let mut perms = std::fs::metadata(&path).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod");
        path
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(claude_bridge_binary_env)]
    async fn run_print_turn_against_fake_binary_aggregates_text_and_session_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = write_fake_claude_script(
            &dir,
            r#"cat <<'EOF'
{"type":"assistant","message":{"content":[{"type":"text","text":"pong"}]}}
{"type":"result","subtype":"success","is_error":false,"result":"pong","session_id":"fake-session-123"}
EOF"#,
        );

        // SAFETY: process-global env var; serialized via #[serial] against the same key
        // used by every test in this module that touches it.
        unsafe {
            std::env::set_var(BINARY_OVERRIDE_ENV, &script);
        }

        let result = run_print_turn(ClaudePrintOptions {
            prompt: "ignored by fake script".to_string(),
            ..Default::default()
        })
        .await;

        unsafe {
            std::env::remove_var(BINARY_OVERRIDE_ENV);
        }

        let outcome = result.expect("run_print_turn should succeed against fake binary");
        assert_eq!(outcome.text.as_deref(), Some("pong"));
        assert_eq!(outcome.session_id.as_deref(), Some("fake-session-123"));
        assert!(!outcome.is_error);
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(claude_bridge_binary_env)]
    async fn run_print_turn_passes_resume_flag_to_fake_binary() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Fake script echoes back the argument that followed --resume, proving the flag
        // actually reached the subprocess rather than just being present in our own args.
        let script = write_fake_claude_script(
            &dir,
            r#"resume_id=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--resume" ]; then
    resume_id="$2"
  fi
  shift
done
printf '{"type":"result","subtype":"success","is_error":false,"result":"resumed","session_id":"%s"}\n' "$resume_id""#,
        );

        unsafe {
            std::env::set_var(BINARY_OVERRIDE_ENV, &script);
        }

        let result = run_print_turn(ClaudePrintOptions {
            prompt: "continue".to_string(),
            resume_session_id: Some("prior-session-456".to_string()),
            ..Default::default()
        })
        .await;

        unsafe {
            std::env::remove_var(BINARY_OVERRIDE_ENV);
        }

        let outcome = result.expect("run_print_turn should succeed against fake binary");
        assert_eq!(outcome.session_id.as_deref(), Some("prior-session-456"));
    }

    /// A process that fails before emitting any stream-json `result` line (e.g. an auth
    /// error) used to surface as total silence: stdout aggregated to nothing, `is_error`
    /// stayed at its `false` default, and stderr was inherited straight to the raw-mode
    /// terminal instead of being captured. This proves the fallback path: exit status is
    /// checked, and stderr is captured and reported.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(claude_bridge_binary_env)]
    async fn run_print_turn_surfaces_stderr_when_process_exits_without_result_event() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = write_fake_claude_script(
            &dir,
            r#"echo "authentication required" 1>&2
exit 1"#,
        );

        unsafe {
            std::env::set_var(BINARY_OVERRIDE_ENV, &script);
        }

        let result = run_print_turn(ClaudePrintOptions {
            prompt: "hi".to_string(),
            ..Default::default()
        })
        .await;

        unsafe {
            std::env::remove_var(BINARY_OVERRIDE_ENV);
        }

        let outcome = result.expect("run_print_turn should not itself error");
        assert!(outcome.text.is_none());
        assert!(outcome.is_error);
        let message = outcome.error_message.expect("expected an error message");
        assert!(message.contains("authentication required"), "{message}");
    }

    /// A hung child (e.g. an interactive prompt with no TTY to answer it) must not leave
    /// the turn unresolved forever; it should time out, get killed, and report why.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(claude_bridge_binary_env)]
    async fn run_print_turn_with_timeout_reports_timeout_and_kills_hung_child() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = write_fake_claude_script(&dir, "sleep 5");

        unsafe {
            std::env::set_var(BINARY_OVERRIDE_ENV, &script);
        }

        let result = run_print_turn_with_timeout(
            ClaudePrintOptions {
                prompt: "hi".to_string(),
                ..Default::default()
            },
            Duration::from_millis(200),
            None,
        )
        .await;

        unsafe {
            std::env::remove_var(BINARY_OVERRIDE_ENV);
        }

        let outcome = result.expect("run_print_turn_with_timeout should not itself error");
        assert!(outcome.is_error);
        let message = outcome.error_message.expect("expected a timeout message");
        assert!(message.contains("did not respond within"), "{message}");
    }

    /// A turn cancelled mid-flight (e.g. the user hits Ctrl+C) must return promptly with
    /// `is_error: true` and the `"cancelled"` sentinel, not wait for the child to finish on
    /// its own — proving the cancel oneshot actually races the aggregation loop and kills
    /// the child rather than being ignored.
    #[cfg(unix)]
    #[tokio::test]
    #[serial_test::serial(claude_bridge_binary_env)]
    async fn run_cancellable_print_turn_returns_quickly_when_cancelled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let script = write_fake_claude_script(&dir, "sleep 5");

        unsafe {
            std::env::set_var(BINARY_OVERRIDE_ENV, &script);
        }

        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(run_cancellable_print_turn(
            ClaudePrintOptions {
                prompt: "hi".to_string(),
                ..Default::default()
            },
            cancel_rx,
        ));

        tokio::time::sleep(Duration::from_millis(50)).await;
        let start = std::time::Instant::now();
        let _ = cancel_tx.send(());

        let result = handle.await.expect("task should not panic");
        let elapsed = start.elapsed();

        unsafe {
            std::env::remove_var(BINARY_OVERRIDE_ENV);
        }

        let outcome = result.expect("run_cancellable_print_turn should not itself error");
        assert!(outcome.is_error);
        assert_eq!(outcome.error_message.as_deref(), Some("cancelled"));
        assert!(
            elapsed < Duration::from_secs(2),
            "expected cancellation to return well before the 5s sleep, took {elapsed:?}"
        );
    }
}
