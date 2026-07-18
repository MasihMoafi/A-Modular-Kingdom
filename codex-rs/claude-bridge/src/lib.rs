//! Minimal, standalone subprocess bridge to the Claude Code CLI's non-interactive
//! (`--print`) mode.
//!
//! This crate does exactly one thing: spawn `claude -p <prompt> --output-format
//! stream-json` and parse its stdout into typed events. It is intentionally NOT wired
//! into Elpis's runtime selection yet — no such selection point exists in the codebase
//! today (Elpis's TUI talks to exactly one backend, `codex_app_server_client::AppServerClient`,
//! via `codex-rs/tui/src/app_server_session.rs`; there is no enum or trait for "which
//! agent backend owns this turn"). Introducing that abstraction is a separate, larger
//! task. This crate is the foundation a future wiring step would build on.
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

use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
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
#[derive(Debug, Clone)]
pub struct ClaudePrintOptions {
    pub prompt: String,
    /// Passed as `--resume <id>` to continue a prior Claude Code session.
    pub resume_session_id: Option<String>,
    pub cwd: Option<PathBuf>,
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

fn build_command(options: &ClaudePrintOptions) -> Command {
    let mut cmd = Command::new("claude");
    cmd.arg("-p")
        .arg(&options.prompt)
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose");
    if let Some(id) = &options.resume_session_id {
        cmd.arg("--resume").arg(id);
    }
    if let Some(cwd) = &options.cwd {
        cmd.current_dir(cwd);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
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
) -> Result<(Child, mpsc::UnboundedReceiver<Result<ClaudeStreamEvent, ParseError>>), ClaudeBridgeError>
{
    let mut child = build_command(&options).spawn().map_err(ClaudeBridgeError::Spawn)?;
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
            cwd: None,
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
}
