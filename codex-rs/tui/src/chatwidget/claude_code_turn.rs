//! Routes a submitted message through `codex-claude-bridge` when
//! `active_runtime == ActiveRuntime::ClaudeCode`, instead of the normal Codex
//! `AppServerClient` path.
//!
//! Scope, stated plainly: whole-message text-in/text-out only. No incremental
//! token-by-token rendering (unlike Codex's `StreamController`), and no bridging of
//! `tool_use`/`tool_result` events onto Elpis's approval/diff/permission surfaces —
//! `codex-claude-bridge` itself doesn't parse those content-block shapes yet either.
//! A Claude Code turn that calls a tool will not show that in the TUI; only its final
//! text (if any) will render, once the process exits.

use codex_claude_bridge::ClaudePrintOptions;
use codex_claude_bridge::run_print_turn;

impl super::ChatWidget {
    pub(super) fn is_claude_code_turn_running(&self) -> bool {
        self.claude_code_turn_running
    }

    /// Entry point from `input_submission.rs` when the active runtime is Claude Code.
    /// Does not touch Codex's `UserInput`/`AppServerClient` machinery at all.
    pub(super) fn submit_user_message_to_claude_code(&mut self, text: String) {
        if text.trim().is_empty() {
            return;
        }
        if self.claude_code_turn_running {
            self.add_error_message(
                "A Claude Code turn is already running; wait for it to finish.".to_string(),
            );
            return;
        }

        self.add_info_message(format!("You (Claude Code): {text}"), None);
        self.claude_code_turn_running = true;
        self.request_redraw();

        let options = ClaudePrintOptions {
            prompt: text,
            resume_session_id: self.claude_code_session_id.clone(),
            cwd: Some(self.config.cwd.clone()),
        };
        let tx = self.app_event_tx.clone();
        tokio::spawn(async move {
            match run_print_turn(options).await {
                Ok(outcome) => {
                    tx.send(crate::app_event::AppEvent::ClaudeCodeTurnCompleted {
                        text: outcome.text,
                        session_id: outcome.session_id,
                        error: if outcome.is_error {
                            Some(outcome.error_message.unwrap_or_else(|| {
                                "Claude Code reported an error with no message".to_string()
                            }))
                        } else {
                            None
                        },
                    });
                }
                Err(err) => {
                    tx.send(crate::app_event::AppEvent::ClaudeCodeTurnCompleted {
                        text: None,
                        session_id: None,
                        error: Some(format!("failed to run Claude Code: {err}")),
                    });
                }
            }
        });
    }

    /// Handles `AppEvent::ClaudeCodeTurnCompleted`, dispatched from `event_dispatch.rs`.
    pub(crate) fn handle_claude_code_turn_completed(
        &mut self,
        text: Option<String>,
        session_id: Option<String>,
        error: Option<String>,
    ) {
        self.claude_code_turn_running = false;
        if let Some(session_id) = session_id {
            self.claude_code_session_id = Some(session_id);
        }
        if let Some(error) = error {
            self.add_error_message(format!("Claude Code turn failed: {error}"));
        } else if let Some(text) = text {
            self.add_info_message(format!("Claude Code: {text}"), None);
        } else {
            self.add_info_message(
                "Claude Code turn completed with no text output (it may have only used \
                 tools, which this bridge doesn't render yet)."
                    .to_string(),
                None,
            );
        }
        self.request_redraw();
    }
}
