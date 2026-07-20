//! Routes a submitted message through `codex-claude-bridge` when
//! `active_runtime == ActiveRuntime::ClaudeCode` — and implements the ace:
//! per-turn context deletion by composition.
//!
//! Every turn is a FRESH `claude -p` call (no `--resume`), so the next request
//! contains exactly what Elpis composes into it. After each turn, a cheap
//! `--model haiku` call distills the turn into a compact outcome record; only
//! those records are re-sent (via `--append-system-prompt`), so working context
//! shrinks instead of growing. The raw transcript is appended to
//! `<codex_home>/claude_turns.jsonl` as durable evidence, and the chars *not*
//! re-sent accumulate into the `saved` field in the ELPIS status line (no
//! per-turn line in the transcript — it crowded the chat history).
//!
//! Still whole-message text-in/text-out only: no incremental streaming render,
//! and `tool_use`/`tool_result` events are not bridged onto Elpis's
//! approval/diff surfaces.

use std::path::Path;

use codex_claude_bridge::ClaudePrintOptions;
use codex_claude_bridge::run_cancellable_print_turn;
use codex_claude_bridge::run_print_turn;

/// Sentinel `TurnOutcome::error_message` used by `codex-claude-bridge` when a turn is
/// cancelled via `cancel_claude_code_turn`. Kept identical to the string literal in
/// `claude-bridge/src/lib.rs`'s `run_print_turn_with_timeout` — matched on below to show
/// a clear "cancelled" message instead of routing it through the generic error path.
const CANCELLED_SENTINEL: &str = "cancelled";

/// Max chars of raw assistant text kept as the fallback record when the haiku
/// distillation call itself fails. ponytail: blunt truncation; smarter fallback
/// only if distillation failures turn out to be common.
const FALLBACK_RECORD_CHARS: usize = 400;

impl super::ChatWidget {
    // pub(crate), not pub(super): `tui::app::event_dispatch` also needs this, to redirect
    // an Esc-triggered `Op::Interrupt` (which means nothing to a Claude Code turn) into
    // `cancel_claude_code_turn` instead of forwarding it to the app-server.
    pub(crate) fn is_claude_code_turn_running(&self) -> bool {
        self.claude_code_turn_running
    }

    /// Cancels the in-flight Claude Code turn, if any. Returns whether one was running.
    pub(crate) fn cancel_claude_code_turn(&mut self) -> bool {
        match self.claude_code_turn_cancel.take() {
            Some(tx) => {
                let _ = tx.send(());
                true
            }
            None => false,
        }
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

        self.add_info_message(format!("You: {text}"), None);
        self.claude_code_turn_running = true;
        self.bottom_pane.set_task_running(true);
        self.request_redraw();

        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        self.claude_code_turn_cancel = Some(cancel_tx);

        let options = ClaudePrintOptions {
            prompt: text.clone(),
            // The ace: never resume. Each turn is fresh; prior turns exist only as
            // the distilled records composed below.
            resume_session_id: None,
            cwd: Some(self.config.cwd.to_path_buf()),
            model: self.claude_model.clone(),
            append_system_prompt: compose_working_set(&self.claude_outcome_records),
        };
        let transcript_path = self.config.codex_home.as_path().join("claude_turns.jsonl");
        let tx = self.app_event_tx.clone();
        tokio::spawn(async move {
            match run_cancellable_print_turn(options, cancel_rx).await {
                Ok(outcome) => {
                    let raw_chars = text.chars().count()
                        + outcome
                            .text
                            .as_deref()
                            .map_or(0, |reply| reply.chars().count());
                    let outcome_record = if outcome.is_error {
                        None
                    } else {
                        Some(match outcome.text.as_deref() {
                            Some(reply) => distill_turn(&text, reply).await,
                            None => "outcome: turn completed with no text output".to_string(),
                        })
                    };
                    append_raw_transcript(
                        &transcript_path,
                        &text,
                        outcome.text.as_deref(),
                        outcome_record.as_deref(),
                        outcome.session_id.as_deref(),
                    );
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
                        outcome_record,
                        raw_chars,
                    });
                }
                Err(err) => {
                    tx.send(crate::app_event::AppEvent::ClaudeCodeTurnCompleted {
                        text: None,
                        session_id: None,
                        error: Some(format!("failed to run Claude Code: {err}")),
                        outcome_record: None,
                        raw_chars: 0,
                    });
                }
            }
        });
    }

    /// Handles `AppEvent::ClaudeCodeTurnCompleted`, dispatched from `event_dispatch.rs`.
    pub(crate) fn handle_claude_code_turn_completed(
        &mut self,
        text: Option<String>,
        _session_id: Option<String>,
        error: Option<String>,
        outcome_record: Option<String>,
        raw_chars: usize,
    ) {
        self.claude_code_turn_running = false;
        self.claude_code_turn_cancel = None;
        self.bottom_pane.set_task_running(false);
        if let Some(error) = error {
            if error == CANCELLED_SENTINEL {
                self.add_info_message("Claude Code turn cancelled.".to_string(), None);
            } else {
                self.add_error_message(format!("Claude Code turn failed: {error}"));
            }
        } else if let Some(text) = text {
            self.add_info_message(format!("elpis: {text}"), None);
        } else {
            self.add_info_message(
                "Claude Code turn completed with no text output (it may have only used \
                 tools, which this bridge doesn't render yet)."
                    .to_string(),
                None,
            );
        }
        if let Some(record) = outcome_record {
            let kept_chars = record.chars().count();
            let saved = raw_chars.saturating_sub(kept_chars);
            self.claude_outcome_records.push(record);
            self.claude_context_saved_chars = self.claude_context_saved_chars.saturating_add(saved);
            // Not printed per-turn (it crowded the transcript); the running total feeds
            // the `saved` field in the ELPIS status line instead, so it's visible without
            // repeating on every message.
            crate::branding::record_context_saved(saved as u64);
        }
        self.request_redraw();
    }
}

/// Composes the `--append-system-prompt` working set from prior turns' outcome
/// records. Returns `None` on the first turn (nothing to carry).
fn compose_working_set(records: &[String]) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    let mut out = String::from(
        "Elpis session continuity: earlier turns of this session were distilled into the \
         outcome records below (newest last). Treat them as accurate history; raw \
         transcripts remain on disk.\n",
    );
    for (index, record) in records.iter().enumerate() {
        out.push_str(&format!("--- turn {} ---\n{record}\n", index + 1));
    }
    Some(out)
}

/// Distills one raw turn into a compact outcome record via a cheap haiku call.
/// Falls back to a truncated raw excerpt if the distillation call fails, so a
/// flaky distiller can never lose the turn entirely.
async fn distill_turn(user_text: &str, assistant_text: &str) -> String {
    let options = ClaudePrintOptions {
        prompt: format!("User said:\n{user_text}\n\nAssistant turn to distill:\n{assistant_text}"),
        model: Some("haiku".to_string()),
        append_system_prompt: Some(
            "You compress one agent-conversation turn into a compact outcome record. \
             Reply with ONLY the record, at most 6 short lines, in this format:\n\
             outcome: what happened\n\
             decisions: choices made, or none\n\
             constraints: new constraints or facts worth remembering, or none\n\
             paths: files or directories touched, or none\n\
             Keep exact identifiers, paths, and numbers. No preamble."
                .to_string(),
        ),
        ..Default::default()
    };
    match run_print_turn(options).await {
        Ok(outcome) if !outcome.is_error => outcome
            .text
            .filter(|record| !record.trim().is_empty())
            .unwrap_or_else(|| fallback_record(assistant_text)),
        _ => fallback_record(assistant_text),
    }
}

fn fallback_record(assistant_text: &str) -> String {
    let excerpt: String = assistant_text.chars().take(FALLBACK_RECORD_CHARS).collect();
    format!("outcome (undistilled excerpt): {excerpt}")
}

/// One JSONL line per turn in `<codex_home>/claude_turns.jsonl` — the durable raw
/// transcript the composed working set points away from. Best-effort: a failed
/// write never fails the turn.
fn append_raw_transcript(
    path: &Path,
    user_text: &str,
    assistant_text: Option<&str>,
    outcome_record: Option<&str>,
    claude_session_id: Option<&str>,
) {
    let entry = serde_json::json!({
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "user": user_text,
        "assistant": assistant_text,
        "record": outcome_record,
        "claude_session_id": claude_session_id,
    });
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write as _;
        let _ = writeln!(file, "{entry}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_working_set_is_none_on_first_turn() {
        assert_eq!(compose_working_set(&[]), None);
    }

    #[test]
    fn compose_working_set_numbers_records_in_order() {
        let records = vec!["outcome: a".to_string(), "outcome: b".to_string()];
        let composed = compose_working_set(&records).expect("records present");
        let a = composed.find("--- turn 1 ---\noutcome: a").expect("turn 1");
        let b = composed.find("--- turn 2 ---\noutcome: b").expect("turn 2");
        assert!(a < b);
    }

    #[test]
    fn fallback_record_truncates() {
        let long = "x".repeat(1_000);
        let record = fallback_record(&long);
        assert!(record.chars().count() <= FALLBACK_RECORD_CHARS + 40);
        assert!(record.starts_with("outcome (undistilled excerpt): "));
    }
}
