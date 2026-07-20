//! Routes a submitted message through `codex-claude-bridge` when
//! `active_runtime == ActiveRuntime::ClaudeCode` — and implements the ace:
//! deterministic per-turn context pruning, no model compression involved.
//!
//! Every turn is a FRESH `claude -p` call (no `--resume`), so the next request
//! contains exactly what Elpis composes into it. Completed turns are kept as raw
//! records; once a record ages out of the most recent turn it's excerpted
//! (head+tail, mirroring the deterministic cleaner in
//! `core/src/context_cleaner.rs`) rather than summarized by a model — a model
//! asked to compress tends to lose exactly the details that matter. The raw
//! transcript is appended to `<codex_home>/claude_turns.jsonl` as durable
//! evidence regardless, and the chars excerpted away accumulate into the `saved`
//! field in the ELPIS status line (no per-turn line in the transcript — it
//! crowded the chat history).
//!
//! Still whole-message text-in/text-out only: no incremental streaming render,
//! and `tool_use`/`tool_result` events are not bridged onto Elpis's
//! approval/diff surfaces.

use std::path::Path;

use codex_claude_bridge::ClaudePrintOptions;
use codex_claude_bridge::run_cancellable_print_turn;

/// Sentinel `TurnOutcome::error_message` used by `codex-claude-bridge` when a turn is
/// cancelled via `cancel_claude_code_turn`. Kept identical to the string literal in
/// `claude-bridge/src/lib.rs`'s `run_print_turn_with_timeout` — matched on below to show
/// a clear "cancelled" message instead of routing it through the generic error path.
const CANCELLED_SENTINEL: &str = "cancelled";

/// How many of the newest turn records stay verbatim in the working set; anything
/// older is excerpted. Mirrors `RECENT_TOOL_OUTPUTS_TO_KEEP` in
/// `core/src/context_cleaner.rs`.
const RECENT_TURNS_TO_KEEP: usize = 1;
/// Records at or under this length are left untouched. Mirrors
/// `MAX_INLINE_TOOL_OUTPUT_CHARS` in `core/src/context_cleaner.rs`.
const MAX_INLINE_RECORD_CHARS: usize = 400;
/// Head/tail chars kept from an excerpted record. Mirrors `RETAINED_EDGE_CHARS` in
/// `core/src/context_cleaner.rs`.
const RETAINED_EDGE_CHARS: usize = 150;

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
                    let outcome_record = if outcome.is_error {
                        None
                    } else {
                        Some(match outcome.text.as_deref() {
                            Some(reply) => format!("User: {text}\nAssistant: {reply}"),
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
                    });
                }
                Err(err) => {
                    tx.send(crate::app_event::AppEvent::ClaudeCodeTurnCompleted {
                        text: None,
                        session_id: None,
                        error: Some(format!("failed to run Claude Code: {err}")),
                        outcome_record: None,
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
            self.claude_outcome_records.push(record);
            // Deterministic excerpting, not model compression: only the newest turn stays
            // verbatim; anything older gets excerpted in place. Idempotent, so running this
            // on every turn never double-counts a record already excerpted.
            let saved = compact_aged_records(&mut self.claude_outcome_records);
            if saved > 0 {
                self.claude_context_saved_chars =
                    self.claude_context_saved_chars.saturating_add(saved);
                // Not printed per-turn (it crowded the transcript); the running total feeds
                // the `saved` field in the ELPIS status line instead, so it's visible without
                // repeating on every message.
                crate::branding::record_context_saved(saved as u64);
            }
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
        "Elpis session continuity: earlier turns of this session are shown below (newest \
         last), excerpted rather than summarized once they age out of the most recent turn. \
         Raw transcripts remain on disk.\n",
    );
    for (index, record) in records.iter().enumerate() {
        out.push_str(&format!("--- turn {} ---\n{record}\n", index + 1));
    }
    Some(out)
}

/// Excerpts every record beyond the newest `RECENT_TURNS_TO_KEEP` in place
/// (head+tail, pointing at the durable raw transcript on disk), returning the
/// chars saved. Idempotent — an already-excerpted record is short enough that
/// re-excerpting it is a no-op — so calling this every turn never double-counts.
fn compact_aged_records(records: &mut [String]) -> usize {
    let keep_from = records.len().saturating_sub(RECENT_TURNS_TO_KEEP);
    let mut saved = 0usize;
    for record in &mut records[..keep_from] {
        let before = record.chars().count();
        let excerpted = excerpt(record);
        saved += before.saturating_sub(excerpted.chars().count());
        *record = excerpted;
    }
    saved
}

/// Deterministic head+tail excerpt — no model call, nothing summarized. Chars in
/// between are dropped with a pointer to `claude_turns.jsonl`, the durable raw
/// archive every turn is already appended to regardless.
fn excerpt(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= MAX_INLINE_RECORD_CHARS {
        return text.to_string();
    }
    let head: String = chars[..RETAINED_EDGE_CHARS].iter().collect();
    let tail: String = chars[chars.len() - RETAINED_EDGE_CHARS..].iter().collect();
    let omitted = chars.len() - 2 * RETAINED_EDGE_CHARS;
    format!("{head}\n… {omitted} chars omitted, see claude_turns.jsonl …\n{tail}")
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
    fn excerpt_leaves_short_text_untouched() {
        let short = "outcome: a".to_string();
        assert_eq!(excerpt(&short), short);
    }

    #[test]
    fn excerpt_shrinks_long_text_and_keeps_head_and_tail() {
        let long = format!("HEAD{}TAIL", "x".repeat(1_000));
        let result = excerpt(&long);
        assert!(result.chars().count() < long.chars().count());
        assert!(result.starts_with("HEAD"));
        assert!(result.ends_with("TAIL"));
        assert!(result.contains("chars omitted, see claude_turns.jsonl"));
    }

    #[test]
    fn excerpt_is_idempotent() {
        let long = "x".repeat(1_000);
        let once = excerpt(&long);
        let twice = excerpt(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn compact_aged_records_keeps_newest_verbatim_and_excerpts_the_rest() {
        let newest = "x".repeat(1_000);
        let mut records = vec!["y".repeat(1_000), newest.clone()];
        let saved = compact_aged_records(&mut records);
        assert!(saved > 0);
        assert_eq!(records[1], newest, "newest record must stay verbatim");
        assert!(
            records[0].chars().count() < 1_000,
            "aged-out record must be excerpted"
        );
    }

    #[test]
    fn compact_aged_records_never_double_counts_an_already_excerpted_record() {
        let mut records = vec!["x".repeat(1_000), "y".repeat(1_000)];
        let first_pass = compact_aged_records(&mut records);
        assert!(first_pass > 0);
        records.push("z".repeat(1_000));
        let second_pass = compact_aged_records(&mut records);
        // Index 0 was already excerpted by the first pass; re-compacting it (still
        // aged-out) must save nothing further, only the newly-aged index 1 counts.
        assert!(second_pass > 0);
        assert!(second_pass < first_pass);
    }
}
