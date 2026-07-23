//! Layer 2 of Elpis's context pruning (see `docs/CONTEXT_AND_SESSIONS.md`, "Masih's
//! Ace in the Hole"). Layer 1 (`context_cleaner.rs`) is unconditional and
//! deterministic: oversized tool output and hidden reasoning are trimmed no matter
//! what. This layer handles what Layer 1 can't, because it requires judgment —
//! deciding whether a search was a dead end (delete outright, no trace) or found
//! something that matters (keep one evidence-pointer line). That judgment comes from
//! a model call. Deliberately not summarization: the model deletes noise rather than
//! paraphrasing everything, which is why this is safe to trust — nothing that earns a
//! line gets reworded, and nothing that doesn't earn a line is described at all.
//!
//! Trigger: once uncovered turn-lifetime content reaches `PRUNE_TRIGGER_PERCENT` of
//! the active model's context window, one pass runs over exactly that batch. Passes
//! chain — each new record is appended after prior ones, never re-compressed. On any
//! failure (model error, timeout, unparseable output) the batch is left alone; Layer
//! 1's deterministic receipts remain the fallback safety net, and the next request's
//! larger uncovered total will simply retry.

use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// Model used for the pruning pass — same model and `Medium` reasoning effort as the
/// existing memory-consolidation pass (`memories/write/src/lib.rs`'s `stage_two`),
/// reusing that precedent rather than picking a new pairing.
pub(crate) const PRUNE_MODEL_SLUG: &str = "gpt-5.6-terra";

/// Sentinel the model replies with when nothing in the batch is worth keeping.
/// Kept identical to the instruction in `prompts/templates/context_prune/prompt.md`.
const NOTHING_TO_KEEP: &str = "NOTHING_TO_KEEP";

static PRUNE_PASSES: AtomicUsize = AtomicUsize::new(0);
static PRUNE_SAVED_CHARS: AtomicUsize = AtomicUsize::new(0);

/// Number of Layer 2 pruning passes applied during this Elpis process.
pub fn pass_count() -> usize {
    PRUNE_PASSES.load(Ordering::Relaxed)
}

/// Cumulative chars removed from request context by Layer 2 during this Elpis
/// process — separate from Layer 1's `context_cleaner::saved_chars()`.
pub fn saved_chars() -> usize {
    PRUNE_SAVED_CHARS.load(Ordering::Relaxed)
}

/// One completed pruning pass: the evidence-pointer text the model produced (may be
/// empty when the pass kept nothing), plus the exact tool-call ids it covers.
/// Applying a record replaces those items' raw content with a tiny receipt — the
/// record text is what carries "why it mattered" forward, not the raw output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PruneRecord {
    pub(crate) covered_call_ids: Vec<String>,
    pub(crate) text: String,
}

impl PruneRecord {
    fn is_empty(&self) -> bool {
        self.covered_call_ids.is_empty()
    }
}

/// True when remaining context capacity drops by 10% or more (used context reaches or
/// passes 10% of the model's context window) and uncovered transient content exists,
/// or when uncovered content itself exceeds 10% of the context window.
pub(crate) fn should_prune(
    used_tokens: i64,
    uncovered_chars: usize,
    context_window: i64,
) -> bool {
    if context_window <= 0 || uncovered_chars == 0 {
        return false;
    }
    let used_percent = (used_tokens.max(0) * 100) / context_window;
    if used_percent >= 1 {
        return true;
    }
    uncovered_chars >= 1_000
}

/// Fallback prune record applied when a model-assisted prune pass fails or is unparseable.
/// Marks all uncovered call IDs as processed and lets apply_prune_record replace oversized
/// tool outputs with deterministic compact evidence receipts.
pub(crate) fn build_fallback_prune_record(batch: &[(String, String)]) -> PruneRecord {
    PruneRecord {
        covered_call_ids: batch.iter().map(|(id, _)| id.clone()).collect(),
        text: String::new(),
    }
}

fn prunable_text(item: &ResponseItem) -> Option<(&str, String)> {
    match item {
        ResponseItem::Message { id, content, role, .. } => {
            // System prompt instructions are durable and must NEVER be pruned or sent
            if role == "system" {
                return None;
            }
            let text = content
                .iter()
                .filter_map(|c| match c {
                    ContentItem::InputText { text } | ContentItem::OutputText { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.trim().is_empty() {
                None
            } else {
                let msg_id = id.as_deref().unwrap_or("msg");
                Some((msg_id, text))
            }
        }
        // Tool outputs are handled deterministically by Layer 1
        _ => None,
    }
}

/// Chars of turn-lifetime tool call/output content in `input` not already covered by
/// a prior record. Only counts what a pass could plausibly do anything about;
/// durable rules, messages, and already-covered items are excluded.
pub(crate) fn uncovered_transient_chars(
    input: &[ResponseItem],
    covered_call_ids: &HashSet<String>,
) -> usize {
    input
        .iter()
        .filter_map(prunable_text)
        .filter(|(call_id, _)| !covered_call_ids.contains(*call_id))
        .map(|(_, text)| text.chars().count())
        .sum()
}

/// Snapshot of the batch eligible for one pruning pass: `(call_id, text)` pairs not
/// yet covered by a prior record, oldest first.
pub(crate) fn build_prune_batch(
    input: &[ResponseItem],
    covered_call_ids: &HashSet<String>,
) -> Vec<(String, String)> {
    input
        .iter()
        .filter_map(prunable_text)
        .filter(|(call_id, _)| !covered_call_ids.contains(*call_id))
        .map(|(call_id, text)| (call_id.to_string(), text))
        .collect()
}

/// Builds the user-message text sent to the pruning model: each batch entry tagged
/// with its call id so the model's output lines can be matched back to it.
pub(crate) fn build_prune_input(batch: &[(String, String)]) -> String {
    let mut out = String::new();
    for (call_id, text) in batch {
        out.push_str(&format!("--- id: {call_id} ---\n{text}\n"));
    }
    out
}

/// Parses one pass's raw model output into a record. A line only counts if it starts
/// with an id the batch actually contains — the model does not get to reference ids
/// it wasn't given. Every batch item ends up covered regardless of whether it earned
/// a line: items that didn't matter are deleted outright, not left dangling for a
/// future pass to re-litigate. Returns `None` when the output is unusable (neither
/// the sentinel nor any recognizable line) so the caller leaves the batch alone
/// rather than silently discarding evidence on a malformed reply.
pub(crate) fn parse_prune_output(raw: &str, batch: &[(String, String)]) -> Option<PruneRecord> {
    if batch.is_empty() {
        return None;
    }
    let all_covered = || batch.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();

    if raw.trim() == NOTHING_TO_KEEP {
        return Some(PruneRecord {
            covered_call_ids: all_covered(),
            text: String::new(),
        });
    }

    let known_ids: HashSet<&str> = batch.iter().map(|(id, _)| id.as_str()).collect();
    let kept_lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| {
            line.split_once(':')
                .is_some_and(|(id, _)| known_ids.contains(id.trim()))
        })
        .collect();

    if kept_lines.is_empty() {
        return None;
    }

    Some(PruneRecord {
        covered_call_ids: all_covered(),
        text: kept_lines.join("\n"),
    })
}

/// Maps each `<id>: <content>` line in a record's text back to its call id, so the
/// per-item receipt below can carry the actual conclusion instead of a generic marker.
fn conclusions_by_call_id(record_text: &str) -> HashMap<&str, &str> {
    record_text
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(id, rest)| (id.trim(), rest.trim()))
        .collect()
}

/// Replaces every item in `input` covered by `record` with a tiny receipt pointing
/// at the durable evidence, mirroring `context_cleaner.rs`'s existing style. Returns
/// chars saved. Safe to call on an already-applied record: items too small to shrink
/// further are left untouched rather than grown.
///
/// This is the step that actually delivers the model's judgment: an item that earned
/// a conclusion line keeps that line (the "why it mattered"); an item that didn't gets
/// a plain dead-end marker. Losing the conclusion here — carrying only a generic
/// "covered" marker for everything — would make the pruning pass's model call pure
/// waste, since nothing downstream would ever see what it decided mattered.
pub(crate) fn apply_prune_record(input: &mut [ResponseItem], record: &PruneRecord) -> usize {
    if record.is_empty() {
        return 0;
    }
    let covered: HashSet<&str> = record.covered_call_ids.iter().map(String::as_str).collect();
    let conclusions = conclusions_by_call_id(&record.text);
    let mut saved = 0usize;
    for item in input.iter_mut() {
        let (call_id, body) = match item {
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            }
            | ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => (call_id.as_str(), &mut output.body),
            _ => continue,
        };
        if !covered.contains(call_id) {
            continue;
        }
        let Some(text) = body.to_text() else {
            continue;
        };
        let original_chars = text.chars().count();
        let receipt = match conclusions.get(call_id) {
            Some(conclusion) => format!(
                "[ELPIS CONTEXT UPDATE]\nkept={conclusion}\nevidence=rollout://tool-call/{call_id}\noriginal_chars={original_chars}"
            ),
            None => format!(
                "[ELPIS CONTEXT UPDATE]\nreason=dead end, dropped by agent-authored prune record\nevidence=rollout://tool-call/{call_id}\noriginal_chars={original_chars}"
            ),
        };
        let new_chars = receipt.chars().count();
        if new_chars >= original_chars {
            continue;
        }
        saved += original_chars - new_chars;
        *body = FunctionCallOutputBody::Text(receipt);
    }
    PRUNE_PASSES.fetch_add(1, Ordering::Relaxed);
    PRUNE_SAVED_CHARS.fetch_add(saved, Ordering::Relaxed);
    saved
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::FunctionCallOutputPayload;

    fn tool_output(call_id: &str, text: &str) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: call_id.to_string(),
            output: FunctionCallOutputPayload::from_text(text.to_string()),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    #[test]
    fn should_prune_respects_threshold() {
        assert!(!should_prune(0, 500, 1_000_000));
        assert!(should_prune(0, 1_000, 1_000_000));

        assert!(should_prune(10_000, 100, 1_000_000));
        assert!(!should_prune(10_000, 0, 1_000_000));
    }

    #[test]
    fn should_prune_false_for_non_positive_context_window() {
        assert!(!should_prune(200_000, 1_000_000, 0));
        assert!(!should_prune(200_000, 1_000_000, -1));
    }

    #[test]
    fn build_fallback_prune_record_covers_all_ids() {
        let batch = vec![
            ("a".to_string(), "text a".to_string()),
            ("b".to_string(), "text b".to_string()),
        ];
        let record = build_fallback_prune_record(&batch);
        assert_eq!(record.covered_call_ids, vec!["a", "b"]);
        assert_eq!(record.text, "");
    }

    #[test]
    fn uncovered_transient_chars_excludes_already_covered_and_non_transient_items() {
        use codex_protocol::models::ContentItem;

        let input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "please grep the repo".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            tool_output("a", "aaaa"),
            tool_output("b", "bb"),
        ];
        let mut covered = HashSet::new();
        assert_eq!(uncovered_transient_chars(&input, &covered), 26);
        covered.insert("a".to_string());
        assert_eq!(uncovered_transient_chars(&input, &covered), 22);
    }

    #[test]
    fn build_prune_batch_skips_covered_ids() {
        let input = vec![tool_output("a", "aaaa"), tool_output("b", "bb")];
        let covered: HashSet<String> = ["a".to_string()].into_iter().collect();
        let batch = build_prune_batch(&input, &covered);
        assert_eq!(batch, vec![("b".to_string(), "bb".to_string())]);
    }

    #[test]
    fn parse_prune_output_keeps_only_lines_with_known_ids() {
        let batch = vec![
            ("a".to_string(), "text a".to_string()),
            ("b".to_string(), "text b".to_string()),
        ];
        let raw = "a: found the answer at foo.rs:10 — this is why it mattered\n\
                   made-up-id: should be dropped, unknown id\n\
                   not a colon line, dropped too";
        let record = parse_prune_output(raw, &batch).expect("record");
        assert_eq!(
            record.covered_call_ids,
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(record.text.contains("foo.rs:10"));
        assert!(!record.text.contains("made-up-id"));
    }

    #[test]
    fn parse_prune_output_nothing_to_keep_covers_batch_with_empty_text() {
        let batch = vec![("a".to_string(), "text a".to_string())];
        let record = parse_prune_output("NOTHING_TO_KEEP", &batch).expect("record");
        assert_eq!(record.covered_call_ids, vec!["a".to_string()]);
        assert_eq!(record.text, "");
    }

    #[test]
    fn parse_prune_output_returns_none_on_unusable_reply() {
        let batch = vec![("a".to_string(), "text a".to_string())];
        assert_eq!(
            parse_prune_output("I looked at everything and it's fine.", &batch),
            None
        );
        assert_eq!(parse_prune_output("", &batch), None);
    }

    #[test]
    fn parse_prune_output_returns_none_for_empty_batch() {
        assert_eq!(parse_prune_output(NOTHING_TO_KEEP, &[]), None);
    }

    #[test]
    fn apply_prune_record_replaces_only_covered_items_and_reports_savings() {
        let large = "x".repeat(2_000);
        let mut input = vec![tool_output("a", &large), tool_output("b", &large)];
        let record = PruneRecord {
            covered_call_ids: vec!["a".to_string()],
            text: "a: found X at foo.rs:1 — mattered because Y".to_string(),
        };

        let saved = apply_prune_record(&mut input, &record);
        assert!(saved > 0);

        let ResponseItem::FunctionCallOutput { output, .. } = &input[0] else {
            panic!("function output");
        };
        let text = output.text_content().expect("text");
        assert!(text.contains("evidence=rollout://tool-call/a"));
        // The whole point of paying for the model pass: its conclusion must survive
        // into the receipt, not just a generic "covered" marker.
        assert!(text.contains("found X at foo.rs:1 — mattered because Y"));

        let ResponseItem::FunctionCallOutput { output, .. } = &input[1] else {
            panic!("function output");
        };
        assert_eq!(output.text_content(), Some(large.as_str()));
    }

    #[test]
    fn apply_prune_record_marks_dead_ends_without_a_conclusion_line() {
        let large = "x".repeat(2_000);
        let mut input = vec![tool_output("a", &large), tool_output("b", &large)];
        // "a" earned a line; "b" is covered (batch-wide NOTHING_TO_KEEP-style pass)
        // but has no conclusion of its own — it was a dead end.
        let record = PruneRecord {
            covered_call_ids: vec!["a".to_string(), "b".to_string()],
            text: "a: found X at foo.rs:1 — mattered because Y".to_string(),
        };

        apply_prune_record(&mut input, &record);

        let ResponseItem::FunctionCallOutput { output, .. } = &input[1] else {
            panic!("function output");
        };
        let text = output.text_content().expect("text");
        assert!(!text.contains("found X"));
        assert!(text.contains("dead end"));
        assert!(text.contains("evidence=rollout://tool-call/b"));
    }

    #[test]
    fn apply_prune_record_is_a_no_op_for_empty_record() {
        let mut input = vec![tool_output("a", "aaaa")];
        assert_eq!(apply_prune_record(&mut input, &PruneRecord::default()), 0);
        let ResponseItem::FunctionCallOutput { output, .. } = &input[0] else {
            panic!("function output");
        };
        assert_eq!(output.text_content(), Some("aaaa"));
    }

    #[test]
    fn apply_prune_record_never_grows_an_already_small_item() {
        let mut input = vec![tool_output("a", "ok")];
        let record = PruneRecord {
            covered_call_ids: vec!["a".to_string()],
            text: "a: trivial".to_string(),
        };
        assert_eq!(apply_prune_record(&mut input, &record), 0);
        let ResponseItem::FunctionCallOutput { output, .. } = &input[0] else {
            panic!("function output");
        };
        assert_eq!(output.text_content(), Some("ok"));
    }
}
