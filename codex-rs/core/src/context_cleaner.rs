use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

const MAX_INLINE_TOOL_OUTPUT_CHARS: usize = 4_000;
const RETAINED_EDGE_CHARS: usize = 700;
const RECENT_TOOL_OUTPUTS_TO_KEEP: usize = 2;
static EVICTED_TOOL_OUTPUTS: AtomicUsize = AtomicUsize::new(0);
static LAST_EVICTION_EVENT: Mutex<Option<String>> = Mutex::new(None);

/// Number of transient tool outputs compacted during this Elpis process.
pub fn eviction_count() -> usize {
    EVICTED_TOOL_OUTPUTS.load(Ordering::Relaxed)
}

/// Most recent visible context-update event, including its durable evidence pointer.
pub fn latest_eviction_event() -> Option<String> {
    LAST_EVICTION_EVENT
        .lock()
        .ok()
        .and_then(|event| event.clone())
}

/// Applies Elpis's deterministic lifecycle to transient tool output.
///
/// The newest outputs remain intact. Older oversized outputs retain bounded head/tail
/// excerpts plus a stable pointer to the complete output in the durable rollout.
pub(crate) fn clean_transient_tool_outputs(input: &mut [ResponseItem]) -> usize {
    let tool_indices = input
        .iter()
        .enumerate()
        .filter_map(|(index, item)| match item {
            ResponseItem::FunctionCallOutput { .. } | ResponseItem::CustomToolCallOutput { .. } => {
                Some(index)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let keep_from = tool_indices
        .len()
        .saturating_sub(RECENT_TOOL_OUTPUTS_TO_KEEP);
    let mut evicted = 0;
    let mut latest_event = None;

    for index in tool_indices.into_iter().take(keep_from) {
        let (call_id, name, body) = match &mut input[index] {
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } => (call_id.as_str(), "function", &mut output.body),
            ResponseItem::CustomToolCallOutput {
                call_id,
                name,
                output,
                ..
            } => (
                call_id.as_str(),
                name.as_deref().unwrap_or("custom-tool"),
                &mut output.body,
            ),
            _ => continue,
        };
        let FunctionCallOutputBody::Text(text) = body else {
            continue;
        };
        let original_chars = text.chars().count();
        if original_chars <= MAX_INLINE_TOOL_OUTPUT_CHARS {
            continue;
        }
        let head = take_chars(text, RETAINED_EDGE_CHARS);
        let tail = take_last_chars(text, RETAINED_EDGE_CHARS);
        let evidence = format!("rollout://tool-call/{call_id}");
        *text = format!(
            "[ELPIS CONTEXT UPDATE]\nreason=older transient tool output exceeded {MAX_INLINE_TOOL_OUTPUT_CHARS} chars\nevidence={evidence}\ntool={name}\noriginal_chars={original_chars}\nretained=head:{RETAINED_EDGE_CHARS}+tail:{RETAINED_EDGE_CHARS}\n--- head ---\n{head}\n--- omitted; full evidence remains in durable rollout ---\n{tail}\n--- tail ---"
        );
        latest_event = Some(format!(
            "ELPIS continuity: compacted older {name} output ({original_chars} chars); exact evidence: {evidence}"
        ));
        evicted += 1;
    }
    if evicted > 0 {
        EVICTED_TOOL_OUTPUTS.fetch_add(evicted, Ordering::Relaxed);
        if let Ok(mut event) = LAST_EVICTION_EVENT.lock() {
            *event = latest_event;
        }
    }
    evicted
}

fn take_chars(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

fn take_last_chars(text: &str, count: usize) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    chars[chars.len().saturating_sub(count)..].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::FunctionCallOutputPayload;

    fn output(call_id: &str, text: String) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: call_id.to_string(),
            output: FunctionCallOutputPayload::from_text(text),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    #[test]
    fn evicts_only_old_large_outputs_and_preserves_evidence_pointer() {
        let large = format!("HEAD{}TAIL", "x".repeat(5_000));
        let mut input = vec![
            output("old", large.clone()),
            output("recent-1", large.clone()),
            output("recent-2", large.clone()),
        ];

        assert_eq!(clean_transient_tool_outputs(&mut input), 1);
        let ResponseItem::FunctionCallOutput { output, .. } = &input[0] else {
            panic!("function output");
        };
        let text = output.text_content().expect("text");
        assert!(text.contains("evidence=rollout://tool-call/old"));
        assert!(text.contains("HEAD"));
        assert!(text.contains("TAIL"));
        assert!(text.contains("full evidence remains in durable rollout"));
        for item in &input[1..] {
            let ResponseItem::FunctionCallOutput { output, .. } = item else {
                panic!("function output");
            };
            assert_eq!(output.text_content(), Some(large.as_str()));
        }
    }

    #[test]
    fn custom_tool_output_uses_tool_name_and_rollout_pointer() {
        let large = "z".repeat(5_000);
        let mut input = vec![
            ResponseItem::CustomToolCallOutput {
                id: None,
                call_id: "custom-old".to_string(),
                name: Some("browser".to_string()),
                output: FunctionCallOutputPayload::from_text(large.clone()),
                internal_chat_message_metadata_passthrough: None,
            },
            output("recent-1", large.clone()),
            output("recent-2", large),
        ];

        assert_eq!(clean_transient_tool_outputs(&mut input), 1);
        let ResponseItem::CustomToolCallOutput { output, .. } = &input[0] else {
            panic!("custom output");
        };
        let text = output.text_content().expect("text");
        assert!(text.contains("tool=browser"));
        assert!(text.contains("evidence=rollout://tool-call/custom-old"));
    }

    #[test]
    fn leaves_small_outputs_untouched() {
        let mut input = vec![output("small", "ok".to_string())];
        assert_eq!(clean_transient_tool_outputs(&mut input), 0);
        let ResponseItem::FunctionCallOutput { output, .. } = &input[0] else {
            panic!("function output");
        };
        assert_eq!(output.text_content(), Some("ok"));
    }
}
