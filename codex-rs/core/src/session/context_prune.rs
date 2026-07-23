//! Triggers Layer 2 context pruning (`crate::context_pruner`) once uncovered
//! turn-lifetime content has grown past the threshold. Mirrors
//! `super::token_budget::maybe_record`: a small, independent, isolated step called
//! from the turn loop. Any failure here is swallowed and never propagated — a broken,
//! slow, or unavailable pruning pass must never break or stall the user's actual
//! turn. Layer 1 (`context_cleaner.rs`) keeps handling oversized output regardless.

use std::sync::Arc;

use crate::client::ModelClientSession;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::context_pruner;
use crate::responses_metadata::CodexResponsesRequestKind;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_rollout_trace::InferenceTraceContext;
use futures::StreamExt;

use super::session::Session;
use super::turn_context::TurnContext;

pub(super) async fn maybe_run_context_prune(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    client_session: &mut ModelClientSession,
) {
    let context_window = turn_context
        .model_info
        .resolved_context_window()
        .unwrap_or(0);
    if context_window <= 0 {
        return;
    }

    let active_context_tokens = sess.get_total_token_usage().await;

    let covered_call_ids = {
        let state = sess.state.lock().await;
        state.context_prune_covered.clone()
    };

    let items = sess.clone_history().await.into_raw_items();
    let uncovered = context_pruner::uncovered_transient_chars(&items, &covered_call_ids);
    if !context_pruner::should_prune(active_context_tokens, uncovered, context_window) {
        return;
    }
    let batch = context_pruner::build_prune_batch(&items, &covered_call_ids);
    if batch.is_empty() {
        return;
    }

    let record = match run_prune_pass(sess, turn_context, client_session, &batch).await {
        Some(raw) => context_pruner::parse_prune_output(&raw, &batch)
            .unwrap_or_else(|| context_pruner::build_fallback_prune_record(&batch)),
        None => context_pruner::build_fallback_prune_record(&batch),
    };

    let mut state = sess.state.lock().await;
    let mut items = state.history.raw_items().to_vec();
    context_pruner::apply_prune_record(&mut items, &record);
    state.history.replace(items);
    state
        .context_prune_covered
        .extend(record.covered_call_ids.iter().cloned());
}

async fn run_prune_pass(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    client_session: &mut ModelClientSession,
    batch: &[(String, String)],
) -> Option<String> {
    let primary_slug = if turn_context.config.model_provider_id
        == codex_model_provider_info::OPENAI_PROVIDER_ID
    {
        context_pruner::PRUNE_MODEL_SLUG
    } else {
        turn_context.model_info.slug.as_str()
    };

    if let Some(output) =
        try_stream_prune_pass(sess, turn_context, client_session, batch, primary_slug).await
    {
        return Some(output);
    }

    if primary_slug != turn_context.model_info.slug.as_str() {
        return try_stream_prune_pass(
            sess,
            turn_context,
            client_session,
            batch,
            turn_context.model_info.slug.as_str(),
        )
        .await;
    }

    None
}

async fn try_stream_prune_pass(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    client_session: &mut ModelClientSession,
    batch: &[(String, String)],
    prune_model_slug: &str,
) -> Option<String> {
    let model_info = sess
        .services
        .models_manager
        .get_model_info(
            prune_model_slug,
            &turn_context.config.to_models_manager_config(),
        )
        .await;

    let input_text = context_pruner::build_prune_input(batch);
    let prompt = Prompt {
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: input_text.clone(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }],
        base_instructions: BaseInstructions {
            text: codex_prompts::CONTEXT_PRUNE_PROMPT.to_string(),
        },
        ..Default::default()
    };

    let responses_metadata = turn_context.turn_metadata_state.to_responses_metadata(
        sess.installation_id.clone(),
        "context-prune".to_string(),
        CodexResponsesRequestKind::ContextPrune,
    );

    let mut stream = match client_session
        .stream(
            &prompt,
            &model_info,
            &turn_context.session_telemetry,
            Some(ReasoningEffort::Medium),
            turn_context.reasoning_summary,
            turn_context.config.service_tier.clone(),
            &responses_metadata,
            &InferenceTraceContext::disabled(),
        )
        .await
    {
        Ok(stream) => stream,
        Err(err) => {
            tracing::warn!("Context prune stream failed for model {prune_model_slug}: {err}");
            log_prune_debug(prune_model_slug, &input_text, None);
            return None;
        }
    };

    let mut collected: Vec<ResponseItem> = Vec::new();
    loop {
        match stream.next().await {
            Some(Ok(ResponseEvent::OutputItemDone(item))) => collected.push(item),
            Some(Ok(ResponseEvent::Completed { .. })) => break,
            Some(Ok(_)) => continue,
            Some(Err(err)) => {
                tracing::warn!("Context prune stream error for model {prune_model_slug}: {err}");
                log_prune_debug(prune_model_slug, &input_text, None);
                return None;
            }
            None => break,
        }
    }
    let result = super::turn::get_last_assistant_message_from_turn(&collected);
    log_prune_debug(prune_model_slug, &input_text, result.as_deref());
    if let Some(ref text) = result {
        tracing::info!("Context prune LLM response received ({prune_model_slug}): {text}");
    } else {
        tracing::warn!("Context prune LLM stream returned no assistant text ({prune_model_slug})");
    }
    result
}

fn log_prune_debug(model_slug: &str, input_text: &str, output_text: Option<&str>) {
    if let Some(home) = std::env::var_os("HOME") {
        let log_dir = std::path::PathBuf::from(home).join(".codex").join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let log_file = log_dir.join("prune_debug.log");
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(log_file) {
            use std::io::Write;
            let ts = chrono::Utc::now().to_rfc3339();
            let out_str = output_text.unwrap_or("<NO OUTPUT / FAILED>");
            let _ = writeln!(
                file,
                "=== LAYER 2 PRUNING PASS [{ts}] ===\nMODEL: {model_slug}\n--- INPUT BATCH SENT TO LLM ---\n{input_text}\n--- LLM RESPONSE RECEIVED ---\n{out_str}\n=========================================\n"
            );
        }
    }
}
