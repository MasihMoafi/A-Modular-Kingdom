//! `/context` command: a colored context-usage grid, a per-category token breakdown,
//! a Checkpoints section backed by Elpis's real backtrack mechanism, and a System
//! files (auto-loaded) section backed by the same admitted-source list the Context
//! Ledger renders.
//!
//! Unlike `/usage`, this card needs no async round trip: its one cross-cutting input
//! (transcript composition, backtrack checkpoint count) is computed by `App` from
//! `transcript_cells` before `add_context_usage_output` is called; everything else is
//! already available synchronously on `ChatWidget` via `self.token_info` and
//! `self.continuity_sources()`.
//!
//! ## Why the category breakdown is a transcript-composition estimate
//!
//! `transcript_cells` is the full rendered history for this session; it is never
//! trimmed. Pruning (both layers — see `docs/CONTEXT_AND_SESSIONS.md`) is deliberately
//! ephemeral: it reduces what goes out on the *next request*, not the durable
//! transcript, and that reduction happens at request-assembly time in `client.rs`,
//! which this module has no visibility into. Reporting "category X is N% of the
//! context window" from transcript-only data is provably wrong whenever pruning has
//! run (the categories can sum to far more than the actual window occupancy — this
//! shipped broken once already: transcript composition claimed 53% of the window while
//! actual usage was 11.8%). To stay honest, category percentages here are **share of
//! the visible transcript** (they sum to ~100% by construction), never "% of window."
//! The one number that *is* exact — total tokens currently in context — comes from
//! `token_info.last_token_usage`, the same source `/usage` uses, and the already-proven
//! pruning-savings stats are surfaced alongside it so the gap between "transcript ever
//! produced" and "tokens actually in context" is explained, not hidden.
//!
//! System tools (MCP tool schemas) are not included: enumerating them requires the
//! async app-server round trip `/mcp` already uses (`AppEvent::FetchMcpInventory`),
//! which this synchronous card does not perform. The section says so explicitly
//! rather than omitting it silently.

use ratatui::style::Color;
use ratatui::style::Style;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::text::Span;

use super::ChatWidget;
use crate::app_backtrack::ContextUsageTranscriptTotals;
use crate::history_cell::PlainHistoryCell;
use crate::legacy_core::elpis_context::ContinuitySourceCategory;

const GRID_COLUMNS: usize = 20;
const GRID_ROWS: usize = 10;
const GRID_CELLS: usize = GRID_COLUMNS * GRID_ROWS;

struct CategoryUsage {
    label: &'static str,
    chars: usize,
    color: Color,
}

impl ChatWidget {
    pub(crate) fn add_context_usage_output(&mut self, totals: ContextUsageTranscriptTotals) {
        let sources = self.continuity_sources();
        let instruction_sources: Vec<_> = sources
            .iter()
            .filter(|source| source.category == ContinuitySourceCategory::Instructions)
            .collect();
        let is_skill_path = |path: &std::path::Path| {
            path.components()
                .any(|component| component.as_os_str() == "skills")
        };
        let system_prompt_chars: u64 = instruction_sources
            .iter()
            .filter(|source| !is_skill_path(&source.path))
            .map(|source| source.bytes)
            .sum();
        let skills_chars: u64 = instruction_sources
            .iter()
            .filter(|source| is_skill_path(&source.path))
            .map(|source| source.bytes)
            .sum();

        let categories = vec![
            CategoryUsage {
                label: "User messages",
                chars: totals.user_message_chars,
                color: Color::Blue,
            },
            CategoryUsage {
                label: "Agent responses",
                chars: totals.agent_response_chars,
                color: Color::Green,
            },
            CategoryUsage {
                label: "Tool calls",
                chars: totals.tool_call_chars,
                color: Color::Yellow,
            },
            CategoryUsage {
                label: "System prompt",
                chars: system_prompt_chars as usize,
                color: Color::Magenta,
            },
            CategoryUsage {
                label: "Skills",
                chars: skills_chars as usize,
                color: Color::Cyan,
            },
        ];

        // Same source `/usage` uses for "X% left (Y used / Z window)" — last_token_usage
        // is the current context occupancy; total_token_usage (deliberately not used
        // here) is the cumulative sum across the whole session and can exceed the
        // window entirely.
        let default_usage = crate::token_usage::TokenUsage::default();
        let last_usage = self
            .token_info
            .as_ref()
            .map(|info| &info.last_token_usage)
            .unwrap_or(&default_usage);
        let context_window = self.status_line_context_window_size();
        let used_percent = context_window
            .map(|window| last_usage.percent_of_context_window_remaining(window))
            .map(|remaining| (100 - remaining).clamp(0, 100))
            .unwrap_or(0);
        let used_tokens = last_usage.tokens_in_context_window();
        let cleaner_saved_chars = crate::legacy_core::context_cleaner::saved_chars();
        let pruner_saved_chars = crate::legacy_core::context_pruner::saved_chars();
        let total_saved_tokens = codex_utils_string::approx_tokens_from_byte_count(
            cleaner_saved_chars + pruner_saved_chars,
        );

        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(" Context Usage".bold().into());
        lines.extend(build_grid_lines(&categories, used_percent));
        lines.push(Line::default());
        let window_label = context_window
            .map(|w| format!("{used_tokens} used / {w} window ({used_percent}%)"))
            .unwrap_or_else(|| format!("{used_tokens} used"));
        lines.push(format!(" {window_label}").dim().into());
        if total_saved_tokens > 0 {
            lines.push(
                format!(
                    " Already pruned from the next request: ~{total_saved_tokens} tokens (see /usage)"
                )
                .dim()
                .into(),
            );
        }
        lines.push(Line::default());

        lines.push(" Token usage by category".bold().into());
        lines.push(
            "   (share of this session's visible transcript, not of the context window —"
                .dim()
                .into(),
        );
        lines.push(
            "   most tool output above is already pruned from what's actually sent)"
                .dim()
                .into(),
        );
        let category_total_chars: usize = categories.iter().map(|c| c.chars).sum();
        for category in &categories {
            let tokens = codex_utils_string::approx_tokens_from_byte_count(category.chars);
            let share_percent = if category_total_chars > 0 {
                (category.chars * 100) / category_total_chars
            } else {
                0
            };
            lines.push(Line::from(vec![
                Span::styled("● ", Style::default().fg(category.color)),
                Span::from(format!(
                    "{}: {tokens} tokens ({share_percent}% of transcript)",
                    category.label
                )),
            ]));
        }
        if category_total_chars == 0 {
            lines.push("   (transcript is empty this session)".dim().into());
        }
        lines.push(Line::from(vec![
            Span::from("● ").dim(),
            Span::from("System tools (MCP): not measured here — run /mcp for the registered tool list")
                .dim(),
        ]));
        let free_percent = (100 - used_percent).clamp(0, 100);
        lines.push(Line::from(vec![
            Span::from("□ ").dim(),
            Span::from(format!("Free space: {free_percent}% of window")).dim(),
        ]));
        lines.push(Line::default());

        lines.push(" Checkpoints · Esc Esc to backtrack".bold().into());
        if totals.checkpoints == 0 {
            lines.push("   No backtrack points yet — send a message first.".dim().into());
        } else {
            lines.push(
                format!(
                    "   {} backtrack point(s) available — Esc Esc jumps to a prior message and forks from it.",
                    totals.checkpoints
                )
                .dim()
                .into(),
            );
        }
        lines.push(Line::default());

        lines.push(" System files · auto-loaded".bold().into());
        if instruction_sources.is_empty() {
            lines.push("   (none admitted)".dim().into());
        } else {
            for source in &instruction_sources {
                lines.push(
                    format!(
                        "   {}: {} tokens",
                        source.path.display(),
                        source.estimated_tokens
                    )
                    .dim()
                    .into(),
                );
            }
        }

        self.add_to_history(PlainHistoryCell::new(lines));
    }
}

fn build_grid_lines(categories: &[CategoryUsage], used_percent: i64) -> Vec<Line<'static>> {
    let used_cells = ((used_percent.clamp(0, 100) as usize) * GRID_CELLS) / 100;
    let total_category_chars: usize = categories.iter().map(|c| c.chars).sum();

    let mut cells: Vec<Option<Color>> = Vec::with_capacity(used_cells);
    if total_category_chars > 0 {
        let mut remaining = used_cells;
        for (index, category) in categories.iter().enumerate() {
            let share = if index + 1 == categories.len() {
                remaining
            } else {
                ((category.chars * used_cells) / total_category_chars).min(remaining)
            };
            cells.extend(std::iter::repeat_n(Some(category.color), share));
            remaining -= share;
        }
    }
    cells.resize(used_cells, None);
    cells.resize(GRID_CELLS, None);

    cells
        .chunks(GRID_COLUMNS)
        .map(|row| {
            Line::from(
                row.iter()
                    .map(|slot| match slot {
                        Some(color) => Span::styled("● ", Style::default().fg(*color)),
                        None => Span::from("□ ").dim(),
                    })
                    .collect::<Vec<Span<'static>>>(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_grid_lines_fills_only_used_share() {
        let categories = vec![CategoryUsage {
            label: "User messages",
            chars: 100,
            color: Color::Blue,
        }];

        let lines = build_grid_lines(&categories, /*used_percent*/ 50);

        assert_eq!(lines.len(), GRID_ROWS);
        let filled: usize = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| span.content.contains('●'))
            .count();
        assert_eq!(filled, GRID_CELLS / 2);
    }

    #[test]
    fn build_grid_lines_never_exceeds_window_even_with_huge_transcript() {
        // Regression: category chars must never inflate the grid's used-cell count
        // beyond used_percent. Only the color split within that budget depends on
        // category totals, never the total fill.
        let categories = vec![CategoryUsage {
            label: "Tool calls",
            chars: 10_000_000,
            color: Color::Yellow,
        }];

        let lines = build_grid_lines(&categories, /*used_percent*/ 12);

        let filled: usize = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| span.content.contains('●'))
            .count();
        assert_eq!(filled, (12 * GRID_CELLS) / 100);
    }
}
