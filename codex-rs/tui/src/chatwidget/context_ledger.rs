//! Persistent, user-controlled view of Elpis-owned portable context.

use super::*;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;

const LEDGER_MIN_TERMINAL_WIDTH: u16 = 100;
const LEDGER_WIDTH: u16 = 52;
/// Rows the widget claims whenever the ledger is showing, so it reads as the tall,
/// always-visible sidebar of the design prototype instead of being clipped to the
/// bottom pane's normal few-row height. Elpis has no alt-screen mode (chat history is
/// terminal scrollback, so `ChatWidget` only ever renders the bottom viewport) — this
/// is the compromise until that changes.
pub(super) const LEDGER_MIN_HEIGHT: u16 = 38;

#[derive(Default)]
pub(super) struct ContextLedgerState {
    focused: bool,
    selected: usize,
    pending_g: bool,
    why_visible: bool,
}

impl ChatWidget {
    /// The ledger is always visible as a sidebar once the terminal is wide enough —
    /// it's meant to be transparent, not something you have to remember to open.
    /// Tab still focuses it for keyboard control (select/toggle/why).
    pub(super) fn context_ledger_width(&self, terminal_width: u16) -> u16 {
        (terminal_width >= LEDGER_MIN_TERMINAL_WIDTH)
            .then_some(LEDGER_WIDTH)
            .unwrap_or(0)
    }

    pub(super) fn handle_context_ledger_key_event(&mut self, key_event: KeyEvent) -> bool {
        if key_event.kind != KeyEventKind::Press {
            return false;
        }
        if matches!(key_event.code, KeyCode::Tab)
            && key_event.modifiers.is_empty()
            && self.bottom_pane.composer_is_empty()
            && self.bottom_pane.no_modal_or_popup_active()
            && self
                .last_rendered_width
                .get()
                .is_some_and(|width| width >= LEDGER_MIN_TERMINAL_WIDTH as usize)
        {
            self.context_ledger.focused = !self.context_ledger.focused;
            self.context_ledger.pending_g = false;
            self.request_redraw();
            return true;
        }
        if !self.context_ledger.focused {
            return false;
        }

        let sources = self.continuity_sources();
        let selectable = sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| source.selectable.then_some(index))
            .collect::<Vec<_>>();
        if selectable.is_empty() {
            if matches!(key_event.code, KeyCode::Esc) {
                self.context_ledger.focused = false;
                self.request_redraw();
                return true;
            }
            return false;
        }
        if !selectable.contains(&self.context_ledger.selected) {
            self.context_ledger.selected = selectable[0];
        }

        if self.context_ledger.pending_g {
            self.context_ledger.pending_g = false;
            match key_event.code {
                KeyCode::Char('i') => {
                    self.set_all_context_sources_admitted(&sources, true);
                    self.request_redraw();
                    return true;
                }
                KeyCode::Char('e') => {
                    self.set_all_context_sources_admitted(&sources, false);
                    self.request_redraw();
                    return true;
                }
                _ => {}
            }
        }

        match key_event.code {
            KeyCode::Esc => {
                self.context_ledger.focused = false;
            }
            KeyCode::Char('g') => {
                self.context_ledger.pending_g = true;
            }
            KeyCode::Char('w') => {
                self.context_ledger.why_visible = !self.context_ledger.why_visible;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_context_ledger_selection(&selectable, -1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_context_ledger_selection(&selectable, 1);
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                let source = &sources[self.context_ledger.selected];
                self.set_context_source_admitted(source, !source.admitted);
            }
            _ => return false,
        }
        self.request_redraw();
        true
    }

    pub(super) fn render_context_ledger(&self, area: Rect, buf: &mut Buffer) {
        // Content width inside the left border, used to right-align each row's
        // tokens/state against the same edge instead of stacking them on their own line.
        let content_width = area.width.saturating_sub(1).max(1) as usize;
        let sources = self.continuity_sources();
        let total_tokens = sources
            .iter()
            .filter(|source| source.admitted)
            .map(|source| source.estimated_tokens)
            .sum::<u64>();
        let cyan = Style::default().fg(Color::Cyan);
        let muted = Style::default().dim();
        let mut lines = vec![Line::from(vec![
            Span::styled("CONTEXT LEDGER", cyan.bold()),
            Span::raw("  "),
            Span::styled(
                format!("Total ≈{} tokens admitted", format_tokens(total_tokens)),
                cyan,
            ),
        ])];
        if self.context_ledger.focused {
            let sequence_hint = if self.context_ledger.pending_g {
                "g … then i include all / e exclude all"
            } else {
                "Space/Enter toggle · g i all in · g e all out"
            };
            lines.push(Line::from(sequence_hint.dim()));
            lines.push(Line::from("w why · ↑↓ select · Tab close".dim()));
            lines.push(Line::from("use /add to add a file or directory".dim()));
        }
        lines.push(Line::from(""));

        if sources.is_empty() {
            lines.push(Line::from("No portable context is available.".dim()));
        }
        let mut source_line_ranges = vec![0..0; sources.len()];
        for category in crate::legacy_core::elpis_context::ContinuitySourceCategory::ALL {
            let category_sources = sources
                .iter()
                .enumerate()
                .filter(|(_, source)| source.category == category)
                .collect::<Vec<_>>();
            let admitted_tokens = category_sources
                .iter()
                .filter(|(_, source)| source.admitted)
                .map(|(_, source)| source.estimated_tokens)
                .sum::<u64>();
            lines.push(Line::from(vec![
                Span::styled(category.display_name(), cyan.bold()),
                Span::raw("  "),
                Span::styled(
                    format!("≈{} tokens admitted", format_tokens(admitted_tokens)),
                    muted,
                ),
            ]));
            if category_sources.is_empty() {
                lines.push(Line::from("  No sources available.".dim()));
            }
            for (index, source) in category_sources {
                let source_line_start = lines.len();
                let selected = self.context_ledger.focused && index == self.context_ledger.selected;
                let marker = if source.selectable {
                    if source.admitted { "[x]" } else { "[ ]" }
                } else {
                    "[-]"
                };
                let state = if source.admitted {
                    "INCLUDED"
                } else {
                    "EXCLUDED"
                };
                let marker_style = if source.admitted { cyan } else { muted };
                let prefix = if selected { "› " } else { "  " };
                let left = format!("{prefix}{marker} {}", source.name);
                let right = format!("≈{} {state}", format_tokens(source.estimated_tokens));
                let pad = content_width
                    .saturating_sub(left.chars().count())
                    .saturating_sub(right.chars().count())
                    .max(1);
                lines.push(Line::from(vec![
                    Span::styled(prefix, cyan),
                    Span::styled(marker, marker_style),
                    Span::raw(" "),
                    Span::styled(
                        source.name.as_str(),
                        if selected {
                            cyan.bold()
                        } else {
                            Style::default()
                        },
                    ),
                    Span::raw(" ".repeat(pad)),
                    Span::styled(right, marker_style),
                ]));

                if selected && self.context_ledger.why_visible {
                    let inclusion = if source.admitted {
                        "Included"
                    } else {
                        "Excluded; when enabled, included"
                    };
                    lines.push(Line::from(Span::styled("WHY INCLUDED", cyan.bold())));
                    lines.push(Line::from(Span::styled(source.name.clone(), cyan)));
                    lines.push(Line::from(
                        format!("{inclusion} because {}.", source.reason).dim(),
                    ));
                    lines.push(Line::from(
                        format!(
                            "Lifetime: {} · Source: {}",
                            source.lifetime,
                            source.path.display()
                        )
                        .dim(),
                    ));
                    lines.push(Line::from(""));
                }

                source_line_ranges[index] = source_line_start..lines.len();
            }
            lines.push(Line::from(""));
        }
        if self.context_ledger.focused {
            lines.push(Line::from("Esc returns to the command composer.".dim()));
        }

        let scroll_lines = self
            .context_ledger
            .focused
            .then(|| {
                source_line_ranges
                    .get(self.context_ledger.selected)
                    .map(|range| {
                        selected_source_scroll_offset(
                            &lines,
                            range.clone(),
                            area.width.saturating_sub(1).max(1),
                            area.height.saturating_sub(2).max(1),
                        )
                    })
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::LEFT)
                    .border_style(cyan)
                    .title(" CONTEXT "),
            )
            .wrap(Wrap { trim: true })
            .scroll((scroll_lines, 0))
            .render(area, buf);
    }

    fn continuity_sources(&self) -> Vec<crate::legacy_core::elpis_context::ContinuitySource> {
        crate::legacy_core::elpis_context::continuity_sources(
            self.config
                .memories
                .root
                .as_ref()
                .map(|root| root.as_path()),
            self.config.cwd.as_path(),
            &self.instruction_source_paths_as_path_bufs(),
        )
    }

    /// The server-reported instruction sources, converted for `elpis_context` — the
    /// same list `/status` renders, so the ledger cannot disagree with it.
    pub(super) fn instruction_source_paths_as_path_bufs(&self) -> Vec<std::path::PathBuf> {
        self.instruction_source_paths
            .iter()
            .filter_map(|uri| uri.to_abs_path().ok())
            .map(|path| path.as_path().to_path_buf())
            .collect()
    }

    fn move_context_ledger_selection(&mut self, selectable: &[usize], delta: isize) {
        let current = selectable
            .iter()
            .position(|index| *index == self.context_ledger.selected)
            .unwrap_or(0) as isize;
        let next = (current + delta).rem_euclid(selectable.len() as isize) as usize;
        self.context_ledger.selected = selectable[next];
    }

    fn set_all_context_sources_admitted(
        &mut self,
        sources: &[crate::legacy_core::elpis_context::ContinuitySource],
        admitted: bool,
    ) {
        for source in sources.iter().filter(|source| source.selectable) {
            if !self.set_context_source_admitted(source, admitted) {
                break;
            }
        }
    }

    fn set_context_source_admitted(
        &mut self,
        source: &crate::legacy_core::elpis_context::ContinuitySource,
        admitted: bool,
    ) -> bool {
        match crate::legacy_core::elpis_context::set_continuity_source_admitted(
            self.config
                .memories
                .root
                .as_ref()
                .map(|root| root.as_path()),
            self.config.cwd.as_path(),
            &source.name,
            admitted,
        ) {
            Ok(()) => true,
            Err(error) => {
                self.add_error_message(format!("Could not update context admission: {error}"));
                false
            }
        }
    }
}

fn format_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    }
}

fn selected_source_scroll_offset(
    lines: &[Line<'_>],
    source_range: std::ops::Range<usize>,
    width: u16,
    visible_rows: u16,
) -> u16 {
    let start = source_range.start.min(lines.len());
    let end = source_range.end.max(start).min(lines.len());
    let selected_start = wrapped_line_count(&lines[..start], width);
    let selected_end = wrapped_line_count(&lines[..end], width);
    selected_end
        .saturating_sub(visible_rows.max(1))
        .min(selected_start)
}

fn wrapped_line_count(lines: &[Line<'_>], width: u16) -> u16 {
    u16::try_from(
        Paragraph::new(lines.to_vec())
            .wrap(Wrap { trim: true })
            .line_count(width.max(1)),
    )
    .unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_source_scrolls_into_a_short_ledger() {
        let lines = (0..8)
            .map(|index| Line::from(format!("line {index}")))
            .collect::<Vec<_>>();
        assert_eq!(selected_source_scroll_offset(&lines, 1..3, 52, 4), 0);
        assert_eq!(selected_source_scroll_offset(&lines, 6..8, 52, 4), 4);
    }

    #[test]
    fn ledger_scroll_accounts_for_wrapped_grouped_lines() {
        let lines = vec![
            Line::from("CONTEXT LEDGER"),
            Line::from("FILES"),
            Line::from("[x] short.rs"),
            Line::from("tokens"),
            Line::from("INSTRUCTIONS"),
            Line::from("[x] a source name that wraps on a narrow ledger"),
            Line::from("tokens"),
        ];
        let wide = selected_source_scroll_offset(&lines, 5..7, 52, 4);
        let narrow = selected_source_scroll_offset(&lines, 5..7, 12, 4);
        assert!(narrow > wide);
        assert_eq!(selected_source_scroll_offset(&[], 0..4, 52, 4), 0);
    }
}
