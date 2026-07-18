//! Persistent, user-controlled view of Elpis-owned portable context.

use super::*;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;

const LEDGER_MIN_TERMINAL_WIDTH: u16 = 100;
const LEDGER_WIDTH: u16 = 52;

#[derive(Default)]
pub(super) struct ContextLedgerState {
    focused: bool,
    selected: usize,
}

impl ChatWidget {
    pub(super) fn context_ledger_width(&self, terminal_width: u16) -> u16 {
        (terminal_width >= LEDGER_MIN_TERMINAL_WIDTH)
            .then_some(LEDGER_WIDTH)
            .unwrap_or(0)
    }

    pub(super) fn handle_context_ledger_key_event(&mut self, key_event: KeyEvent) -> bool {
        if key_event.kind != KeyEventKind::Press {
            return false;
        }
        if matches!(key_event.code, KeyCode::BackTab)
            && self
                .last_rendered_width
                .get()
                .is_some_and(|width| width >= LEDGER_MIN_TERMINAL_WIDTH as usize)
        {
            self.context_ledger.focused = !self.context_ledger.focused;
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

        match key_event.code {
            KeyCode::Esc => {
                self.context_ledger.focused = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_context_ledger_selection(&selectable, -1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_context_ledger_selection(&selectable, 1);
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                let source = &sources[self.context_ledger.selected];
                if let Err(error) =
                    crate::legacy_core::elpis_context::set_continuity_source_admitted(
                        self.config
                            .memories
                            .root
                            .as_ref()
                            .map(|root| root.as_path()),
                        self.config.cwd.as_path(),
                        &source.name,
                        !source.admitted,
                    )
                {
                    self.add_error_message(format!("Could not update context admission: {error}"));
                }
            }
            _ => return false,
        }
        self.request_redraw();
        true
    }

    pub(super) fn render_context_ledger(&self, area: Rect, buf: &mut Buffer) {
        let sources = self.continuity_sources();
        let total_bytes = sources
            .iter()
            .filter(|source| source.admitted)
            .map(|source| source.bytes)
            .sum::<u64>();
        let cyan = Style::default().fg(Color::Cyan);
        let muted = Style::default().dim();
        let mut lines = vec![Line::from(vec![
            Span::styled("CONTEXT LEDGER", cyan.bold()),
            Span::raw("  "),
            Span::styled(format!("{} admitted", format_bytes(total_bytes)), cyan),
        ])];
        if self.context_ledger.focused {
            let selected_source = sources.get(self.context_ledger.selected);
            let action = selected_source.map_or("enable", |source| {
                if source.admitted { "disable" } else { "enable" }
            });
            lines.push(Line::from(
                format!("Space {action} · ↑↓ select · Shift+Tab close").dim(),
            ));
        }
        lines.push(Line::from(""));

        if sources.is_empty() {
            lines.push(Line::from("No portable context is available.".dim()));
        }
        for (index, source) in sources.iter().enumerate() {
            let selected = self.context_ledger.focused && index == self.context_ledger.selected;
            let marker = if source.selectable {
                if source.admitted { "[x]" } else { "[ ]" }
            } else {
                "[-]"
            };
            let state = if source.admitted {
                "ENABLED"
            } else {
                "DISABLED"
            };
            let marker_style = if source.admitted {
                cyan
            } else {
                Style::default().fg(Color::Yellow)
            };
            let prefix = if selected { "› " } else { "  " };
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
            ]));
            lines.push(Line::from(vec![
                Span::raw("      "),
                Span::styled(
                    format!("{} · {} · ", format_bytes(source.bytes), source.lifetime),
                    muted,
                ),
                Span::styled(state, marker_style),
            ]));
            lines.push(Line::from(vec![
                Span::raw("      "),
                Span::styled(source.reason, muted),
            ]));
            lines.push(Line::from(""));
        }
        if self.context_ledger.focused {
            lines.push(Line::from("Esc returns to the command composer.".dim()));
        }

        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::LEFT)
                    .border_style(cyan)
                    .title(" CONTEXT "),
            )
            .wrap(Wrap { trim: true })
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
        )
    }

    fn move_context_ledger_selection(&mut self, selectable: &[usize], delta: isize) {
        let current = selectable
            .iter()
            .position(|index| *index == self.context_ledger.selected)
            .unwrap_or(0) as isize;
        let next = (current + delta).rem_euclid(selectable.len() as isize) as usize;
        self.context_ledger.selected = selectable[next];
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1_024 {
        format!("{bytes} B")
    } else {
        format!("{:.1}k", bytes as f64 / 1_024.0)
    }
}
