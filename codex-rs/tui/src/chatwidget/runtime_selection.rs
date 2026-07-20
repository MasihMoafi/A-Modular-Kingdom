//! Which agent backend owns the active session.
//!
//! Both `/claude-code` and provider-grouped model-picker selections converge on this
//! runtime switch. Turn submission consults the selected value before choosing the Codex
//! app-server or Claude Code CLI path.

use crate::app_event::RuntimeSelection;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ActiveRuntime {
    #[default]
    Codex,
    ClaudeCode,
}

impl ActiveRuntime {
    pub(crate) fn display_name(self) -> &'static str {
        match self {
            ActiveRuntime::Codex => "Codex",
            ActiveRuntime::ClaudeCode => "Claude Code",
        }
    }
}

impl super::ChatWidget {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn active_runtime(&self) -> ActiveRuntime {
        self.active_runtime
    }

    /// Records the selected runtime and confirms the switch in the transcript.
    pub(crate) fn switch_active_runtime(&mut self, runtime: ActiveRuntime) {
        self.active_runtime = runtime;
        let hint = match runtime {
            ActiveRuntime::Codex => "Turns run through the configured Codex provider.",
            ActiveRuntime::ClaudeCode => {
                "Turns now run through the Claude Code CLI (text in/out; no streaming, \
                 tool, or approval bridging yet)."
            }
        };
        self.add_info_message(
            format!("Active runtime switched to {}.", runtime.display_name()),
            Some(hint.to_string()),
        );
        self.sync_branding_for_active_runtime();
    }

    pub(crate) fn switch_active_runtime_selection(&mut self, selection: RuntimeSelection) {
        let runtime = match selection {
            RuntimeSelection::Codex => ActiveRuntime::Codex,
        };
        if self.active_runtime != runtime {
            self.switch_active_runtime(runtime);
        }
    }

    /// Selects the `--model` alias Claude Code turns use (`None` = account default) and
    /// makes sure Claude Code is the active runtime.
    pub(crate) fn select_claude_code_model(&mut self, model: Option<String>) {
        self.claude_model = model;
        if self.active_runtime == ActiveRuntime::ClaudeCode {
            self.sync_branding_for_active_runtime();
        } else {
            self.switch_active_runtime(ActiveRuntime::ClaudeCode);
        }
    }

    /// The status line ("ELPIS · provider · model · …") is normally kept in sync by
    /// Codex app-server notifications (`protocol.rs`). Those notifications keep arriving
    /// even while Claude Code is the active runtime and would otherwise stomp the
    /// display back to the Codex model, so this is the one place that sets it directly
    /// for both runtimes.
    fn sync_branding_for_active_runtime(&mut self) {
        match self.active_runtime {
            ActiveRuntime::ClaudeCode => {
                let model = self.claude_model.as_deref().unwrap_or("account default");
                crate::branding::record_provider_switch("claude-code", model);
            }
            ActiveRuntime::Codex => {
                let provider = self.model_provider_display_name();
                let model = self.current_model().to_string();
                crate::branding::record_provider_switch(&provider, &model);
            }
        }
        self.refresh_status_line();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_is_codex() {
        assert_eq!(ActiveRuntime::default(), ActiveRuntime::Codex);
    }

    #[test]
    fn display_names_are_stable() {
        assert_eq!(ActiveRuntime::Codex.display_name(), "Codex");
        assert_eq!(ActiveRuntime::ClaudeCode.display_name(), "Claude Code");
    }
}
