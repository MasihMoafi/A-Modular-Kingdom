//! Which agent backend owns the active session.
//!
//! This is the first slice of R11 (`REQUIREMENTS.md`): today this enum only records the
//! user's selection and drives `/claude-code`'s confirmation message. It does NOT yet
//! change how a turn is executed — submitting a message still always goes through
//! `codex_app_server_client::AppServerClient` regardless of this value. Routing an actual
//! turn through `codex-claude-bridge` when `ActiveRuntime::ClaudeCode` is selected is the
//! next, larger step (touches turn submission and the Codex event-bridging path), not
//! done here.

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
