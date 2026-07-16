#!/usr/bin/env python3
from pathlib import Path


def replace_exact(path: str, old: str, new: str, expected: int = 1) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} matches, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new))


# Give /model an Elpis-owned, provider-aware first layer while retaining the inherited catalog.
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    "const ULTRA_REASONING_CONCURRENCY_WARNING_THRESHOLD: usize = 8;\n",
    '''const ULTRA_REASONING_CONCURRENCY_WARNING_THRESHOLD: usize = 8;

const OPENROUTER_MODELS: &[(&str, &str, &str)] = &[
    (
        "~anthropic/claude-sonnet-latest",
        "Claude Sonnet Latest",
        "Anthropic's current Sonnet family through OpenRouter",
    ),
    (
        "~google/gemini-pro-latest",
        "Gemini Pro Latest",
        "Google's current Gemini Pro family through OpenRouter",
    ),
    (
        "~google/gemini-flash-latest",
        "Gemini Flash Latest",
        "Google's current Gemini Flash family through OpenRouter",
    ),
    (
        "~openai/gpt-latest",
        "OpenAI GPT Latest",
        "OpenAI's current GPT family through OpenRouter",
    ),
];
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    "        self.open_model_popup_with_presets(presets);\n",
    "        let presets = self.with_elpis_provider_models(presets);\n        self.open_model_popup_with_presets(presets);\n",
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '''    fn custom_openai_base_url(&self) -> Option<String> {
        if !self.config.model_provider.is_openai() {
            return None;
        }

        let base_url = self.config.model_provider.base_url.as_ref()?;
        let trimmed = base_url.trim();
        if trimmed.is_empty() {
            return None;
        }

        let normalized = trimmed.trim_end_matches('/');
        if normalized == DEFAULT_OPENAI_BASE_URL {
            return None;
        }

        Some(trimmed.to_string())
    }
''',
    '''    fn custom_openai_base_url(&self) -> Option<String> {
        if !self.config.model_provider.is_openai() {
            return None;
        }

        let base_url = self.config.model_provider.base_url.as_ref()?;
        let trimmed = base_url.trim();
        if trimmed.is_empty() {
            return None;
        }

        let normalized = trimmed.trim_end_matches('/');
        if normalized == DEFAULT_OPENAI_BASE_URL {
            return None;
        }

        Some(trimmed.to_string())
    }

    fn is_openrouter_provider(&self) -> bool {
        self.config
            .model_provider
            .name
            .eq_ignore_ascii_case("openrouter")
    }

    fn with_elpis_provider_models(&self, presets: Vec<ModelPreset>) -> Vec<ModelPreset> {
        if !self.is_openrouter_provider() {
            return presets;
        }

        let Some(template) = presets.iter().find(|preset| preset.show_in_picker).cloned() else {
            return presets;
        };

        let mut models = OPENROUTER_MODELS
            .iter()
            .map(|(slug, display_name, description)| {
                let mut preset = template.clone();
                preset.id = (*slug).to_string();
                preset.model = (*slug).to_string();
                preset.display_name = (*display_name).to_string();
                preset.description = (*description).to_string();
                preset.additional_speed_tiers.clear();
                preset.service_tiers.clear();
                preset.default_service_tier = None;
                preset.is_default = false;
                preset.upgrade = None;
                preset.availability_nux = None;
                preset.supported_in_api = true;
                preset
            })
            .collect::<Vec<_>>();
        models.extend(presets.into_iter().filter(|preset| {
            !OPENROUTER_MODELS
                .iter()
                .any(|(slug, _, _)| preset.model == *slug)
        }));
        models
    }
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '''                SelectionItem {
                    name: model.clone(),
''',
    '''                SelectionItem {
                    name: preset.display_name.clone(),
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '''        let header = self.model_menu_header(
            "Select Model",
            "Pick a quick auto mode or browse all models.",
        );
''',
    '''        let subtitle = if self.is_openrouter_provider() {
            "Choose a mind. Elpis keeps the context, memory, permissions, and evidence."
        } else {
            "Choose the model for this Elpis runtime."
        };
        let header = self.model_menu_header("Choose a mind", subtitle);
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '''    fn is_auto_model(model: &str) -> bool {
        model.starts_with("codex-auto-")
    }
''',
    '''    fn is_auto_model(model: &str) -> bool {
        model.starts_with("codex-auto-")
            || OPENROUTER_MODELS
                .iter()
                .any(|(slug, _, _)| model == *slug)
    }
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '''            "codex-auto-thorough" => 2,
            _ => 3,
''',
    '''            "codex-auto-thorough" => 2,
            "~anthropic/claude-sonnet-latest" => 10,
            "~google/gemini-pro-latest" => 11,
            "~google/gemini-flash-latest" => 12,
            "~openai/gpt-latest" => 13,
            _ => 20,
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '''        let header = self.model_menu_header(
            "Select Model and Effort",
            "Access legacy models by running codex -m <model_name> or in your config.toml",
        );
''',
    '''        let header = self.model_menu_header(
            "Choose model and reasoning",
            "Use elpis -m <model_name> or config.toml for a model not shown here.",
        );
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/model_popups.rs",
    '            format!("Select Reasoning Level for {model_slug}").bold(),\n',
    '            format!("Reasoning for {model_slug}").bold(),\n',
)

# Remove two unreachable memory-maintenance placeholders, not the real memory implementation.
replace_exact(
    "codex-rs/tui/src/slash_command.rs",
    '''    // Debugging commands.
    #[strum(serialize = "debug-m-drop")]
    MemoryDrop,
    #[strum(serialize = "debug-m-update")]
    MemoryUpdate,
''',
    "",
)
replace_exact(
    "codex-rs/tui/src/slash_command.rs",
    '''            SlashCommand::MemoryDrop => "DO NOT USE",
            SlashCommand::MemoryUpdate => "DO NOT USE",
''',
    "",
)
replace_exact(
    "codex-rs/tui/src/slash_command.rs",
    '''            | SlashCommand::Logout
            | SlashCommand::MemoryDrop
            | SlashCommand::MemoryUpdate => false,
''',
    '''            | SlashCommand::Logout => false,
''',
)
replace_exact(
    "codex-rs/tui/src/slash_command.rs",
    '''            SlashCommand::Archive
            | SlashCommand::Memories
            | SlashCommand::MemoryDrop
            | SlashCommand::MemoryUpdate
            | SlashCommand::Mention
''',
    '''            SlashCommand::Archive
            | SlashCommand::Memories
            | SlashCommand::Mention
''',
)
replace_exact(
    "codex-rs/tui/src/slash_command.rs",
    '''    #[test]
    fn certain_commands_are_available_during_task() {
''',
    '''    #[test]
    fn removed_memory_debug_commands_do_not_parse() {
        assert!(SlashCommand::from_str("debug-m-drop").is_err());
        assert!(SlashCommand::from_str("debug-m-update").is_err());
    }

    #[test]
    fn certain_commands_are_available_during_task() {
''',
)
replace_exact(
    "codex-rs/tui/src/chatwidget/slash_dispatch.rs",
    '''            SlashCommand::MemoryDrop => {
                self.add_app_server_stub_message("Memory maintenance");
            }
            SlashCommand::MemoryUpdate => {
                self.add_app_server_stub_message("Memory maintenance");
            }
''',
    "",
)

# Remove inherited product wording where Elpis owns the visible surface.
for old, new in [
    ("instructions for Codex", "instructions for Elpis agents"),
    ("continue this session in Codex Desktop", "continue this session in the desktop client"),
    ("how Codex performs specific tasks", "how Elpis agents perform specific tasks"),
    ("communication style for Codex", "communication style for the active agent"),
    ("what Codex is allowed to do", "what the active agent is allowed to do"),
    ("log out of Codex", "log out of the active account"),
]:
    replace_exact("codex-rs/tui/src/slash_command.rs", old, new)
replace_exact(
    "codex-rs/tui/src/chatwidget/slash_dispatch.rs",
    "archive the current session and exit Codex",
    "archive the current session and exit Elpis",
)
replace_exact(
    "codex-rs/models-manager/prompt.md",
    "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.",
    "You are a coding agent running inside Elpis, a terminal environment that preserves the user's context, memory, permissions, and evidence across supported model runtimes. Elpis contains a modified Codex-derived execution foundation. You are expected to be precise, safe, truthful, and helpful.",
)
replace_exact(
    "codex-rs/models-manager/prompt.md",
    "Within this context, Codex refers to the open-source agentic coding interface (not the old Codex language model built by OpenAI).",
    "Within this context, Elpis is the surrounding agent environment; Codex refers only to the contained open-source execution foundation (not the old Codex language model).",
)
replace_exact(
    "codex-rs/models-manager/src/model_info.rs",
    "You are Codex, a coding agent based on GPT-5. You and the user share the same workspace and collaborate to achieve the user's goals.",
    "You are an agent running inside Elpis. You and the user share the same workspace and collaborate to achieve the user's goals while preserving visible context, permissions, and evidence.",
)

# Keep the release workflow aligned with the expanded binary tests.
replace_exact(
    ".github/workflows/embedded-elpis-linux.yml",
    "run: cargo test -p codex-tui --bin elpis --locked provider_flag_is_limited_and_becomes_a_config_override",
    "run: cargo test -p codex-tui --bin elpis --locked",
)
