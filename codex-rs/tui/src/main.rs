use clap::Parser;
use codex_arg0::Arg0DispatchPaths;
use codex_arg0::arg0_dispatch_or_else;
use codex_config::LoaderOverrides;
use codex_tui::AppExitInfo;
use codex_tui::Cli;
use codex_tui::ExitReason;
use codex_tui::run_main;
use codex_utils_cli::CliConfigOverrides;
use std::io::Write;
use std::path::PathBuf;
use supports_color::Stream;

const OPENROUTER_CLAUDE_MODEL: &str = "~anthropic/claude-sonnet-latest";
const OPENROUTER_GEMINI_MODEL: &str = "~google/gemini-pro-latest";
const OPENROUTER_GEMINI_FLASH_MODEL: &str = "~google/gemini-flash-latest";

fn format_exit_messages(exit_info: AppExitInfo, color_enabled: bool) -> Vec<String> {
    let is_fatal = matches!(&exit_info.exit_reason, ExitReason::Fatal(_));
    let AppExitInfo {
        token_usage,
        thread_id,
        resume_hint,
        ..
    } = exit_info;

    let mut lines = Vec::new();
    if !token_usage.is_zero() {
        lines.push(token_usage.to_string());
    }

    if let Some(resume_cmd) = resume_hint {
        let command = if color_enabled {
            format!("\u{1b}[36m{resume_cmd}\u{1b}[39m")
        } else {
            resume_cmd
        };
        lines.push(format!("To continue this session, run {command}"));
    } else if is_fatal && let Some(thread_id) = thread_id {
        lines.push(format!("Session ID: {thread_id}"));
    }

    lines
}

#[derive(Parser, Debug)]
#[command(name = "elpis")]
struct TopCli {
    /// Select an Elpis provider or a curated OpenRouter model family without editing configuration.
    #[arg(
        long,
        value_parser = [
            "openai",
            "openrouter",
            "claude",
            "gemini",
            "gemini-flash",
            "amazon-bedrock",
            "ollama",
            "lmstudio",
        ]
    )]
    provider: Option<String>,

    #[clap(flatten)]
    config_overrides: CliConfigOverrides,

    #[clap(flatten)]
    inner: Cli,
}

fn prepend_elpis_memories_defaults(
    config_overrides: &mut CliConfigOverrides,
    home_dir: Option<PathBuf>,
) {
    let Some(home_dir) = home_dir else {
        return;
    };
    let memories_root = home_dir.join(".elpis/memories");
    let state_root = home_dir.join(".elpis/state");
    let memories_value = toml::Value::String(memories_root.to_string_lossy().into_owned());
    let state_value = toml::Value::String(state_root.to_string_lossy().into_owned());
    config_overrides.raw_overrides.splice(
        0..0,
        [
            format!("memories.root={memories_value}"),
            format!("memories.state_root={state_value}"),
        ],
    );
}

fn push_string_override(config_overrides: &mut CliConfigOverrides, key: &str, value: &str) {
    let value = toml::Value::String(value.to_string());
    config_overrides
        .raw_overrides
        .push(format!("{key}={value}"));
}

fn append_provider_override(config_overrides: &mut CliConfigOverrides, provider: Option<&str>) {
    let Some(provider) = provider else {
        return;
    };

    match provider {
        "claude" => {
            push_string_override(config_overrides, "model_provider", "openrouter");
            push_string_override(config_overrides, "model", OPENROUTER_CLAUDE_MODEL);
        }
        "gemini" => {
            push_string_override(config_overrides, "model_provider", "openrouter");
            push_string_override(config_overrides, "model", OPENROUTER_GEMINI_MODEL);
        }
        "gemini-flash" => {
            push_string_override(config_overrides, "model_provider", "openrouter");
            push_string_override(
                config_overrides,
                "model",
                OPENROUTER_GEMINI_FLASH_MODEL,
            );
        }
        provider => push_string_override(config_overrides, "model_provider", provider),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elpis_memories_defaults_precede_user_config() {
        let mut overrides = CliConfigOverrides {
            raw_overrides: vec![
                "memories.root=\"/tmp/custom-memories\"".to_string(),
                "memories.state_root=\"/tmp/custom-state\"".to_string(),
            ],
        };

        prepend_elpis_memories_defaults(&mut overrides, Some(PathBuf::from("/tmp/home")));

        assert_eq!(
            overrides.raw_overrides,
            vec![
                "memories.root=\"/tmp/home/.elpis/memories\"",
                "memories.state_root=\"/tmp/home/.elpis/state\"",
                "memories.root=\"/tmp/custom-memories\"",
                "memories.state_root=\"/tmp/custom-state\"",
            ]
        );
    }

    #[test]
    fn provider_flag_becomes_a_config_override() {
        let parsed = TopCli::try_parse_from(["elpis", "--provider", "openrouter"])
            .expect("OpenRouter provider flag");
        let mut overrides = parsed.config_overrides;
        append_provider_override(&mut overrides, parsed.provider.as_deref());
        assert_eq!(
            overrides.raw_overrides,
            vec!["model_provider=\"openrouter\"".to_string()]
        );
    }

    #[test]
    fn model_family_aliases_select_openrouter_and_a_model() {
        for (provider, model) in [
            ("claude", OPENROUTER_CLAUDE_MODEL),
            ("gemini", OPENROUTER_GEMINI_MODEL),
            ("gemini-flash", OPENROUTER_GEMINI_FLASH_MODEL),
        ] {
            let parsed = TopCli::try_parse_from(["elpis", "--provider", provider])
                .expect("curated OpenRouter family flag");
            let mut overrides = parsed.config_overrides;
            append_provider_override(&mut overrides, parsed.provider.as_deref());
            assert_eq!(
                overrides.raw_overrides,
                vec![
                    "model_provider=\"openrouter\"".to_string(),
                    format!("model=\"{model}\""),
                ]
            );
        }
    }

    #[test]
    fn provider_flag_accepts_all_built_in_provider_ids() {
        for provider in [
            "openai",
            "openrouter",
            "amazon-bedrock",
            "ollama",
            "lmstudio",
        ] {
            assert!(TopCli::try_parse_from(["elpis", "--provider", provider]).is_ok());
        }
        assert!(TopCli::try_parse_from(["elpis", "--provider", "unknown"]).is_err());
    }
}

fn main() -> anyhow::Result<()> {
    arg0_dispatch_or_else(|arg0_paths: Arg0DispatchPaths| async move {
        let mut top_cli = TopCli::parse();
        let provider = top_cli.provider.clone();
        append_provider_override(&mut top_cli.config_overrides, provider.as_deref());
        let mut inner = top_cli.inner;
        inner
            .config_overrides
            .raw_overrides
            .splice(0..0, top_cli.config_overrides.raw_overrides);
        prepend_elpis_memories_defaults(&mut inner.config_overrides, dirs::home_dir());
        let exit_info = run_main(
            inner,
            arg0_paths,
            LoaderOverrides::default(),
            /*explicit_remote_endpoint*/ None,
        )
        .await?;
        let is_fatal = match &exit_info.exit_reason {
            ExitReason::Fatal(message) => {
                eprintln!("ERROR: {message}");
                true
            }
            ExitReason::UserRequested => false,
        };

        let color_enabled = supports_color::on(Stream::Stdout).is_some();
        for line in format_exit_messages(exit_info, color_enabled) {
            println!("{line}");
        }
        if is_fatal {
            std::io::stdout().flush()?;
            std::process::exit(1);
        }
        Ok(())
    })
}
