use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
use std::path::PathBuf;

const MAX_GOAL_CHARS: usize = 6_000;
const MAX_CHECKPOINT_CHARS: usize = 8_000;

pub fn workspace_context_dir(memories_root: Option<&Path>, cwd: &Path) -> Option<PathBuf> {
    let elpis_home = memories_root?.parent()?;
    Some(
        elpis_home
            .join("context")
            .join("workspaces")
            .join(workspace_key(cwd)),
    )
}

pub async fn build_continuity_prompt(memories_root: Option<&Path>, cwd: &Path) -> Option<String> {
    let workspace_dir = workspace_context_dir(memories_root, cwd)?;
    let mut sections = Vec::new();
    for (name, limit) in [("GOAL.md", MAX_GOAL_CHARS), ("ES.md", MAX_CHECKPOINT_CHARS)] {
        let path = workspace_dir.join(name);
        let Ok(content) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        let content = truncate_chars(content.trim(), limit);
        if !content.is_empty() {
            sections.push(format!(
                "### Source: {} ({} characters)\n\n{}",
                path.display(),
                content.chars().count(),
                content
            ));
        }
    }
    if sections.is_empty() {
        return None;
    }
    Some(format!(
        "## Elpis Continuity\n\n\
         Use these small, user-owned files to continue the current workspace. They are not a full\n\
         transcript. Verify mutable repository state before acting, and prefer the current user\n\
         message when it changes the task.\n\n{}",
        sections.join("\n\n")
    ))
}

fn workspace_key(cwd: &Path) -> String {
    let slug = cwd
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("workspace")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .take(40)
        .collect::<String>();
    let slug = if slug.is_empty() { "workspace" } else { &slug };
    let digest = Sha256::digest(cwd.to_string_lossy().as_bytes());
    let suffix = digest[..6]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{slug}-{suffix}")
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn workspace_path_is_stable_readable_and_path_specific() {
        let memories = Path::new("/tmp/home/.elpis/memories");
        let first = workspace_context_dir(Some(memories), Path::new("/tmp/My Project"))
            .expect("workspace path");
        assert_eq!(
            first,
            workspace_context_dir(Some(memories), Path::new("/tmp/My Project"))
                .expect("workspace path")
        );
        assert!(
            first
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("My-Project-"))
        );
        assert_ne!(
            workspace_context_dir(Some(memories), Path::new("/a/project")),
            workspace_context_dir(Some(memories), Path::new("/b/project"))
        );
    }

    #[tokio::test]
    async fn prompt_loads_only_portable_goal_and_checkpoint() -> anyhow::Result<()> {
        let home = tempdir()?;
        let memories = home.path().join(".elpis/memories");
        let cwd = Path::new("/tmp/project");
        let workspace = workspace_context_dir(Some(&memories), cwd).expect("workspace path");
        tokio::fs::create_dir_all(&workspace).await?;
        tokio::fs::write(workspace.join("GOAL.md"), "Ship Elpis").await?;
        tokio::fs::write(workspace.join("ES.md"), "Next: visible context").await?;
        tokio::fs::write(workspace.join("raw.log"), "must not load").await?;

        let prompt = build_continuity_prompt(Some(&memories), cwd)
            .await
            .expect("continuity prompt");
        assert!(prompt.contains("Ship Elpis"));
        assert!(prompt.contains("Next: visible context"));
        assert!(!prompt.contains("must not load"));
        Ok(())
    }
}
