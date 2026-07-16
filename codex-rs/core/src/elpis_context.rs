use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
use std::path::PathBuf;

const MAX_GOAL_CHARS: usize = 6_000;
const MAX_CHECKPOINT_CHARS: usize = 8_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuitySource {
    pub name: &'static str,
    pub path: PathBuf,
    pub bytes: u64,
}

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

pub fn continuity_sources(memories_root: Option<&Path>, cwd: &Path) -> Vec<ContinuitySource> {
    let Some(workspace_dir) = workspace_context_dir(memories_root, cwd) else {
        return Vec::new();
    };
    ["GOAL.md", "ES.md"]
        .into_iter()
        .filter_map(|name| {
            let path = workspace_dir.join(name);
            let metadata = std::fs::metadata(&path).ok()?;
            (metadata.is_file() && metadata.len() > 0).then_some(ContinuitySource {
                name,
                path,
                bytes: metadata.len(),
            })
        })
        .collect()
}

pub async fn sync_continuity_before_compaction(
    memories_root: Option<&Path>,
    cwd: &Path,
) -> std::io::Result<()> {
    let Some(workspace_dir) = workspace_context_dir(memories_root, cwd) else {
        return Ok(());
    };
    for name in ["GOAL.md", "ES.md"] {
        let path = workspace_dir.join(name);
        match tokio::fs::OpenOptions::new().read(true).open(&path).await {
            Ok(file) => file.sync_all().await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
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

    #[test]
    fn source_list_contains_only_nonempty_portable_context_files() -> anyhow::Result<()> {
        let home = tempdir()?;
        let memories = home.path().join(".elpis/memories");
        let cwd = Path::new("/tmp/project");
        let workspace = workspace_context_dir(Some(&memories), cwd).expect("workspace path");
        std::fs::create_dir_all(&workspace)?;
        std::fs::write(workspace.join("GOAL.md"), "Ship Elpis")?;
        std::fs::write(workspace.join("ES.md"), "")?;
        std::fs::write(workspace.join("raw.log"), "hidden")?;

        let sources = continuity_sources(Some(&memories), cwd);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].name, "GOAL.md");
        assert_eq!(sources[0].bytes, 10);
        Ok(())
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

    #[tokio::test]
    async fn pre_compaction_sync_accepts_present_or_missing_files() -> anyhow::Result<()> {
        let home = tempdir()?;
        let memories = home.path().join(".elpis/memories");
        let cwd = Path::new("/tmp/project");
        let workspace = workspace_context_dir(Some(&memories), cwd).expect("workspace path");
        tokio::fs::create_dir_all(&workspace).await?;
        tokio::fs::write(workspace.join("GOAL.md"), "Ship Elpis").await?;

        sync_continuity_before_compaction(Some(&memories), cwd).await?;
        assert_eq!(
            tokio::fs::read_to_string(workspace.join("GOAL.md")).await?,
            "Ship Elpis"
        );
        Ok(())
    }
}
