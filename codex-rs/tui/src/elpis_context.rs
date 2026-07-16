use anyhow::Context;
use anyhow::Result;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
use std::path::PathBuf;

const CONTEXT_DIR: &str = "context";
const WORKSPACES_DIR: &str = "workspaces";
const GOAL_FILE: &str = "GOAL.md";

pub(crate) async fn write_goal(
    memories_root: Option<&Path>,
    cwd: &Path,
    thread_id: &str,
    objective: &str,
    status: &str,
    updated_at: i64,
) -> Result<Option<PathBuf>> {
    let Some(goal_path) = goal_path(memories_root, cwd) else {
        return Ok(None);
    };
    let parent = goal_path.parent().context("Elpis goal path has no parent")?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("create Elpis context directory {}", parent.display()))?;

    let content = format!(
        "# Elpis Goal\n\n\
         - Workspace: `{}`\n\
         - Thread: `{thread_id}`\n\
         - Status: {status}\n\
         - Updated: {updated_at}\n\n\
         ## Objective\n\n\
         {}\n",
        cwd.display(),
        objective.trim(),
    );
    let temporary_path = goal_path.with_extension(format!("md.tmp-{thread_id}"));
    tokio::fs::write(&temporary_path, content)
        .await
        .with_context(|| format!("write Elpis goal temporary file {}", temporary_path.display()))?;
    tokio::fs::rename(&temporary_path, &goal_path)
        .await
        .with_context(|| format!("replace Elpis goal file {}", goal_path.display()))?;
    Ok(Some(goal_path))
}

pub(crate) async fn clear_goal(
    memories_root: Option<&Path>,
    cwd: &Path,
    thread_id: &str,
) -> Result<Option<PathBuf>> {
    let Some(goal_path) = goal_path(memories_root, cwd) else {
        return Ok(None);
    };
    let content = match tokio::fs::read_to_string(&goal_path).await {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err)
                .with_context(|| format!("read Elpis goal file {}", goal_path.display()));
        }
    };
    if !content.contains(&format!("- Thread: `{thread_id}`")) {
        return Ok(None);
    }
    tokio::fs::remove_file(&goal_path)
        .await
        .with_context(|| format!("remove Elpis goal file {}", goal_path.display()))?;
    Ok(Some(goal_path))
}

fn goal_path(memories_root: Option<&Path>, cwd: &Path) -> Option<PathBuf> {
    let elpis_home = memories_root?.parent()?;
    Some(
        elpis_home
            .join(CONTEXT_DIR)
            .join(WORKSPACES_DIR)
            .join(workspace_key(cwd))
            .join(GOAL_FILE),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn goal_is_written_under_elpis_and_cleared_only_by_its_thread() -> Result<()> {
        let home = tempdir()?;
        let memories_root = home.path().join(".elpis/memories");
        let cwd = Path::new("/tmp/My Project");

        let path = write_goal(
            Some(&memories_root),
            cwd,
            "thread-one",
            "Ship the context layer",
            "active",
            42,
        )
        .await?
        .context("goal path")?;

        let content = tokio::fs::read_to_string(&path).await?;
        assert!(path.starts_with(home.path().join(".elpis/context/workspaces")));
        assert!(content.contains("Ship the context layer"));
        assert!(content.contains("- Thread: `thread-one`"));
        assert_eq!(clear_goal(Some(&memories_root), cwd, "thread-two").await?, None);
        assert!(path.exists());
        assert_eq!(
            clear_goal(Some(&memories_root), cwd, "thread-one").await?,
            Some(path.clone())
        );
        assert!(!path.exists());
        Ok(())
    }

    #[test]
    fn workspace_key_is_stable_readable_and_path_specific() {
        let first = workspace_key(Path::new("/tmp/My Project"));
        assert_eq!(first, workspace_key(Path::new("/tmp/My Project")));
        assert!(first.starts_with("My-Project-"));
        assert_ne!(
            workspace_key(Path::new("/a/project")),
            workspace_key(Path::new("/b/project"))
        );
    }
}
