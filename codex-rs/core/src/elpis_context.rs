use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

const MAX_GOAL_CHARS: usize = 6_000;
const MAX_CHECKPOINT_CHARS: usize = 8_000;
const MAX_RULE_CHARS: usize = 8_000;
const ADMISSION_FILE: &str = "admission.toml";

const GLOBAL_RULES: &str = "Global AGENTS.md";
const PROJECT_RULES: &str = "Project AGENTS.md";
const DEV_AGENTS: &str = "dev/AGENTS.md";
const DEV_ARTIFACT_RULES: &str = "dev/ARTIFACT_RULES.md";
const DEV_CODING_GUIDELINES: &str = "dev/CODING_GUIDELINES.md";
const DEV_TERMINAL_RULES: &str = "dev/TERMINAL_AND_GIT_RULES.md";

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
struct ContinuityAdmission {
    global_rules: bool,
    project_rules: bool,
    goal: bool,
    checkpoint: bool,
    dev_agents: bool,
    dev_artifact_rules: bool,
    dev_coding_guidelines: bool,
    dev_terminal_rules: bool,
    custom_sources: BTreeMap<String, bool>,
}

impl Default for ContinuityAdmission {
    fn default() -> Self {
        Self {
            global_rules: true,
            project_rules: true,
            goal: true,
            checkpoint: true,
            dev_agents: true,
            dev_artifact_rules: true,
            dev_coding_guidelines: true,
            dev_terminal_rules: true,
            custom_sources: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuitySource {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub lifetime: &'static str,
    pub reason: &'static str,
    pub admitted: bool,
    pub selectable: bool,
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
    let mut sections = Vec::new();
    for source in continuity_sources(memories_root, cwd) {
        if !source.admitted {
            continue;
        }
        let Ok(content) = tokio::fs::read_to_string(&source.path).await else {
            continue;
        };
        let content = truncate_chars(content.trim(), source_char_limit(&source.name));
        if !content.is_empty() {
            sections.push(format!(
                "### Source: {} ({} characters)\n\n{}",
                source.path.display(),
                content.chars().count(),
                content
            ));
        }
    }
    if sections.is_empty() {
        return None;
    }
    Some(format!(
        "## Elpis Admitted Context\n\n\
         These are the user-visible sources Elpis admitted for this workspace. They are not a full\n\
         transcript. Verify mutable repository state before acting, and prefer the current user\n\
         message when it changes the task.\n\n{}",
        sections.join("\n\n")
    ))
}

pub fn continuity_sources(memories_root: Option<&Path>, cwd: &Path) -> Vec<ContinuitySource> {
    let Some(memories_root) = memories_root else {
        return Vec::new();
    };
    let Some(workspace_dir) = workspace_context_dir(Some(memories_root), cwd) else {
        return Vec::new();
    };
    let admission = read_admission(&workspace_dir);
    let dev_dir = cwd
        .parent()
        .map(|parent| parent.join("skills/dev"));
    let global_rules = memories_root
        .parent()
        .and_then(Path::parent)
        .map(|home| home.join(".codex/AGENTS.md"));
    let mut sources = [
        (
            GLOBAL_RULES,
            global_rules,
            "every turn",
            "applicable global rules",
            admission.global_rules,
            true,
        ),
        (
            PROJECT_RULES,
            Some(cwd.join("AGENTS.md")),
            "every turn",
            "applicable project rules",
            admission.project_rules,
            true,
        ),
        (
            DEV_AGENTS,
            dev_dir.as_ref().map(|dir| dir.join("AGENTS.md")),
            "every turn",
            "configured development rules",
            admission.dev_agents,
            true,
        ),
        (
            DEV_ARTIFACT_RULES,
            dev_dir.as_ref().map(|dir| dir.join("ARTIFACT_RULES.md")),
            "every turn",
            "configured development rules",
            admission.dev_artifact_rules,
            true,
        ),
        (
            DEV_CODING_GUIDELINES,
            dev_dir.as_ref().map(|dir| dir.join("CODING_GUIDELINES.md")),
            "every turn",
            "configured development rules",
            admission.dev_coding_guidelines,
            true,
        ),
        (
            DEV_TERMINAL_RULES,
            dev_dir
                .as_ref()
                .map(|dir| dir.join("TERMINAL_AND_GIT_RULES.md")),
            "every turn",
            "configured development rules",
            admission.dev_terminal_rules,
            true,
        ),
        (
            "GOAL.md",
            Some(workspace_dir.join("GOAL.md")),
            "every turn",
            "active workspace goal",
            admission.goal,
            true,
        ),
        (
            "ES.md",
            Some(workspace_dir.join("ES.md")),
            "every turn",
            "lean session checkpoint",
            admission.checkpoint,
            true,
        ),
    ]
    .into_iter()
    .filter_map(|(name, path, lifetime, reason, admitted, selectable)| {
        let path = path?;
        let metadata = std::fs::metadata(&path).ok()?;
        (metadata.is_file() && metadata.len() > 0).then_some(ContinuitySource {
            name: name.to_string(),
            path,
            bytes: metadata.len(),
            lifetime,
            reason,
            admitted,
            selectable,
        })
    })
    .collect::<Vec<_>>();
    sources.extend(
        admission
            .custom_sources
            .iter()
            .filter_map(|(path, admitted)| {
                let path = PathBuf::from(path);
                let metadata = std::fs::metadata(&path).ok()?;
                (metadata.is_file() && metadata.len() > 0).then_some(ContinuitySource {
                    name: path.display().to_string(),
                    path,
                    bytes: metadata.len(),
                    lifetime: "every turn",
                    reason: "manually added file",
                    admitted: *admitted,
                    selectable: true,
                })
            }),
    );
    sources
}

pub fn set_continuity_source_admitted(
    memories_root: Option<&Path>,
    cwd: &Path,
    source_name: &str,
    admitted: bool,
) -> std::io::Result<()> {
    let Some(workspace_dir) = workspace_context_dir(memories_root, cwd) else {
        return Ok(());
    };
    let mut selection = read_admission(&workspace_dir);
    match source_name {
        GLOBAL_RULES => selection.global_rules = admitted,
        PROJECT_RULES => selection.project_rules = admitted,
        "GOAL.md" => selection.goal = admitted,
        "ES.md" => selection.checkpoint = admitted,
        DEV_AGENTS => selection.dev_agents = admitted,
        DEV_ARTIFACT_RULES => selection.dev_artifact_rules = admitted,
        DEV_CODING_GUIDELINES => selection.dev_coding_guidelines = admitted,
        DEV_TERMINAL_RULES => selection.dev_terminal_rules = admitted,
        _ => {
            let source_path = PathBuf::from(source_name);
            let canonical = source_path.canonicalize()?;
            if !selection
                .custom_sources
                .contains_key(&canonical.to_string_lossy().to_string())
            {
                return Ok(());
            }
            selection
                .custom_sources
                .insert(canonical.to_string_lossy().to_string(), admitted);
        }
    }
    write_admission(&workspace_dir, &selection)
}

pub fn add_continuity_source(
    memories_root: Option<&Path>,
    cwd: &Path,
    requested_path: &Path,
) -> std::io::Result<PathBuf> {
    let Some(workspace_dir) = workspace_context_dir(memories_root, cwd) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Elpis context storage is unavailable",
        ));
    };
    let path = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        cwd.join(requested_path)
    };
    let path = path.canonicalize()?;
    let metadata = std::fs::metadata(&path)?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "context source must be a non-empty file",
        ));
    }
    let mut selection = read_admission(&workspace_dir);
    selection
        .custom_sources
        .insert(path.to_string_lossy().to_string(), true);
    write_admission(&workspace_dir, &selection)?;
    Ok(path)
}

fn write_admission(workspace_dir: &Path, selection: &ContinuityAdmission) -> std::io::Result<()> {
    std::fs::create_dir_all(&workspace_dir)?;
    let path = workspace_dir.join(ADMISSION_FILE);
    let temporary_path = path.with_extension("toml.tmp");
    let contents = toml::to_string_pretty(selection)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(&temporary_path, contents)?;
    std::fs::rename(temporary_path, path)
}

fn read_admission(workspace_dir: &Path) -> ContinuityAdmission {
    let Ok(content) = std::fs::read_to_string(workspace_dir.join(ADMISSION_FILE)) else {
        return ContinuityAdmission::default();
    };
    if let Ok(admission) = toml::from_str(&content) {
        return admission;
    }
    let mut admission = ContinuityAdmission::default();
    for line in content.lines().map(str::trim) {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let admitted = matches!(value.trim(), "true");
        match key.trim() {
            GLOBAL_RULES => admission.global_rules = admitted,
            PROJECT_RULES => admission.project_rules = admitted,
            "GOAL.md" => admission.goal = admitted,
            "ES.md" => admission.checkpoint = admitted,
            DEV_AGENTS => admission.dev_agents = admitted,
            DEV_ARTIFACT_RULES => admission.dev_artifact_rules = admitted,
            DEV_CODING_GUIDELINES => admission.dev_coding_guidelines = admitted,
            DEV_TERMINAL_RULES => admission.dev_terminal_rules = admitted,
            _ => {}
        }
    }
    admission
}

fn source_char_limit(name: &str) -> usize {
    match name {
        "GOAL.md" => MAX_GOAL_CHARS,
        "ES.md" => MAX_CHECKPOINT_CHARS,
        _ => MAX_RULE_CHARS,
    }
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
        assert_eq!(sources[0].lifetime, "every turn");
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
    async fn admission_selection_excludes_a_source_from_the_next_prompt() -> anyhow::Result<()> {
        let home = tempdir()?;
        let memories = home.path().join(".elpis/memories");
        let cwd = Path::new("/tmp/project");
        let workspace = workspace_context_dir(Some(&memories), cwd).expect("workspace path");
        tokio::fs::create_dir_all(&workspace).await?;
        tokio::fs::write(workspace.join("GOAL.md"), "Ship the ledger").await?;
        tokio::fs::write(workspace.join("ES.md"), "Keep the checkpoint").await?;

        set_continuity_source_admitted(Some(&memories), cwd, "ES.md", false)?;
        let prompt = build_continuity_prompt(Some(&memories), cwd)
            .await
            .expect("prompt");

        assert!(prompt.contains("Ship the ledger"));
        assert!(!prompt.contains("Keep the checkpoint"));
        let sources = continuity_sources(Some(&memories), cwd);
        assert!(
            sources
                .iter()
                .any(|source| source.name == "ES.md" && !source.admitted)
        );
        Ok(())
    }

    #[tokio::test]
    async fn custom_source_is_visible_enabled_and_can_be_disabled() -> anyhow::Result<()> {
        let home = tempdir()?;
        let memories = home.path().join(".elpis/memories");
        let cwd = home.path().join("project");
        let custom = cwd.join("notes.md");
        tokio::fs::create_dir_all(&cwd).await?;
        tokio::fs::write(&custom, "Keep this visible").await?;

        let added = add_continuity_source(Some(&memories), &cwd, Path::new("notes.md"))?;
        let sources = continuity_sources(Some(&memories), &cwd);
        assert!(
            sources
                .iter()
                .any(|source| source.path == added && source.admitted)
        );
        assert!(
            build_continuity_prompt(Some(&memories), &cwd)
                .await
                .expect("prompt")
                .contains("Keep this visible")
        );

        set_continuity_source_admitted(Some(&memories), &cwd, &added.display().to_string(), false)?;
        assert!(
            !build_continuity_prompt(Some(&memories), &cwd)
                .await
                .is_some_and(|prompt| prompt.contains("Keep this visible"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn applicable_rules_are_visible_and_toggleable() -> anyhow::Result<()> {
        let home = tempdir()?;
        let memories = home.path().join(".elpis/memories");
        let cwd = home.path().join("projects/Elpis");
        let dev = home.path().join("projects/skills/dev");
        tokio::fs::create_dir_all(home.path().join(".codex")).await?;
        tokio::fs::create_dir_all(&cwd).await?;
        tokio::fs::create_dir_all(&dev).await?;
        tokio::fs::write(home.path().join(".codex/AGENTS.md"), "Global rule").await?;
        tokio::fs::write(cwd.join("AGENTS.md"), "Project rule").await?;
        tokio::fs::write(dev.join("AGENTS.md"), "Dev rule").await?;

        let sources = continuity_sources(Some(&memories), &cwd);
        assert!(
            sources
                .iter()
                .all(|source| source.selectable && source.admitted)
        );
        set_continuity_source_admitted(Some(&memories), &cwd, GLOBAL_RULES, false)?;
        let prompt = build_continuity_prompt(Some(&memories), &cwd)
            .await
            .expect("prompt");
        assert!(!prompt.contains("Global rule"));
        assert!(prompt.contains("Project rule"));
        assert!(prompt.contains("Dev rule"));
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
