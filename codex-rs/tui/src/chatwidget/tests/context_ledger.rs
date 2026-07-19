use super::*;

fn render_ledger(chat: &ChatWidget, height: u16) -> String {
    let area = ratatui::layout::Rect::new(0, 0, 52, height);
    let mut buf = ratatui::buffer::Buffer::empty(area);
    chat.render_context_ledger(area, &mut buf);

    (0..area.height)
        .map(|y| {
            (0..area.width)
                .fold(String::new(), |mut line, x| {
                    line.push_str(buf[(x, y)].symbol());
                    line
                })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn configure_ledger_sources(
    chat: &mut ChatWidget,
    root: &std::path::Path,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let memories = root.join(".elpis/memories");
    let cwd = root.join("projects/Elpis");
    let dev = root.join("projects/skills/dev");
    let workspace = crate::legacy_core::elpis_context::workspace_context_dir(
        Some(&memories),
        &cwd,
    )
    .expect("workspace path");

    std::fs::create_dir_all(root.join(".codex"))?;
    std::fs::create_dir_all(&cwd)?;
    std::fs::create_dir_all(&dev)?;
    std::fs::create_dir_all(&workspace)?;
    std::fs::write(root.join(".codex/AGENTS.md"), "Global instructions")?;
    std::fs::write(cwd.join("AGENTS.md"), "Project instructions")?;
    std::fs::write(dev.join("SKILL.md"), "Development instructions")?;
    std::fs::write(workspace.join("GOAL.md"), "Ship the grouped ledger")?;
    std::fs::write(workspace.join("ES.md"), "Command evidence")?;

    chat.config.memories.root = Some(memories.clone().abs());
    chat.config.cwd = cwd.clone().abs();
    chat.instruction_source_paths = vec![
        codex_utils_path_uri::PathUri::from_abs_path(&root.join(".codex/AGENTS.md").abs()),
        codex_utils_path_uri::PathUri::from_abs_path(&cwd.join("AGENTS.md").abs()),
        codex_utils_path_uri::PathUri::from_abs_path(&dev.join("SKILL.md").abs()),
    ];
    chat.last_rendered_width.set(Some(120));
    Ok((memories, cwd))
}

#[tokio::test]
async fn ledger_groups_real_sources_and_exposes_selected_reason() -> anyhow::Result<()> {
    let root = tempdir()?;
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(None).await;
    configure_ledger_sources(&mut chat, root.path())?;

    assert!(chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::BackTab)));
    assert!(chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('w'))));
    let rendered = render_ledger(&chat, 80);

    for heading in ["ACTIVE FILES", "DURABLE MEMORY", "INSTRUCTIONS", "TOOL EVIDENCE"] {
        assert!(rendered.contains(heading), "missing {heading}:\n{rendered}");
    }
    assert!(rendered.contains("≈"), "token estimates must be labeled");
    assert!(rendered.contains("WHY INCLUDED"));
    assert!(rendered.contains("applicable global rules"));

    let short = render_ledger(&chat, 16);
    assert!(short.contains("Global AGENTS.md"));
    assert!(short.contains("WHY INCLUDED"));
    assert!(short.contains("applicable global rules"));
    Ok(())
}

#[tokio::test]
async fn ledger_g_sequences_exclude_and_include_all_selectable_sources() -> anyhow::Result<()> {
    let root = tempdir()?;
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(None).await;
    let (memories, cwd) = configure_ledger_sources(&mut chat, root.path())?;
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::BackTab));

    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('g')));
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('e')));
    assert!(
        crate::legacy_core::elpis_context::continuity_sources(
            Some(&memories),
            &cwd,
            &chat
                .instruction_source_paths
                .iter()
                .filter_map(|uri| uri.to_abs_path().ok())
                .collect::<Vec<_>>(),
        )
            .iter()
            .filter(|source| source.selectable)
            .all(|source| !source.admitted)
    );

    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('g')));
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('i')));
    assert!(
        crate::legacy_core::elpis_context::continuity_sources(
            Some(&memories),
            &cwd,
            &chat
                .instruction_source_paths
                .iter()
                .filter_map(|uri| uri.to_abs_path().ok())
                .collect::<Vec<_>>(),
        )
            .iter()
            .filter(|source| source.selectable)
            .all(|source| source.admitted)
    );
    Ok(())
}

#[tokio::test]
async fn test_deduplication_of_manually_added_files() -> anyhow::Result<()> {
    let root = tempdir()?;
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(None).await;
    let (memories, cwd) = configure_ledger_sources(&mut chat, root.path())?;

    // Manually add the dev rule file
    let dev_rule = root.path().join("projects/skills/dev/SKILL.md");
    crate::legacy_core::elpis_context::add_continuity_source(
        Some(&memories),
        &cwd,
        &dev_rule,
    )?;

    let instruction_paths = chat
        .instruction_source_paths
        .iter()
        .filter_map(|uri| uri.to_abs_path().ok())
        .collect::<Vec<_>>();

    let sources = crate::legacy_core::elpis_context::continuity_sources(
        Some(&memories),
        &cwd,
        &instruction_paths,
    );

    // Count occurrences of the dev rule file
    let count = sources
        .iter()
        .filter(|source| source.path == dev_rule)
        .count();

    assert_eq!(count, 1, "The dev rule file should only appear once after being manually added");

    Ok(())
}

#[tokio::test]
async fn test_totals_equal_sum_of_rows() -> anyhow::Result<()> {
    let root = tempdir()?;
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(None).await;
    let (memories, cwd) = configure_ledger_sources(&mut chat, root.path())?;

    // Make sure we have some active custom source
    let notes = cwd.join("notes.md");
    tokio::fs::write(&notes, "Some random notes with tokens").await?;
    crate::legacy_core::elpis_context::add_continuity_source(
        Some(&memories),
        &cwd,
        &notes,
    )?;

    // Make sure all sources are admitted
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::BackTab));
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('g')));
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('i')));

    let instruction_paths = chat
        .instruction_source_paths
        .iter()
        .filter_map(|uri| uri.to_abs_path().ok())
        .collect::<Vec<_>>();

    let sources = crate::legacy_core::elpis_context::continuity_sources(
        Some(&memories),
        &cwd,
        &instruction_paths,
    );

    let total_tokens_from_sources: u64 = sources
        .iter()
        .filter(|source| source.admitted)
        .map(|source| source.estimated_tokens)
        .sum();

    let mut category_totals: std::collections::HashMap<crate::legacy_core::elpis_context::ContinuitySourceCategory, u64> = std::collections::HashMap::new();

    for category in crate::legacy_core::elpis_context::ContinuitySourceCategory::ALL {
        let cat_total: u64 = sources
            .iter()
            .filter(|source| source.admitted && source.category == category)
            .map(|source| source.estimated_tokens)
            .sum();
        category_totals.insert(category, cat_total);
    }

    let sum_of_categories: u64 = category_totals.values().sum();

    assert_eq!(total_tokens_from_sources, sum_of_categories, "Total tokens should equal the sum of tokens in each category");

    Ok(())
}
