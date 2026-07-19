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

    for heading in ["FILES", "MEMORY", "INSTRUCTIONS", "EVIDENCE"] {
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
        crate::legacy_core::elpis_context::continuity_sources(Some(&memories), &cwd)
            .iter()
            .filter(|source| source.selectable)
            .all(|source| !source.admitted)
    );

    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('g')));
    chat.handle_context_ledger_key_event(KeyEvent::from(KeyCode::Char('i')));
    assert!(
        crate::legacy_core::elpis_context::continuity_sources(Some(&memories), &cwd)
            .iter()
            .filter(|source| source.selectable)
            .all(|source| source.admitted)
    );
    Ok(())
}
