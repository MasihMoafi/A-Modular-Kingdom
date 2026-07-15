# Foundation Codex Baseline Evidence

**Date:** 2026-07-15
**Branch:** `agent/foundation-codex-baseline`
**Imported binary:** `codex-rs/target/debug/codex-tui`

## Acceptance Turn

The repository-built TUI was launched from the worktree in inline mode with the
existing authenticated ChatGPT/Codex account:

```text
./codex-rs/target/debug/codex-tui --no-alt-screen \
  -C /home/masih/Desktop/f/p/Elpis-foundation \
  -s workspace-write -a never '<acceptance prompt>'
```

The prompt required one `pwd`, creation of
`.elpis-foundation-acceptance-test.txt` with one exact marker line, and a final
`pwd` plus `cat` verification. The imported TUI visibly rendered the command
activity and final result:

```text
Working directory: /home/masih/Desktop/f/p/Elpis-foundation
File content: ELPIS_FOUNDATION_CODEX_BASELINE_OK
```

The recorded Codex turn `019f6616-842c-7b90-9b48-844406cd496f` confirms these
three successful tool calls:

```text
pwd
printf '%s\n' 'ELPIS_FOUNDATION_CODEX_BASELINE_OK' > .elpis-foundation-acceptance-test.txt
pwd
cat .elpis-foundation-acceptance-test.txt
```

The final tool output was:

```text
/home/masih/Desktop/f/p/Elpis-foundation
ELPIS_FOUNDATION_CODEX_BASELINE_OK
```

An independent shell check confirmed the file path and exact content. The test
file was then removed, and `git status --short` confirmed it was no longer present.

## Acceptance Result

`foundation-codex-baseline` passes all three acceptance items when combined with
the prior locked build and donor-isolation launch evidence in
`docs/LAUNCH_SMOKE_EVIDENCE.md`:

1. the imported TUI builds and launches from repository-contained source;
2. an authenticated Codex turn runs commands and creates a workspace file;
3. runtime checks show zero file access to the donor clone.
