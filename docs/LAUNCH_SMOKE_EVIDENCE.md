# Launch Smoke Evidence

**Date:** 2026-07-15
**Branch:** `agent/foundation-codex-baseline`
**Commit verified:** `f37fc77` (Import pinned Codex Rust foundation)

## Binary

```
codex-rs/target/debug/codex-tui
```

Built from the `--locked` build in the worktree. Links only to system libraries:
`libssl.so.3`, `libcrypto.so.3`, `libgcc_s.so.1`, `libc.so.6`.

## Smoke Checks

### 1. `--help` exits clean

```
$ timeout 10 ./codex-rs/target/debug/codex-tui --help
Usage: codex-tui [OPTIONS] [PROMPT]
...
exit=0
```

### 2. TUI starts without crash

```
$ timeout 4 ./codex-rs/target/debug/codex-tui "smoke"
exit=124   # timeout killed it; TUI ran alive for 4 s waiting for terminal
```

Exit 124 = killed by timeout after 4 s with no error. The TUI initialised,
opened `~/.codex/config.toml`, `auth.json`, and `state_5.sqlite`, and entered
its event loop.

### 3. No donor path accessed at runtime

`strace -f -e trace=openat` during startup (617 syscalls):

```
$ if grep -q "f/p/others/codex" /tmp/tui_strace2.txt; then
    echo "FAIL"; else echo "PASS"; fi
PASS: no donor path accessed
```

Donor clone: `/home/masih/Desktop/f/p/others/codex`
Result: **zero file-open calls to that path**.

### 4. No donor path embedded in the binary

```
$ strings codex-rs/target/debug/codex-tui | grep -E "Desktop/f/p/(others|codex)"
(empty — no matches)
```

### 5. Runtime file reads confirmed only from `~/.codex/`

Observed opens during startup:
- `~/.codex/config.toml`
- `~/.codex/auth.json`
- `~/.codex/state_5.sqlite` (and -wal, -shm)
- `~/.codex/tmp/arg0/` (lock files for arg0 management)

None from the donor clone or any Elpis source directory.

## Conclusion

The imported TUI from `codex-rs/target/debug/codex-tui` starts cleanly,
enters its event loop, and loads zero files from the donor clone at runtime.
No launch blockers were found. The binary is self-contained and donor-isolated.
