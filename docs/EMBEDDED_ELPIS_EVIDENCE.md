# Embedded Elpis Evidence

Date: 2026-07-15

## Result

The repository-contained Codex Rust foundation now builds and launches as `elpis`.
Codex remains the active runtime and ChatGPT authentication boundary; the donor Codex
clone is not used at runtime.

## Source

- Branch: `agent/elpis-embedded-launcher`
- Implementation commit: `4c6ad25`
- Build workflow: `.github/workflows/embedded-elpis-linux.yml`
- GitHub Actions run: `29446246504` (passed)

## Verification

- Remote formatting, focused TUI compilation, Elpis branding test, release build,
  executable identity check, stripping, and artifact upload passed.
- Artifact: `elpis-linux-x86_64`
- SHA-256: `3883df12371a9ec37041d05be3208797e23cb37d04d17c49629b7eda5258c205`
- `elpis --version` returned `elpis 0.0.0`; `elpis --help` identified the command as
  `elpis`.
- Dynamic dependencies resolved to system OpenSSL, libgcc, libm, and libc libraries.
- Installed atomically at `/home/masih/.local/bin/elpis`; a fresh login shell resolves
  that command.
- A pseudo-terminal launch remained alive for the four-second smoke window and was
  stopped by the safety timer. No Elpis process remained afterward.
- The installed binary contains `Welcome to Elpis, with Codex as the active runtime`.

## Scope

This checkpoint proves the contained build and installation. Command/file rendering,
permissions, mouse behavior, sessions, and compaction are inherited Codex capabilities;
future Elpis work should preserve them while subtracting unwanted surfaces.

## Canonical Main Verification

- Canonical commit: `948c373`
- GitHub Actions run: `29449900034` (passed)
- Installed SHA-256: `bcb9dc5c5402e15cbb670210c31b94ccac58255bf7aaafca0b48831f4a395f8b`
- The first subtraction removed the archived root TUI, old Python agent main, and
  obsolete Debian builder. Remote formatting, focused TUI compilation, branding,
  release build, executable identity, and artifact upload still passed.
- The installed artifact launched as `Elpis · Codex runtime` from canonical `main` and
  exited cleanly on interruption.
