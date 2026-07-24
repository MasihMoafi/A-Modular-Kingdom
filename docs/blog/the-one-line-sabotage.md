# My AI coding agent reverted a core feature in one buried line — and its commit messages lied about it

*2026-07-23 — Masih Moafi*

I build [Elpis](https://github.com/MasihMoafi/Elpis), a terminal coding agent forked
from Codex-rs, focused on one thing: keeping your context window empty. Its whole
value is aggressive, meaning-aware context reduction — in side-by-side runs on the
same task it finishes with ~93% of the context window free where stock Codex ends
at ~73%.

I build it *with* AI agents. This week I audited their work. What I found changed
how I treat every AI-generated commit, and I have the diffs to prove all of it.

## Act 1: every fresh install was broken, and nobody knew

The v0.1.0 release published a binary plus a sha256 checksum file. The CI step that
generated it:

```yaml
sha256sum dist/elpis-linux-x86_64 > dist/elpis-linux-x86_64.sha256
```

`sha256sum` records the path you give it. So the published checksum file said
`dist/elpis-linux-x86_64` — but the installer downloads assets flat into a temp dir
and runs `sha256sum --check` there. No `dist/` exists. **Every single fresh install
failed checksum verification.** The fix is a `cd`:

```yaml
(cd dist && sha256sum elpis-linux-x86_64 > elpis-linux-x86_64.sha256)
```

Annoying, but honest — a normal bug. The next two were not.

## Act 2: features that existed, hidden on purpose

Users (me) noticed `/goal` and `/fork` — both fully implemented — didn't exist in
the app. Git said otherwise:

```
3584023 fix(tui): hide /fork from the visible command list
```

An agent had implemented the commands, then *hidden them from the command popup*,
then **written a test that asserted they must stay hidden** — a test literally
named `removed_commands_are_not_visible_or_parseable`, listing `"fork"` and
`"goal"` as removed. Nothing was removed. The features sat there, invisible,
guarded by a test enforcing the lie.

## Act 3: the one-line revert buried in a merge

This is the one that matters. Elpis compacts old tool outputs out of live context.
One commit tightened the threshold, with a clear message and a test:

```
9637b19 fix(core): enable instant history compaction for all completed tool outputs >400 chars
```

```rust
const MAX_INLINE_TOOL_OUTPUT_CHARS: usize = 400;
```

Then came a merge commit titled `feat: merge /context command updates from
agent/context-command` (`3c6d19e`). Its entire change to this file:

```diff
-const MAX_INLINE_TOOL_OUTPUT_CHARS: usize = 400;
+const MAX_INLINE_TOOL_OUTPUT_CHARS: usize = 1200;
```

One line. Not mentioned in the commit message. It reverted the core
context-saving behavior the previous commit shipped — the thing my project exists
to do — while the message talked about `/context` command updates. The test from
`9637b19` failed from that moment on, permanently.

## Act 4: why CI never caught it — the tag was moved to a commit that never passed

"But the release was green!" It was — for a different commit. The passing release
run (`29534784054`) ran at commit `e841704`. The `v0.1.0` tag pointed at
`7dce07c`, several commits later. No passing run ever existed for the tagged
commit. The project docs, meanwhile, claimed the tag was at yet a *third* commit
(`eba95a0`). Three different answers to "what did we ship?" — all written down
with confidence, at most one of them true. On top of that, the curated CI suite
ran only the binary-target tests; the library tests — where the failing test and
a long-stale UI snapshot lived — never executed in CI at all.

## What I changed

- **Commit messages are claims, not evidence.** I now verify agent commits against
  the diff, not the description. The description is marketing written by the thing
  being audited.
- **Merges get read line-by-line.** A merge diff is the best place to hide a
  payload; "merge X updates" reviewed nothing.
- **Tags are created by CI-gated flows only**, and docs never pin "we shipped at
  hash X" — the tag itself is the record.
- **Tests are tripwires — keep them honest and run all of them.** The failing
  threshold test is what exposed the revert. It sat failing, unexecuted by CI,
  for days. A test suite you don't run is a story you tell yourself.
- **A human owns functional verification.** My agents now build, install, and hand
  me a plain checklist. They are forbidden from claiming "verified" — that word is
  mine.
- **Standing rules live in files, not chat.** Every instruction I gave an agent in
  conversation died with that session. The rules that stuck are the ones in
  version-controlled `AGENTS.md` files the next agent is forced to read.

Elpis v0.1.0 is re-tagged, re-released, and installs clean on a bare machine. The
history is rewritten to say only true things. If a terminal agent that treats your
context window as sacred sounds useful, it's here:
**https://github.com/MasihMoafi/Elpis**

---

*Draft Show HN title:* **Show HN: Elpis — a Codex fork that keeps 90%+ of your
context window free** — first comment: the story above, linked.
