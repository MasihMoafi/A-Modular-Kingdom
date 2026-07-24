# Elpis documentation

Elpis gives terminal coding agents portable context, bounded local memory, and a
continuity model that survives long sessions without blindly resending an expanding
conversation history.

![Elpis demo](assets/elpis-demo.gif)

## Start here

```bash
mkdir -p "$HOME/.local/bin" && curl -fsSL https://github.com/MasihMoafi/Elpis/releases/latest/download/elpis-linux-x86_64 | install -m 755 /dev/stdin "$HOME/.local/bin/elpis"
elpis
```

This Linux x86_64 command assumes `~/.local/bin` is on your `PATH`. On first launch,
choose a provider and follow its sign-in prompt. Use the sections below when you need
the implementation and operating model behind the interface.

- [Context and sessions](context-and-sessions.md) explains what Elpis admits,
  retains, and prunes.
- [Memory](memory.md) explains bounded local memory, recall provenance, and the
  fail-closed archive.
- [Providers](providers.md) explains provider selection and credentials.
- [Visual walkthrough](visual-walkthrough.md) points to the screenshot-led guide in
  the repository README.
