# Elpis Provider Boundary

Elpis owns context admission, durable memory, continuity, permissions, evidence, and the
terminal interface. The selected provider owns inference. Provider changes must not discard the
Elpis state around them, and a native-provider selection must never be redirected through another
provider.

## Built-in hosted routes

| Provider ID | API base URL | Credential environment variable | Wire protocol | Default model |
| --- | --- | --- | --- | --- |
| `openai` | `https://api.openai.com/v1` (or the existing configured OpenAI/Codex base) | `OPENAI_API_KEY` when API-key auth is used | OpenAI Responses | `gpt-5.4` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | OpenAI Responses compatibility | `openai/gpt-5.4` |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` | Anthropic Messages | `claude-sonnet-4-6` |
| `google-gemini` | `https://generativelanguage.googleapis.com/v1beta` | `GEMINI_API_KEY` | Gemini GenerateContent | `gemini-3.5-flash` |

The OpenAI Responses implementation is unchanged. OpenRouter also continues to use its existing
Responses-compatible path.

## Compatibility aliases

These launcher aliases are intentionally **not native routes**:

| Alias | Actual provider | Model alias | Label |
| --- | --- | --- | --- |
| `--provider claude` | OpenRouter | `~anthropic/claude-sonnet-latest` | Claude via OpenRouter (compatibility) |
| `--provider gemini` | OpenRouter | `~google/gemini-pro-latest` | Gemini Pro via OpenRouter (compatibility) |
| `--provider gemini-flash` | OpenRouter | `~google/gemini-flash-latest` | Gemini Flash via OpenRouter (compatibility) |

Use `--provider anthropic` or `--provider google-gemini` for direct vendor routing. Those native
IDs never install an OpenRouter model override.

## Native authentication boundaries

- Anthropic sends `ANTHROPIC_API_KEY` only as `x-api-key` and sends
  `anthropic-version: 2023-06-01`.
- Gemini sends `GEMINI_API_KEY` only as `x-goog-api-key`.
- OpenAI and OpenRouter API keys retain their existing `Authorization: Bearer ...` behavior.
- Provider credentials are read from the provider's configured environment variable. They are not
  copied between providers and are never translated into an OpenRouter credential.

## Native request and stream translation

The native adapters translate the canonical Elpis turn representation as follows:

- system and developer text becomes Anthropic `system` blocks or Gemini `systemInstruction`;
- user and assistant text becomes vendor-native message/content blocks;
- function definitions become Anthropic `tools` or Gemini `functionDeclarations`;
- function calls and text-only function results round-trip through native tool-use/function-call
  blocks;
- streamed text, tool calls, vendor errors, token usage, model/version identifiers, and completion
  state are translated back into the existing `ResponseEvent` stream;
- dropping the response stream cancels the parser task and drops the upstream response body;
- provider stream-idle timeouts are surfaced as stream errors.

The static native catalogs are supplied to the model manager, so `/model` consumes the native
provider's default model instead of attempting an OpenAI `/models` request.

## Honest protocol limitations

The current native boundary intentionally rejects rather than silently approximates unsupported
history or tool shapes.

- Text and function tools are supported. Image inputs and image-bearing tool results are currently
  rejected even though both vendors have image-capable APIs.
- OpenAI Responses-only items (encrypted reasoning state, remote compaction controls, custom/freeform
  tools, tool-search items, built-in web search, image generation, and namespace tools) are not
  translated.
- Vendor-native thinking/reasoning signatures, citations, prompt-cache controls, structured-output
  strictness, Anthropic server tools, and Gemini built-in tools/code execution are not preserved.
- Anthropic requests currently use an explicit `max_tokens` value of 8192 because the canonical
  request has no provider-neutral output-token limit.
- Gemini emits only the first candidate. Repeated full function-call chunks are de-duplicated.
- The canonical completion event exposes `end_turn`, not a raw vendor finish-reason field. Known
  finish reasons are mapped explicitly; unknown reasons remain unknown. A parsed tool call always
  maps to `end_turn = false`.
- Native stream reconnection is not attempted after partial output. HTTP and SSE failures are
  surfaced to the existing provider error path.

## Validation commands

Run these from `codex-rs` after changing provider metadata, routing, authentication, translation, or
model selection:

```sh
cargo fmt --all --check
cargo test -p codex-model-provider-info --locked
cargo test -p codex-core --lib --locked chat_completions
cargo test -p codex-tui --bin elpis --locked
cargo build -p codex-tui --bin elpis --locked
```

## Manual smoke tests

Anthropic:

```sh
export ANTHROPIC_API_KEY='...'
cargo run -p codex-tui --bin elpis -- --provider anthropic
# In the TUI: run /model and confirm Claude Sonnet 4.6 is listed, then ask for a simple
# answer and a task that invokes a local function tool.
```

Gemini:

```sh
export GEMINI_API_KEY='...'
cargo run -p codex-tui --bin elpis -- --provider google-gemini
# In the TUI: run /model and confirm Gemini 3.5 Flash is listed, then exercise text and a
# function-tool turn.
```

Compatibility-route check:

```sh
export OPENROUTER_API_KEY='...'
cargo run -p codex-tui --bin elpis -- --provider claude
# Confirm logs/config show model_provider=openrouter and the compatibility model alias.
```
