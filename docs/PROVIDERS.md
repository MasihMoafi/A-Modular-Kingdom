# Elpis Provider Boundary

Elpis owns the surrounding environment: context admission, durable memory, session continuity, permissions, evidence, and the terminal interface. A selected provider owns model inference; selecting another model must not discard the Elpis state around it.

## Supported entry points

| Selection | Runtime path | Credential |
| --- | --- | --- |
| `--provider openai` | OpenAI/Codex foundation | ChatGPT login or configured OpenAI auth |
| `--provider openrouter` | OpenRouter Responses API | `OPENROUTER_API_KEY` |
| `--provider claude` | OpenRouter, latest Claude Sonnet family | `OPENROUTER_API_KEY` |
| `--provider gemini` | OpenRouter, latest Gemini Pro family | `OPENROUTER_API_KEY` |
| `--provider gemini-flash` | OpenRouter, latest Gemini Flash family | `OPENROUTER_API_KEY` |
| `--provider amazon-bedrock` | Amazon Bedrock Mantle path | configured Bedrock/AWS auth |
| `--provider ollama` | local Ollama path | local service |
| `--provider lmstudio` | local LM Studio path | local service |

The Claude and Gemini shortcuts are model-family aliases routed through OpenRouter. They are not native Anthropic or Google adapters, and the UI and documentation must say so.

## Model selection

When OpenRouter is active, `/model` should offer a small curated set of stable family aliases before the inherited catalog:

- `~anthropic/claude-sonnet-latest`
- `~google/gemini-pro-latest`
- `~google/gemini-flash-latest`
- `~openai/gpt-latest`

A user may still select an explicit OpenRouter model slug through configuration or `elpis -m <model>`.

## Native adapter roadmap

Direct Anthropic and Google support requires protocol adapters rather than renamed OpenAI endpoints. Each native adapter must own:

1. request and streaming translation;
2. tool-call and tool-result mapping;
3. reasoning/thinking preservation across turns;
4. provider-specific authentication and error mapping;
5. a truthful model catalog and capability matrix;
6. continuity acceptance across a provider switch.

The first native-adapter acceptance check is one visible coding task, including a tool call and a resumed session, while Elpis preserves the same goal, admitted context, memory provenance, permissions, and evidence before and after the provider change.
