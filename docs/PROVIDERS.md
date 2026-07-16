# Elpis Provider Boundary

Elpis owns context admission, durable memory, continuity, permissions, evidence, and the
terminal interface. A selected provider owns inference; changing provider must not discard
the Elpis state around it.

## Implementation Status

Remotely tested at launcher/configuration level:

- OpenAI, OpenRouter, Amazon Bedrock, Ollama, and LM Studio provider IDs;
- Claude Sonnet and Gemini Pro/Flash shortcuts through OpenRouter;
- separate OpenRouter credentials through `OPENROUTER_API_KEY`.

Not yet accepted end to end:

- an authenticated OpenAI task and resume;
- an authenticated OpenRouter task and resume;
- continuity evidence across a provider change;
- the provider-aware `Choose a mind` `/model` surface.

Native Anthropic and Google adapters are not implemented. Claude and Gemini shortcuts are
OpenRouter compatibility routes and must be labelled that way.

## Supported Entry Points

| Selection | Runtime path | Credential |
| --- | --- | --- |
| `--provider openai` | OpenAI/Codex foundation | ChatGPT login or configured OpenAI auth |
| `--provider openrouter` | OpenRouter Responses API | `OPENROUTER_API_KEY` |
| `--provider claude` | OpenRouter Claude Sonnet family | `OPENROUTER_API_KEY` |
| `--provider gemini` | OpenRouter Gemini Pro family | `OPENROUTER_API_KEY` |
| `--provider gemini-flash` | OpenRouter Gemini Flash family | `OPENROUTER_API_KEY` |
| `--provider amazon-bedrock` | Bedrock path | configured AWS/Bedrock auth |
| `--provider ollama` | local Ollama | local service |
| `--provider lmstudio` | local LM Studio | local service |

An explicit OpenRouter slug may be selected with configuration or `elpis -m <model>`.

## Model Selection

Today, provider-family aliases are selected through the launcher. The inherited `/model`
picker has not yet become the Elpis provider-aware surface.

The future `Choose a mind` layer should show provider, runtime path, credential source,
capabilities, and whether the route is native or compatibility-based.

## Native Adapter Roadmap

Direct Anthropic and Google support requires real protocol adapters covering:

1. request and streaming translation;
2. tool-call and tool-result mapping;
3. reasoning/thinking preservation;
4. authentication and error mapping;
5. truthful model/capability catalogs;
6. continuity acceptance across a provider switch.
