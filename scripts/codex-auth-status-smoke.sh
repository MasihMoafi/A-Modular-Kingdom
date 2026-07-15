#!/usr/bin/env bash
set -euo pipefail

command -v codex >/dev/null
command -v jq >/dev/null

{
  printf '%s\n' \
    '{"method":"initialize","id":0,"params":{"clientInfo":{"name":"elpis-auth-smoke","title":"Elpis auth smoke","version":"0.1.0"}}}' \
    '{"method":"initialized","params":{}}' \
    '{"method":"account/read","id":1,"params":{"refreshToken":false}}'
  sleep 1
} | timeout 8s codex app-server --listen stdio:// |
  jq -e -c '
    select(.id == 1)
    | {
        account_type: (.result.account.type // null),
        requires_openai_auth: .result.requiresOpenaiAuth
      }
  '
