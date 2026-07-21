#!/usr/bin/env bash
# Provisions the elpis-rag MCP sidecar for THIS machine and THIS checkout: creates the
# venv in-place and (re)writes the mcp_servers.elpis-rag block in config.toml with
# absolute paths computed right now, from wherever this script actually lives. Never
# hand-edit those paths — a path baked in on one machine/clone breaks the moment the
# repo moves or the config is copied to another device. Re-run this after moving the
# repo or on a fresh machine.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
codex_home="${CODEX_HOME:-$HOME/.codex}"
config_file="$codex_home/config.toml"

command -v uv >/dev/null 2>&1 || {
  printf 'uv is required (https://docs.astral.sh/uv/); install it first.\n' >&2
  exit 1
}

printf 'Setting up elpis-rag venv at %s/.venv ...\n' "$repo_root"
(cd "$repo_root" && uv sync)

venv_python="$repo_root/.venv/bin/python"
host_py="$repo_root/src/agent/host.py"
test -x "$venv_python"
test -f "$host_py"

mkdir -p "$codex_home"
touch "$config_file"

REPO_ROOT="$repo_root" VENV_PYTHON="$venv_python" HOST_PY="$host_py" CONFIG_FILE="$config_file" \
python3 - <<'PY'
import os
import re

config_file = os.environ["CONFIG_FILE"]
repo_root = os.environ["REPO_ROOT"]
venv_python = os.environ["VENV_PYTHON"]
host_py = os.environ["HOST_PY"]

block = (
    "[mcp_servers.elpis-rag]\n"
    f'command = "{venv_python}"\n'
    f'args = ["{host_py}"]\n'
    "\n"
    "[mcp_servers.elpis-rag.env]\n"
    f'ELPIS_WORKSPACE_ROOT = "{repo_root}"\n'
    f'PYTHONPATH = "{repo_root}/src"\n'
)

with open(config_file, encoding="utf-8") as f:
    text = f.read()

# Replace an existing elpis-rag block (both [mcp_servers.elpis-rag] and its .env
# sub-table, up to the next table header that isn't part of this block) or append
# a fresh one.
combined_pattern = re.compile(
    r"\[mcp_servers\.elpis-rag\]\n.*?(?=\n\[(?!mcp_servers\.elpis-rag\.env\])|\Z)",
    re.DOTALL,
)

if "[mcp_servers.elpis-rag]" in text:
    new_text = combined_pattern.sub(block.rstrip("\n") + "\n", text, count=1)
else:
    sep = "" if text.endswith("\n") or not text else "\n"
    new_text = text + sep + "\n" + block

with open(config_file, "w", encoding="utf-8") as f:
    f.write(new_text)

print(f"Wrote elpis-rag MCP config to {config_file}")
PY

printf 'Done. elpis-rag now points at this checkout: %s\n' "$repo_root"
