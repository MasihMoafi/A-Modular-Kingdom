#!/usr/bin/env python3
import os

config_path = os.path.expanduser("~/.codex/config.toml")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        content = f.read()
    
    if "[mcp_servers.modular_kingdom_host]" in content:
        print("[AMK Setup] modular_kingdom_host already configured in config.toml.")
    elif "[disabled_mcp_servers.modular_kingdom_host]" in content:
        print("[AMK Setup] Enabling modular_kingdom_host in config.toml...")
        new_content = content.replace("[disabled_mcp_servers.modular_kingdom_host]", "[mcp_servers.modular_kingdom_host]")
        with open(config_path, "w") as f:
            f.write(new_content)
    else:
        print("[AMK Setup] Appending modular_kingdom_host to config.toml...")
        mcp_block = """

[mcp_servers.modular_kingdom_host]
command = "/home/masih/Desktop/p/A-Modular-Kingdom/scripts/thermal_runner.py"
args = ["--threshold", "85", "--", "/home/masih/Desktop/p/A-Modular-Kingdom/.venv/bin/python", "-u", "/home/masih/Desktop/p/A-Modular-Kingdom/src/agent/host.py"]
startup_timeout_sec = 60
tool_timeout_sec = 30
env = { "PYTHONUNBUFFERED" = "1", "MCP_LOG_FILE" = "/tmp/modular_kingdom_mcp.log", "MCP_DEBUG_PROTOCOL" = "1" }
"""
        with open(config_path, "a") as f:
            f.write(mcp_block)
else:
    print(f"[AMK Setup] Codex config.toml not found at {config_path}. Skipping Codex setup.")
