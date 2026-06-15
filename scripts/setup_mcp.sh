#!/bin/bash
set -e

# Get workspace absolute path
WORKSPACE_DIR="/home/masih/Desktop/p/A-Modular-Kingdom"

echo "🏰 [AMK-Harness] Initializing A-Modular-Kingdom MCP harness..."

# 1. Run Codex config update
python3 "$WORKSPACE_DIR/scripts/register_mcp.py"

# 2. Check if running inside/for Claude Code
# Claude Code automatically triggers SessionStart.sh.
# Here we check if the 'claude' CLI command is available or if we are executing as a hook.
if command -v claude &> /dev/null; then
    echo "🤖 [AMK-Harness] Claude CLI detected. Configuring Claude Code MCP..."
    if claude mcp list 2>/dev/null | grep -q "a-modular-kingdom"; then
        echo "🤖 [AMK-Harness] a-modular-kingdom is already registered in Claude Code."
    else
        echo "🤖 [AMK-Harness] Registering a-modular-kingdom with Claude Code..."
        # Add server using local python and thermal runner wrapper
        claude mcp add a-modular-kingdom \
            "$WORKSPACE_DIR/.venv/bin/python" \
            "$WORKSPACE_DIR/scripts/thermal_runner.py" \
            --threshold 85 -- \
            "$WORKSPACE_DIR/.venv/bin/python" -u \
            "$WORKSPACE_DIR/src/agent/host.py"
    fi
else
    echo "⚠️ [AMK-Harness] 'claude' command not found in current PATH. Skipping Claude Code CLI configuration."
    echo "   (This is normal if you are running this from outside a Claude Code session)"
fi

echo "✅ [AMK-Harness] Setup completed successfully."
