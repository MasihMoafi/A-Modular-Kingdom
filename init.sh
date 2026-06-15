#!/bin/bash
set -e

# Get workspace absolute path
WORKSPACE_DIR="/home/masih/Desktop/p/A-Modular-Kingdom"

echo "🏰 [AMK-Harness] Initializing workspace..."
echo "📍 CWD: $(pwd)"

if [ "$(pwd)" != "$WORKSPACE_DIR" ]; then
    echo "❌ Error: init.sh must be run from the repository root: $WORKSPACE_DIR"
    exit 1
fi

# 1. Ensure virtual environment is ready
if [ ! -d ".venv" ]; then
    echo "📦 Virtual environment not found. Synchronizing workspace via uv..."
    if command -v uv &> /dev/null; then
        uv sync --group dev
    else
        echo "❌ Error: 'uv' is not installed. Please install 'uv' or create a virtual environment in .venv manually."
        exit 1
    fi
fi

# 2. Run Codex configuration registration
echo "🤖 Registering A-Modular-Kingdom MCP host in Codex config..."
.venv/bin/python "$WORKSPACE_DIR/scripts/register_mcp.py"

# 3. Run a quick baseline verification
echo "🧪 Running baseline verification test..."
if .venv/bin/pytest tests/ &> /dev/null; then
    echo "✅ Baseline check: Pytest suite passed successfully."
else
    echo "⚠️ Baseline check: Pytest failures detected, but server dependencies are installed."
fi

echo "✅ [AMK-Harness] Initialization completed successfully."
