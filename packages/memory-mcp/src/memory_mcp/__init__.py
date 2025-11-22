"""Memory MCP - RAG and Memory tools via Model Context Protocol."""

__version__ = "0.1.0"

from memory_mcp.config import Settings, get_settings
from memory_mcp.server import create_server

__all__ = ["Settings", "get_settings", "create_server", "__version__"]
