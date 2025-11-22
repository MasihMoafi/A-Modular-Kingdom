"""CLI entry point for Memory MCP server."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="memory-mcp",
        description="Memory MCP - RAG and Memory tools via Model Context Protocol",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    serve_parser = subparsers.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument(
        "--docs",
        "-d",
        nargs="+",
        help="Paths to documents to index for RAG",
        default=[],
    )
    serve_parser.add_argument(
        "--embed-provider",
        "-e",
        choices=["ollama", "sentence-transformers", "openai", "anthropic", "cohere"],
        help="Embedding provider to use",
    )
    serve_parser.add_argument(
        "--embed-model",
        "-m",
        help="Embedding model name",
    )

    init_parser = subparsers.add_parser("init", help="Initialize configuration directory")

    config_parser = subparsers.add_parser("config", help="Show current configuration")

    index_parser = subparsers.add_parser("index", help="Index documents for RAG")
    index_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to documents to index",
    )
    index_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reindex even if files haven't changed",
    )

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args)
    elif args.command == "init":
        run_init()
    elif args.command == "config":
        show_config()
    elif args.command == "index":
        run_index(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_server(args):
    """Start the MCP server."""
    from memory_mcp.config import Settings, get_settings
    from memory_mcp.server import create_server

    overrides = {}
    if args.embed_provider:
        overrides["embed_provider"] = args.embed_provider
    if args.embed_model:
        overrides["embed_model"] = args.embed_model

    if overrides:
        settings = Settings(**overrides)
    else:
        settings = get_settings()

    document_paths = args.docs if args.docs else None

    print(f"Starting Memory MCP server...", file=sys.stderr)
    print(f"  Embedding provider: {settings.embed_provider}", file=sys.stderr)
    print(f"  Embedding model: {settings.embed_model or '(default)'}", file=sys.stderr)
    if document_paths:
        print(f"  Document paths: {document_paths}", file=sys.stderr)

    server = create_server(settings=settings, document_paths=document_paths)
    server.run()


def run_init():
    """Initialize configuration directory."""
    from memory_mcp.config import init_config_dir, CONFIG_FILE

    created = init_config_dir()
    if created:
        print(f"Configuration initialized at: {CONFIG_FILE}")
        print("Edit this file to configure embedding providers, API keys, etc.")
    else:
        print(f"Configuration already exists at: {CONFIG_FILE}")


def show_config():
    """Show current configuration."""
    from memory_mcp.config import Settings, get_settings

    settings = get_settings()
    print("Current Memory MCP Configuration:")
    print("-" * 40)
    for field_name in Settings.model_fields:
        value = getattr(settings, field_name)
        if "api_key" in field_name.lower() and value:
            value = value[:8] + "..." if len(value) > 8 else "***"
        print(f"  {field_name}: {value}")


def run_index(args):
    """Index documents for RAG."""
    from memory_mcp.config import get_settings
    from memory_mcp.rag import RAGPipeline

    settings = get_settings()
    paths = [str(Path(p).expanduser().resolve()) for p in args.paths]

    print(f"Indexing documents from: {paths}")
    print(f"  Embedding provider: {settings.embed_provider}")

    pipeline = RAGPipeline(
        settings=settings,
        document_paths=paths,
    )

    count = pipeline.index(force=args.force)
    print(f"Indexed {count} chunks")


if __name__ == "__main__":
    main()
