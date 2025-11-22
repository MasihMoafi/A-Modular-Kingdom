# Memory MCP Docker Image
# Provides RAG and Memory tools via MCP protocol

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy package files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package
RUN uv pip install --system -e .

# Create data directories
RUN mkdir -p /data/qdrant /data/chroma

# Set environment variables
ENV MEMORY_MCP_QDRANT_PATH=/data/qdrant
ENV MEMORY_MCP_CHROMA_PATH=/data/chroma

# Expose no ports - MCP uses stdio
# Mount volumes for persistent data

# Default command
ENTRYPOINT ["memory-mcp"]
CMD ["serve"]

# Usage:
# docker build -t memory-mcp .
# docker run -it --rm \
#   -v ./documents:/docs:ro \
#   -v ./data:/data \
#   -e MEMORY_MCP_EMBED_PROVIDER=openai \
#   -e MEMORY_MCP_OPENAI_API_KEY=sk-... \
#   memory-mcp serve --docs /docs
