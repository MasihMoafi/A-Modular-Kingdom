# Multi-stage Dockerfile for A-Modular-Kingdom MCP Server
FROM python:3.12-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./
COPY requirements.txt* ./

# Install dependencies (prefer uv, fallback to pip)
RUN if [ -f "uv.lock" ]; then \
        uv sync --frozen; \
    elif [ -f "requirements.txt" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install fastmcp ollama qdrant-client sentence-transformers; \
    fi

# Development stage - includes all dev tools
FROM base as development

COPY . .

# Install dev dependencies if available
RUN if [ -f "uv.lock" ]; then \
        uv sync --frozen --all-extras; \
    elif [ -f "requirements.txt" ]; then \
        pip install --no-cache-dir pytest pytest-cov black ruff; \
    fi

CMD ["python", "src/agent/host.py"]

# Production stage - minimal image
FROM base as production

COPY src/ ./src/
COPY packages/ ./packages/

# Create directory for global memories
RUN mkdir -p /root/.modular_kingdom/legacy_memories

# Expose default MCP port if needed
EXPOSE 8000

CMD ["python", "src/agent/host.py"]

# Testing stage - for running tests
FROM development as testing

COPY tests/ ./tests/

CMD ["pytest", "tests/", "-v"]
