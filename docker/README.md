# Docker Setup for A-Modular-Kingdom

Complete Docker configuration for running the MCP server, development environment, and tests.

## Quick Start

### Production Server

```bash
# Start MCP server
docker-compose up mcp-server

# With environment variables
QDRANT_API_KEY=your_key docker-compose up mcp-server
```

### Development Environment

```bash
# Start development environment with Jupyter
docker-compose up dev

# Access Jupyter at http://localhost:8888
```

### Running Tests

```bash
# Run all tests
docker-compose run test

# Run specific test file
docker-compose run test pytest tests/test_rag_real.py -v
```

## Services

### mcp-server (Production)
- Minimal production image
- Runs `src/agent/host.py` MCP server
- Persists memories to volume
- Port: 8000

### dev (Development)
- Full development environment
- Jupyter notebook included
- Hot reload with volume mounts
- Ports: 8888 (Jupyter), 8001 (MCP)

### test
- Testing environment
- Runs pytest suite
- Includes all fixtures and test data

### ollama (Optional)
- Local Ollama instance for embeddings
- Port: 11434
- Requires `ollama pull embeddinggemma` after startup

## Environment Variables

Create `.env` file in project root:

```bash
# Gemini API (optional - uses Ollama if not set)
GEMINI_API_KEY=your_gemini_key

# Qdrant Cloud (for RAG V2/V3)
QDRANT_API_KEY=your_qdrant_key
QDRANT_URL=your_qdrant_url
QDRANT_CLUSTER_NAME=your_cluster_name
```

## Volumes

- **mcp-memories**: Persistent storage for global memories (`~/.modular_kingdom`)
- **qdrant-data**: Local Qdrant database (if using local instance)
- **ollama-data**: Ollama models storage

## Build Targets

### Base
```bash
docker build -f docker/Dockerfile --target base -t amk:base .
```

### Development
```bash
docker build -f docker/Dockerfile --target development -t amk:dev .
```

### Production
```bash
docker build -f docker/Dockerfile --target production -t amk:prod .
```

### Testing
```bash
docker build -f docker/Dockerfile --target testing -t amk:test .
```

## Usage Examples

### Start MCP Server Only
```bash
docker-compose up -d mcp-server
docker-compose logs -f mcp-server
```

### Interactive Development
```bash
# Start dev environment
docker-compose up -d dev

# Enter container
docker exec -it a-modular-kingdom-dev bash

# Run Python interactively
python src/agent/main.py
```

### Test Workflow
```bash
# Run all tests
docker-compose run --rm test

# Run with coverage
docker-compose run --rm test pytest --cov=src tests/

# Run specific test
docker-compose run --rm test pytest tests/test_rag_v2.py -k "test_napoleon"
```

### Production Deployment
```bash
# Build production image
docker-compose build mcp-server

# Run with restart policy
docker-compose up -d mcp-server

# View logs
docker-compose logs -f mcp-server

# Stop
docker-compose down
```

## Troubleshooting

### Ollama Models
```bash
# Pull embeddings model
docker exec -it a-modular-kingdom-ollama ollama pull embeddinggemma

# List models
docker exec -it a-modular-kingdom-ollama ollama list
```

### Reset Volumes
```bash
# Remove all volumes (CAUTION: deletes memories)
docker-compose down -v

# Remove specific volume
docker volume rm a-modular-kingdom_mcp-memories
```

### Build Issues
```bash
# Clean build (no cache)
docker-compose build --no-cache

# Rebuild specific service
docker-compose build mcp-server
```
