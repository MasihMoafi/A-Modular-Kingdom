# Qdrant Cloud Setup for RAG

## Why Qdrant Cloud?

Qdrant Cloud solves the concurrency issue with local Qdrant:
- ✅ **Multiple concurrent queries** - Written in Rust
- ✅ **Blazing Speed** - Written in Rust
- ✅ **Generous free tier** - No credit card required
- ✅ **Production-ready** 
- ✅ **Local-mode Available**
- ✅ **Supports Multimodal Databases.**
- ✅ **Same API** - No code changes needed

## Setup Steps

### 1. Create Free Qdrant Cloud Cluster

1. Go to https://cloud.qdrant.io/signup
2. Sign up 
3. Create a new cluster:

### 2. Get Your Credentials

After cluster creation, you'll get:
- **Cluster URL**: `https://xyz-abc-123.qdrant.io`
- **API Key**: Click "Generate API Key" in the dashboard

### 3. Configure RAG to Use Cloud

Edit `src/rag/fetch_2.py`:

```python
RAG_CONFIG = {
    # ... other settings ...
    "qdrant_mode": "cloud",  # Change from "local" to "cloud"
    "qdrant_url": "https://your-cluster.qdrant.io",  # Your cluster URL
    "qdrant_api_key": "your-api-key-here",  # Your API key
}
```

**Security Note**: Store API key in environment variable:

```python
import os

RAG_CONFIG = {
    # ... other settings ...
    "qdrant_mode": "cloud",
    "qdrant_url": os.getenv("QDRANT_URL"),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
}
```

Then set in your shell:
```bash
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"
```

### 4. Test It

```bash
# Query should work without locking issues now
python -c "
from rag.fetch_2 import fetchExternalKnowledge
result = fetchExternalKnowledge('test query')
print(result)
"
```

## Switching Back to Local

To use local mode:

```python
RAG_CONFIG = {
    # ... other settings ...
    "qdrant_mode": "local",
    "qdrant_url": None,
    "qdrant_api_key": None,
}
```

## Monitoring Usage

- Dashboard: https://cloud.qdrant.io/
- Monitor query performance

## Migration Notes

- Existing local collections won't automatically sync to cloud
- First query to cloud will re-index documents
- Cloud and local collections are separate
