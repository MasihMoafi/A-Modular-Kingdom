# Qdrant Cloud Setup for RAG

## Why Qdrant Cloud?

Qdrant Cloud solves the concurrency issue with local Qdrant:
- ✅ **Multiple concurrent queries** - No file locking issues
- ✅ **1GB free tier** - No credit card required
- ✅ **Production-ready** - Managed service with high availability
- ✅ **Same API** - No code changes needed

## Setup Steps

### 1. Create Free Qdrant Cloud Cluster

1. Go to https://cloud.qdrant.io/signup
2. Sign up (no credit card required)
3. Create a new cluster:
   - Choose **Free tier** (1GB)
   - Select a region close to you
   - Wait for cluster to be ready (~2 minutes)

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

To use local mode again (for development without internet):

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
- Check storage: Free tier includes 1GB
- Monitor query performance
- View collections and data

## Migration Notes

- Existing local collections won't automatically sync to cloud
- First query to cloud will re-index documents
- Cloud and local collections are separate
- Collection names are same, but data is independent
