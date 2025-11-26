# RAG Performance Guide

## Benchmark Results

Based on comprehensive testing with GPU acceleration (CUDA):

### Performance Summary

| Version | Docs | Cold Start | Warm Query | Use Case |
|---------|------|------------|------------|----------|
| **V2** | 10 | 30.7s | 0.32s | Production, standard retrieval |
| **V3** | 10 | 13.7s | **0.01s** | High-accuracy, research |
| **V2** | 100 | 26.8s | 0.31s | Production, standard retrieval |
| **V3** | 100 | 13.9s | **0.02s** | High-accuracy, research |

### Key Findings

1. **V3 is faster on warm queries** (0.01-0.02s vs 0.31-0.32s for V2)
2. **V3 has faster cold starts** (13-14s vs 26-30s for V2)
3. **Both use GPU acceleration** (CUDA) for embeddings and reranking
4. **Indexing is one-time cost** - amortized over many queries

## When to Use Each Version

### RAG V2 (Qdrant + BM25 + CrossEncoder)
- **Best for:** Production apps, standard accuracy requirements
- **Strengths:** Proven, stable, cloud-ready (Qdrant Cloud)
- **Speed:** 0.31s per query (100 docs)
- **Supports:** .pdf, .txt, .py, .md, .ipynb, .js, .ts

### RAG V3 (Custom Vector + BM25 + RRF + LLM Reranking)
- **Best for:** Research, maximum accuracy, fast queries
- **Strengths:** Advanced techniques (RRF fusion, contextual retrieval)
- **Speed:** 0.02s per query (100 docs) - **15x faster than V2!**
- **Supports:** .pdf, .txt, .py, .md, .ipynb, .json (newly added)

## Scale Recommendations

### Small Projects (< 50 documents)
- **Use:** Either V2 or V3
- **Indexing:** < 15s
- **Query:** < 0.1s
- **Memory:** ~500MB

### Medium Projects (50-500 documents)
- **Use:** V3 for speed, V2 for cloud deployment
- **Indexing:** 15-60s (one-time)
- **Query:** < 0.5s
- **Memory:** ~1-2GB

### Large Projects (500-5000 documents)
- **Use:** V2 with Qdrant Cloud
- **Indexing:** 1-5min (one-time)
- **Query:** < 2s
- **Memory:** ~2-4GB
- **Note:** Max 100 files enforced by default (configurable)

### Very Large Projects (> 5000 documents)
- **Use:** V2 with distributed Qdrant cluster
- **Considerations:**
  - Batch indexing recommended
  - Consider document chunking strategy
  - Monitor memory usage
  - Use Qdrant Cloud for scalability

## GPU Acceleration

### How GPU is Used

Both versions automatically detect and use GPU (CUDA) if available:

**V2:**
- Embeddings: `SentenceTransformer` on CUDA
- Reranking: `CrossEncoder` on CUDA
- Logs: `[RAG V2] Using device: cuda`

**V3:**
- Embeddings: `SentenceTransformer` on CUDA
- Reranking: `CrossEncoder` on CUDA
- Logs: `[RAG V3] Using device: cuda`

### Verifying GPU Usage

Check logs for:
```
[RAG V2/V3] Using device: cuda
```

If GPU not available, falls back to CPU automatically (slower but functional).

### GPU Requirements

- **NVIDIA GPU** with CUDA support
- **2GB+ VRAM** for small datasets (< 100 docs)
- **4GB+ VRAM** for medium datasets (100-500 docs)
- **8GB+ VRAM** for large datasets (> 500 docs)

## Performance Tips

### 1. Warm Up the System
First query includes model loading (~10-30s). Subsequent queries are fast.

### 2. Use Caching
Both versions cache loaded pipelines. Reusing same `doc_path` avoids re-indexing.

### 3. Exclude Patterns
Default excludes: `['test_*.py', '*_test.py', '*__pycache__*', '*.pyc']`

These patterns prevent test files from being indexed.

### 4. Batch Operations
For bulk indexing, use the Python API directly instead of repeated MCP calls.

### 5. Monitor Memory
- V2: Uses Qdrant Cloud (minimal local memory)
- V3: Uses local Qdrant (stores vectors in `rag_db_v3/`)

## Testing

Run benchmarks yourself:

```bash
python tests/benchmark_rag.py
```

Run unit tests:

```bash
pytest tests/test_rag_v2.py -v
pytest tests/test_rag_v3.py -v
```

## Troubleshooting

### Slow Queries
- Check if GPU is being used (`cuda` in logs)
- Verify model cache (`~/.cache/huggingface/`)
- Check network (V2 uses Qdrant Cloud)

### Memory Issues
- V3 stores full vector database locally
- Consider V2 with cloud backend for large datasets

### Indexing Failures
- Check file permissions
- Verify supported file types
- Check exclude patterns aren't blocking your files

## Architecture Comparison

### V2: Production Stack
```
Query → SentenceTransformer (CUDA) → Qdrant Cloud
     → BM25 Search → Ensemble
     → CrossEncoder Reranking (CUDA) → Results
```

### V3: Advanced Stack
```
Query → SentenceTransformer (CUDA) → Custom Vector Index
     → BM25 Search → RRF Fusion
     → CrossEncoder Reranking (CUDA)
     → Contextual Enhancement → Results
```

## Changelog

**2025-11-26:**
- Added .ipynb support to V3
- Fixed exclude patterns (now allows test fixtures)
- Benchmark results show V3 15x faster on warm queries
- Both versions verified to use GPU acceleration
