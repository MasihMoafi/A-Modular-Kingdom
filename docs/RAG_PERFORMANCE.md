# RAG Performance Guide

## Real-World Benchmark Results

Tested with **9 real documents** (3MB total) including:
- Napoleon.pdf (3MB historical text)
- Python code files (core_2.py, core_3.py, host.py, memory_core.py, multi_intent_rag.py)
- Markdown documentation (RAG_PERFORMANCE.md, README_V2.md, forex_docs.md)

**10 challenging queries** testing semantic understanding, keyword matching, and mixed retrieval.

### Performance Summary

| Metric | V2 (Qdrant + BM25) | V3 (Custom + RRF) |
|--------|-------------------|-------------------|
| **Avg Time/Query** | 6.82s | **1.90s** (3.6x faster) |
| **Overall Accuracy** | **33.3%** | 30.0% |
| **Semantic Accuracy** | **45.8%** | 37.5% |
| **Keyword Accuracy** | 25.0% | 25.0% |
| **Mixed Accuracy** | 25.0% | 25.0% |
| **Cold Start** | 64.6s | 18.2s |
| **Warm Query** | 0.39-0.41s | 0.06-0.11s |

### Key Findings

1. **V3 is 3.6x faster** on average (1.90s vs 6.82s)
2. **V2 is slightly more accurate** (33.3% vs 30.0%)
3. **Both struggle with keyword-only queries** (25% accuracy)
4. **V2 better for semantic queries** (45.8% vs 37.5%)
5. **Warm queries are fast** for both (<0.5s)

## When to Use Each Version

### RAG V2 (Qdrant + BM25 + CrossEncoder)
- **Best for:** Semantic understanding, production apps
- **Strengths:** 
  - Better semantic accuracy (45.8%)
  - Cloud-ready (Qdrant Cloud)
  - Proven, stable
- **Speed:** 0.39s per warm query
- **Use when:** You need reliable semantic search

### RAG V3 (Custom Vector + BM25 + RRF + CrossEncoder)
- **Best for:** Speed-critical applications
- **Strengths:**
  - 3.6x faster overall
  - Advanced RRF fusion
  - Local storage
- **Speed:** 0.08s per warm query
- **Use when:** Speed is priority over accuracy

## Detailed Query Analysis

### Semantic Queries (Understanding Context)
**Example:** "How does the RAG system handle reranking and what models does it use?"

- **V2:** 33% accuracy (found core_2.py, missed core_3.py and docs)
- **V3:** 0% accuracy (missed all expected docs)
- **Winner:** V2 for semantic understanding

### Keyword Queries (Exact Matching)
**Example:** "CrossEncoder ms-marco-MiniLM"

- **V2:** 0% accuracy (missed both code files)
- **V3:** 0% accuracy (missed both code files)
- **Winner:** Tie (both need improvement)

### Mixed Queries (Semantic + Keywords)
**Example:** "What GPU acceleration features are available and how do they improve performance?"

- **V2:** 0% accuracy
- **V3:** 0% accuracy
- **Winner:** Tie (challenging for both)

## Performance Tips

### 1. Warm Up the System
First query includes model loading and indexing. Subsequent queries are much faster:
- **V2:** 64.6s cold → 0.39s warm (165x faster)
- **V3:** 18.2s cold → 0.08s warm (228x faster)

### 2. Use Appropriate Version
- **Semantic-heavy workload?** → Use V2
- **Speed-critical?** → Use V3
- **Keyword matching?** → Both need improvement, consider adding more BM25 weight

### 3. GPU Acceleration
Both versions use CUDA automatically:
- Embeddings: `SentenceTransformer` on CUDA
- Reranking: `CrossEncoder` on CUDA
- Check logs for: `[RAG V2/V3] Using device: cuda`

### 4. Document Chunking
- V2: 1230 chunks from 9 files
- V3: 1206 chunks from 9 files
- Similar chunking strategies

## Accuracy Challenges

Both versions struggled with:
1. **Multi-document queries** - Expected 3 docs, found 0-1
2. **Keyword-only queries** - BM25 not weighted enough
3. **Code file retrieval** - Python files harder to match

**Potential improvements:**
- Increase BM25 weight for keyword queries
- Better chunking for code files
- Hybrid scoring adjustments

## Testing

Run the real-world benchmark yourself:

```bash
python tests/benchmark_real.py
```

This tests with actual documents from the project and challenging real-world queries.

## Architecture Comparison

### V2 Architecture
```
Query → Qdrant Vector Search → BM25 → CrossEncoder Rerank → Results
        (Cloud/Local)           (Hybrid)  (GPU)
```

### V3 Architecture
```
Query → Vector Search → BM25 → RRF Fusion → CrossEncoder → Results
        (Local Qdrant)  (Hybrid) (Advanced)   (GPU)
```

## Recommendations

**For Production:**
- Use V2 if semantic accuracy is critical
- Use V3 if speed is priority
- Both are production-ready with GPU acceleration

**For Development:**
- V3 faster iteration (3.6x faster)
- V2 better for testing semantic understanding

**For Research:**
- V3 has more advanced techniques (RRF fusion)
- V2 more stable and proven

## Future Improvements

1. **Tune BM25 weights** - Improve keyword matching
2. **Better code chunking** - Improve Python file retrieval
3. **Hybrid scoring** - Balance semantic vs keyword
4. **Query expansion** - Handle multi-document queries better
5. **Contextual retrieval** - Add document context to chunks
