# RAG Performance Tuning Guide

## Current Performance

- **FAQ Dataset (focused):** 100% accuracy
- **Mixed Documents:** 60-70% accuracy
- **Speed:** V3 is 1.5x faster than V2

## Tuning Parameters

### 1. Retrieval Parameters (fetch_2.py, fetch_3.py)

```python
CONFIG = {
    "top_k": 5,              # Number of chunks to retrieve
    "rerank_top_k": 5,       # Number after reranking
}
```

**To improve accuracy:**
- Increase `top_k` to 10-15 (retrieve more candidates)
- Keep `rerank_top_k` at 5 (final results)

### 2. BM25 Weight (for keyword matching)

Currently hybrid search uses equal weights. To favor keywords:

**In core_2.py:**
```python
# Line ~250: Adjust hybrid search weights
vector_weight = 0.4  # Decrease semantic
bm25_weight = 0.6    # Increase keyword
```

**In core_3.py (RRF fusion):**
```python
# Line ~300: Adjust RRF k parameter
"rrf_k": 60  # Lower = more aggressive fusion (default: 60)
```

### 3. Chunk Size

**In core_2.py and core_3.py:**
```python
chunk_size = 500      # Current default
chunk_overlap = 50    # Overlap between chunks
```

**To improve accuracy:**
- Increase to 800-1000 for more context
- Increase overlap to 100 for better continuity

### 4. Reranking Model

Current: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, 6 layers)

**For better accuracy (slower):**
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (12 layers)
- `cross-encoder/ms-marco-TinyBERT-L-6` (smaller, faster)

## Quick Wins

### Option 1: More Retrieval Candidates
```python
# In fetch_2.py and fetch_3.py
CONFIG = {
    "top_k": 10,         # Retrieve 10 instead of 5
    "rerank_top_k": 5,   # Still return 5 best
}
```
**Impact:** +10-15% accuracy, minimal speed cost

### Option 2: Increase BM25 Weight
For keyword-heavy queries (technical docs, FAQs):
```python
# Favor exact keyword matches
bm25_weight = 0.7
vector_weight = 0.3
```
**Impact:** +15-20% on keyword queries

### Option 3: Larger Chunks
```python
chunk_size = 800
chunk_overlap = 100
```
**Impact:** +5-10% accuracy, better context

## Testing Your Changes

Run benchmarks after tuning:

```bash
# Test with FAQ dataset
python tests/benchmark_faq_fixed.py

# Test with mixed documents
python tests/benchmark_challenging.py
```

## Advanced Techniques

### 1. Query Expansion
Rephrase query into multiple variations:
```python
queries = [
    original_query,
    f"Explain {original_query}",
    f"What is {original_query}",
]
# Search with all, merge results
```

### 2. Semantic Chunking
Instead of fixed-size chunks, split by:
- Paragraphs
- Sections (markdown headers)
- Semantic similarity

### 3. Metadata Filtering
Tag chunks with:
- Document type (code, docs, FAQ)
- Topic/category
- Date/version

Then filter before searching.

### 4. Two-Stage Retrieval
1. Fast retrieval (BM25 only) → 50 candidates
2. Slow reranking (CrossEncoder) → 5 results

## Recommended Settings

### For FAQs / Structured Data
```python
top_k = 10
bm25_weight = 0.7
chunk_size = 500
```

### For Technical Documentation
```python
top_k = 15
bm25_weight = 0.5
chunk_size = 800
```

### For Code Search
```python
top_k = 10
bm25_weight = 0.6
chunk_size = 1000  # Larger for full functions
```

## Monitoring

Track these metrics:
- **Accuracy:** % of correct answers
- **Latency:** Time per query
- **Relevance:** User feedback on results

## Next Steps

1. Start with Quick Win #1 (more candidates)
2. Test on your specific dataset
3. Adjust BM25 weight based on query type
4. Consider larger chunks if context is important
