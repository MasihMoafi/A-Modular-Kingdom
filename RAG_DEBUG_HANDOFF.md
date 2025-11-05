# RAG Debugging Handoff

## Issues Found

### 1. CRITICAL BUG - Line 110 in fetch_2.py
```python
if not query_words or not topic:  # ❌ 'topic' is undefined
```
**Impact:** Causes crash when query has no keywords. Returns empty/silent failure.

### 2. GPU Not Used
**File:** `core_2.py` lines 14-22
- OllamaEmbeddings: No device specified (defaults to CPU)
- CrossEncoder: No device specified (defaults to CPU)

**Fix needed:**
```python
# Line 18
self.embeddings = OllamaEmbeddings(model=self.config.get("embed_model"), device='cuda')

# Line 22
self.reranker = CrossEncoder(self.config.get("reranker_model"), device='cuda')
```

### 3. Empty Tool Output
**Symptom:** `mcp__memory-host__query_knowledge_base` returns nothing
**Cause:** Likely silent crash from bug #1

## Debugging Checklist

- [ ] Fix `topic` undefined error (line 110 fetch_2.py)
- [ ] Add GPU device specification to embeddings
- [ ] Add GPU device specification to reranker
- [ ] Verify CUDA available: `torch.cuda.is_available()`
- [ ] Test with simple query: "What is prompt engineering?"
- [ ] Check all 3 RAG versions (v1, v2, v3) for same issues
- [ ] Verify files directory has content: `rag/files/`
- [ ] Test end-to-end from MCP tool call
- [ ] Add error logging to catch silent failures

## Files to Review

Priority:
1. `/A-Modular-Kingdom/rag/fetch_2.py` (main interface)
2. `/A-Modular-Kingdom/rag/core_2.py` (pipeline logic)
3. `/A-Modular-Kingdom/agent/host.py` (MCP integration)
4. `/A-Modular-Kingdom/rag/fetch.py` (v1 - compare)
5. `/A-Modular-Kingdom/rag/fetch_3.py` (v3 - compare)

## Test Query
```python
mcp__memory-host__query_knowledge_base(
    query="What are best practices for prompt engineering?",
    version="v2"
)
```

Expected: Should return document chunks about prompt engineering from files in `rag/files/`

## Context Notes
- Session used ~100k tokens on prompt engineering work
- New session recommended for full RAG audit (need ~100k for thorough debugging)
- Load `~/Juliette/CLAUDE.md` for master assistant mode
