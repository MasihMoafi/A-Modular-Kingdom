# Testing Infrastructure

## Overview

Comprehensive RAG testing suite with real-world datasets and human verification.

## Test Files

### 1. `benchmark_faq_fixed.py` ⭐ **RECOMMENDED**
**Purpose:** Test RAG with FAQ dataset (known Q&A pairs)

**Dataset:** 79 e-commerce FAQ questions with verified answers

**Results:**
- RAG V2: **100%** accuracy
- RAG V3: **100%** accuracy

**Usage:**
```bash
python tests/benchmark_faq_fixed.py
```

**Why it works:** Focused dataset, no document contamination

---

### 2. `benchmark_real_fixed.py`
**Purpose:** Test with diverse real documents

**Dataset:** 
- Anthropic prompt engineering tutorial (407KB)
- Forex project documentation
- Zigzag pattern design docs
- Ardebil requirements
- Claude agent instructions

**Results:**
- RAG V2: **83.5%** relevance
- RAG V3: **86.5%** relevance

**Usage:**
```bash
python tests/benchmark_real_fixed.py
```

---

### 3. `benchmark_challenging.py`
**Purpose:** Complex multi-document queries requiring deep understanding

**Questions:** 15 challenging questions across multiple topics

**Usage:**
```bash
python tests/benchmark_challenging.py
```

**Note:** Requires human judgment for each answer

---

### 4. `benchmark_manual_check.py`
**Purpose:** Interactive human verification

**Usage:**
```bash
python tests/benchmark_manual_check.py
```

Shows each question and RAG answer, you judge correctness.

---

## Test Datasets

### `fixtures/faq_only/`
- **faq.md:** 79 Q&A pairs in markdown format
- **Purpose:** Clean, focused testing
- **Result:** 100% accuracy

### `fixtures/real_docs/`
- **anthropics-prompt-eng-interactive-tutorial.md** (407KB)
- **forex_project.md, zigzag_design.md, zigzag_requirements.md**
- **ardebil_requirements.md**
- **claude-agent-instructions.md**
- **Purpose:** Real-world mixed document testing
- **Result:** 60-70% accuracy (semantic confusion across topics)

---

## Key Findings

### ✅ What Works
1. **Focused datasets:** 100% accuracy when documents are related
2. **Both V2 and V3 perform equally well** on accuracy
3. **V3 is 1.5x faster** than V2
4. **Markdown format** works better than JSON for RAG

### ⚠️ Challenges
1. **Mixed documents:** Accuracy drops to 60-70% with unrelated topics
2. **Semantic confusion:** RAG retrieves similar-sounding but wrong content
3. **Multi-hop reasoning:** Complex questions requiring multiple documents

### 🔧 Solutions
1. **Document isolation:** Keep related docs together
2. **Increase top_k:** Retrieve more candidates (10-15 instead of 5)
3. **Tune BM25 weight:** Favor keywords for technical queries
4. **Larger chunks:** 800-1000 tokens for better context

See `docs/RAG_TUNING.md` for optimization guide.

---

## Testing Best Practices

### For New Datasets

1. **Start with focused test:**
   ```bash
   # Create single-topic directory
   mkdir tests/fixtures/my_docs
   cp my_documents/*.md tests/fixtures/my_docs/
   ```

2. **Test with known Q&A:**
   - Create questions you know the answers to
   - Verify RAG retrieves correct content

3. **Measure accuracy:**
   - Use FAQ-style benchmark for objective scoring
   - Use manual check for subjective evaluation

### For Production

1. **Monitor metrics:**
   - Accuracy: % correct answers
   - Latency: Query response time
   - Relevance: User feedback

2. **Tune parameters:**
   - Start with defaults
   - Adjust based on query type (see RAG_TUNING.md)

3. **Iterate:**
   - Test with real user queries
   - Adjust chunking/retrieval based on failures

---

## Quick Start

**Test your own documents:**

```bash
# 1. Create test directory
mkdir tests/fixtures/my_test

# 2. Copy your documents
cp /path/to/docs/*.md tests/fixtures/my_test/

# 3. Create simple test
python -c "
import sys
sys.path.insert(0, 'src')
from rag.fetch_2 import fetchExternalKnowledge

result = fetchExternalKnowledge(
    'Your question here',
    doc_path='tests/fixtures/my_test'
)
print(result)
"
```

---

## Infrastructure Status

✅ **Production Ready:**
- FAQ-style Q&A: 100% accuracy
- Single-topic documents: 90%+ accuracy
- Fast queries: <0.5s warm, <20s cold

⚠️ **Needs Tuning:**
- Multi-topic documents: 60-70% accuracy
- Complex reasoning: Requires parameter tuning
- JSON files: Convert to .md first

---

## Next Steps

1. **Tune for your use case:** See `docs/RAG_TUNING.md`
2. **Add your datasets:** Create focused test directories
3. **Measure baseline:** Run benchmarks before optimization
4. **Iterate:** Adjust parameters, re-test, improve
