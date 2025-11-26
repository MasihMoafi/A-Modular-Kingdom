"""
Real-world RAG benchmark with PROPER content verification
Checks if the CONTENT is relevant, not just filenames
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledge as fetchV3

DOCS_PATH = Path(__file__).parent / "fixtures" / "real_docs"

# Real-world queries with CONTENT-BASED verification
QUERIES = [
    {
        "query": "How does the RAG system handle reranking and what models does it use?",
        "type": "semantic",
        "keywords": ["rerank", "CrossEncoder", "ms-marco", "score"]
    },
    {
        "query": "What is Napoleon's strategy for military campaigns?",
        "type": "semantic",
        "keywords": ["Napoleon", "military", "campaign", "strategy", "battle"]
    },
    {
        "query": "Explain the memory system architecture and scoping mechanism",
        "type": "semantic",
        "keywords": ["memory", "scope", "global", "project", "context"]
    },
    {
        "query": "How does multi-intent RAG work and what are its components?",
        "type": "semantic",
        "keywords": ["multi", "intent", "RAG", "component"]
    },
    {
        "query": "CrossEncoder ms-marco-MiniLM",
        "type": "keyword",
        "keywords": ["CrossEncoder", "ms-marco", "MiniLM", "rerank"]
    },
    {
        "query": "Qdrant collection vector database",
        "type": "keyword",
        "keywords": ["Qdrant", "collection", "vector", "database"]
    },
    {
        "query": "MCP protocol tools server",
        "type": "keyword",
        "keywords": ["MCP", "protocol", "tool", "server"]
    },
    {
        "query": "forex zigzag pattern trading",
        "type": "keyword",
        "keywords": ["forex", "zigzag", "pattern", "trading"]
    },
    {
        "query": "What GPU acceleration features are available and how do they improve performance?",
        "type": "mixed",
        "keywords": ["GPU", "CUDA", "acceleration", "performance", "device"]
    },
    {
        "query": "How to implement scoped memory with global and project-specific contexts?",
        "type": "mixed",
        "keywords": ["scope", "memory", "global", "project", "context"]
    }
]


def verify_content_relevance(query_info, result):
    """Check if result contains relevant keywords (case-insensitive)"""
    result_lower = result.lower()
    found_keywords = []
    missing_keywords = []
    
    for keyword in query_info["keywords"]:
        if keyword.lower() in result_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    relevance = len(found_keywords) / len(query_info["keywords"]) if query_info["keywords"] else 0
    
    return {
        "found": found_keywords,
        "missing": missing_keywords,
        "relevance": relevance,
        "result_length": len(result)
    }


def benchmark_version(version_name, fetch_fn):
    """Benchmark a RAG version with content-based verification"""
    print(f"\n{'='*70}")
    print(f"Benchmarking {version_name}")
    print(f"{'='*70}")
    
    results = []
    total_time = 0
    
    for i, query_info in enumerate(QUERIES, 1):
        query = query_info["query"]
        query_type = query_info["type"]
        
        print(f"\n[{i}/{len(QUERIES)}] {query_type.upper()}: {query[:60]}...")
        
        start = time.time()
        result = fetch_fn(query, doc_path=str(DOCS_PATH))
        elapsed = time.time() - start
        total_time += elapsed
        
        verification = verify_content_relevance(query_info, result)
        
        print(f"  ⏱️  Time: {elapsed:.2f}s")
        print(f"  ✓ Keywords: {len(verification['found'])}/{len(query_info['keywords'])}")
        print(f"  📊 Relevance: {verification['relevance']*100:.0f}%")
        print(f"  📝 Result: {verification['result_length']} chars")
        
        if verification['missing']:
            print(f"  ⚠️  Missing: {', '.join(verification['missing'])}")
        
        results.append({
            "query": query,
            "type": query_type,
            "time": elapsed,
            "relevance": verification["relevance"],
            "found": verification["found"],
            "missing": verification["missing"]
        })
    
    # Statistics
    avg_time = total_time / len(QUERIES)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    
    semantic_results = [r for r in results if r["type"] == "semantic"]
    keyword_results = [r for r in results if r["type"] == "keyword"]
    mixed_results = [r for r in results if r["type"] == "mixed"]
    
    semantic_rel = sum(r["relevance"] for r in semantic_results) / len(semantic_results) if semantic_results else 0
    keyword_rel = sum(r["relevance"] for r in keyword_results) / len(keyword_results) if keyword_results else 0
    mixed_rel = sum(r["relevance"] for r in mixed_results) / len(mixed_results) if mixed_results else 0
    
    print(f"\n{'='*70}")
    print(f"{version_name} Summary")
    print(f"{'='*70}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time/Query: {avg_time:.2f}s")
    print(f"Overall Relevance: {avg_relevance*100:.1f}%")
    print(f"  - Semantic: {semantic_rel*100:.1f}%")
    print(f"  - Keyword: {keyword_rel*100:.1f}%")
    print(f"  - Mixed: {mixed_rel*100:.1f}%")
    
    return {
        "version": version_name,
        "total_time": total_time,
        "avg_time": avg_time,
        "overall_relevance": avg_relevance,
        "semantic_relevance": semantic_rel,
        "keyword_relevance": keyword_rel,
        "mixed_relevance": mixed_rel,
        "results": results
    }


def main():
    """Run fixed benchmarks"""
    print(f"\n{'#'*70}")
    print(f"# Real-World RAG Benchmark (FIXED)")
    print(f"# Documents: {len(list(DOCS_PATH.glob('*')))} files")
    print(f"# Queries: {len(QUERIES)} (content-based verification)")
    print(f"{'#'*70}")
    
    v2_results = benchmark_version("RAG V2", fetchV2)
    v3_results = benchmark_version("RAG V3", fetchV3)
    
    # Comparison
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}\n")
    print(f"{'Metric':<25} {'V2':<20} {'V3':<20}")
    print("-" * 70)
    print(f"{'Avg Time/Query':<25} {v2_results['avg_time']:<20.2f} {v3_results['avg_time']:<20.2f}")
    print(f"{'Overall Relevance':<25} {v2_results['overall_relevance']*100:<20.1f} {v3_results['overall_relevance']*100:<20.1f}")
    print(f"{'Semantic Relevance':<25} {v2_results['semantic_relevance']*100:<20.1f} {v3_results['semantic_relevance']*100:<20.1f}")
    print(f"{'Keyword Relevance':<25} {v2_results['keyword_relevance']*100:<20.1f} {v3_results['keyword_relevance']*100:<20.1f}")
    print(f"{'Mixed Relevance':<25} {v2_results['mixed_relevance']*100:<20.1f} {v3_results['mixed_relevance']*100:<20.1f}")
    
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}\n")
    
    if v2_results['avg_time'] < v3_results['avg_time']:
        print(f"⚡ V2 is {v3_results['avg_time']/v2_results['avg_time']:.1f}x faster")
    else:
        print(f"⚡ V3 is {v2_results['avg_time']/v3_results['avg_time']:.1f}x faster")
    
    if v2_results['overall_relevance'] > v3_results['overall_relevance']:
        print(f"🎯 V2 is more relevant ({v2_results['overall_relevance']*100:.1f}% vs {v3_results['overall_relevance']*100:.1f}%)")
    else:
        print(f"🎯 V3 is more relevant ({v3_results['overall_relevance']*100:.1f}% vs {v2_results['overall_relevance']*100:.1f}%)")


if __name__ == "__main__":
    main()
