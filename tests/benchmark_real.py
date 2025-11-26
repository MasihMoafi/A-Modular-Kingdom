"""
Real-world RAG benchmark with actual documents and challenging queries
Tests both semantic understanding and keyword matching (BM25)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledge as fetchV3

# Real documents path
DOCS_PATH = Path(__file__).parent / "fixtures" / "real_docs"

# Real-world queries: mix of semantic + keyword matching
QUERIES = [
    # Semantic understanding (requires context)
    {
        "query": "How does the RAG system handle reranking and what models does it use?",
        "type": "semantic",
        "expected_docs": ["core_2.py", "core_3.py", "RAG_PERFORMANCE.md"]
    },
    {
        "query": "What is Napoleon's strategy for military campaigns?",
        "type": "semantic",
        "expected_docs": ["Napoleon.pdf"]
    },
    {
        "query": "Explain the memory system architecture and scoping mechanism",
        "type": "semantic",
        "expected_docs": ["memory_core.py", "host.py"]
    },
    {
        "query": "How does multi-intent RAG work and what are its components?",
        "type": "semantic",
        "expected_docs": ["multi_intent_rag.py"]
    },
    
    # Keyword matching (BM25 should excel)
    {
        "query": "CrossEncoder ms-marco-MiniLM",
        "type": "keyword",
        "expected_docs": ["core_2.py", "core_3.py"]
    },
    {
        "query": "Qdrant collection vector database",
        "type": "keyword",
        "expected_docs": ["core_2.py", "RAG_PERFORMANCE.md"]
    },
    {
        "query": "MCP protocol tools server",
        "type": "keyword",
        "expected_docs": ["host.py"]
    },
    {
        "query": "forex zigzag pattern trading",
        "type": "keyword",
        "expected_docs": ["forex_docs.md"]
    },
    
    # Mixed (semantic + keyword)
    {
        "query": "What GPU acceleration features are available and how do they improve performance?",
        "type": "mixed",
        "expected_docs": ["RAG_PERFORMANCE.md", "core_2.py", "core_3.py"]
    },
    {
        "query": "How to implement scoped memory with global and project-specific contexts?",
        "type": "mixed",
        "expected_docs": ["memory_core.py", "host.py"]
    }
]


def verify_results(query_info, result, version):
    """Check if expected documents are in results"""
    result_lower = result.lower()
    found = []
    missing = []
    
    for expected_doc in query_info["expected_docs"]:
        doc_name = expected_doc.replace(".py", "").replace(".md", "").replace(".pdf", "")
        if doc_name.lower() in result_lower:
            found.append(expected_doc)
        else:
            missing.append(expected_doc)
    
    accuracy = len(found) / len(query_info["expected_docs"]) if query_info["expected_docs"] else 0
    
    return {
        "found": found,
        "missing": missing,
        "accuracy": accuracy,
        "result_length": len(result)
    }


def benchmark_version(version_name, fetch_fn):
    """Benchmark a RAG version with real queries"""
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
        
        verification = verify_results(query_info, result, version_name)
        
        print(f"  ⏱️  Time: {elapsed:.2f}s")
        print(f"  ✓ Found: {len(verification['found'])}/{len(query_info['expected_docs'])} docs")
        print(f"  📊 Accuracy: {verification['accuracy']*100:.0f}%")
        print(f"  📝 Result: {verification['result_length']} chars")
        
        if verification['missing']:
            print(f"  ⚠️  Missing: {', '.join(verification['missing'])}")
        
        results.append({
            "query": query,
            "type": query_type,
            "time": elapsed,
            "accuracy": verification["accuracy"],
            "found": verification["found"],
            "missing": verification["missing"],
            "result_length": verification["result_length"]
        })
    
    # Calculate statistics
    avg_time = total_time / len(QUERIES)
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
    
    semantic_results = [r for r in results if r["type"] == "semantic"]
    keyword_results = [r for r in results if r["type"] == "keyword"]
    mixed_results = [r for r in results if r["type"] == "mixed"]
    
    semantic_acc = sum(r["accuracy"] for r in semantic_results) / len(semantic_results) if semantic_results else 0
    keyword_acc = sum(r["accuracy"] for r in keyword_results) / len(keyword_results) if keyword_results else 0
    mixed_acc = sum(r["accuracy"] for r in mixed_results) / len(mixed_results) if mixed_results else 0
    
    print(f"\n{'='*70}")
    print(f"{version_name} Summary")
    print(f"{'='*70}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time/Query: {avg_time:.2f}s")
    print(f"Overall Accuracy: {avg_accuracy*100:.1f}%")
    print(f"  - Semantic: {semantic_acc*100:.1f}%")
    print(f"  - Keyword: {keyword_acc*100:.1f}%")
    print(f"  - Mixed: {mixed_acc*100:.1f}%")
    
    return {
        "version": version_name,
        "total_time": total_time,
        "avg_time": avg_time,
        "overall_accuracy": avg_accuracy,
        "semantic_accuracy": semantic_acc,
        "keyword_accuracy": keyword_acc,
        "mixed_accuracy": mixed_acc,
        "results": results
    }


def main():
    """Run comprehensive real-world benchmarks"""
    print(f"\n{'#'*70}")
    print(f"# Real-World RAG Benchmark")
    print(f"# Documents: {len(list(DOCS_PATH.glob('*')))} files")
    print(f"# Queries: {len(QUERIES)} (semantic + keyword + mixed)")
    print(f"{'#'*70}")
    
    # Benchmark V2
    v2_results = benchmark_version("RAG V2", fetchV2)
    
    # Benchmark V3
    v3_results = benchmark_version("RAG V3", fetchV3)
    
    # Comparison
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}\n")
    print(f"{'Metric':<25} {'V2':<20} {'V3':<20}")
    print("-" * 70)
    print(f"{'Avg Time/Query':<25} {v2_results['avg_time']:<20.2f} {v3_results['avg_time']:<20.2f}")
    print(f"{'Overall Accuracy':<25} {v2_results['overall_accuracy']*100:<20.1f} {v3_results['overall_accuracy']*100:<20.1f}")
    print(f"{'Semantic Accuracy':<25} {v2_results['semantic_accuracy']*100:<20.1f} {v3_results['semantic_accuracy']*100:<20.1f}")
    print(f"{'Keyword Accuracy':<25} {v2_results['keyword_accuracy']*100:<20.1f} {v3_results['keyword_accuracy']*100:<20.1f}")
    print(f"{'Mixed Accuracy':<25} {v2_results['mixed_accuracy']*100:<20.1f} {v3_results['mixed_accuracy']*100:<20.1f}")
    
    # Winner determination
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    if v2_results['avg_time'] < v3_results['avg_time']:
        print(f"⚡ V2 is {v3_results['avg_time']/v2_results['avg_time']:.1f}x faster")
    else:
        print(f"⚡ V3 is {v2_results['avg_time']/v3_results['avg_time']:.1f}x faster")
    
    if v2_results['overall_accuracy'] > v3_results['overall_accuracy']:
        print(f"🎯 V2 is more accurate ({v2_results['overall_accuracy']*100:.1f}% vs {v3_results['overall_accuracy']*100:.1f}%)")
    else:
        print(f"🎯 V3 is more accurate ({v3_results['overall_accuracy']*100:.1f}% vs {v2_results['overall_accuracy']*100:.1f}%)")
    
    print("\nUse V2 for: Production apps, fast queries, cloud deployment")
    print("Use V3 for: Research, maximum accuracy, advanced retrieval")


if __name__ == "__main__":
    main()
