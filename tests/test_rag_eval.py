"""
RAG Evaluation with LLM-as-Judge

Metrics:
- Groundedness: Is retrieved content factual and from source?
- Relevance: Does it answer the question?
- Completeness: Is it comprehensive?

Results (qwen3:8b judge):
- V2: 95% average (G:99% R:100% C:88%)
- V3: 96% average (G:95% R:95% C:98%)
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ollama

# Curated test queries - specific, not general
EVAL_QUERIES = [
    {
        "query": "What year was Napoleon crowned Emperor of France?",
        "expected": "1804",
        "doc": "pdf",
    },
    {
        "query": "Where was Napoleon exiled after Waterloo?",
        "expected": "Saint Helena",
        "doc": "pdf",
    },
    {
        "query": "What is the recommended chunk size for RAG in the documentation?",
        "expected": "700",
        "doc": "md",
    },
    {
        "query": "What reranking model does V2 use?",
        "expected": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "doc": "md",
    },
]


def judge_with_llm(query: str, retrieved: str, expected: str) -> dict:
    """Use qwen3:8b as judge"""

    prompt = f"""You are evaluating RAG (Retrieval-Augmented Generation) quality.

QUERY: {query}
EXPECTED ANSWER: {expected}

RETRIEVED CONTEXT:
{retrieved[:2500]}

Rate 0.0 to 1.0:
1. GROUNDEDNESS: Is the retrieved text from a real document (not hallucinated)?
2. RELEVANCE: Does the context contain information to answer the query?
3. COMPLETENESS: Does it have the expected answer or equivalent?

Return JSON only:
{{"groundedness": 0.X, "relevance": 0.X, "completeness": 0.X, "found_answer": true/false, "reasoning": "brief"}}"""

    try:
        response = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        text = response["message"]["content"]
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "groundedness": 0, "relevance": 0, "completeness": 0}


def run_eval(version: str = "v2"):
    """Run evaluation for specified RAG version"""

    if version == "v2":
        from rag.fetch_2 import fetchExternalKnowledge as fetch
    else:
        from rag.fetch_3 import fetchExternalKnowledgeV3 as fetch

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = os.path.join(project_root, "src", "rag")

    print(f"\n{'='*60}")
    print(f"RAG {version.upper()} EVALUATION - Judge: qwen3:8b")
    print(f"{'='*60}\n")

    results = []

    for item in EVAL_QUERIES:
        print(f"Q: {item['query'][:50]}...")

        start = time.time()
        if item["doc"] == "md":
            retrieved = fetch(item["query"], doc_path=md_path)
        else:
            retrieved = fetch(item["query"])
        latency = (time.time() - start) * 1000

        scores = judge_with_llm(item["query"], retrieved or "", item["expected"])

        if "error" in scores:
            print(f"   ERROR: {scores['error']}")
            continue

        avg = (scores["groundedness"] + scores["relevance"] + scores["completeness"]) / 3
        icon = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
        found = "✓" if scores.get("found_answer") else "✗"

        print(f"   {icon} G:{scores['groundedness']:.0%} R:{scores['relevance']:.0%} C:{scores['completeness']:.0%} | Answer:{found} | {latency:.0f}ms")

        results.append({
            "query": item["query"],
            "groundedness": scores["groundedness"],
            "relevance": scores["relevance"],
            "completeness": scores["completeness"],
            "found_answer": scores.get("found_answer", False),
            "latency_ms": latency,
        })

    if results:
        avg_g = sum(r["groundedness"] for r in results) / len(results)
        avg_r = sum(r["relevance"] for r in results) / len(results)
        avg_c = sum(r["completeness"] for r in results) / len(results)
        found_count = sum(1 for r in results if r["found_answer"])

        print(f"\n{'='*60}")
        print(f"SUMMARY - RAG {version.upper()}")
        print(f"{'='*60}")
        print(f"Groundedness:  {avg_g:.0%}")
        print(f"Relevance:     {avg_r:.0%}")
        print(f"Completeness:  {avg_c:.0%}")
        print(f"Average:       {(avg_g + avg_r + avg_c) / 3:.0%}")
        print(f"Answers Found: {found_count}/{len(results)}")
        print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v2", "v3", "both"], default="both")
    args = parser.parse_args()

    if args.version == "both":
        run_eval("v2")
        run_eval("v3")
    else:
        run_eval(args.version)
