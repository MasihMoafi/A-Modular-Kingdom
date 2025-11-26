#!/usr/bin/env python3
"""
RAG Quality Showcase - Direct Evidence

Shows what RAG actually retrieves with clear evidence of quality.
No LLM judge needed - the results speak for themselves.
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def showcase_rag():
    """Run showcase queries and display results"""

    from rag.fetch_2 import fetchExternalKnowledge as fetchV2
    from rag.fetch_3 import fetchExternalKnowledgeV3 as fetchV3

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    real_docs = os.path.join(project_root, "tests", "fixtures", "real_docs")

    # Test cases with expected evidence
    tests = [
        {
            "name": "Napoleon Military",
            "query": "Napoleon's military campaigns and battles",
            "doc_path": "",
            "expected": ["battle", "war", "campaign", "army", "victory"],
        },
        {
            "name": "Napoleon Exile",
            "query": "What happened to Napoleon after his defeat?",
            "doc_path": "",
            "expected": ["elba", "helena", "exile", "island", "death"],
        },
        {
            "name": "Prompt Engineering",
            "query": "How to write effective prompts for Claude",
            "doc_path": real_docs,
            "expected": ["prompt", "claude", "instruction", "example"],
        },
        {
            "name": "Project Requirements",
            "query": "zigzag project design requirements",
            "doc_path": real_docs,
            "expected": ["zigzag", "requirement", "design", "feature"],
        },
    ]

    print("\n" + "="*80)
    print("RAG QUALITY SHOWCASE - A-Modular-Kingdom")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)

    results = {"v2": [], "v3": []}

    for test in tests:
        print(f"\n{'─'*80}")
        print(f"📋 TEST: {test['name']}")
        print(f"   Query: \"{test['query']}\"")
        print(f"{'─'*80}")

        for version, fetch_fn in [("V2", fetchV2), ("V3", fetchV3)]:
            print(f"\n   🔍 RAG {version}:")

            start = time.time()
            try:
                if test['doc_path']:
                    result = fetch_fn(test['query'], doc_path=test['doc_path'])
                else:
                    result = fetch_fn(test['query'])
            except Exception as e:
                result = f"ERROR: {e}"
            latency = (time.time() - start) * 1000

            # Check for expected keywords
            result_lower = (result or "").lower()
            found = [kw for kw in test['expected'] if kw.lower() in result_lower]
            score = len(found) / len(test['expected']) if test['expected'] else 0

            # Display results
            print(f"      ⏱️  Latency: {latency:.0f}ms")
            print(f"      📊 Keywords found: {len(found)}/{len(test['expected'])} ({score:.0%})")
            print(f"      ✓  Found: {', '.join(found) if found else 'None'}")
            missing = [kw for kw in test['expected'] if kw.lower() not in result_lower]
            if missing:
                print(f"      ✗  Missing: {', '.join(missing)}")

            # Show snippet of retrieved content
            if result and len(result) > 100:
                snippet = result[:300].replace('\n', ' ')
                print(f"      📄 Retrieved: \"{snippet}...\"")

            results[version.lower()].append({
                "test": test['name'],
                "score": score,
                "latency": latency,
                "found": found,
            })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for version in ["v2", "v3"]:
        scores = [r['score'] for r in results[version]]
        latencies = [r['latency'] for r in results[version]]
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\n   RAG {version.upper()}:")
        print(f"      Average Score:   {avg_score:.0%}")
        print(f"      Average Latency: {avg_latency:.0f}ms")
        print(f"      Tests Passed:    {sum(1 for s in scores if s >= 0.6)}/{len(scores)}")

    # Winner
    v2_avg = sum(r['score'] for r in results['v2']) / len(results['v2'])
    v3_avg = sum(r['score'] for r in results['v3']) / len(results['v3'])

    print(f"\n   🏆 WINNER: RAG {'V2' if v2_avg >= v3_avg else 'V3'} ({max(v2_avg, v3_avg):.0%} accuracy)")
    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    showcase_rag()
