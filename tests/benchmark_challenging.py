"""
Challenging RAG benchmark with real documentation
Questions require deep understanding and multi-hop reasoning
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledge as fetchV3

DOCS_PATH = Path(__file__).parent / "fixtures" / "real_docs"

# Challenging questions from actual documentation
QUESTIONS = [
    # Anthropic prompt engineering
    "What are the key techniques for reducing hallucinations in Claude according to Anthropic's guide?",
    "Explain the difference between zero-shot and few-shot prompting with examples from the tutorial",
    "What is chain-of-thought prompting and when should it be used?",
    "How does XML tagging improve prompt clarity and why is it recommended?",
    
    # Forex zigzag project
    "What are the three main components of the zigzag pattern detection system?",
    "Explain the difference between swing highs and swing lows in the zigzag algorithm",
    "What MT5 functions are used for drawing zigzag lines and what are their parameters?",
    "What are the known issues with the current zigzag implementation?",
    
    # Ardebil multi-intent RAG
    "What is the purpose of the multi-intent RAG system in the Ardebil project?",
    "How does the system handle multiple user intents in a single query?",
    "What evaluation metrics are used to measure RAG performance?",
    
    # Claude agent instructions
    "What are the best practices for structuring agent instructions according to the guide?",
    "How should agents handle ambiguous user requests?",
    "What safety considerations should be implemented in agent systems?"
]


def test_rag(version_name, fetch_fn):
    """Test RAG with challenging questions"""
    print(f"\n{'='*80}")
    print(f"Testing {version_name}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question[:70]}...")
        
        result = fetch_fn(question, doc_path=str(DOCS_PATH))
        
        # Show snippet
        print(f"  📄 Answer (first 300 chars):")
        print(f"     {result[:300].replace(chr(10), ' ')}")
        
        # Manual judgment
        print(f"\n  ❓ Does this answer the question? (y/n/partial): ", end='')
        judgment = input().strip().lower()
        
        score = {"y": 1.0, "yes": 1.0, "p": 0.5, "partial": 0.5, "n": 0.0, "no": 0.0}.get(judgment, 0.0)
        results.append({"question": question, "score": score, "result": result})
        
        print()
    
    # Summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n{'='*80}")
    print(f"{version_name} Score: {avg_score*100:.1f}%")
    print(f"{'='*80}\n")
    
    return results, avg_score


def main():
    """Run challenging benchmark"""
    print(f"\n{'#'*80}")
    print(f"# CHALLENGING RAG BENCHMARK")
    print(f"# Real documentation: {len(list(DOCS_PATH.glob('*')))} files")
    print(f"# Questions: {len(QUESTIONS)} (requires deep understanding)")
    print(f"{'#'*80}\n")
    
    print("You will judge each answer as:")
    print("  y/yes = correct (1.0)")
    print("  p/partial = partially correct (0.5)")
    print("  n/no = wrong (0.0)")
    
    # Test V2
    print(f"\n\n{'*'*80}")
    print("TESTING RAG V2")
    print(f"{'*'*80}")
    v2_results, v2_score = test_rag("RAG V2", fetchV2)
    
    # Test V3
    print(f"\n\n{'*'*80}")
    print("TESTING RAG V3")
    print(f"{'*'*80}")
    v3_results, v3_score = test_rag("RAG V3", fetchV3)
    
    # Final comparison
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"RAG V2: {v2_score*100:.1f}%")
    print(f"RAG V3: {v3_score*100:.1f}%")
    
    if v2_score > v3_score:
        print(f"\n🏆 V2 wins by {(v2_score-v3_score)*100:.1f} points")
    elif v3_score > v2_score:
        print(f"\n🏆 V3 wins by {(v3_score-v2_score)*100:.1f} points")
    else:
        print(f"\n🤝 Tie!")


if __name__ == "__main__":
    main()
