"""
Human-verifiable RAG benchmark
Shows actual results so YOU can judge if they're correct
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledge as fetchV3

DOCS_PATH = Path(__file__).parent / "fixtures" / "real_docs"

# Questions with VERIFIABLE answers from the documents
QUESTIONS = [
    {
        "q": "What embedding model does RAG V2 use?",
        "expected": "all-MiniLM-L6-v2 or SentenceTransformer",
        "file": "core_2.py"
    },
    {
        "q": "What reranking model is used in RAG V3?",
        "expected": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "file": "core_3.py"
    },
    {
        "q": "What is Napoleon's view on offensive vs defensive warfare?",
        "expected": "Every offensive war is invasion warfare; defensive warfare does not exclude attacking",
        "file": "Napoleon.pdf"
    },
    {
        "q": "What device does RAG use for GPU acceleration?",
        "expected": "cuda or CUDA",
        "file": "core_2.py or core_3.py"
    },
    {
        "q": "What MCP tools are exposed by the host server?",
        "expected": "query_knowledge_base, save_memory, search_memories, etc.",
        "file": "host.py"
    },
    {
        "q": "What are the memory scopes in the system?",
        "expected": "global_rules, global_preferences, project_context, etc.",
        "file": "memory_core.py"
    },
    {
        "q": "What is the forex zigzag pattern used for?",
        "expected": "Trading pattern detection or technical analysis",
        "file": "forex_docs.md"
    },
    {
        "q": "What is RRF fusion in RAG V3?",
        "expected": "Reciprocal Rank Fusion - combines vector and BM25 results",
        "file": "core_3.py"
    }
]


def test_query(version_name, fetch_fn, question):
    """Test a single query and show results"""
    print(f"\n{'='*80}")
    print(f"[{version_name}] Question: {question['q']}")
    print(f"Expected answer: {question['expected']}")
    print(f"Should be in: {question['file']}")
    print(f"{'='*80}")
    
    result = fetch_fn(question['q'], doc_path=str(DOCS_PATH))
    
    print(f"\n📄 RESULT (first 800 chars):")
    print("-" * 80)
    print(result[:800])
    if len(result) > 800:
        print("\n... (truncated) ...\n")
        print(result[-200:])
    print("-" * 80)
    
    # Let user verify
    print(f"\n❓ YOUR JUDGMENT:")
    print(f"   Does this answer the question correctly? (You decide!)")
    print(f"   Expected: {question['expected']}")
    
    return result


def main():
    """Run manual verification benchmark"""
    print(f"\n{'#'*80}")
    print(f"# MANUAL VERIFICATION BENCHMARK")
    print(f"# You will see each question and the RAG's answer")
    print(f"# Judge for yourself if the answer is correct!")
    print(f"{'#'*80}")
    
    print(f"\nDocuments indexed: {len(list(DOCS_PATH.glob('*')))} files")
    print(f"Questions: {len(QUESTIONS)}")
    
    # Test V2
    print(f"\n\n{'*'*80}")
    print(f"TESTING RAG V2")
    print(f"{'*'*80}")
    
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[Question {i}/{len(QUESTIONS)}]")
        test_query("V2", fetchV2, q)
        input("\nPress Enter to continue to next question...")
    
    # Test V3
    print(f"\n\n{'*'*80}")
    print(f"TESTING RAG V3")
    print(f"{'*'*80}")
    
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[Question {i}/{len(QUESTIONS)}]")
        test_query("V3", fetchV3, q)
        input("\nPress Enter to continue to next question...")
    
    print(f"\n\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print("Now YOU tell me: Which version answered better?")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
