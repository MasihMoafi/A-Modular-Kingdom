"""
FAQ Dataset Benchmark - FIXED to use only FAQ file
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledge as fetchV3

# Use FAQ-ONLY directory
DOCS_PATH = Path(__file__).parent / "fixtures" / "faq_only"
FAQ_FILE = DOCS_PATH / "Ecommerce_FAQ_Chatbot_dataset.json"

with open(FAQ_FILE) as f:
    faq_data = json.load(f)['questions']

# Test 10 diverse questions
TEST_QUESTIONS = [
    faq_data[0],   # How can I create an account?
    faq_data[5],   # How long does shipping take?
    faq_data[11],  # What is your price matching policy?
    faq_data[15],  # Do you have a loyalty program?
    faq_data[20],  # Can I use multiple promo codes?
    faq_data[25],  # Can I return if I changed my mind?
    faq_data[30],  # Do you offer installation services?
    faq_data[40],  # Can I request product reservation?
    faq_data[50],  # Can I return damaged product?
    faq_data[60],  # Can I request custom order?
]


def check_answer(expected, result):
    """Simple check: is expected answer in result?"""
    # Check if first 50 chars of expected answer appear in result
    expected_snippet = expected[:50].lower()
    result_lower = result.lower()
    
    if expected_snippet in result_lower:
        return 1.0, "✓ Correct"
    elif expected[:20].lower() in result_lower:
        return 0.5, "⚠ Partial"
    else:
        return 0.0, "✗ Wrong"


def test_rag(version_name, fetch_fn):
    """Test RAG with FAQ questions"""
    print(f"\n{'='*80}")
    print(f"Testing {version_name}")
    print(f"{'='*80}\n")
    
    scores = []
    
    for i, qa in enumerate(TEST_QUESTIONS, 1):
        question = qa['question']
        expected = qa['answer']
        
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
        
        result = fetch_fn(question, doc_path=str(DOCS_PATH))
        score, status = check_answer(expected, result)
        
        print(f"  Expected: {expected[:60]}...")
        print(f"  Got: {result[:80]}...")
        print(f"  {status} (Score: {score})\n")
        
        scores.append(score)
    
    avg = sum(scores) / len(scores)
    print(f"\n{'='*80}")
    print(f"{version_name} Score: {avg*100:.1f}%")
    print(f"{'='*80}\n")
    
    return avg


def main():
    print(f"\n{'#'*80}")
    print(f"# FAQ BENCHMARK (FIXED - FAQ ONLY)")
    print(f"# Path: {DOCS_PATH}")
    print(f"# Files: {list(DOCS_PATH.glob('*'))}")
    print(f"# Questions: {len(TEST_QUESTIONS)}")
    print(f"{'#'*80}\n")
    
    v2_score = test_rag("RAG V2", fetchV2)
    v3_score = test_rag("RAG V3", fetchV3)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"RAG V2: {v2_score*100:.1f}%")
    print(f"RAG V3: {v3_score*100:.1f}%")
    
    if v2_score > v3_score:
        print(f"\n🏆 V2 wins by {(v2_score-v3_score)*100:.1f} points")
    elif v3_score > v2_score:
        print(f"\n🏆 V3 wins by {(v3_score-v2_score)*100:.1f} points")
    else:
        print(f"\n🤝 Tie at {v2_score*100:.1f}%")


if __name__ == "__main__":
    main()
