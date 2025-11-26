"""
FAQ Dataset Benchmark - Test RAG with known Q&A pairs
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.fetch_2 import fetchExternalKnowledge as fetchV2
from rag.fetch_3 import fetchExternalKnowledge as fetchV3

DOCS_PATH = Path(__file__).parent / "fixtures" / "real_docs"
FAQ_FILE = DOCS_PATH / "Ecommerce_FAQ_Chatbot_dataset.json"

# Load FAQ dataset
with open(FAQ_FILE) as f:
    faq_data = json.load(f)['questions']

# Select 10 diverse questions to test
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


def check_answer_quality(question, expected_answer, rag_result):
    """Check if RAG result contains the expected answer"""
    expected_lower = expected_answer.lower()
    result_lower = rag_result.lower()
    
    # Extract key phrases from expected answer
    key_phrases = []
    if "30 days" in expected_lower:
        key_phrases.append("30 days")
    if "3-5 business days" in expected_lower:
        key_phrases.append("3-5")
    if "customer support" in expected_lower:
        key_phrases.append("customer support")
    if "sign up" in expected_lower:
        key_phrases.append("sign up")
    if "loyalty program" in expected_lower:
        key_phrases.append("loyalty")
    
    # Check if key info is present
    found = sum(1 for phrase in key_phrases if phrase in result_lower)
    
    # Also check if the exact answer is there
    if expected_answer[:50].lower() in result_lower:
        return 1.0, "✓ Exact match"
    elif found >= len(key_phrases) * 0.5:
        return 0.7, f"⚠ Partial ({found}/{len(key_phrases)} phrases)"
    else:
        return 0.0, "✗ Wrong answer"


def test_rag(version_name, fetch_fn):
    """Test RAG with FAQ questions"""
    print(f"\n{'='*80}")
    print(f"Testing {version_name}")
    print(f"{'='*80}\n")
    
    scores = []
    
    for i, qa in enumerate(TEST_QUESTIONS, 1):
        question = qa['question']
        expected = qa['answer']
        
        print(f"[{i}/{len(TEST_QUESTIONS)}] Q: {question}")
        print(f"  Expected: {expected[:80]}...")
        
        result = fetch_fn(question, doc_path=str(DOCS_PATH))
        score, status = check_answer_quality(question, expected, result)
        
        print(f"  RAG: {result[:100]}...")
        print(f"  {status} (Score: {score})")
        print()
        
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    print(f"\n{'='*80}")
    print(f"{version_name} Average Score: {avg_score*100:.1f}%")
    print(f"{'='*80}\n")
    
    return avg_score


def main():
    """Run FAQ benchmark"""
    print(f"\n{'#'*80}")
    print(f"# FAQ DATASET BENCHMARK")
    print(f"# Dataset: {len(faq_data)} Q&A pairs")
    print(f"# Testing: {len(TEST_QUESTIONS)} questions")
    print(f"{'#'*80}\n")
    
    # Test V2
    print("\n" + "*"*80)
    print("TESTING RAG V2")
    print("*"*80)
    v2_score = test_rag("RAG V2", fetchV2)
    
    # Test V3
    print("\n" + "*"*80)
    print("TESTING RAG V3")
    print("*"*80)
    v3_score = test_rag("RAG V3", fetchV3)
    
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
