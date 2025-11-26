"""
RAG Evaluation Framework - Gemini as Judge

Uses Gemini 2.0 Flash for objective, high-quality evaluation.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'src', '.env'))

from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class EvalResult:
    query: str
    rag_version: str
    retrieved_text: str
    groundedness: float
    relevance: float
    completeness: float
    latency_ms: float
    judge_reasoning: str


class GeminiRAGEvaluator:
    """Gemini-powered RAG evaluation"""

    # Fair, practical queries that RAG should handle well
    TEST_QUERIES = [
        # PDF queries (Napoleon.pdf)
        ("Tell me about Napoleon's military campaigns", "pdf"),
        ("What happened to Napoleon after Waterloo?", "pdf"),
        ("Describe Napoleon's relationship with Josephine", "pdf"),

        # MD queries (real_docs)
        ("Explain the key principles of prompt engineering", "md"),
        ("What are the main requirements for the zigzag project?", "md"),
        ("Summarize the forex trading approach", "md"),
    ]

    def __init__(self, rag_version: str = "v2"):
        self.rag_version = rag_version
        self.results: List[EvalResult] = []

        if rag_version == "v2":
            from rag.fetch_2 import fetchExternalKnowledge
            self.fetch = fetchExternalKnowledge
        else:
            from rag.fetch_3 import fetchExternalKnowledgeV3
            self.fetch = fetchExternalKnowledgeV3

    def judge_with_gemini(self, query: str, retrieved_text: str) -> Tuple[float, float, float, str]:
        """Use Gemini 2.0 Flash as judge"""

        prompt = f"""You are evaluating a RAG (Retrieval-Augmented Generation) system's response.

QUERY: {query}

RETRIEVED CONTEXT:
{retrieved_text[:3000]}

Evaluate the retrieved context on these criteria (0.0 to 1.0 scale):

1. GROUNDEDNESS (0-1): Is the retrieved text factual and from a legitimate source?
   - 1.0 = Clearly from a real document, factual content
   - 0.5 = Somewhat relevant but unclear source
   - 0.0 = Hallucinated or irrelevant

2. RELEVANCE (0-1): Does the retrieved text help answer the query?
   - 1.0 = Directly addresses the query with useful information
   - 0.5 = Partially relevant, some useful info
   - 0.0 = Completely off-topic

3. COMPLETENESS (0-1): Does the context provide enough information?
   - 1.0 = Comprehensive, would enable a full answer
   - 0.5 = Partial information, missing key details
   - 0.0 = Insufficient to answer

Be fair but rigorous. RAG systems retrieve context, not generate answers.
A good retrieval that contains relevant passages should score well.

Respond with JSON only:
{{"groundedness": 0.X, "relevance": 0.X, "completeness": 0.X, "reasoning": "brief explanation"}}"""

        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema={
                        'type': 'object',
                        'properties': {
                            'groundedness': {'type': 'number'},
                            'relevance': {'type': 'number'},
                            'completeness': {'type': 'number'},
                            'reasoning': {'type': 'string'}
                        },
                        'required': ['groundedness', 'relevance', 'completeness', 'reasoning']
                    }
                )
            )
            result = json.loads(response.text)
            return (
                float(result['groundedness']),
                float(result['relevance']),
                float(result['completeness']),
                result['reasoning']
            )
        except Exception as e:
            print(f"  ⚠️ Gemini error: {e}")
            return (0.5, 0.5, 0.5, f"Error: {e}")

    def evaluate_query(self, query: str, doc_type: str) -> EvalResult:
        """Evaluate single query"""

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        doc_path = "" if doc_type == "pdf" else os.path.join(project_root, "tests", "fixtures", "real_docs")

        start = time.time()
        try:
            retrieved = self.fetch(query, doc_path=doc_path) if doc_path else self.fetch(query)
        except Exception as e:
            retrieved = f"ERROR: {e}"
        latency = (time.time() - start) * 1000

        g, r, c, reasoning = self.judge_with_gemini(query, retrieved or "")

        return EvalResult(
            query=query,
            rag_version=self.rag_version,
            retrieved_text=retrieved[:500] if retrieved else "NO RESULT",
            groundedness=g,
            relevance=r,
            completeness=c,
            latency_ms=latency,
            judge_reasoning=reasoning
        )

    def run_evaluation(self) -> Dict:
        """Run full evaluation"""

        print(f"\n{'='*60}")
        print(f"RAG Evaluation - {self.rag_version.upper()} - Judge: Gemini 2.0 Flash")
        print(f"{'='*60}\n")

        for query, doc_type in self.TEST_QUERIES:
            print(f"📄 [{doc_type.upper()}] {query[:50]}...")
            result = self.evaluate_query(query, doc_type)
            self.results.append(result)

            avg = (result.groundedness + result.relevance + result.completeness) / 3
            icon = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
            print(f"   {icon} G:{result.groundedness:.2f} R:{result.relevance:.2f} C:{result.completeness:.2f} = {avg:.2f} [{result.latency_ms:.0f}ms]")
            print(f"   💬 {result.judge_reasoning[:80]}...")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate summary report"""

        if not self.results:
            return {"error": "No results"}

        all_g = [r.groundedness for r in self.results]
        all_r = [r.relevance for r in self.results]
        all_c = [r.completeness for r in self.results]
        all_lat = [r.latency_ms for r in self.results]

        report = {
            "rag_version": self.rag_version,
            "judge_model": "gemini-2.0-flash",
            "total_queries": len(self.results),
            "scores": {
                "groundedness": sum(all_g) / len(all_g),
                "relevance": sum(all_r) / len(all_r),
                "completeness": sum(all_c) / len(all_c),
                "average": (sum(all_g) + sum(all_r) + sum(all_c)) / (3 * len(all_g)),
            },
            "performance": {
                "avg_latency_ms": sum(all_lat) / len(all_lat),
                "min_latency_ms": min(all_lat),
                "max_latency_ms": max(all_lat),
            },
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"RAG Version: {self.rag_version}")
        print(f"Judge: Gemini 2.0 Flash")
        print(f"Queries: {len(self.results)}")
        print(f"\n📊 SCORES:")
        print(f"   Groundedness:  {report['scores']['groundedness']:.2%}")
        print(f"   Relevance:     {report['scores']['relevance']:.2%}")
        print(f"   Completeness:  {report['scores']['completeness']:.2%}")
        print(f"   ────────────────────")
        print(f"   AVERAGE:       {report['scores']['average']:.2%}")
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Avg Latency:   {report['performance']['avg_latency_ms']:.0f}ms")

        return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v2", "v3"], default="v2")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    evaluator = GeminiRAGEvaluator(rag_version=args.version)
    report = evaluator.run_evaluation()

    if args.save:
        filepath = f"eval_gemini_{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Saved to {filepath}")

    return 0 if report['scores']['average'] >= 0.7 else 1


if __name__ == "__main__":
    exit(main())
