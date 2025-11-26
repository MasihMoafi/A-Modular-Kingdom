"""
RAG Evaluation Framework - LLM as Judge + Multi-format Testing

Metrics:
- Groundedness: Is the answer supported by retrieved context?
- Relevance: Does the answer address the question?
- Completeness: Does the answer cover all aspects?
- Faithfulness: No hallucination beyond source material

Usage:
    python tests/evaluation/rag_evaluator.py --version v2 --format all
"""

import os
import sys
import json
import time
import ollama
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@dataclass
class EvalResult:
    query: str
    rag_version: str
    file_format: str
    retrieved_text: str
    groundedness: float  # 0-1: Is answer in context?
    relevance: float     # 0-1: Does it answer the question?
    completeness: float  # 0-1: Full answer or partial?
    latency_ms: float
    judge_reasoning: str
    timestamp: str


class RAGEvaluator:
    """LLM-as-Judge evaluation for RAG systems"""

    JUDGE_MODEL = "qwen3:8b"

    # Multi-format test corpus
    TEST_QUERIES = {
        "pdf": [
            ("What battles did Napoleon fight?", ["battle", "war", "campaign"]),
            ("Who was Napoleon's wife?", ["josephine", "marie", "wife"]),
            ("When did Napoleon become emperor?", ["emperor", "1804", "coronation"]),
        ],
        "md": [
            ("How does prompt engineering work?", ["prompt", "instruction", "claude"]),
            ("What are the forex trading strategies?", ["forex", "trading", "strategy"]),
            ("What are the zigzag requirements?", ["zigzag", "requirement", "design"]),
        ],
        "py": [
            ("How does the RAG pipeline work?", ["rag", "retriev", "chunk"]),
            ("What tools are available in the MCP server?", ["tool", "mcp", "function"]),
        ],
        "ipynb": [
            ("What machine learning concepts are covered?", ["model", "train", "data"]),
            ("What sklearn methods are used?", ["sklearn", "scikit", "fit"]),
        ],
    }

    def __init__(self, rag_version: str = "v2"):
        self.rag_version = rag_version
        self.results: List[EvalResult] = []

        # Import appropriate RAG function
        if rag_version == "v2":
            from rag.fetch_2 import fetchExternalKnowledge
            self.fetch = fetchExternalKnowledge
        else:
            from rag.fetch_3 import fetchExternalKnowledgeV3
            self.fetch = fetchExternalKnowledgeV3

    def judge_response(
        self,
        query: str,
        retrieved_text: str,
        expected_keywords: List[str]
    ) -> Tuple[float, float, float, str]:
        """
        Use LLM as judge to evaluate RAG response

        Returns: (groundedness, relevance, completeness, reasoning)
        """
        judge_prompt = f"""You are a RAG evaluation judge. Evaluate the retrieved text for the given query.

QUERY: {query}

RETRIEVED TEXT:
{retrieved_text[:2000]}  # Truncate for context

EXPECTED TOPICS (hints): {expected_keywords}

Rate each metric 0.0 to 1.0:
1. GROUNDEDNESS: Is all information in the retrieved text factual and supported?
2. RELEVANCE: Does the text actually answer the query?
3. COMPLETENESS: Does it cover the topic thoroughly or just partially?

Respond with JSON only:
{{"groundedness": 0.X, "relevance": 0.X, "completeness": 0.X, "reasoning": "brief explanation"}}"""

        try:
            response = ollama.chat(
                model=self.JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                format="json"
            )
            content = response["message"]["content"]

            # Clean and parse JSON
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()

            result = json.loads(content)
            return (
                float(result.get("groundedness", 0.5)),
                float(result.get("relevance", 0.5)),
                float(result.get("completeness", 0.5)),
                result.get("reasoning", "No reasoning provided")
            )
        except Exception as e:
            # Fallback to keyword-based evaluation
            found = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text.lower())
            score = found / len(expected_keywords) if expected_keywords else 0.5
            return (score, score, score, f"Fallback eval (error: {e})")

    def evaluate_query(
        self,
        query: str,
        expected_keywords: List[str],
        file_format: str,
        doc_path: str = ""
    ) -> EvalResult:
        """Evaluate a single query"""

        # Time the RAG call
        start = time.time()
        try:
            retrieved_text = self.fetch(query, doc_path=doc_path) if doc_path else self.fetch(query)
        except Exception as e:
            retrieved_text = f"ERROR: {e}"
        latency_ms = (time.time() - start) * 1000

        # Judge the response
        groundedness, relevance, completeness, reasoning = self.judge_response(
            query, retrieved_text or "", expected_keywords
        )

        return EvalResult(
            query=query,
            rag_version=self.rag_version,
            file_format=file_format,
            retrieved_text=retrieved_text[:500] if retrieved_text else "NO RESULT",
            groundedness=groundedness,
            relevance=relevance,
            completeness=completeness,
            latency_ms=latency_ms,
            judge_reasoning=reasoning,
            timestamp=datetime.now().isoformat()
        )

    def run_evaluation(
        self,
        formats: List[str] = None,
        doc_paths: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation across formats

        Args:
            formats: List of formats to test ["pdf", "md", "py", "ipynb"]
            doc_paths: Map format -> doc_path for custom test locations
        """
        formats = formats or ["pdf", "md"]
        doc_paths = doc_paths or {}

        print(f"\n{'='*60}")
        print(f"RAG Evaluation - Version: {self.rag_version}")
        print(f"{'='*60}\n")

        for fmt in formats:
            if fmt not in self.TEST_QUERIES:
                print(f"⚠️  No test queries for format: {fmt}")
                continue

            print(f"\n📄 Testing {fmt.upper()} files...")
            doc_path = doc_paths.get(fmt, "")

            for query, keywords in self.TEST_QUERIES[fmt]:
                print(f"  → Query: {query[:50]}...")
                result = self.evaluate_query(query, keywords, fmt, doc_path)
                self.results.append(result)

                # Print inline result
                avg_score = (result.groundedness + result.relevance + result.completeness) / 3
                status = "✅" if avg_score > 0.6 else "⚠️" if avg_score > 0.3 else "❌"
                print(f"    {status} Score: {avg_score:.2f} (G:{result.groundedness:.2f} R:{result.relevance:.2f} C:{result.completeness:.2f}) [{result.latency_ms:.0f}ms]")

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report with aggregate metrics"""

        if not self.results:
            return {"error": "No results"}

        # Aggregate by format
        by_format = {}
        for r in self.results:
            if r.file_format not in by_format:
                by_format[r.file_format] = []
            by_format[r.file_format].append(r)

        format_scores = {}
        for fmt, results in by_format.items():
            format_scores[fmt] = {
                "groundedness": sum(r.groundedness for r in results) / len(results),
                "relevance": sum(r.relevance for r in results) / len(results),
                "completeness": sum(r.completeness for r in results) / len(results),
                "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
                "count": len(results),
            }

        # Overall metrics
        all_g = [r.groundedness for r in self.results]
        all_r = [r.relevance for r in self.results]
        all_c = [r.completeness for r in self.results]

        report = {
            "rag_version": self.rag_version,
            "total_queries": len(self.results),
            "overall": {
                "groundedness": sum(all_g) / len(all_g),
                "relevance": sum(all_r) / len(all_r),
                "completeness": sum(all_c) / len(all_c),
                "avg_score": (sum(all_g) + sum(all_r) + sum(all_c)) / (3 * len(all_g)),
            },
            "by_format": format_scores,
            "detailed_results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"RAG Version: {self.rag_version}")
        print(f"Total Queries: {len(self.results)}")
        print(f"\nOverall Scores:")
        print(f"  Groundedness:  {report['overall']['groundedness']:.2f}")
        print(f"  Relevance:     {report['overall']['relevance']:.2f}")
        print(f"  Completeness:  {report['overall']['completeness']:.2f}")
        print(f"  Average Score: {report['overall']['avg_score']:.2f}")
        print(f"\nBy Format:")
        for fmt, scores in format_scores.items():
            print(f"  {fmt}: G={scores['groundedness']:.2f} R={scores['relevance']:.2f} C={scores['completeness']:.2f} ({scores['avg_latency_ms']:.0f}ms)")

        return report

    def save_report(self, filepath: str = None):
        """Save report to JSON file"""
        filepath = filepath or f"eval_report_{self.rag_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = self.generate_report()

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Report saved to: {filepath}")
        return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument("--version", choices=["v2", "v3"], default="v2", help="RAG version to test")
    parser.add_argument("--format", nargs="+", default=["pdf", "md"], help="File formats to test")
    parser.add_argument("--save", action="store_true", help="Save report to JSON")
    args = parser.parse_args()

    # Setup paths for different formats
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_paths = {
        "pdf": "",  # Default path (Napoleon.pdf)
        "md": os.path.join(project_root, "tests", "fixtures", "real_docs"),
        "py": os.path.join(project_root, "src"),
        "ipynb": os.path.join(project_root, "notebooks"),
    }

    evaluator = RAGEvaluator(rag_version=args.version)
    report = evaluator.run_evaluation(formats=args.format, doc_paths=doc_paths)

    if args.save:
        evaluator.save_report()

    # Return exit code based on score
    avg_score = report.get("overall", {}).get("avg_score", 0)
    return 0 if avg_score > 0.5 else 1


if __name__ == "__main__":
    exit(main())
