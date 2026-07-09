import os
import sys
import time
import csv
import json
import argparse
from pathlib import Path

# Add project root to path so we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, project_root)

from rag.core import RAGPipeline

def run_test(test_dir: str):
    test_path = Path(test_dir).resolve()
    pdf_path = test_path / "document.pdf"
    csv_path = test_path / "queries.csv"
    
    if not pdf_path.exists() or not csv_path.exists():
        print(f"Error: Missing document.pdf or queries.csv in {test_dir}")
        return

    print(f"=== Starting Experiment for {test_path.name} using Unified RAG ===")
    
    # Configure isolated RAG
    persist_dir = test_path / f"rag_db_unified"
    config = {
        "document_paths": [str(pdf_path)],
        "persist_dir": str(persist_dir),
        "embed_provider": "ollama",
        "embed_model": "qwen3-embedding:8b",
        "qdrant_mode": "local",
        "distance_metric": "cosine",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "rerank_top_k": 5
    }
    
    print("Initializing Unified RAG Pipeline (This will index the PDF if not already done)...")
    start_init = time.time()
    pipeline = RAGPipeline(config)
    print(f"Initialization took {time.time() - start_init:.2f} seconds.")

    # Load queries
    queries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)

    output_md_path = test_path / f"results_unified.md"
    
    print(f"Running {len(queries)} queries...")
    
    with open(output_md_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"# Unified RAG Experiment Results: {test_path.name}\n\n")
        out_f.write(f"**Target Document:** `{pdf_path.name}`\n")
        out_f.write(f"**Embedding Model:** `{config['embed_model']}`\n")
        out_f.write(f"**Reranker Model:** `{config.get('reranker_model', 'None')}`\n\n")
        out_f.write("---\n\n")
        
        for idx, q in enumerate(queries, 1):
            category = q.get('Category', 'Unknown')
            question = q.get('Question', '')
            expected_location = q.get('Location', 'Unknown')
            expected_answer = q.get('Answer', '')
            
            print(f"[{idx}/{len(queries)}] Querying: {question[:50]}...")
            
            start_q = time.time()
            try:
                # Query the pipeline
                results = pipeline.search(question)
            except Exception as e:
                print(f"Error querying: {e}")
                results = []
                
            q_time = time.time() - start_q
            
            # Write to markdown
            out_f.write(f"## {idx}. [{category}] {question}\n")
            out_f.write(f"- **Expected Location:** {expected_location}\n")
            out_f.write(f"- **Expected Answer:** {expected_answer}\n")
            out_f.write(f"- **Retrieval Time:** {q_time:.3f}s\n")
            out_f.write(f"- **Chunks Retrieved:** {len(results)}\n\n")
            
            out_f.write("### Top Retrieved Chunks:\n")
            for r_idx, res in enumerate(results, 1):
                # Safely extract content and metadata depending on how RAG pipeline returns it
                content = res.get('content', '') if isinstance(res, dict) else getattr(res, 'page_content', str(res))
                metadata = res.get('metadata', {}) if isinstance(res, dict) else getattr(res, 'metadata', {})
                
                # Try to extract page/source info if it exists
                page = metadata.get('page', 'Unknown')
                source = metadata.get('source', 'Unknown')
                
                out_f.write(f"#### Chunk {r_idx} (Page: {page})\n")
                out_f.write(f"```text\n{content.strip()}\n```\n\n")
            
            out_f.write("---\n\n")

    print(f"\nExperiment complete. Results saved to {output_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG ablation tests.")
    parser.add_argument("test_dir", help="Path to the test directory (e.g., ../../tests/test1/napoleon)")
    
    args = parser.parse_args()
    run_test(args.test_dir)
