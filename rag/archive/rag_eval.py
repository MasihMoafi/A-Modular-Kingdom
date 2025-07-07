#DISABLE PROXY
import os

def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

import json
import numpy as np
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# --- THE CONFIGURABLE SWITCH ---
# Change this variable to 'v1' or 'v2' to test the desired pipeline.
PIPELINE_VERSION = "v2"

# --- Dynamic Import ---
if PIPELINE_VERSION == "v1":
    from fetch import get_rag_pipeline
elif PIPELINE_VERSION == "v2":
    from fetch_2 import get_rag_pipeline
else:
    raise ValueError("Invalid PIPELINE_VERSION specified.")

EVAL_CONFIG = {
    "eval_file": "evaluation_data.json",
    "evaluation_mode": "llm_judge",
    "similarity_threshold": 0.6,
    "llm_judge_model": 'qwen3:8b'
}

# --- Evaluation Functions (Unchanged) ---
def evaluate_simple(query_data, retriever, embeddings):
    retrieved_docs = retriever.get_relevant_documents(query_data["query"])
    if not retrieved_docs:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "retrieved_docs": []}
    gt_embedding = embeddings.embed_query(query_data["ground_truth"])
    relevance_scores = [cosine_similarity([gt_embedding], [embeddings.embed_query(doc.page_content)])[0][0] for doc in retrieved_docs]
    relevant_found = [score > EVAL_CONFIG["similarity_threshold"] for score in relevance_scores]
    precision = sum(relevant_found) / len(relevant_found) if relevant_found else 0.0
    recall = 1.0 if any(relevant_found) else 0.0
    mrr = 1.0 / (relevant_found.index(True) + 1) if any(relevant_found) else 0.0
    return {"precision": precision, "recall": recall, "mrr": mrr, "retrieved_docs": retrieved_docs}

async def evaluate_llm_judge(query_data, retriever):
    retrieved_docs = retriever.invoke(query_data["query"])
    if not retrieved_docs:
        return {"context_recall": 0.0, "retrieved_docs": []}
    context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    gt_sentences = [s.strip() for s in query_data["ground_truth"].split('.') if s.strip()]
    if not gt_sentences:
        return {"context_recall": 0.0, "retrieved_docs": retrieved_docs}
    verified_sentences = 0
    for sentence in gt_sentences:
        prompt = f'Context:\n---\n{context_str}\n---\nStatement: "{sentence}"\nCan this statement be verified by the context? Answer Yes or No:'
        response = await ollama.AsyncClient().chat(model=EVAL_CONFIG["llm_judge_model"], messages=[{'role': 'user', 'content': prompt}])
        if 'yes' in response['message']['content'].strip().lower():
            verified_sentences += 1
    context_recall = verified_sentences / len(gt_sentences)
    return {"context_recall": context_recall, "retrieved_docs": retrieved_docs}

# --- Main Execution (Now with dynamic report names) ---
async def main():
    pipeline = get_rag_pipeline()
    if not pipeline:
        print("Failed to initialize RAG pipeline.")
        return

    with open(EVAL_CONFIG["eval_file"], 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)

    mode = EVAL_CONFIG["evaluation_mode"]
    print(f"Running evaluation for {PIPELINE_VERSION} in '{mode}' mode...")
    
    query_details = []
    retriever = pipeline.vector_db.as_retriever()

    for query_id, data in evaluation_data.items():
        if mode == 'simple':
            result = evaluate_simple(data, retriever, pipeline.embeddings)
        elif mode == 'llm_judge':
            result = await evaluate_llm_judge(data, retriever)
        query_details.append({"id": query_id, "data": data, "result": result})

    if not query_details:
        print("No results to display.")
        return

    # --- THE FIX: Unique Report Filenames ---
    report_filename = f"rag_report_{PIPELINE_VERSION}_{mode}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"--- RAG Evaluation Report ({PIPELINE_VERSION} - {mode}) ---\n\n")
        f.write("--- Overall Metrics ---\n")
        if mode == 'simple':
            avg_precision = np.mean([r['result']['precision'] for r in query_details])
            avg_recall = np.mean([r['result']['recall'] for r in query_details])
            avg_mrr = np.mean([r['result']['mrr'] for r in query_details])
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}\n")
            print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, MRR: {avg_mrr:.4f}")
        elif mode == 'llm_judge':
            avg_context_recall = np.mean([r['result']['context_recall'] for r in query_details])
            f.write(f"Average Context Recall: {avg_context_recall:.4f}\n")
            print(f"Context Recall: {avg_context_recall:.4f}")
        
        f.write("\n--- Per-Query Details ---\n\n")
        for detail in query_details:
            f.write(f"--- Query ID: {detail['id']} ---\n")
            f.write(f"Query: {detail['data']['query']}\n")
            f.write(f"Ground Truth: {detail['data']['ground_truth']}\n")
            if mode == 'simple':
                f.write(f"Precision: {detail['result']['precision']:.4f}, Recall: {detail['result']['recall']:.4f}, MRR: {detail['result']['mrr']:.4f}\n")
            elif mode == 'llm_judge':
                f.write(f"Context Recall: {detail['result']['context_recall']:.4f}\n")
            f.write("--- Retrieved Chunks ---\n")
            if detail['result']['retrieved_docs']:
                for i, doc in enumerate(detail['result']['retrieved_docs']):
                    f.write(f"  --- Chunk {i+1} ---\n")
                    f.write(f"  {doc.page_content}\n\n")
            else:
                f.write("  No documents retrieved.\n\n")

    print(f"Detailed report saved to {report_filename}")

if __name__ == "__main__":
    asyncio.run(main())
