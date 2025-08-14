import os
from .core_2 import RAGPipeline

RAG_CONFIG = {
    "persist_dir": "./rag_db_v2",
    "document_paths": ["./files/"],
    "embed_model": "all-MiniLM-L6-v2",
    "top_k": 20,
    "chunk_size": 350,
    "chunk_overlap": 75,
    "ensemble_weights": [0.7, 0.3],
    "reranker_model": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    "rerank_top_k": 3,
    "force_reindex": True
}

_rag_system_instance = None

def get_rag_pipeline():
    global _rag_system_instance
    if _rag_system_instance is not None:
        return _rag_system_instance
    print("Initializing RAG system V2 for the first time...")
    try:
        config = RAG_CONFIG.copy()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config["persist_dir"] = os.path.join(current_dir, config["persist_dir"])
        config["document_paths"] = [os.path.join(current_dir, path) for path in config["document_paths"]]
        _rag_system_instance = RAGPipeline(config=config)
        print("RAG system V2 initialized successfully.")
        return _rag_system_instance
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize RAGPipeline: {e}")
        raise

def fetchExternalKnowledge(query: str) -> str:
    try:
        pipeline = get_rag_pipeline()
        if not isinstance(query, str) or not query:
            return "Error: Invalid or empty query provided."
        return pipeline.search(query)
    except Exception as e:
        return f"Sorry, an error occurred while searching: {e}"
