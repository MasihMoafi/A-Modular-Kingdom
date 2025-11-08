import os
from .core import RAGPipeline

# --- Configuration ---
# This is the main control panel for the RAG system.
# Change these settings to experiment with the pipeline.
RAG_CONFIG = {
    "persist_dir": "./rag_db",
    "document_paths": ["./files/"],
    "embed_model": "all-MiniLM-L6-v2",
    "top_k": 3,
    "chunk_size": 500,
    "chunk_overlap": 100,
    "ensemble_weights": [0.7, 0.3], # [Vector retriever, BM25] 
    # Developer tool: Set to True to force the database to be rebuilt.
    "force_reindex": False
}

# --- System Singleton ---
# This global variable will hold our single, initialized RAGPipeline instance
# to ensure we don't do the slow setup work more than once.
_rag_system_instance = None

def get_rag_pipeline():
    """
    Initializes and returns the singleton RAGPipeline instance.
    This function ensures the expensive setup runs only once.
    """
    global _rag_system_instance
    # If the instance already exists, return it immediately.
    if _rag_system_instance is not None:
        return _rag_system_instance

    # --- First-time setup ---
    print("Initializing RAG system for the first time...")
    try:
        config = RAG_CONFIG.copy()

        # --- Path Correction ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config["persist_dir"] = os.path.join(current_dir, "rag_db")
        config["document_paths"] = [os.path.join(current_dir, "files")]

        # This is the slow step: creating the RAGPipeline instance.
        _rag_system_instance = RAGPipeline(config=config)
        print("RAG system initialized successfully.")
        return _rag_system_instance
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize RAGPipeline: {e}")
        raise 

def get_retriever():
    """Returns the retriever interface for the evaluation script."""
    pipeline = get_rag_pipeline()
    return pipeline.vector_db.as_retriever()
        
def fetchExternalKnowledge(query: str) -> str:
    """
    Fetches external knowledge based on a query.
    This is the main entry point for using the RAG system.
    """
    try:
        pipeline = get_rag_pipeline()
        # Safety checks
        if not isinstance(query, str) or not query:
            return "Error: Invalid or empty query provided."

        return pipeline.search(query)
    except Exception as e:
        return f"Sorry, an error occurred while searching: {e}"

