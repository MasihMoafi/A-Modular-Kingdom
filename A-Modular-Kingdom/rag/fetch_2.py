import os
import hashlib
from typing import Optional, Dict
from .core_2 import RAGPipeline

RAG_CONFIG = {
    "persist_dir": "./rag_db_v2",
    "document_paths": ["./files/"],
    "embed_provider": "ollama",
    "embed_model": "embeddinggemma",
    "top_k": 5,
    "chunk_size": 200,
    "chunk_overlap": 25,
    "ensemble_weights": [0.7, 0.3],
    "reranker_model": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    "rerank_top_k": 2,
    "force_reindex": False
}

_rag_system_instances: Dict[str, RAGPipeline] = {}

def resolve_path(path: str) -> str:
    """Resolve user-friendly paths to absolute paths"""
    if not path:
        return None
    
    # Handle common shortcuts
    shortcuts = {
        "desktop": "~/Desktop",
        "documents": "~/Documents",
        "downloads": "~/Downloads",
    }
    
    lower = path.lower().strip()
    if lower in shortcuts:
        path = shortcuts[lower]
    
    # Expand ~ and environment variables
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    
    # Make absolute if relative
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    return path

def get_rag_pipeline(doc_path: Optional[str] = None):
    """Get or create RAG pipeline for given path"""
    # Resolve path
    if doc_path:
        doc_path = resolve_path(doc_path)
        if not os.path.exists(doc_path):
            raise ValueError(f"Path does not exist: {doc_path}")
    
    # Use path as cache key
    key = doc_path if doc_path else "__DEFAULT__"
    
    if key in _rag_system_instances:
        print(f"[RAG V2] Using cached instance for {key}")
        return _rag_system_instances[key]
    
    print(f"[RAG V2] Creating new instance for {key}...")
    try:
        config = RAG_CONFIG.copy()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if doc_path:
            # Use provided path
            config["document_paths"] = [doc_path]
            # Create unique persist dir for this path
            path_hash = hashlib.md5(doc_path.encode()).hexdigest()[:8]
            scope_dir = os.path.join(current_dir, "rag_db_v2", f"scope_{path_hash}")
            os.makedirs(scope_dir, exist_ok=True)
            config["persist_dir"] = scope_dir
        else:
            # Use default paths
            config["persist_dir"] = os.path.join(current_dir, config["persist_dir"])
            config["document_paths"] = [os.path.join(current_dir, path) for path in config["document_paths"]]
        
        instance = RAGPipeline(config=config)
        _rag_system_instances[key] = instance
        print("[RAG V2] Initialization complete")
        return instance
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize RAGPipeline: {e}")
        raise

def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None) -> str:
    try:
        pipeline = get_rag_pipeline(doc_path=doc_path)
        if not isinstance(query, str) or not query:
            return "Error: Invalid or empty query provided."
        return pipeline.search(query)
    except Exception as e:
        return f"Sorry, an error occurred while searching: {e}"
