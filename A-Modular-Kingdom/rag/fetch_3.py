import os
import hashlib
from typing import Optional, Dict
from .core_3 import RAGPipelineV3

RAG_CONFIG_V3 = {
    "version": "v3",
    "persist_dir": "./rag_db_v3", 
    "document_paths": ["./files/"],
    "embed_provider": "ollama",
    "embed_model": "embeddinggemma",
    "top_k": 5,
    "chunk_size": 200,
    "chunk_overlap": 25,
    "rrf_k": 60,  # RRF parameter instead of ensemble weights
    "rerank_top_k": 2,  # Enable LLM reranking
    "force_reindex": False,
    "use_contextual": False,  # Disable contextual retrieval for performance
    "llm_model": "qwen3:8b",  # For LLM reranking
    "bm25_k1": 1.5,  # BM25 parameters
    "bm25_b": 0.75,
    "distance_metric": "cosine"  # Vector distance metric
}

_rag_system_v3_instances: Dict[str, RAGPipelineV3] = {}

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

def _safe_dir_name(path: str) -> str:
    abs_path = os.path.abspath(path)
    h = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:8]
    base = os.path.basename(abs_path.rstrip(os.sep)) or "root"
    return f"{base}_{h}"

def get_rag_pipeline_v3(doc_path: Optional[str] = None):
    import sys
    
    # Resolve path
    if doc_path:
        doc_path = resolve_path(doc_path)
        if not os.path.exists(doc_path):
            raise ValueError(f"Path does not exist: {doc_path}")
    
    key = doc_path if doc_path else "__DEFAULT__"
    if key in _rag_system_v3_instances:
        sys.stderr.write(f"[RAG V3] Using cached instance for {key}\n")
        sys.stderr.flush()
        return _rag_system_v3_instances[key]
    sys.stderr.write(f"[RAG V3] Creating new instance for {key}...\n")
    sys.stderr.flush()
    try:
        config = RAG_CONFIG_V3.copy()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if doc_path:
            # Scope documents to the provided directory
            config["document_paths"] = [doc_path]
            # Create a unique persist dir per doc scope
            scope_dir = os.path.join(current_dir, "rag_db_v3", _safe_dir_name(doc_path))
            os.makedirs(scope_dir, exist_ok=True)
            config["persist_dir"] = scope_dir
        else:
            config["persist_dir"] = os.path.join(current_dir, config["persist_dir"])
            config["document_paths"] = [os.path.join(current_dir, path) for path in config["document_paths"]]
        
        sys.stderr.write("[RAG] About to create RAGPipelineV3...\n")
        sys.stderr.flush()
        instance = RAGPipelineV3(config=config)
        _rag_system_v3_instances[key] = instance
        sys.stderr.write("[RAG] V3 initialization complete\n")
        sys.stderr.flush()
        return instance
    except Exception as e:
        sys.stderr.write(f"[RAG] FATAL ERROR: {e}\n")
        sys.stderr.flush()
        raise

# class V3RetrieverAdapter:
#     """
#     Adapts the V3 pipeline's custom output to the standard retriever interface
#     expected by the evaluation script.
#     """
#     def __init__(self, pipeline):
#         self.pipeline = pipeline

#     def get_relevant_documents(self, query: str):
#         """
#         Calls the V3 search and wraps the string results in Document objects.
#         """
#         # The V3 search function returns a single string with results separated by "---".
#         search_result_str = self.pipeline.search(query)
        
#         # We need to parse this string back into a list of Document objects.
#         # This is a simplified parsing based on the V3 search function's output format.
#         if "Document search results:" in search_result_str:
#             content = search_result_str.split("Document search results:\n\n")[1]
#             chunks = content.split("\n\n---\n\n")
#             # langchain_core.documents.Document
#             from langchain_core.documents import Document
#             return [Document(page_content=chunk) for chunk in chunks]
#         return []

# def get_retriever():
#     """Returns the retriever interface for the evaluation script."""
#     pipeline = get_rag_pipeline_v3()
#     return V3RetrieverAdapter(pipeline)

def find_relevant_files(query: str, directory: str, max_files: int = 5) -> list:
    """Find files in directory whose names match query keywords"""
    if not os.path.isdir(directory):
        return []
    
    # Remove common location words from query
    stop_words = {'in', 'on', 'at', 'from', 'the', 'a', 'an', 'desktop', 'documents', 'downloads', 'folder', 'directory', 'file'}
    query_words = [w for w in query.lower().split() if w not in stop_words]
    
    # If no words left after filtering, return all indexable files (up to max)
    if not query_words:
        all_files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.pdf', '.txt', '.py', '.md')):
                all_files.append(file_path)
        return all_files[:max_files]
    
    scored_files = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if not os.path.isfile(file_path):
            continue
        if not file.lower().endswith(('.pdf', '.txt', '.py', '.md')):
            continue
        
        # Score based on filename match
        filename_lower = file.lower()
        score = sum(1 for word in query_words if word in filename_lower)
        
        if score > 0:
            scored_files.append((score, file_path))
    
    # Return top matches
    scored_files.sort(reverse=True, key=lambda x: x[0])
    return [path for score, path in scored_files[:max_files]]

def fetchExternalKnowledgeV3(query: str, doc_path: Optional[str] = None) -> str:
    try:
        if not isinstance(query, str) or not query:
            return "Error: Invalid or empty query provided."
        
        # If custom path provided, find relevant files first
        if doc_path:
            resolved_path = resolve_path(doc_path)
            if not os.path.exists(resolved_path):
                return f"Error: Path does not exist: {resolved_path}"
            
            if os.path.isdir(resolved_path):
                # Find relevant files based on query
                relevant_files = find_relevant_files(query, resolved_path)
                
                if not relevant_files:
                    return f"No files matching '{query}' found in {resolved_path}"
                
                print(f"[RAG V3] Found {len(relevant_files)} relevant files: {[os.path.basename(f) for f in relevant_files]}")
                
                # Use first relevant file for now (TODO: support multiple)
                doc_path = relevant_files[0]
        
        pipeline = get_rag_pipeline_v3(doc_path=doc_path)
        return pipeline.search(query)
    except Exception as e:
        return f"Sorry, an error occurred while searching: {e}"

# For compatibility with a unified interface
def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None) -> str:
    return fetchExternalKnowledgeV3(query, doc_path=doc_path)