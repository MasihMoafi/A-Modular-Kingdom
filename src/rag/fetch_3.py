import os
import hashlib
from typing import Optional, Dict
from .core_3 import RAGPipelineV3

RAG_CONFIG_V3 = {
    "version": "v3",
    "persist_dir": "./rag_db_v3",
    "document_paths": ["./files/"],
    "embed_provider": "sentencetransformer",  # Fast GPU-based embeddings
    "embed_model": "all-MiniLM-L6-v2",  # Same as V2 for consistency
    "top_k": 5,
    "chunk_size": 700,
    "chunk_overlap": 100,
    "rrf_k": 60,  # RRF parameter for fusion (better than weighted ensemble)
    "rerank_top_k": 5,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast CrossEncoder reranking
    "force_reindex": False,
    "use_contextual": False,  # Disable contextual retrieval for performance
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

def find_all_indexable_files(
    directory: str,
    max_depth: int = 5,
    include_patterns: Optional[list] = None,
    exclude_patterns: Optional[list] = None,
    max_files: Optional[int] = None
) -> list:
    """Recursively find indexable files with selective filtering"""
    if not os.path.isdir(directory):
        return []

    indexable_extensions = ('.pdf', '.txt', '.py', '.md', '.ipynb', '.js', '.ts', '.tsx', '.jsx')
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', '.ipynb_checkpoints', 'migrations'}

    all_files = []

    def matches_pattern(path: str, patterns: list) -> bool:
        """Check if path matches any glob pattern"""
        import fnmatch
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False

    def walk_dir(path: str, depth: int = 0):
        if depth > max_depth:
            return
        if max_files and len(all_files) >= max_files:
            return

        try:
            for entry in os.listdir(path):
                # Skip hidden files and excluded directories
                if entry.startswith('.') or entry in exclude_dirs:
                    continue

                entry_path = os.path.join(path, entry)

                # Apply exclude patterns
                if exclude_patterns and matches_pattern(entry_path, exclude_patterns):
                    continue

                if os.path.isfile(entry_path):
                    if entry.lower().endswith(indexable_extensions):
                        # Apply include patterns
                        if include_patterns:
                            if matches_pattern(entry_path, include_patterns):
                                all_files.append(entry_path)
                        else:
                            all_files.append(entry_path)

                        if max_files and len(all_files) >= max_files:
                            return
                elif os.path.isdir(entry_path):
                    walk_dir(entry_path, depth + 1)
        except PermissionError:
            print(f"[RAG] Permission denied: {path}")

    print(f"[RAG] Scanning: {directory}")
    if include_patterns:
        print(f"[RAG] Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"[RAG] Exclude patterns: {exclude_patterns}")
    if max_files:
        print(f"[RAG] Max files: {max_files}")

    walk_dir(directory)
    print(f"[RAG] Found {len(all_files)} indexable files")

    return all_files


def find_relevant_files(query: str, directory: str, max_files: int = 5) -> list:
    """Find files in directory whose names match query keywords"""
    if not os.path.isdir(directory):
        return []
    
    # Query should already be cleaned by main.py's Instructor parsing
    print(f"[RAG V3] Searching for files matching: '{query}' in {directory}")
    
    query_words = query.lower().split()
    
    # If no meaningful query, return first few indexable files
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

        # If custom path provided, find all indexable files
        if doc_path:
            resolved_path = resolve_path(doc_path)
            if not os.path.exists(resolved_path):
                return f"Error: Path does not exist: {resolved_path}"

            if os.path.isdir(resolved_path):
                # Find ALL indexable files recursively (limit to 100 like v2)
                all_files = find_all_indexable_files(
                    resolved_path,
                    max_files=100,
                    exclude_patterns=['test_*.py', '*_test.py', '*__pycache__*', '*.pyc']
                )

                if not all_files:
                    return f"No indexable files (.pdf, .txt, .py, .md) found in {resolved_path}"

                print(f"[RAG V3] Indexing {len(all_files)} files from {resolved_path}")

                # Pass directory path - v3 will handle multiple files
                doc_path = resolved_path

        pipeline = get_rag_pipeline_v3(doc_path=doc_path)
        return pipeline.search(query)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Sorry, an error occurred while searching: {e}"

# For compatibility with a unified interface
def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None) -> str:
    return fetchExternalKnowledgeV3(query, doc_path=doc_path)