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
    "chunk_size": 700,
    "chunk_overlap": 100,
    "ensemble_weights": [0.7, 0.3],
    "reranker_model": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    "rerank_top_k": 5,
    "force_reindex": False,
    "distance_metric": "cosine"  # Qdrant distance metric
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

def get_rag_pipeline(doc_path: Optional[str] = None, file_list: Optional[list] = None):
    """Get or create RAG pipeline for given path or file list"""
    # Resolve path
    if doc_path:
        doc_path = resolve_path(doc_path)
        if not os.path.exists(doc_path):
            raise ValueError(f"Path does not exist: {doc_path}")

    # Use path or file list hash as cache key
    if file_list:
        key = hashlib.md5(str(sorted(file_list)).encode()).hexdigest()[:8]
    else:
        key = doc_path if doc_path else "__DEFAULT__"

    # Check if cached instance exists AND if files haven't changed
    if key in _rag_system_instances:
        cached_pipeline = _rag_system_instances[key]
        persist_dir = cached_pipeline.config.get("persist_dir")

        # Check if files changed since last index
        if not cached_pipeline._files_changed(persist_dir):
            print(f"[RAG V2] Using cached instance for {key} (no file changes)")
            return cached_pipeline
        else:
            print(f"[RAG V2] Files changed, invalidating cache for {key}")
            del _rag_system_instances[key]
    
    print(f"[RAG V2] Creating new instance for {key}...")
    try:
        config = RAG_CONFIG.copy()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if file_list:
            # Use specific file list
            config["document_paths"] = file_list
            path_hash = hashlib.md5(str(sorted(file_list)).encode()).hexdigest()[:8]
            scope_dir = os.path.join(current_dir, "rag_db_v2", f"scope_{path_hash}")
            os.makedirs(scope_dir, exist_ok=True)
            config["persist_dir"] = scope_dir
        elif doc_path:
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

def find_all_indexable_files(directory: str, max_depth: int = 5) -> list:
    """Recursively find all indexable files in directory

    Args:
        directory: Root directory to search
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        List of file paths that can be indexed
    """
    if not os.path.isdir(directory):
        return []

    indexable_extensions = ('.pdf', '.txt', '.py', '.md')
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}

    all_files = []

    def walk_dir(path: str, depth: int = 0):
        if depth > max_depth:
            return

        try:
            for entry in os.listdir(path):
                # Skip hidden files and excluded directories
                if entry.startswith('.') or entry in exclude_dirs:
                    continue

                entry_path = os.path.join(path, entry)

                if os.path.isfile(entry_path):
                    if entry.lower().endswith(indexable_extensions):
                        all_files.append(entry_path)
                elif os.path.isdir(entry_path):
                    walk_dir(entry_path, depth + 1)
        except PermissionError:
            print(f"[RAG] Permission denied: {path}")

    print(f"[RAG] Recursively scanning directory: {directory}")
    walk_dir(directory)
    print(f"[RAG] Found {len(all_files)} indexable files")

    return all_files

def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None) -> str:
    try:
        if not isinstance(query, str) or not query:
            return "Error: Invalid or empty query provided."

        # If custom path provided, find all indexable files
        file_list = None
        if doc_path:
            resolved_path = resolve_path(doc_path)
            if not os.path.exists(resolved_path):
                return f"Error: Path does not exist: {resolved_path}"

            if os.path.isdir(resolved_path):
                # Find ALL indexable files recursively
                all_files = find_all_indexable_files(resolved_path)

                if not all_files:
                    return f"No indexable files (.pdf, .txt, .py, .md) found in {resolved_path}"

                # Limit files to prevent excessive indexing (can be tuned)
                max_files = 500
                if len(all_files) > max_files:
                    print(f"[RAG] Warning: Found {len(all_files)} files, limiting to {max_files}")
                    all_files = all_files[:max_files]

                print(f"[RAG] Indexing {len(all_files)} files from {resolved_path}")

                # Use all found files for indexing
                file_list = all_files
                doc_path = None  # Clear doc_path, use file_list instead

        pipeline = get_rag_pipeline(doc_path=doc_path, file_list=file_list)
        return pipeline.search(query)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Sorry, an error occurred while searching: {e}"
