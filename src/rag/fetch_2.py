import os
import hashlib
from typing import Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment on import
def _validate_rag_dependencies():
    """Validate RAG dependencies are available"""
    try:
        import torch
        import sentence_transformers
        from qdrant_client import QdrantClient
        return True
    except ImportError as e:
        logger.error(f"Missing RAG dependency: {e}")
        return False

_DEPENDENCIES_OK = _validate_rag_dependencies()

try:
    from .core_2 import RAGPipeline
except Exception as e:
    logger.error(f"Failed to import RAGPipeline: {e}")
    RAGPipeline = None

RAG_CONFIG = {
    "persist_dir": "./rag_db_v2",
    "document_paths": ["./files/"],
    "embed_provider": "sentencetransformer",
    "embed_model": "all-MiniLM-L6-v2",  # 384-dim, 80MB model (fast)
    "top_k": 5,
    "chunk_size": 700,
    "chunk_overlap": 100,
    "ensemble_weights": [0.7, 0.3],
    "reranker_model": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    "rerank_top_k": 5,
    "force_reindex": False,
    "distance_metric": "cosine",  # Qdrant distance metric
    # Qdrant backend options
    "qdrant_mode": "cloud",  # "local" or "cloud"
    "qdrant_url": "https://5c99b123-9ead-4adb-b715-d10743893daf.us-west-2-0.aws.cloud.qdrant.io:6333",
    "qdrant_api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.jPT9n6OF6yTtc_nZoL50d8sxA67GM_VDjznulfl87Sg",
    # Speed optimization
    "batch_size": 500,  # Process 500 chunks at once
    "max_files": 100,  # Limit to prevent massive indexing
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

    # Check if cached instance exists
    if key in _rag_system_instances:
        print(f"[RAG V2] Using cached instance for {key}")
        return _rag_system_instances[key]
    
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

def find_all_indexable_files(
    directory: str,
    max_depth: int = 5,
    include_patterns: Optional[list] = None,
    exclude_patterns: Optional[list] = None,
    max_files: Optional[int] = None
) -> list:
    """Recursively find indexable files with selective filtering

    Args:
        directory: Root directory to search
        max_depth: Maximum recursion depth
        include_patterns: Only include files matching these patterns (e.g., ['*.py', 'src/**'])
        exclude_patterns: Exclude files matching these patterns (e.g., ['*test*', '*__pycache__*'])
        max_files: Stop after finding this many files

    Returns:
        List of file paths that can be indexed
    """
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

def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None) -> str:
    """Query RAG system with robust error handling

    Args:
        query: Search query string
        doc_path: Optional path to documents (file or directory)

    Returns:
        Search results or error message
    """
    # Validate dependencies
    if not _DEPENDENCIES_OK:
        return "Error: RAG dependencies not properly installed. Run: uv pip install torch sentence-transformers qdrant-client"

    if RAGPipeline is None:
        return "Error: RAGPipeline failed to import. Check logs for details."

    try:
        # Validate query
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query provided")
            return "Error: Query must be a non-empty string."

        # If custom path provided, find all indexable files
        file_list = None
        if doc_path:
            try:
                resolved_path = resolve_path(doc_path)
            except Exception as e:
                logger.error(f"Path resolution failed: {e}")
                return f"Error: Could not resolve path: {doc_path}"

            if not os.path.exists(resolved_path):
                logger.error(f"Path does not exist: {resolved_path}")
                return f"Error: Path does not exist: {resolved_path}"

            if os.path.isdir(resolved_path):
                # Smart file discovery with limits
                max_files = RAG_CONFIG.get("max_files", 100)

                try:
                    all_files = find_all_indexable_files(
                        resolved_path,
                        max_files=max_files,
                        exclude_patterns=['*test*', '*__pycache__*', '*.pyc']
                    )
                except Exception as e:
                    logger.error(f"File discovery failed: {e}")
                    return f"Error: Failed to scan directory: {e}"

                if not all_files:
                    logger.warning(f"No indexable files found in {resolved_path}")
                    return f"No indexable files found in {resolved_path}"

                logger.info(f"Indexing {len(all_files)} files from {resolved_path}")

                # Use all found files for indexing
                file_list = all_files
                doc_path = None  # Clear doc_path, use file_list instead

        # Get or create pipeline
        try:
            pipeline = get_rag_pipeline(doc_path=doc_path, file_list=file_list)
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            return f"Error: Failed to initialize RAG system: {e}"

        # Perform search
        try:
            result = pipeline.search(query)
            return result
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return f"Error: Search failed: {e}"

    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
        return "Search interrupted by user"
    except Exception as e:
        logger.error(f"Unexpected error in fetchExternalKnowledge: {e}", exc_info=True)
        return f"Error: Unexpected failure: {e}"
