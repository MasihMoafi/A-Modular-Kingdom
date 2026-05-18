import os
import sys
import hashlib
from typing import Optional, Dict
import logging
from pathlib import Path
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Offline/local-first by default. Loading src/.env is opt-in because stale
# Qdrant Cloud credentials can break local RAG when cloud projects are gone.
_env_path = Path(__file__).parent.parent / ".env"
if os.getenv("AMK_LOAD_DOTENV", "0").lower() in ("1", "true", "yes", "y", "on") and _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

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
    "ensemble_weights": [0.5, 0.5],  # Balanced vector + BM25
    "reranker_model": 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    "rerank_top_k": 5,
    "force_reindex": False,
    "distance_metric": "cosine",  # Qdrant distance metric
    # Qdrant backend options
    "qdrant_mode": os.getenv("QDRANT_MODE", "local"),
    "qdrant_url": os.getenv("QDRANT_URL", ""),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
    # Speed optimization
    "batch_size": 500,  # Process 500 chunks at once
    "max_files": 100,  # Limit to prevent massive indexing
}

_rag_system_instances: Dict[str, RAGPipeline] = {}

_STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "from","into","over","under","not","no","yes","do","does","did","done","can",
}


def _tokenize_query(query: str) -> list[str]:
    toks = [t.lower() for t in re.findall(r"[a-zA-Z0-9_./-]{2,}", query)]
    return [t for t in toks if t not in _STOPWORDS]


def _fast_search_files(
    query: str,
    file_list: list[str],
    *,
    top_k: int = 5,
    max_bytes_per_file: int = 250_000,
    max_total_bytes: int = 5_000_000,
) -> str:
    """Quick lexical search over provided files (no embeddings/Qdrant).

    This is used to keep MCP tool calls responsive when indexing a new scope would
    require heavyweight model loads or network access.
    """
    tokens = _tokenize_query(query)
    if not tokens:
        tokens = _tokenize_query(query.strip()[:80]) or [query.strip().lower()[:32]]

    results = []
    total = 0
    started = time.monotonic()

    for path in file_list:
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue

        if size <= 0:
            continue

        # Cap total bytes scanned to avoid burning CPU.
        if total >= max_total_bytes:
            break

        read_n = min(size, max_bytes_per_file, max_total_bytes - total)
        total += read_n

        try:
            with open(path, "rb") as f:
                blob = f.read(read_n)
        except OSError:
            continue

        try:
            text = blob.decode("utf-8", errors="ignore")
        except Exception:
            continue

        low = text.lower()
        score = sum(low.count(t) for t in tokens[:8])
        if score <= 0:
            continue

        # Find best matching line.
        best = None
        best_ln = None
        for i, line in enumerate(text.splitlines(), start=1):
            l = line.lower()
            ls = sum(l.count(t) for t in tokens[:8])
            if ls > 0 and (best is None or ls > best):
                best = ls
                best_ln = (i, line.strip()[:240])
                if ls >= 5:
                    break

        results.append((score, path, best_ln))

        # Keep fast path fast.
        if time.monotonic() - started > 2.5 and len(results) >= top_k:
            break

    results.sort(key=lambda x: x[0], reverse=True)
    out_lines = ["Fast search results (lexical, scoped):"]
    for score, path, best_ln in results[:top_k]:
        if best_ln:
            ln, snippet = best_ln
            out_lines.append(f"- {path}:{ln} (score={score}): {snippet}")
        else:
            out_lines.append(f"- {path} (score={score})")

    if len(out_lines) == 1:
        return "Fast search: no matches."
    return "\n".join(out_lines)


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
        sys.stderr.write(f"[RAG V2] Using cached instance for {key}\n")
        return _rag_system_instances[key]
    
    sys.stderr.write(f"[RAG V2] Creating new instance for {key}...\n")
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
        sys.stderr.write("[RAG V2] Initialization complete\n")
        return instance
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR: Could not initialize RAGPipeline: {e}\n")
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

    indexable_extensions = ('.pdf', '.txt', '.py', '.md', '.html', '.htm', '.ipynb', '.js', '.ts', '.tsx', '.jsx')
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
            sys.stderr.write(f"[RAG] Permission denied: {path}\n")

    sys.stderr.write(f"[RAG] Scanning: {directory}\n")
    if include_patterns:
        sys.stderr.write(f"[RAG] Include patterns: {include_patterns}\n")
    if exclude_patterns:
        sys.stderr.write(f"[RAG] Exclude patterns: {exclude_patterns}\n")
    if max_files:
        sys.stderr.write(f"[RAG] Max files: {max_files}\n")

    walk_dir(directory)
    sys.stderr.write(f"[RAG] Found {len(all_files)} indexable files\n")

    return all_files

def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None, file_list: Optional[list] = None) -> str:
    """Query RAG system with robust error handling

    Args:
        query: Search query string
        doc_path: Optional path to documents (file or directory)
        file_list: Optional list of specific file paths to index

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

        # For ad hoc scopes (explicit file list or doc_path), default to a fast
        # lexical search to keep MCP tool calls responsive and offline-friendly.
        # Opt into semantic RAG with `RAG_SEARCH_MODE=semantic`.
        search_mode = os.getenv("RAG_SEARCH_MODE", "fast").strip().lower()

        # If file_list provided, use it directly
        if file_list and len(file_list) > 0:
            # Validate files exist
            valid_files = [f for f in file_list if os.path.isfile(f)]
            if not valid_files:
                return f"Error: No valid files found in provided list"
            file_list = valid_files
            doc_path = None
        # If custom path provided, find all indexable files
        elif doc_path:
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
                        exclude_patterns=['test_*.py', '*_test.py', '*__pycache__*', '*.pyc']
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

        if search_mode != "semantic" and file_list:
            return _fast_search_files(query, file_list, top_k=RAG_CONFIG.get("top_k", 5))

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
