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

def extract_search_topic(full_query: str) -> str:
    """
    Use LLM to extract the actual search topic from a query that may include location info.
    
    Examples:
    "Napoleon in Downloads" -> "Napoleon"
    "AI research on Desktop" -> "AI research"
    "machine learning" -> "machine learning"
    """
    import ollama
    
    prompt = f"""Extract only the topic/subject that the user wants to search for, removing any location/path information.

Examples:
- "Napoleon in Downloads" → "Napoleon"
- "AI research on Desktop" → "AI research"  
- "python code in tools folder" → "python code"
- "machine learning" → "machine learning"
- "what is in my documents" → "" (no specific topic)

Query: "{full_query}"

Return ONLY the topic, nothing else:"""
    
    try:
        response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}])
        topic = response['message']['content'].strip().strip('"').strip("'")
        return topic if topic else full_query
    except Exception as e:
        print(f"[RAG] LLM topic extraction failed: {e}, using original query")
        return full_query

def find_relevant_files(query: str, directory: str, max_files: int = 5) -> list:
    """Find files in directory whose names match query keywords"""
    if not os.path.isdir(directory):
        return []
    
    # Extract just the topic from the full query
    topic = extract_search_topic(query)
    print(f"[RAG] Extracted topic: '{topic}' from query: '{query}'")
    
    query_words = topic.lower().split()
    
    # If no meaningful topic, return first few indexable files
    if not query_words or not topic:
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

def fetchExternalKnowledge(query: str, doc_path: Optional[str] = None) -> str:
    try:
        if not isinstance(query, str) or not query:
            return "Error: Invalid or empty query provided."
        
        # If custom path provided, find relevant files first
        file_list = None
        if doc_path:
            resolved_path = resolve_path(doc_path)
            if not os.path.exists(resolved_path):
                return f"Error: Path does not exist: {resolved_path}"
            
            if os.path.isdir(resolved_path):
                # Find relevant files based on query
                relevant_files = find_relevant_files(query, resolved_path)
                
                if not relevant_files:
                    return f"No files matching '{query}' found in {resolved_path}"
                
                print(f"[RAG] Found {len(relevant_files)} relevant files: {[os.path.basename(f) for f in relevant_files]}")
                
                # Use only relevant files for indexing
                file_list = relevant_files
                doc_path = None  # Clear doc_path, use file_list instead
        
        pipeline = get_rag_pipeline(doc_path=doc_path, file_list=file_list)
        return pipeline.search(query)
    except Exception as e:
        return f"Sorry, an error occurred while searching: {e}"
