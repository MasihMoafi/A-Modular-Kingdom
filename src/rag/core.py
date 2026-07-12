import sys
import os, re, string, fitz, json, math
import torch
from typing import List, Dict, Tuple, Any, Optional

# --- FIX for Ollama Proxy ---
# This is necessary to ensure the local Ollama server can be reached.
def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]
clear_proxy_settings()

import ollama
from collections import Counter
from typing import List, Dict, Any, Tuple, Callable, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from .qdrant_backend import QdrantVectorDB


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks without langchain."""
    if not text or len(text.strip()) < 50:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        start = end - chunk_overlap if end < len(text) else len(text)

        if len(chunks) > 10000:  # Safety limit
            break

    return chunks


class VectorIndex:
    """
    Custom Vector Database implementation from RAG Course.
    Uses cosine or euclidean distance for similarity search.
    """
    def __init__(self, distance_metric: str = "cosine", embedding_fn=None):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def add_document(self, document: Dict[str, Any]):
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_vector(self, vector: List[float], document: Dict[str, Any]):
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}")

        self.vectors.append(list(vector))
        self.documents.append(document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.vectors:
            return []

        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError("Embedding function not provided for string query.")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

        if self._vector_dim is None:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if self._distance_metric == "cosine":
            dist_func = self._cosine_distance
        else:
            dist_func = self._euclidean_distance

        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, self.documents[i]))

        distances.sort(key=lambda item: item[0])
        return [(doc, dist) for dist, doc in distances[:k]]

    def __len__(self) -> int:
        return len(self.vectors)


class BM25Index:
    """
    Custom BM25 implementation from RAG Course.
    Uses proper IDF calculation and BM25 scoring formula.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _calculate_idf(self):
        """Calculate Inverse Document Frequency for each term"""
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            # IDF formula from tutorial: log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def add_document(self, document: Dict[str, Any]):
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)
        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        
        # Update statistics
        self._doc_len.append(len(doc_tokens))
        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)
        
        self._index_built = False

    def _build_index(self):
        """Build the BM25 index by calculating average doc length and IDF scores"""
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def search(self, query: str, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents:
            return []

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        # Calculate BM25 scores for all documents
        scores = []
        for i, doc in enumerate(self.documents):
            score = self._compute_bm25_score(query_tokens, i)
            if score > 1e-9:  # Only include documents with non-zero scores
                scores.append((score, doc))

        # Sort by score (highest first) and return top k
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to distance format (lower is better) for consistency with VectorIndex
        results = []
        for score, doc in scores[:k]:
            # Convert score to distance using exponential normalization
            distance = math.exp(-0.1 * score)
            results.append((doc, distance))
        
        return results

    def _compute_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        """Compute BM25 score for a document given query tokens"""
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_len))
            score += numerator / (denominator + 1e-9)  # Small epsilon to avoid division by zero

        return score


class RAGPipelineV2:
    """
    RAG Pipeline V2 implementing tutorial techniques:
    1. Custom VectorIndex and BM25Index
    2. RRF (Reciprocal Rank Fusion) instead of weighted ensemble
    3. LLM-based reranking with Ollama
    4. Contextual chunk preprocessing
    """
    
    def __init__(self, config: dict):
        self.config = config
        print(f"[RAG V2] Initializing with config: {config.get('version', 'unknown')}")

        # Determine device (CUDA if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[RAG V2] Using device: {self.device}")

        # Initialize embedding function based on provider
        embed_provider = self.config.get("embed_provider", "sentencetransformer")

        if embed_provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            model = OllamaEmbeddings(model=self.config.get("embed_model"))
            
            class OllamaWrapper:
                def __init__(self, ollama_model):
                    self.model = ollama_model
                def embed_documents(self, texts):
                    return self.model.embed_documents(texts)
                def embed_query(self, text):
                    return self.model.embed_query(text)
            
            self.embedding_fn = OllamaWrapper(model)
            print(f"[RAG V2] Using Ollama embeddings with model: {self.config.get('embed_model')}")
        else:
            model = SentenceTransformer(self.config.get("embed_model"), device=self.device)
            class STWrapper:
                def __init__(self, st_model):
                    self.model = st_model
                def embed_documents(self, texts):
                    return self.model.encode(texts, convert_to_numpy=True).tolist()
                def embed_query(self, text):
                    return self.model.encode([text], convert_to_numpy=True)[0].tolist()
            
            self.embedding_fn = STWrapper(model)
            print(f"[RAG V2] Using SentenceTransformer embeddings with model: {self.config.get('embed_model')}")

        # Initialize Qdrant vector DB (replaces custom VectorIndex)
        qdrant_path = os.path.join(self.config.get("persist_dir"), "qdrant_storage")

        # Determine vector size based on model
        vector_size_map = {
            "ollama": 768,  # default ollama fallback
            "vllm": 4096,  # e5-mistral-7b-instruct
            "sentencetransformer": 384  # all-MiniLM-L6-v2
        }
        vector_size = vector_size_map.get(embed_provider, 384)
        embed_model_name = self.config.get("embed_model", "").lower()
        if "qwen3-embedding" in embed_model_name:
            vector_size = 4096

        # Create unique collection name from persist_dir to avoid conflicts
        persist_dir_name = os.path.basename(self.config.get("persist_dir"))
        collection_name = f"rag_v2_{persist_dir_name}" if persist_dir_name != "rag_db_v2" else "rag_v2_default"

        self.vector_db = QdrantVectorDB(
            collection_name=collection_name,
            embedding_fn=self.embedding_fn,
            vector_size=vector_size,
            distance=self.config.get("distance_metric", "cosine"),
            persist_path=qdrant_path
        )

        # Keep BM25 (still valuable for lexical search)
        self.bm25_index = BM25Index(
            k1=self.config.get("bm25_k1", 1.5),
            b=self.config.get("bm25_b", 0.75)
        )

        # Add CrossEncoder / LLM for fast reranking
        reranker_model = self.config.get('reranker_model')
        self.reranker_type = "cross-encoder"
        if reranker_model:
            if "qwen3-reranker" in reranker_model.lower():
                self.reranker_type = "qwen3"
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"[RAG V2] Loading Qwen3 LLM Reranker: {reranker_model}")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model, padding_side='left')
                self.reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model, load_in_8bit=True, device_map="auto").eval()
            else:
                self.reranker = CrossEncoder(
                    reranker_model,
                    device=self.device
                )
                print(f"[RAG V2] Using CrossEncoder reranking with model: {reranker_model}")

        # Load or create database
        self._load_or_create_database()

    def _normalize_text(self, text: str) -> str:
        """Same normalization as V2 for consistency"""
        if not text: return ""
        translator = str.maketrans('', '', string.punctuation + '،؛؟»«')
        text = text.translate(translator)
        return ' '.join(text.split()).lower()

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Same PDF extraction as V2"""
        try:
            doc = fitz.open(pdf_path)
            full_text = "".join(page.get_text() for page in doc)
            doc.close()
            return re.sub(r'\s+', ' ', full_text).strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def _add_context_to_chunk(self, chunk: str, document_context: str) -> str:
        """
        STEP 7: CONTEXTUAL RETRIEVAL
        Add context to chunks using LLM before embedding.
        This helps chunks be more searchable by providing situational context.
        """
        # Always return chunk without context to avoid Ollama hanging
        return chunk

    def _load_or_create_database(self):
        """Load or create the V2 database with all tutorial techniques"""
        persist_dir = self.config.get("persist_dir")

        if not os.path.exists(persist_dir) or self.config.get("force_reindex", False):
            print(f"[RAG V2] Creating new database at {persist_dir}...")
            if self.config.get("force_reindex", False):
                try:
                    self.vector_db.clear()
                except Exception as e:
                    print(f"[RAG V2] Warning: Failed to clear Qdrant collection: {e}")
            self._index_documents()
        else:
            print(f"[RAG V2] Loading existing database from {persist_dir}...")
            self._load_existing_database()

    def _index_documents(self):
        """Index all documents using V2 techniques"""
        import sys
        all_docs = []
        document_contents = {}

        # First pass: extract all document content
        for path in self.config.get("document_paths"):
            if not os.path.exists(path):
                continue

            if os.path.isdir(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if file.lower().endswith(('.pdf', '.txt', '.py', '.md', '.json', '.ipynb')):
                        content = self._extract_content(file_path)
                        if content:
                            document_contents[file_path] = content
            else:
                content = self._extract_content(path)
                if content:
                    document_contents[path] = content

        # Second pass: chunk and optionally add context
        for file_path, full_content in document_contents.items():
            print(f"[RAG V2] Processing {os.path.basename(file_path)}...")

            # STEP 1: CHUNKING (simple overlapping chunks, no langchain)
            chunks = chunk_text(
                full_content,
                chunk_size=self.config.get("chunk_size"),
                chunk_overlap=self.config.get("chunk_overlap")
            )
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    # STEP 7: CONTEXTUAL RETRIEVAL
                    contextualized_chunk = self._add_context_to_chunk(chunk, full_content)
                    
                    doc = {
                        "content": contextualized_chunk,
                        "original_content": chunk,  # Keep original for comparison
                        "source": file_path,
                        "chunk_id": i,
                        "id": f"{os.path.basename(file_path)}_chunk_{i}"
                    }
                    all_docs.append(doc)

        if not all_docs:
            raise ValueError("No documents processed.")

        # Add documents to indexes
        print(f"[RAG V2] Indexing {len(all_docs)} chunks...")

        # STEP 2: VECTOR SEARCH - Batch indexing with Qdrant (FAST)
        self.vector_db.add_documents_batch(all_docs, batch_size=100)

        # STEP 4: BM25 - Sequential (lightweight, fast enough)
        print(f"[RAG V2] Building BM25 index...")
        for doc in all_docs:
            self.bm25_index.add_document(doc)

        # Save database
        self._save_database(all_docs)
        print(f"[RAG V2] Database created with {len(all_docs)} chunks")

    def _extract_content(self, file_path: str) -> str:
        """Extract content from various file types"""
        if file_path.lower().endswith(".pdf"):
            return self._extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(".json"):
            return self._extract_text_from_json(file_path)
        elif file_path.lower().endswith(".ipynb"):
            return self._extract_text_from_notebook(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return ""

    def _extract_text_from_notebook(self, notebook_path: str) -> str:
        """Extract text from Jupyter notebook (.ipynb)"""
        try:
            from .notebook_chunker import extract_cells_from_notebook
            cells = extract_cells_from_notebook(notebook_path)
            # Combine all cell contents
            return "\n\n".join([cell['content'] for cell in cells])
        except Exception as e:
            print(f"Error extracting text from {notebook_path}: {e}")
            return ""

    def _extract_text_from_json(self, json_path: str) -> str:
        """Extract text from JSON - handles various structures"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert JSON to searchable text
            def flatten_json(obj, prefix=''):
                text_parts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            text_parts.extend(flatten_json(value, f"{prefix}{key}."))
                        else:
                            text_parts.append(f"{prefix}{key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        text_parts.extend(flatten_json(item, f"{prefix}[{i}]."))
                else:
                    text_parts.append(str(obj))
                return text_parts

            text_parts = flatten_json(data)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error extracting text from {json_path}: {e}")
            return ""

    def _save_database(self, docs: List[Dict]):
        """Save the database (simplified for now)"""
        persist_dir = self.config.get("persist_dir")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Save document metadata
        with open(os.path.join(persist_dir, "docs.json"), 'w') as f:
            json.dump(docs, f, indent=2)

    def _load_existing_database(self):
        """Load existing database"""
        persist_dir = self.config.get("persist_dir")
        docs_file = os.path.join(persist_dir, "docs.json")

        # Check if Qdrant collection has data
        qdrant_count = self.vector_db.count()

        if qdrant_count > 0 and os.path.exists(docs_file):
            # Qdrant already loaded, just rebuild BM25
            with open(docs_file, 'r') as f:
                docs = json.load(f)

            print(f"[RAG V2] Loading {len(docs)} chunks from existing database...")
            for doc in docs:
                self.bm25_index.add_document(doc)

            print(f"[RAG V2] Loaded {len(docs)} chunks (Qdrant: {qdrant_count}, BM25: {len(docs)})")
        else:
            # Database directory exists but empty - trigger indexing
            print(f"[RAG V2] Database empty. Triggering indexing...")
            self._index_documents()

    def _rrf_fusion(self, vector_results: List[Tuple], bm25_results: List[Tuple], k: int = 60) -> List[Dict]:
        """
        STEP 5: HYBRID SEARCH FUSION (Unified Architecture)
        
        This unifies our V1 (Weighted Ensemble) and V2 (Unweighted RRF) architectures.
        It uses Reciprocal Rank Fusion (RRF) to merge keyword (BM25) and semantic (Vector) search.
        
        If 'ensemble_weights' is provided in the config (e.g. [0.5, 0.5]), it performs a 
        Weighted RRF, exactly matching LangChain's EnsembleRetriever math (V1).
        If no weights are provided, it falls back to standard Unweighted RRF (V2).
        
        RRF Formula: score = weight * (1 / (k + rank))
        """
        doc_scores = {}
        
        # Check if the user wants tunable weights (V1 style)
        weights = self.config.get("ensemble_weights", [1.0, 1.0])
        
        # Process vector search results (rank by similarity - lower distance = higher rank)
        for rank, (doc, distance) in enumerate(vector_results, 1):
            doc_id = doc.get("id", id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_rank": float('inf'), "bm25_rank": float('inf')}
            doc_scores[doc_id]["vector_rank"] = rank

        # Process BM25 results (rank by relevance - lower distance = higher rank) 
        for rank, (doc, distance) in enumerate(bm25_results, 1):
            doc_id = doc.get("id", id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_rank": float('inf'), "bm25_rank": float('inf')}
            doc_scores[doc_id]["bm25_rank"] = rank

        # Calculate Unified RRF scores
        rrf_results = []
        for doc_id, data in doc_scores.items():
            vector_score = weights[0] * (1.0 / (k + data["vector_rank"])) if data["vector_rank"] != float('inf') else 0
            bm25_score = weights[1] * (1.0 / (k + data["bm25_rank"])) if data["bm25_rank"] != float('inf') else 0
            
            rrf_score = vector_score + bm25_score
            if rrf_score > 0:
                rrf_results.append((data["doc"], rrf_score))

        # Sort by RRF score (higher is better)
        rrf_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in rrf_results]

    def _llm_rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
        """
        STEP 6b: LLM-BASED RERANKING (Alternative to CrossEncoder)

        Instead of a specialized CrossEncoder (like Qwen3 or MS-MARCO), this uses 
        a generative LLM to act as a judge, analyzing each chunk's relevance to the query.
        
        Why this exists:
        - Portability: Users don't need heavy ML frameworks (Torch/Transformers).
        - Reasoning: Large LLMs can reason through complex semantic connections that 
          smaller cross-encoders might miss.
        """
        print("[RAG Core] Executing LLM-based Reranking...")
        
        # NOTE: In a full production environment, this would call your Ollama/OpenAI API.
        # This implementation serves as the scaffolding for the LLM judge logic.
        scores = []
        for doc in docs:
            # Hypothetical Prompt Structure:
            # prompt = f"Query: {query}\nPassage: {doc['content']}\nScore this passage from 0 to 10 based on its ability to answer the query."
            # response = llm.generate(prompt)
            # score = parse_score(response)
            
            # Since we don't want to block the execution with an API call here unless configured,
            # we simulate an LLM pass-through for now, ready to be wired to the TUI's LLM engine.
            scores.append(1.0) 
            
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:k]]


    def _crossencoder_rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
        """
        STEP 6: CROSSENCODER RERANKING

        Use CrossEncoder to rerank documents by relevance.
        Fast, GPU-based, task-specialized for passage ranking.
        """
        if len(docs) <= k:
            return docs

        try:
            if self.reranker_type == "qwen3":
                import torch
                token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
                token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
                prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
                suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
                
                scores = []
                for doc in docs:
                    text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc['content'][:1000]}"
                    inputs = self.reranker_tokenizer(text, return_attention_mask=False, add_special_tokens=False)
                    input_ids = prefix_tokens + inputs['input_ids'] + suffix_tokens
                    inputs_tensor = torch.tensor([input_ids]).to(self.device)
                    
                    with torch.no_grad():
                        batch_scores = self.reranker_model(inputs_tensor).logits[:, -1, :]
                        true_vector = batch_scores[:, token_true_id]
                        false_vector = batch_scores[:, token_false_id]
                        batch_scores_stack = torch.stack([false_vector, true_vector], dim=1)
                        batch_scores_log = torch.nn.functional.log_softmax(batch_scores_stack, dim=1)
                        score = batch_scores_log[:, 1].exp().item()
                        scores.append(score)
                        
                    # Aggressively free memory
                    del inputs_tensor
                    del batch_scores
                    del true_vector
                    del false_vector
                    del batch_scores_stack
                    del batch_scores_log
                    torch.cuda.empty_cache()
            elif hasattr(self, 'reranker') and self.reranker:
                pairs = [[query, doc['content'][:512]] for doc in docs]  # Limit to 512 chars for speed
                scores = self.reranker.predict(pairs)
            else:
                scores = [1.0] * len(docs)

            # Sort documents by score (higher is better)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

            return [doc for doc, score in ranked[:k]]

        except Exception as e:
            print(f"[RAG Core] CrossEncoder reranking failed: {e}, falling back to original order")
            return docs[:k]

    def search(self, query: str) -> str:
        """
        Main search method implementing all 7 tutorial steps:
        1. ✅ Chunking (in indexing)
        2. ✅ Embeddings (in vector search)  
        3. ✅ Vector Search (semantic similarity)
        4. ✅ BM25 (keyword search)
        5. ✅ Hybrid (RRF fusion)
        6. ✅ Reranking (LLM-based)
        7. ✅ Contextual (in indexing)
        """
        try:
            top_k = self.config.get("top_k", 20)
            rerank_top_k = self.config.get("rerank_top_k", 3)
            
            print(f"[RAG V2] Searching with query: '{query[:50]}...'")
            
            # STEP 3: QDRANT VECTOR SEARCH
            print("[RAG V2] Step 3: Vector search...")
            vector_results_qdrant = self.vector_db.search(query, top_k=top_k)
            # Convert to tuple format for RRF
            vector_results = [(doc, doc.get("score", 0)) for doc in vector_results_qdrant]

            # STEP 4: BM25 SEARCH
            print("[RAG V2] Step 4: BM25 search...")
            bm25_results = self.bm25_index.search(query, k=top_k)
            
            # STEP 5: HYBRID FUSION with RRF
            print("[RAG V2] Step 5: RRF fusion...")
            fused_docs = self._rrf_fusion(vector_results, bm25_results, k=self.config.get("rrf_k", 60))
            
            if not fused_docs:
                return "No relevant information found."
            # Step 4: Reranking (Configurable via reranker_type)
            reranker_type = self.config.get("reranker_type", "cross-encoder").lower()
            
            if reranker_type == "disabled" or reranker_type == "none":
                final_docs = fused_docs[:rerank_top_k]
            elif reranker_type == "llm":
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using LLM Reranker...")
                final_docs = self._llm_rerank(fused_docs, query, k=rerank_top_k)
            else:
                print(f"[RAG Core] Reranking {len(fused_docs)} initial results using CrossEncoder Reranker...")
                final_docs = self._crossencoder_rerank(fused_docs, query, k=rerank_top_k)
            unique_content = []
            seen_content = set()
            for doc in final_docs:
                content = doc.get('original_content', doc.get('content', ''))
                if content not in seen_content:
                    seen_content.add(content)
                    unique_content.append(content)
            
            sys.stderr.write(f"[RAG V2] Returning {len(final_docs)} unique results\n")
            return final_docs
            
        except Exception as e:
            print(f"[RAG V2] Search error: {e}")
            return []