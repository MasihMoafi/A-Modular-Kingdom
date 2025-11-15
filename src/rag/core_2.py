import os, re, string, fitz, json, hashlib
import torch
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

# Import Qdrant backend
from .qdrant_backend import QdrantVectorDB

# Import notebook chunker
from .notebook_chunker import process_notebook_for_rag

class RAGPipeline:
    def __init__(self, config: dict):
        self.config = config

        # Determine device (CUDA if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[RAG V2] Using device: {self.device}")

        # Use Ollama embeddings if specified
        embed_provider = self.config.get("embed_provider", "sentencetransformer")
        if embed_provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            # Note: OllamaEmbeddings runs on Ollama server, not local GPU
            embeddings_model = OllamaEmbeddings(model=self.config.get("embed_model"))
            print(f"[RAG V2] Using Ollama embeddings with model: {self.config.get('embed_model')}")
        else:
            embeddings_model = SentenceTransformerEmbeddings(
                model_name=self.config.get("embed_model"),
                model_kwargs={'device': self.device}
            )
            print(f"[RAG V2] Using SentenceTransformer embeddings with model: {self.config.get('embed_model')}")

        self.embeddings = embeddings_model

        # CrossEncoder with GPU support
        self.reranker = CrossEncoder(
            self.config.get("reranker_model"),
            device=self.device
        )

        # Initialize Qdrant vector DB (replaces FAISS)
        qdrant_path = os.path.join(self.config.get("persist_dir"), "qdrant_storage")

        # Determine vector size based on provider
        vector_size_map = {
            "ollama": 768,  # embeddinggemma
            "sentencetransformer": 384  # all-MiniLM-L6-v2
        }
        vector_size = vector_size_map.get(embed_provider, 384)

        # Create unique collection name from persist_dir to avoid conflicts
        persist_dir_name = os.path.basename(self.config.get("persist_dir"))
        collection_name = f"rag_v2_{persist_dir_name}" if persist_dir_name != "rag_db_v2" else "rag_v2_default"

        self.vector_db = QdrantVectorDB(
            collection_name=collection_name,
            embedding_fn=self.embeddings,
            vector_size=vector_size,
            distance=self.config.get("distance_metric", "cosine"),
            persist_path=qdrant_path
        )

        # Store documents for BM25
        self.all_documents: List[Document] = []

        # Load or create database
        self._load_or_create_database()

    def _normalize_text(self, text: str) -> str:
        if not text: return ""
        translator = str.maketrans('', '', string.punctuation + '،؛؟»«')
        text = text.translate(translator)
        return ' '.join(text.split()).lower()

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            full_text = "".join(page.get_text() for page in doc)
            doc.close()
            return re.sub(r'\s+', ' ', full_text).strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
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

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file modification time and size for change detection"""
        try:
            stat = os.stat(file_path)
            # Use mtime and size for faster checking than full content hash
            hash_input = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}")
            return ""

    def _get_all_source_files(self) -> list:
        """Get list of all source files from document_paths"""
        all_files = []
        for path in self.config.get("document_paths"):
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(('.pdf', '.txt', '.py', '.md', '.json', '.ipynb')):
                        all_files.append(file_path)
            else:
                all_files.append(path)
        return all_files

    def _build_manifest(self) -> dict:
        """Build manifest of all source files with their hashes"""
        files = self._get_all_source_files()
        return {f: self._get_file_hash(f) for f in files}

    def _save_manifest(self, persist_dir: str, manifest: dict):
        """Save manifest to disk"""
        manifest_path = os.path.join(persist_dir, "manifest.json")
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Error saving manifest: {e}")

    def _load_manifest(self, persist_dir: str) -> dict:
        """Load manifest from disk"""
        manifest_path = os.path.join(persist_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return {}
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading manifest: {e}")
            return {}

    def _files_changed(self, persist_dir: str) -> bool:
        """Check if any source files have changed since last index"""
        old_manifest = self._load_manifest(persist_dir)
        new_manifest = self._build_manifest()

        # Check if file lists differ
        old_files = set(old_manifest.keys())
        new_files = set(new_manifest.keys())

        if old_files != new_files:
            added = new_files - old_files
            removed = old_files - new_files
            if added:
                print(f"[RAG V2] New files detected: {[os.path.basename(f) for f in added]}")
            if removed:
                print(f"[RAG V2] Removed files detected: {[os.path.basename(f) for f in removed]}")
            return True

        # Check if any file hashes changed
        for file_path, new_hash in new_manifest.items():
            old_hash = old_manifest.get(file_path)
            if old_hash != new_hash:
                print(f"[RAG V2] File changed: {os.path.basename(file_path)}")
                return True

        return False

    def _load_or_create_database(self):
        """Load or create Qdrant database"""
        persist_dir = self.config.get("persist_dir")
        docs_file = os.path.join(persist_dir, "docs.json")

        # Check if database exists and is populated
        qdrant_count = self.vector_db.count()
        needs_reindex = (
            qdrant_count == 0 or
            self.config.get("force_reindex", False) or
            self._files_changed(persist_dir)
        )

        if needs_reindex:
            print(f"[RAG V2] Creating new Qdrant database at {persist_dir}...")
            all_docs = []
            all_qdrant_docs = []  # For Qdrant batch indexing

            for path in self.config.get("document_paths"):
                if not os.path.exists(path): continue
                if os.path.isdir(path):
                    for file in os.listdir(path):
                        file_path = os.path.join(path, file)

                        # Handle Jupyter notebooks specially
                        if file.lower().endswith('.ipynb'):
                            print(f"[RAG V2] Processing notebook: {file}")
                            notebook_chunks = process_notebook_for_rag(file_path, max_chunk_size=2000)
                            for cell_data in notebook_chunks:
                                content = cell_data['content']
                                if len(content.strip()) > 50:
                                    # Build rich metadata
                                    metadata = {
                                        "source": file_path,
                                        "cell_type": cell_data.get('cell_type'),
                                        "cell_id": cell_data.get('cell_id'),
                                        "cell_index": cell_data.get('cell_index'),
                                        "exercise_number": cell_data.get('exercise_number'),
                                        "chunk_id": cell_data.get('chunk_id'),
                                        "is_partial": cell_data.get('is_partial', False)
                                    }
                                    # Langchain Document for BM25
                                    all_docs.append(Document(
                                        page_content=content,
                                        metadata=metadata
                                    ))
                                    # Dict for Qdrant
                                    all_qdrant_docs.append({
                                        "content": content,
                                        **metadata
                                    })

                        # Handle other file types
                        elif file.lower().endswith(('.pdf', '.txt', '.py', '.md', '.json')):
                            if file.lower().endswith(".pdf"):
                                text = self._extract_text_from_pdf(file_path)
                            elif file.lower().endswith(".json"):
                                text = self._extract_text_from_json(file_path)
                            else:
                                text = open(file_path, 'r', encoding='utf-8').read()
                            if text:
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=self.config.get("chunk_size"),
                                    chunk_overlap=self.config.get("chunk_overlap")
                                )
                                chunks = text_splitter.split_text(text)
                                for i, chunk in enumerate(chunks):
                                    if len(chunk.strip()) > 50:
                                        # Langchain Document for BM25
                                        all_docs.append(Document(
                                            page_content=chunk,
                                            metadata={"source": file_path, "chunk_id": i}
                                        ))
                                        # Dict for Qdrant
                                        all_qdrant_docs.append({
                                            "content": chunk,
                                            "source": file_path,
                                            "chunk_id": i
                                        })
                else:
                    text = self._extract_text_from_pdf(path) if path.lower().endswith(".pdf") else open(path, 'r', encoding='utf-8').read()
                    if not text: continue
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.config.get("chunk_size"),
                        chunk_overlap=self.config.get("chunk_overlap")
                    )
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:
                            all_docs.append(Document(
                                page_content=chunk,
                                metadata={"source": path, "chunk_id": i}
                            ))
                            all_qdrant_docs.append({
                                "content": chunk,
                                "source": path,
                                "chunk_id": i
                            })

            if not all_docs:
                raise ValueError("No documents processed.")

            # Index with Qdrant (batch mode - FAST)
            self.vector_db.add_documents_batch(all_qdrant_docs, batch_size=100)

            # Store documents for BM25
            self.all_documents = all_docs

            # Save manifest and docs for future loading
            manifest = self._build_manifest()
            self._save_manifest(persist_dir, manifest)
            os.makedirs(persist_dir, exist_ok=True)
            with open(docs_file, 'w') as f:
                json.dump([{"content": doc.page_content, **doc.metadata} for doc in all_docs], f)

            print(f"[RAG V2] Indexed {len(all_docs)} chunks from {len(manifest)} files")
        else:
            print(f"[RAG V2] Loading existing Qdrant database from {persist_dir}...")
            # Load documents for BM25
            if os.path.exists(docs_file):
                with open(docs_file, 'r') as f:
                    docs_data = json.load(f)
                self.all_documents = [
                    Document(page_content=doc["content"], metadata={k: v for k, v in doc.items() if k != "content"})
                    for doc in docs_data
                ]
                print(f"[RAG V2] Loaded {len(self.all_documents)} documents for BM25")
            else:
                print("[RAG V2] Warning: No docs.json found, BM25 will be empty")

    def search(self, query: str) -> str:
        try:
            print(f"[RAG V2] Searching for: '{query[:50]}...'")
            top_k = self.config.get("top_k", 5)

            # Step 1: Qdrant vector search
            vector_results = self.vector_db.search(query, top_k=top_k)
            # Convert to Langchain Documents
            vector_docs = [
                Document(page_content=doc["content"], metadata={k: v for k, v in doc.items() if k not in ["content", "score"]})
                for doc in vector_results
            ]

            # Step 2: BM25 search
            if not self.all_documents:
                print("[RAG V2] Warning: No documents loaded for BM25, using vector results only")
                initial_results = vector_docs
            else:
                bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                bm25_retriever.k = top_k
                bm25_docs = bm25_retriever.invoke(query)

                # Combine results (simple union - could use RRF like V3)
                combined_docs = vector_docs + bm25_docs
                # Deduplicate by content
                seen = set()
                initial_results = []
                for doc in combined_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        initial_results.append(doc)

            if not initial_results:
                print("[RAG V2] No results found in initial search")
                return "No relevant information found."

            print(f"[RAG V2] Re-ranking {len(initial_results)} initial results...")
            rerank_top_k = self.config.get("rerank_top_k", 3)
            pairs = [[query, doc.page_content] for doc in initial_results]
            scores = self.reranker.predict(pairs)
            scored_results = zip(scores, initial_results)
            sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
            final_results = [doc for score, doc in sorted_results[:rerank_top_k]]
            unique_content = list({doc.page_content for doc in final_results})

            print(f"[RAG V2] Returning {len(unique_content)} unique results")
            return "Document search results:\n\n" + "\n\n---\n\n".join(unique_content)
        except Exception as e:
            error_msg = f"[RAG V2] Search error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"Search failed: {str(e)}"
