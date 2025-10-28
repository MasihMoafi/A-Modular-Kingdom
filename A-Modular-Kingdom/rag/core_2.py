import os, re, string, fitz
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        
        # Use Ollama embeddings if specified
        embed_provider = self.config.get("embed_provider", "sentencetransformer")
        if embed_provider == "ollama":
            from .embeddings import OllamaEmbeddingFunction
            embeddings_model = OllamaEmbeddingFunction(model=self.config.get("embed_model"))
            self.embeddings = embeddings_model
        else:
            self.embeddings = SentenceTransformerEmbeddings(model_name=self.config.get("embed_model"))
        
        self.reranker = CrossEncoder(self.config.get("reranker_model"))
        self.vector_db = self._load_or_create_database()

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

    def _load_or_create_database(self) -> FAISS:
        persist_dir = self.config.get("persist_dir")
        force_reindex = self.config.get("force_reindex", False)
        
        # Check if documents have changed
        if os.path.exists(persist_dir) and not force_reindex:
            if self._has_new_documents():
                print(f"[RAG V2] New documents detected, triggering re-indexing...")
                force_reindex = True
        
        if not os.path.exists(persist_dir) or force_reindex:
            print(f"Creating new FAISS database at {persist_dir}...")
            all_docs = []
            for path in self.config.get("document_paths"):
                print(f"[RAG V2] Checking path: {path}")
                if not os.path.exists(path):
                    print(f"[RAG V2] Path does not exist: {path}")
                    continue
                if os.path.isdir(path):
                    for file in os.listdir(path):
                        file_path = os.path.join(path, file)
                        if file.lower().endswith(('.pdf', '.txt', '.py', '.md')):
                            text = self._extract_text_from_pdf(file_path) if file.lower().endswith(".pdf") else open(file_path, 'r', encoding='utf-8').read()
                            if text:
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.get("chunk_size"), chunk_overlap=self.config.get("chunk_overlap"))
                                chunks = text_splitter.split_text(text)
                                for i, chunk in enumerate(chunks):
                                    if len(chunk.strip()) > 50:
                                        all_docs.append(Document(page_content=chunk, metadata={"source": file_path, "chunk_id": i}))
                else:
                    text = self._extract_text_from_pdf(path) if path.lower().endswith(".pdf") else open(path, 'r', encoding='utf-8').read()
                    if not text: continue
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.get("chunk_size"), chunk_overlap=self.config.get("chunk_overlap"))
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:
                            all_docs.append(Document(page_content=chunk, metadata={"source": path, "chunk_id": i}))
            if not all_docs: raise ValueError("No documents processed.")
            # Create FAISS with proper embedding function
            if self.config.get("embed_provider") == "ollama":
                # For Ollama, we need to wrap the embedding function
                from langchain_community.embeddings import OllamaEmbeddings
                embeddings_wrapper = OllamaEmbeddings(model=self.config.get("embed_model"))
                vector_db = FAISS.from_documents(all_docs, embeddings_wrapper)
            else:
                vector_db = FAISS.from_documents(all_docs, self.embeddings)
            vector_db.save_local(persist_dir)
            self._save_document_hashes()
            return vector_db
        else:
            print(f"Loading existing FAISS database from {persist_dir}...")
            # Load with proper embedding function
            if self.config.get("embed_provider") == "ollama":
                from langchain_community.embeddings import OllamaEmbeddings
                embeddings_wrapper = OllamaEmbeddings(model=self.config.get("embed_model"))
                return FAISS.load_local(persist_dir, embeddings_wrapper, allow_dangerous_deserialization=True)
            else:
                return FAISS.load_local(persist_dir, self.embeddings, allow_dangerous_deserialization=True)
    
    def _get_document_hashes(self):
        """Get hash of each document file for change detection."""
        import hashlib
        doc_hashes = {}
        
        for path in self.config.get("document_paths", []):
            if not os.path.exists(path):
                continue
                
            if os.path.isdir(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if file.lower().endswith(('.pdf', '.txt', '.py', '.md')) and os.path.isfile(file_path):
                        doc_hashes[file_path] = self._hash_file(file_path)
            elif os.path.isfile(path):
                doc_hashes[path] = self._hash_file(path)
        
        return doc_hashes
    
    def _hash_file(self, file_path):
        """Generate hash of file modification time and size."""
        import hashlib
        try:
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            return hashlib.md5(f"{file_path}:{mtime}:{size}".encode()).hexdigest()
        except Exception:
            return ""
    
    def _save_document_hashes(self):
        """Save document hashes for change detection."""
        import json
        persist_dir = self.config.get("persist_dir")
        hash_file = os.path.join(persist_dir, "doc_hashes.json")
        
        doc_hashes = self._get_document_hashes()
        with open(hash_file, 'w') as f:
            json.dump(doc_hashes, f, indent=2)
    
    def _has_new_documents(self):
        """Check if any documents have been added or modified."""
        import json
        persist_dir = self.config.get("persist_dir")
        hash_file = os.path.join(persist_dir, "doc_hashes.json")
        
        if not os.path.exists(hash_file):
            return True
        
        try:
            with open(hash_file, 'r') as f:
                stored_hashes = json.load(f)
        except Exception:
            return True
        
        current_hashes = self._get_document_hashes()
        
        # Check for new or modified files
        for path, hash_val in current_hashes.items():
            if path not in stored_hashes or stored_hashes[path] != hash_val:
                print(f"[RAG V2] Detected change in: {os.path.basename(path)}")
                return True
        
        # Check for deleted files
        for path in stored_hashes:
            if path not in current_hashes:
                print(f"[RAG V2] Detected deletion: {os.path.basename(path)}")
                return True
        
        return False

    def search(self, query: str) -> str:
        top_k = self.config.get("top_k", 5)
        vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": top_k})
        docs_for_bm25 = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in self.vector_db.docstore._dict.values()]
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = top_k
        ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=self.config.get("ensemble_weights"))
        initial_results = ensemble_retriever.get_relevant_documents(query)
        if not initial_results:
            return "No relevant information found."
        
        # Boost code files if query contains code-related keywords
        code_keywords = ['def ', 'async ', 'class ', 'import ', 'function', 'playwright', 'await ', 'return ', '.py']
        is_code_query = any(keyword in query.lower() for keyword in code_keywords)
        
        if is_code_query:
            print(f"[RAG V2] Code query detected, boosting .py files...")
            # Boost scores for Python files
            boosted_results = []
            for doc in initial_results:
                source = doc.metadata.get('source', '')
                if source.endswith('.py'):
                    # Add Python files multiple times to boost their presence
                    boosted_results.extend([doc] * 3)
                else:
                    boosted_results.append(doc)
            initial_results = boosted_results[:top_k * 2]  # Keep reasonable size
        
        print(f"[RAG V2] Re-ranking {len(initial_results)} initial results...")
        rerank_top_k = self.config.get("rerank_top_k", 3)
        pairs = [[query, doc.page_content] for doc in initial_results]
        scores = self.reranker.predict(pairs)
        scored_results = zip(scores, initial_results)
        sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
        final_results = [doc for score, doc in sorted_results[:rerank_top_k]]
        
        # Format results with source information for code files
        formatted_results = []
        seen_content = set()
        for doc in final_results:
            content = doc.page_content
            if content in seen_content:
                continue
            seen_content.add(content)
            
            source = doc.metadata.get('source', '')
            if source.endswith('.py'):
                filename = os.path.basename(source)
                formatted_results.append(f"# From: {filename}\n\n{content}")
            else:
                formatted_results.append(content)
        
        return "Document search results:\n\n" + "\n\n---\n\n".join(formatted_results)
