"""RAG Pipeline for Memory MCP.

This is the main RAG implementation using:
- Qdrant for vector storage
- BM25 for lexical search
- CrossEncoder for reranking (optional)
"""

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from memory_mcp.embeddings import EmbeddingProvider, get_embedding_provider
from memory_mcp.rag.chunkers import (
    chunk_text,
    extract_text_from_json,
    extract_text_from_pdf,
    process_notebook,
)
from memory_mcp.rag.qdrant_backend import QdrantVectorDB

if TYPE_CHECKING:
    from memory_mcp.config import Settings


class RAGPipeline:
    """RAG Pipeline with hybrid search (vector + BM25) and optional reranking."""

    def __init__(
        self,
        settings: "Settings",
        document_paths: list[str] | None = None,
        collection_name: str | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            settings: Application settings
            document_paths: List of paths to index (files or directories)
            collection_name: Optional custom collection name
        """
        self.settings = settings
        self.document_paths = document_paths or []

        self.embedding_provider = get_embedding_provider(settings)

        persist_path = Path(settings.qdrant_path).expanduser()
        persist_path.mkdir(parents=True, exist_ok=True)

        if collection_name:
            self.collection_name = collection_name
        else:
            path_hash = hashlib.md5(
                "|".join(sorted(self.document_paths)).encode()
            ).hexdigest()[:8]
            self.collection_name = f"rag_{path_hash}"

        self.vector_db = QdrantVectorDB(
            collection_name=self.collection_name,
            embedding_provider=self.embedding_provider,
            distance="cosine",
            persist_path=str(persist_path / "qdrant_storage"),
            mode=settings.qdrant_mode,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        self.all_documents: list[Document] = []
        self._reranker = None

        if settings.rag_rerank:
            self._init_reranker()

        self._manifest_path = persist_path / f"{self.collection_name}_manifest.json"
        self._docs_path = persist_path / f"{self.collection_name}_docs.json"

    def _init_reranker(self):
        """Initialize CrossEncoder reranker if available."""
        try:
            from sentence_transformers import CrossEncoder

            device = self.settings.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._reranker = CrossEncoder(
                self.settings.rag_reranker_model,
                device=device,
            )
        except ImportError:
            pass

    def index(self, force: bool = False) -> int:
        """Index documents from configured paths.

        Args:
            force: Force reindex even if files haven't changed

        Returns:
            Number of chunks indexed
        """
        if not self.document_paths:
            return 0

        needs_reindex = (
            force
            or self.vector_db.count() == 0
            or self._files_changed()
        )

        if not needs_reindex:
            self._load_documents()
            return len(self.all_documents)

        all_docs = []
        all_qdrant_docs = []

        for path in self.document_paths:
            if not os.path.exists(path):
                continue

            if os.path.isdir(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path):
                        docs = self._process_file(file_path)
                        for doc in docs:
                            all_docs.append(
                                Document(
                                    page_content=doc["content"],
                                    metadata={k: v for k, v in doc.items() if k != "content"},
                                )
                            )
                            all_qdrant_docs.append(doc)
            else:
                docs = self._process_file(path)
                for doc in docs:
                    all_docs.append(
                        Document(
                            page_content=doc["content"],
                            metadata={k: v for k, v in doc.items() if k != "content"},
                        )
                    )
                    all_qdrant_docs.append(doc)

        if not all_docs:
            return 0

        self.vector_db.clear()
        self.vector_db.add_documents_batch(all_qdrant_docs, batch_size=100)
        self.all_documents = all_docs
        self._save_manifest()
        self._save_documents()

        return len(all_docs)

    def _process_file(self, file_path: str) -> list[dict[str, Any]]:
        """Process a single file and return chunks."""
        file_lower = file_path.lower()

        if file_lower.endswith(".ipynb"):
            return process_notebook(file_path, max_chunk_size=2000)
        elif file_lower.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_lower.endswith(".json"):
            text = extract_text_from_json(file_path)
        elif file_lower.endswith((".txt", ".py", ".md", ".js", ".ts", ".html", ".css")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                return []
        else:
            return []

        return chunk_text(
            text,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
            source=file_path,
        )

    def search(self, query: str, top_k: int | None = None) -> str:
        """Search the indexed documents.

        Args:
            query: Search query
            top_k: Number of results (defaults to settings.rag_top_k)

        Returns:
            Formatted search results
        """
        if top_k is None:
            top_k = self.settings.rag_top_k

        vector_results = self.vector_db.search(query, top_k=top_k)
        vector_docs = [
            Document(
                page_content=doc["content"],
                metadata={k: v for k, v in doc.items() if k not in ["content", "score"]},
            )
            for doc in vector_results
        ]

        if self.all_documents:
            try:
                from langchain_community.retrievers import BM25Retriever

                bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                bm25_retriever.k = top_k
                bm25_docs = bm25_retriever.invoke(query)

                combined_docs = vector_docs + bm25_docs
                seen = set()
                initial_results = []
                for doc in combined_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        initial_results.append(doc)
            except ImportError:
                initial_results = vector_docs
        else:
            initial_results = vector_docs

        if not initial_results:
            return "No relevant information found."

        if self._reranker and len(initial_results) > 1:
            pairs = [[query, doc.page_content] for doc in initial_results]
            scores = self._reranker.predict(pairs)
            scored_results = zip(scores, initial_results)
            sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
            final_results = [doc for score, doc in sorted_results[: top_k]]
        else:
            final_results = initial_results[:top_k]

        unique_content = list({doc.page_content for doc in final_results})
        return "Document search results:\n\n" + "\n\n---\n\n".join(unique_content)

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for change detection."""
        try:
            stat = os.stat(file_path)
            hash_input = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            return ""

    def _build_manifest(self) -> dict:
        """Build manifest of all source files with hashes."""
        manifest = {}
        for path in self.document_paths:
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path):
                        manifest[file_path] = self._get_file_hash(file_path)
            else:
                manifest[path] = self._get_file_hash(path)
        return manifest

    def _files_changed(self) -> bool:
        """Check if any source files have changed since last index."""
        if not self._manifest_path.exists():
            return True

        try:
            with open(self._manifest_path, "r") as f:
                old_manifest = json.load(f)
        except Exception:
            return True

        new_manifest = self._build_manifest()
        return old_manifest != new_manifest

    def _save_manifest(self):
        """Save manifest to disk."""
        manifest = self._build_manifest()
        try:
            with open(self._manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

    def _save_documents(self):
        """Save documents for BM25 reload."""
        try:
            docs_data = [
                {"content": doc.page_content, **doc.metadata}
                for doc in self.all_documents
            ]
            with open(self._docs_path, "w") as f:
                json.dump(docs_data, f)
        except Exception:
            pass

    def _load_documents(self):
        """Load documents from disk for BM25."""
        if not self._docs_path.exists():
            return

        try:
            with open(self._docs_path, "r") as f:
                docs_data = json.load(f)
            self.all_documents = [
                Document(
                    page_content=doc["content"],
                    metadata={k: v for k, v in doc.items() if k != "content"},
                )
                for doc in docs_data
            ]
        except Exception:
            pass
