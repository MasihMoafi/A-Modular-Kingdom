"""RAG Pipeline for Memory MCP.

This is the main RAG implementation using:
- Qdrant for vector storage
- BM25 for lexical search (rank_bm25, no langchain)
- CrossEncoder for reranking (optional)
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memory_mcp.embeddings import get_embedding_provider
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

        self.all_documents: list[dict[str, Any]] = []
        self._reranker = None
        self._bm25 = None
        self._bm25_corpus: list[list[str]] = []

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

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        text = (text or "").lower()
        return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]

    def _rebuild_bm25(self):
        """Build BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_corpus = [self._tokenize(doc["content"]) for doc in self.all_documents]
            self._bm25 = BM25Okapi(self._bm25_corpus) if self._bm25_corpus else None
        except ImportError:
            self._bm25 = None

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
            self._rebuild_bm25()
            return len(self.all_documents)

        all_docs = []

        for path in self.document_paths:
            if not os.path.exists(path):
                continue

            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        docs = self._process_file(file_path)
                        all_docs.extend(docs)
            else:
                docs = self._process_file(path)
                all_docs.extend(docs)

        if not all_docs:
            return 0

        self.vector_db.clear()
        self.vector_db.add_documents_batch(all_docs, batch_size=100)
        self.all_documents = all_docs
        self._rebuild_bm25()
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
            {"content": doc["content"], **{k: v for k, v in doc.items() if k not in ["content", "score"]}}
            for doc in vector_results
        ]

        bm25_docs = []
        if self._bm25 is not None and self.all_documents:
            q_tokens = self._tokenize(query)
            scores = self._bm25.get_scores(q_tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            for idx, score in ranked[:top_k]:
                if score > 0:
                    bm25_docs.append(self.all_documents[idx])

        combined_docs = vector_docs + bm25_docs
        seen = set()
        initial_results = []
        for doc in combined_docs:
            content = doc["content"]
            if content not in seen:
                seen.add(content)
                initial_results.append(doc)

        if not initial_results:
            return "No relevant information found."

        if self._reranker and len(initial_results) > 1:
            pairs = [[query, doc["content"]] for doc in initial_results]
            scores = self._reranker.predict(pairs)
            scored_results = list(zip(scores, initial_results))
            sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
            final_results = [doc for score, doc in sorted_results[:top_k]]
        else:
            final_results = initial_results[:top_k]

        unique_content = list({doc["content"] for doc in final_results})
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
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
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
            with open(self._docs_path, "w") as f:
                json.dump(self.all_documents, f)
        except Exception:
            pass

    def _load_documents(self):
        """Load documents from disk for BM25."""
        if not self._docs_path.exists():
            return

        try:
            with open(self._docs_path, "r") as f:
                self.all_documents = json.load(f)
        except Exception:
            pass
