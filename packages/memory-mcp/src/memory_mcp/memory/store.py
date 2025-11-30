"""Memory Store for persistent fact/memory storage.

This module provides:
- ChromaDB-based persistent storage
- BM25 lexical search (primary, fast)
- Vector similarity search (fallback)
- Optional LLM-based fact extraction (requires external LLM)
"""

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_mcp.config import Settings


class MemoryStore:
    """Persistent memory store using ChromaDB with BM25 search.

    This is a simple, LLM-agnostic memory store. Users can:
    1. Store memories directly with `add()`
    2. Search memories with `search()`
    3. Delete memories with `delete()`
    4. List all memories with `get_all()`

    For LLM-based fact extraction, use your own LLM to extract facts
    from conversations, then store them here.
    """

    def __init__(self, settings: "Settings"):
        """Initialize memory store.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._bm25 = None
        self._bm25_docs: list[list[str]] = []
        self._bm25_ids: list[str] = []
        self._bm25_dirty: bool = True

        self._chroma_path = Path(settings.chroma_path).expanduser()
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        self._collection_name = settings.memory_collection

    def _get_collection(self):
        """Get ChromaDB collection (lazy initialization)."""
        import chromadb

        client = chromadb.PersistentClient(path=str(self._chroma_path))
        return client.get_or_create_collection(name=self._collection_name)

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a memory to the store.

        Args:
            content: The memory content to store
            metadata: Optional metadata dictionary

        Returns:
            The ID of the stored memory
        """
        collection = self._get_collection()
        memory_id = str(uuid.uuid4())
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata] if metadata else None,
        )
        self._bm25_dirty = True
        return memory_id

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if deletion was successful
        """
        try:
            collection = self._get_collection()
            collection.delete(ids=[memory_id])
            self._bm25_dirty = True
            return True
        except Exception:
            return False

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Search memories using BM25 (primary) with vector fallback.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of matching memories with id, content, and metadata
        """
        try:
            if self._bm25_dirty or self._bm25 is None:
                self._rebuild_bm25()

            if self._bm25 is not None and self._bm25_docs:
                q_tokens = self._tokenize(query)
                scores = self._bm25.get_scores(q_tokens)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                top = ranked[:k]

                collection = self._get_collection()
                results: list[dict[str, Any]] = []

                for idx, score in top:
                    if score <= 0:
                        continue
                    doc_id = self._bm25_ids[idx]
                    got = collection.get(ids=[doc_id], include=["documents", "metadatas"])
                    if got.get("documents"):
                        results.append(
                            {
                                "id": doc_id,
                                "content": got["documents"][0],
                                "metadata": (got.get("metadatas", [None])[0] or {}),
                                "score": float(score),
                            }
                        )
                if results:
                    return results
        except Exception:
            pass

        return self._vector_search(query, k)

    def _vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Fallback vector similarity search via ChromaDB."""
        try:
            collection = self._get_collection()
            results = collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            if results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted.append(
                        {
                            "id": doc_id,
                            "content": results["documents"][0][i],
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"] and results["metadatas"][0]
                                else {}
                            ),
                        }
                    )
            return formatted
        except Exception:
            return []

    def get_all(self) -> list[dict[str, Any]]:
        """Retrieve all memories from the store.

        Returns:
            List of all memories with id, content, and metadata
        """
        try:
            collection = self._get_collection()
            results = collection.get(include=["documents", "metadatas"])

            formatted = []
            if results.get("ids"):
                for i, doc_id in enumerate(results["ids"]):
                    formatted.append(
                        {
                            "id": doc_id,
                            "content": results["documents"][i],
                            "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        }
                    )
            return formatted
        except Exception:
            return []

    def count(self) -> int:
        """Get total number of memories.

        Returns:
            Number of memories in the store
        """
        try:
            collection = self._get_collection()
            return collection.count()  # type: ignore[no-any-return]
        except Exception:
            return 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        text = (text or "").lower()
        return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]

    def _rebuild_bm25(self):
        """Rebuild BM25 index from ChromaDB."""
        try:
            from rank_bm25 import BM25Okapi

            collection = self._get_collection()
            results = collection.get(include=["documents"])
            ids = results.get("ids") or []
            docs = results.get("documents") or []
            tokenized = [self._tokenize(d or "") for d in docs]

            self._bm25_ids = ids
            self._bm25_docs = tokenized
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            self._bm25_dirty = False
        except Exception:
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_ids = []
            self._bm25_dirty = True
