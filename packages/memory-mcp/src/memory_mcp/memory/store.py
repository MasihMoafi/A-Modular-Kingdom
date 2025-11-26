"""Memory Store using Qdrant for persistent storage.

Unified vector DB - same backend as RAG for consistency.
Provides:
- Qdrant-based persistent storage (local or cloud)
- BM25 lexical search (primary, fast)
- Vector similarity search (fallback)
"""

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

if TYPE_CHECKING:
    from memory_mcp.config import Settings


class MemoryStore:
    """Persistent memory store using Qdrant with BM25 search.

    Uses same vector DB as RAG for architectural consistency.
    BM25 is primary search (fast, no embedding needed).
    Vector search is fallback for semantic queries.
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

        self._embedding_provider = None
        self._collection_name = settings.memory_collection

        persist_path = Path(settings.qdrant_path).expanduser()
        persist_path.mkdir(parents=True, exist_ok=True)

        if settings.qdrant_mode == "cloud" and settings.qdrant_url:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        elif settings.qdrant_mode == "memory":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(path=str(persist_path / "memory_storage"))

        self._init_collection()

    def _get_embedding_provider(self):
        """Lazy load embedding provider."""
        if self._embedding_provider is None:
            from memory_mcp.embeddings import get_embedding_provider
            self._embedding_provider = get_embedding_provider(self.settings)
        return self._embedding_provider

    def _init_collection(self):
        """Initialize Qdrant collection for memories."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            provider = self._get_embedding_provider()
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=provider.dimension,
                    distance=Distance.COSINE
                ),
            )

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a memory to the store.

        Args:
            content: The memory content to store
            metadata: Optional metadata dictionary

        Returns:
            The ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        provider = self._get_embedding_provider()
        vector = provider.embed_query(content)

        payload = {
            "memory_id": memory_id,
            "content": content,
            **(metadata or {})
        }

        point_id = abs(hash(memory_id)) % (2**63)

        self._client.upsert(
            collection_name=self._collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
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
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
                )
            )
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

                results: list[dict[str, Any]] = []
                for idx, score in top:
                    if score <= 0:
                        continue
                    doc_id = self._bm25_ids[idx]
                    memory = self._get_by_id(doc_id)
                    if memory:
                        memory["score"] = float(score)
                        results.append(memory)

                if results:
                    return results
        except Exception:
            pass

        return self._vector_search(query, k)

    def _get_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Get memory by ID."""
        try:
            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
                ),
                limit=1,
                with_payload=True,
            )
            if results[0]:
                point = results[0][0]
                return {
                    "id": point.payload.get("memory_id"),
                    "content": point.payload.get("content", ""),
                    "metadata": {k: v for k, v in point.payload.items()
                               if k not in ("memory_id", "content")},
                }
        except Exception:
            pass
        return None

    def _vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Fallback vector similarity search."""
        try:
            provider = self._get_embedding_provider()
            query_vector = provider.embed_query(query)

            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            )

            formatted = []
            for hit in results.points:
                formatted.append({
                    "id": hit.payload.get("memory_id"),
                    "content": hit.payload.get("content", ""),
                    "metadata": {k: v for k, v in hit.payload.items()
                               if k not in ("memory_id", "content")},
                    "score": hit.score,
                })
            return formatted
        except Exception:
            return []

    def get_all(self) -> list[dict[str, Any]]:
        """Retrieve all memories from the store.

        Returns:
            List of all memories with id, content, and metadata
        """
        try:
            results = self._client.scroll(
                collection_name=self._collection_name,
                limit=10000,
                with_payload=True,
            )

            formatted = []
            for point in results[0]:
                formatted.append({
                    "id": point.payload.get("memory_id"),
                    "content": point.payload.get("content", ""),
                    "metadata": {k: v for k, v in point.payload.items()
                               if k not in ("memory_id", "content")},
                })
            return formatted
        except Exception:
            return []

    def count(self) -> int:
        """Get total number of memories.

        Returns:
            Number of memories in the store
        """
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count
        except Exception:
            return 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        text = (text or "").lower()
        return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]

    def _rebuild_bm25(self):
        """Rebuild BM25 index from Qdrant."""
        try:
            from rank_bm25 import BM25Okapi

            results = self._client.scroll(
                collection_name=self._collection_name,
                limit=10000,
                with_payload=True,
            )

            ids = []
            docs = []
            for point in results[0]:
                ids.append(point.payload.get("memory_id", ""))
                docs.append(point.payload.get("content", ""))

            tokenized = [self._tokenize(d) for d in docs]

            self._bm25_ids = ids
            self._bm25_docs = tokenized
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            self._bm25_dirty = False
        except Exception:
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_ids = []
            self._bm25_dirty = True
