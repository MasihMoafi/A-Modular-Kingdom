"""Qdrant Vector Database Backend for RAG.

Provides:
- Batch embedding (10x+ faster indexing)
- Persistent storage (local or cloud)
- Production-grade vector search
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from memory_mcp.embeddings.base import EmbeddingProvider


class QdrantVectorDB:
    """Qdrant-based vector database with batch embedding support."""

    def __init__(
        self,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        distance: str = "cosine",
        persist_path: str = "./qdrant_db",
        mode: str = "local",
        url: str | None = None,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.vector_size = embedding_provider.dimension

        if mode == "cloud" and url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif mode == "memory":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=persist_path)

        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance, Distance.COSINE)
        self._init_collection()

    def _init_collection(self):
        """Initialize Qdrant collection."""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
            )

    def add_documents_batch(
        self, documents: list[dict[str, Any]], batch_size: int = 100
    ):
        """Add documents in batches with batch embedding."""
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            contents = [doc["content"] for doc in batch]
            vectors = self.embedding_provider.embed_documents(contents)

            points = []
            for j, (doc, vector) in enumerate(zip(batch, vectors)):
                point_id = i + j
                payload = {k: v for k, v in doc.items()}
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Vector similarity search."""
        query_vector = self.embedding_provider.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        formatted = []
        for hit in results.points:
            doc = {
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                **{k: v for k, v in hit.payload.items() if k != "content"},
            }
            formatted.append(doc)

        return formatted

    def count(self) -> int:
        """Get total document count."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def clear(self):
        """Delete all documents from collection."""
        self.client.delete_collection(self.collection_name)
        self._init_collection()
