"""
Qdrant Vector Database Backend for RAG V2

Replaces custom VectorIndex with production-ready Qdrant.
Key improvements:
- Batch embedding (10x+ faster indexing)
- Persistent storage
- Production-grade vector search
"""

import os
import sys

# Central proxy manager — strip SOCKS, keep HTTP(S) for cloud
from utils.proxy import strip_socks as _strip_socks
_strip_socks()

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

@dataclass
class QdrantConfig:
    """Configuration for QdrantVectorDB."""
    collection_name: str
    vector_size: int = 768
    distance: str = "cosine"
    persist_path: str = "./qdrant_db"
    mode: str = "local"
    url: Optional[str] = None
    api_key: Optional[str] = None

class QdrantVectorDB:
    """Qdrant-based vector database with batch embedding support."""

    def __init__(
        self,
        config: QdrantConfig,
        embedding_fn: Callable
    ):
        self.collection_name = config.collection_name
        self.embedding_fn = embedding_fn
        self.vector_size = config.vector_size

        # Initialize Qdrant client (local or cloud)
        if config.mode == "cloud" and config.url:
            sys.stderr.write(f"[Qdrant] Connecting to cloud: {config.url}" + "\n")
            self.client = QdrantClient(url=config.url, api_key=config.api_key, timeout=30)
        else:
            sys.stderr.write(f"[Qdrant] Using local storage: {config.persist_path}" + "\n")
            self.client = QdrantClient(path=config.persist_path)

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(config.distance, Distance.COSINE)

        # Create or get collection
        self._init_collection()

    def _init_collection(self):
        """Initialize Qdrant collection."""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            sys.stderr.write(f"[Qdrant] Created collection: {self.collection_name}" + "\n")
        else:
            sys.stderr.write(f"[Qdrant] Using existing collection: {self.collection_name}" + "\n")

    def add_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 500):
        """
        Add documents in batches with batch embedding.

        Large batch size (500) for maximum embedding throughput.
        """
        total = len(documents)
        sys.stderr.write(f"[Qdrant] Batch indexing {total} documents..." + "\n")

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]

            # Extract content for batch embedding
            contents = [doc["content"] for doc in batch]

            # Batch embed (FAST)
            vectors = self._batch_embed(contents)

            # Prepare points for Qdrant
            points = []
            for j, (doc, vector) in enumerate(zip(batch, vectors)):
                point_id = i + j
                payload = {k: v for k, v in doc.items() if k != "content"}
                payload["content"] = doc["content"]

                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                sys.stderr.write(f"[Qdrant] Indexed {min(i + batch_size, total)}/{total} documents..." + "\n")

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding with parallel processing for speed.
        """
        # Check if embedding_fn has batch method (vLLM, Gemini, etc.)
        if hasattr(self.embedding_fn, 'embed_documents'):
            try:
                return self.embedding_fn.embed_documents(texts)
            except Exception as e:
                sys.stderr.write(f"[Qdrant] Batch embedding via embed_documents failed: {e}" + "\n")

        # Try calling with list directly (some APIs support this)
        try:
            result = self.embedding_fn(texts)
            if isinstance(result, list) and len(result) == len(texts):
                return result
        except:
            pass

        # Parallel processing fallback (FAST)
        from concurrent.futures import ThreadPoolExecutor
        import os

        max_workers = min(os.cpu_count() or 4, 8)  # Use up to 8 cores

        def embed_chunk(text):
            return self._embed_single(text)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            vectors = list(executor.map(embed_chunk, texts))

        return vectors

    def _embed_single(self, text: str) -> List[float]:
        """Embed single text - handles different embedding APIs"""
        # Try method-based APIs (Ollama, etc.)
        if hasattr(self.embedding_fn, 'embed_query'):
            return self.embedding_fn.embed_query(text)
        # Try callable
        elif callable(self.embedding_fn):
            return self.embedding_fn(text)
        else:
            raise ValueError(f"Unsupported embedding function type: {type(self.embedding_fn)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        query_vector = self._embed_single(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        # Convert to standard format
        formatted = []
        for hit in results:
            doc = {
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                **{k: v for k, v in hit.payload.items() if k != "content"}
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
        sys.stderr.write(f"[Qdrant] Cleared collection: {self.collection_name}" + "\n")
