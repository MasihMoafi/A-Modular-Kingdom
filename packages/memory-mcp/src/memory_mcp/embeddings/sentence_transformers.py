"""SentenceTransformers embedding provider."""

from memory_mcp.embeddings.base import EmbeddingProvider

DIMENSION_MAP = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-mpnet-base-dot-v1": 768,
    "all-distilroberta-v1": 768,
}


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """SentenceTransformers-based embeddings using HuggingFace models."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model_name = model
        self.device = device
        self._model = None
        self._dimension = DIMENSION_MAP.get(model, 384)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "SentenceTransformers support requires the 'sentence-transformers' package. "
                    "Install with: pip install memory-mcp[sentence-transformers]"
                )
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
