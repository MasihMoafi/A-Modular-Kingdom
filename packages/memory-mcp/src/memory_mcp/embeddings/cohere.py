"""Cohere embedding provider."""

from memory_mcp.embeddings.base import EmbeddingProvider

DIMENSION_MAP = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbeddings(EmbeddingProvider):
    """Cohere-based embeddings using the Cohere API."""

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
    ):
        self.model_name = model
        self.api_key = api_key
        self._client = None
        self._dimension = DIMENSION_MAP.get(model, 1024)

    @property
    def client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Cohere support requires the 'cohere' package. "
                    "Install with: pip install memory-mcp[cohere]"
                )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document",
        )
        return response.embeddings  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query",
        )
        return response.embeddings[0]  # type: ignore[no-any-return]
