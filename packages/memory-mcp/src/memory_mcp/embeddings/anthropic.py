"""Anthropic/Voyage embedding provider.

Note: Anthropic partners with Voyage AI for embeddings.
The voyage-3 model is recommended for most use cases.
"""

from memory_mcp.embeddings.base import EmbeddingProvider

DIMENSION_MAP = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
}


class AnthropicEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings (Anthropic's partner for embeddings).

    Note: Requires voyageai package and API key from voyageai.com
    """

    def __init__(
        self,
        model: str = "voyage-3",
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
                import voyageai
                self._client = voyageai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Voyage AI support requires the 'voyageai' package. "
                    "Install with: pip install voyageai"
                )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(texts, model=self.model_name, input_type="document")
        return result.embeddings  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.embed([text], model=self.model_name, input_type="query")
        return result.embeddings[0]  # type: ignore[no-any-return]
