"""OpenAI embedding provider."""

from memory_mcp.embeddings.base import EmbeddingProvider

DIMENSION_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI-based embeddings using the OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self.model_name = model
        self.api_key = api_key
        self._client = None
        self._dimension = DIMENSION_MAP.get(model, 1536)

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI support requires the 'openai' package. "
                    "Install with: pip install memory-mcp[openai]"
                )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding  # type: ignore[no-any-return]
