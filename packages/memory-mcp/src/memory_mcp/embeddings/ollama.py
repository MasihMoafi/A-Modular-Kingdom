"""Ollama embedding provider."""

from memory_mcp.embeddings.base import EmbeddingProvider

DIMENSION_MAP = {
    "embeddinggemma": 768,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama-based embeddings using local models."""

    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._dimension = DIMENSION_MAP.get(model, 768)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "Ollama support requires the 'ollama' package. "
                    "Install with: pip install memory-mcp[ollama]"
                )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]
