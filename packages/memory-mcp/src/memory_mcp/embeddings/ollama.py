"""Ollama embedding provider using httpx (no extra dependencies)."""

import httpx

from memory_mcp.embeddings.base import EmbeddingProvider

DIMENSION_MAP = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
}


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama-based embeddings using local models via HTTP API.

    Uses httpx directly - no extra dependencies required.
    Requires Ollama server running locally.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension = DIMENSION_MAP.get(model, 768)
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]  # type: ignore[no-any-return]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}.\n\n"
                "OPTION 1 - Start Ollama (free, local):\n"
                "  1. Install from https://ollama.ai\n"
                "  2. Run: ollama pull nomic-embed-text\n"
                "  3. Start ollama (it runs in background)\n\n"
                "OPTION 2 - Use offline embeddings (downloads PyTorch ~2GB once):\n"
                "  pip install rag-mem[local]\n"
                "  Then set: MEMORY_MCP_EMBED_PROVIDER=sentence-transformers\n\n"
                "OPTION 3 - Use OpenAI API:\n"
                "  pip install rag-mem[openai]\n"
                "  Then set: MEMORY_MCP_EMBED_PROVIDER=openai MEMORY_MCP_OPENAI_API_KEY=sk-..."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.model}' not found. Pull it with: ollama pull {self.model}"
                )
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
