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
                    "Embedding provider 'sentence-transformers' requires extra dependencies.\n\n"
                    "Install with: pip install rag-mem[local]\n\n"
                    "Or use a different provider:\n"
                    "  - ollama: Free, local (requires Ollama running)\n"
                    "  - openai: pip install rag-mem[openai] + API key\n"
                    "  - anthropic: pip install rag-mem[anthropic] + API key\n"
                    "  - cohere: pip install rag-mem[cohere] + API key\n\n"
                    "Configure in ~/.memory-mcp/config.toml or via MEMORY_MCP_EMBED_PROVIDER env var"
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
