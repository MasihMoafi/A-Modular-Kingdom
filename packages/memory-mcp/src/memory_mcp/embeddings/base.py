"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_mcp.config import Settings


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Implement this interface to add support for custom embedding models.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        pass


def get_embedding_provider(settings: "Settings") -> EmbeddingProvider:
    """Factory function to create embedding provider based on settings.

    Args:
        settings: Application settings containing provider configuration.

    Returns:
        Configured embedding provider instance.

    Raises:
        ValueError: If the provider is not supported or required dependencies missing.
    """
    provider = settings.embed_provider.lower()

    if provider == "ollama":
        from memory_mcp.embeddings.ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.embed_model if settings.embed_model != "all-MiniLM-L6-v2" else "nomic-embed-text",
            base_url=settings.ollama_base_url,
        )

    elif provider == "sentence-transformers" or provider == "sentencetransformers":
        from memory_mcp.embeddings.sentence_transformers import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(
            model=settings.embed_model or "all-MiniLM-L6-v2",
            device=settings.device,
        )

    elif provider == "openai":
        from memory_mcp.embeddings.openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embed_model or "text-embedding-3-small",
            api_key=settings.openai_api_key,
        )

    elif provider == "anthropic":
        from memory_mcp.embeddings.anthropic import AnthropicEmbeddings
        return AnthropicEmbeddings(
            model=settings.embed_model or "voyage-3",
            api_key=settings.anthropic_api_key,
        )

    elif provider == "cohere":
        from memory_mcp.embeddings.cohere import CohereEmbeddings
        return CohereEmbeddings(
            model=settings.embed_model or "embed-english-v3.0",
            api_key=settings.cohere_api_key,
        )

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: ollama, sentence-transformers, openai, anthropic, cohere"
        )
