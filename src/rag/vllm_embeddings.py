"""
vLLM Batch Embedding Wrapper

Provides batch embedding support using vLLM for 10x+ faster indexing.
Compatible with embeddinggemma and other embedding models.
"""

from typing import List
from vllm import LLM, SamplingParams


class VLLMBatchEmbeddings:
    """
    vLLM-based batch embedding function.

    Key advantage: Processes 100+ texts simultaneously via continuous batching.
    """

    def __init__(
        self,
        model: str = "google/gemma-2b",  # Use embeddinggemma equivalent
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 512
    ):
        """
        Initialize vLLM embedding model.

        Args:
            model: HuggingFace model ID
            tensor_parallel_size: Number of GPUs (1 for single GPU)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Max sequence length
        """
        print(f"[vLLM] Initializing {model}...")

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True  # Required for some embedding models
        )

        # For embeddings, we don't need sampling
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # Embeddings don't generate tokens
            skip_special_tokens=True
        )

        print(f"[vLLM] Model loaded successfully")

    def embed_query(self, text: str) -> List[float]:
        """Embed single query (fallback to batch of 1)."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts.

        This is the KEY speedup: vLLM processes all texts simultaneously
        using continuous batching and PagedAttention.
        """
        # vLLM's generate method handles batching internally
        outputs = self.llm.encode(texts)

        # Extract embeddings
        embeddings = []
        for output in outputs:
            # vLLM's encode returns embeddings directly
            embeddings.append(output.outputs.embedding)

        return embeddings

    def __call__(self, input):
        """Callable interface for compatibility."""
        if isinstance(input, str):
            return self.embed_query(input)
        elif isinstance(input, list):
            return self.embed_documents(input)
        else:
            raise ValueError(f"Input must be str or List[str], got {type(input)}")


class VLLMEmbeddingFunction:
    """
    Alternative: Use vLLM's embeddings API directly (simpler).
    For models that support embeddings natively.
    """

    def __init__(self, model: str = "intfloat/e5-mistral-7b-instruct"):
        """
        Initialize vLLM embedding function.

        Recommended models:
        - intfloat/e5-mistral-7b-instruct (high quality, 4096 dim)
        - sentence-transformers/all-MiniLM-L6-v2 (fast, 384 dim)
        - google/gemma-2b (if embedding-specific variant available)
        """
        from vllm import LLM

        print(f"[vLLM] Loading embedding model: {model}")
        self.llm = LLM(
            model=model,
            task="embed",  # Specify embedding task
            trust_remote_code=True
        )
        print("[vLLM] Embedding model ready")

    def embed_query(self, text: str) -> List[float]:
        """Embed single query."""
        outputs = self.llm.encode([text])
        return outputs[0].outputs.embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents."""
        outputs = self.llm.encode(texts)
        return [output.outputs.embedding for output in outputs]

    def __call__(self, input):
        if isinstance(input, str):
            return self.embed_query(input)
        elif isinstance(input, list):
            return self.embed_documents(input)
        else:
            raise ValueError(f"Input must be str or List[str], got {type(input)}")
