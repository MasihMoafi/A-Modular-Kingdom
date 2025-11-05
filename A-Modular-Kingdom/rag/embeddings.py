"""
Ollama Embedding Interface for A-Modular-Kingdom

Provides a unified embedding interface using Ollama's embedding API,
compatible with both RAG and memory systems.
"""

import ollama
from typing import List, Union
import sys


class OllamaEmbeddingFunction:
    """
    Embedding function using Ollama's embeddinggemma model.
    Compatible with both VectorIndex and ChromaDB interfaces.
    """
    
    def __init__(self, model: str = "embeddinggemma", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama embedding function.
        
        Args:
            model: Ollama embedding model name (default: embeddinggemma)
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is available and model exists."""
        try:
            # Test connection with a simple embedding
            ollama.embeddings(model=self.model, prompt="test")
            sys.stderr.write(f"[Embeddings] Successfully connected to Ollama with model: {self.model}\n")
            sys.stderr.flush()
        except Exception as e:
            error_msg = f"""
[Embeddings] ERROR: Cannot connect to Ollama or model '{self.model}' not available.

Please ensure:
1. Ollama is running: Check if Ollama service is active
2. Model is pulled: Run 'ollama pull {self.model}'

Error details: {str(e)}
"""
            sys.stderr.write(error_msg)
            sys.stderr.flush()
            raise ConnectionError(error_msg)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
            ConnectionError: If Ollama is unavailable
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            embedding = response.get('embedding', [])
            
            if not embedding:
                raise ValueError(f"Empty embedding returned for text: {text[:50]}...")
            
            return embedding
            
        except Exception as e:
            sys.stderr.write(f"[Embeddings] Error embedding query: {str(e)}\n")
            sys.stderr.flush()
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is empty or invalid
            ConnectionError: If Ollama is unavailable
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")
        
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_query(text)
                embeddings.append(embedding)
                
                # Progress logging for large batches
                if (i + 1) % 10 == 0:
                    sys.stderr.write(f"[Embeddings] Processed {i + 1}/{len(texts)} documents\n")
                    sys.stderr.flush()
                    
            except Exception as e:
                sys.stderr.write(f"[Embeddings] Error embedding document {i}: {str(e)}\n")
                sys.stderr.flush()
                raise
        
        return embeddings
    
    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Callable interface for compatibility with ChromaDB.
        
        Args:
            input: Single text or list of texts
            
        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(input, str):
            return self.embed_query(input)
        elif isinstance(input, list):
            return self.embed_documents(input)
        else:
            raise ValueError(f"Input must be str or List[str], got {type(input)}")


# ChromaDB-compatible embedding function
class ChromaOllamaEmbeddingFunction:
    """
    ChromaDB-specific embedding function wrapper.
    Implements the interface expected by ChromaDB's collection.
    """
    
    def __init__(self, model_name: str = "embeddinggemma"):
        """
        Initialize ChromaDB-compatible Ollama embedding function.
        
        Args:
            model_name: Ollama embedding model name
        """
        self.model_name = model_name
        self._ollama_fn = OllamaEmbeddingFunction(model=model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed documents for ChromaDB.
        
        Args:
            input: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._ollama_fn.embed_documents(input)
