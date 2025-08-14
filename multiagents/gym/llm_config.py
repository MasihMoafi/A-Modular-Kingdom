import os
from typing import Any
from crewai import LLM

def get_llm():
    """Get LLM instance for CrewAI"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "qwen3:8b")
    
    if provider == "ollama":
        return LLM(
            model=f"ollama/{model}",
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 