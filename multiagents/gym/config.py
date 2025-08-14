"""
Configuration file for the AI Gym Assistant

Set these environment variables or modify the defaults below:
"""

import os

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama or google
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")  # For ollama: qwen3:8b, For google: gemini-pro

# Ollama Configuration (for local LLM)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Google AI Configuration (for Google models)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key_here")

# Database Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "gym.db")

# Web Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Example commands to set environment variables:
# export LLM_PROVIDER=google
# export LLM_MODEL=gemini-1.5-flash
# export LLM_MODEL=qwen3:8b
# export GOOGLE_API_KEY=your_actual_key_here 