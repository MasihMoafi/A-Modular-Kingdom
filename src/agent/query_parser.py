"""
Query parsing using Pydantic with Gemini's native structured output
"""
from pydantic import BaseModel, Field
from typing import Literal
import os

class QueryIntent(BaseModel):
    """Structured representation of user query intent"""
    tool: Literal['rag', 'web_search', 'none'] = Field(
        description="The tool to use: 'rag' for document search, 'web_search' for internet, 'none' for direct answer"
    )
    search_query: str = Field(
        description="The clean search query/topic without any location information. Examples: 'Napoleon', 'machine learning', 'python code'"
    )
    doc_path: str = Field(
        default="",
        description="Full path to search location. Convert shortcuts: Desktop→~/Desktop, Documents→~/Documents, Downloads→~/Downloads. Keep absolute paths as-is. Empty if not specified."
    )
    version: Literal['v1', 'v2', 'v3'] = Field(
        default='v2',
        description="RAG version to use if tool is 'rag'"
    )

def parse_query_gemini(user_input: str) -> QueryIntent:
    """
    Parse user query using Gemini's native structured output (Pydantic)
    
    Args:
        user_input: Raw user query like "Napoleon in Downloads" or "current weather"
    
    Returns:
        QueryIntent with tool, search_query, doc_path, version
    """
    from google import genai
    from google.genai import types
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Extract structured information from the user's query:

User query: "{user_input}"

Extract:
1. tool: 'rag' if asking about documents/files, 'web_search' if asking about current events, 'none' for general questions
2. search_query: ONLY the topic without location (e.g., "Napoleon in Downloads" → "Napoleon")  
3. doc_path: Convert shortcuts (Desktop→~/Desktop, Documents→~/Documents, Downloads→~/Downloads), keep absolute paths, empty if not mentioned
4. version: RAG version if mentioned (v1/v2/v3), default 'v2'"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=QueryIntent
            )
        )
        
        # Parse JSON response into Pydantic model
        import json
        data = json.loads(response.text)
        return QueryIntent(**data)
        
    except Exception as e:
        print(f"[QueryParser] Gemini error: {e}, using defaults")
        return QueryIntent(
            tool='none',
            search_query=user_input,
            doc_path='',
            version='v2'
        )

def parse_query_ollama(user_input: str) -> QueryIntent:
    """
    Parse user query using Ollama (fallback if no Gemini API key)
    
    Args:
        user_input: Raw user query
    
    Returns:
        QueryIntent
    """
    import ollama
    import json
    
    prompt = f"""Extract JSON:
{{
  "tool": "rag" or "web_search" or "none",
  "search_query": "topic only",
  "doc_path": "path if mentioned",
  "version": "v2"
}}

Query: "{user_input}"

Rules:
- tool: 'rag' for docs/files
- search_query: Topic only ("Napoleon in Downloads" → "Napoleon")
- doc_path: Convert "Desktop"→"~/Desktop", "Downloads"→"~/Downloads"

JSON:"""
    
    try:
        response = ollama.chat(
            model='qwen3:8b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        
        data = json.loads(response['message']['content'])
        return QueryIntent(**data)
        
    except Exception as e:
        print(f"[QueryParser] Ollama error: {e}, using defaults")
        return QueryIntent(
            tool='none',
            search_query=user_input,
            doc_path='',
            version='v2'
        )

def parse_query(user_input: str) -> QueryIntent:
    """
    Parse user query with automatic fallback: Gemini (preferred) → Ollama
    """
    if os.getenv("GEMINI_API_KEY"):
        return parse_query_gemini(user_input)
    else:
        return parse_query_ollama(user_input)
