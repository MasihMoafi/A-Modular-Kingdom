"""
Query parsing using Pydantic for structured extraction
"""
from pydantic import BaseModel, Field
from typing import Literal
import ollama
import json

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

def parse_query(user_input: str) -> QueryIntent:
    """
    Parse user query into structured intent using Pydantic
    
    Args:
        user_input: Raw user query like "Napoleon in Downloads" or "current weather"
    
    Returns:
        QueryIntent with tool, search_query, doc_path, version
    """
    # Create extraction prompt with schema
    schema = QueryIntent.model_json_schema()
    
    prompt = f"""Extract structured information from the user's query and return valid JSON matching this schema:

{json.dumps(schema, indent=2)}

Rules:
- tool: 'rag' for document/file questions, 'web_search' for current events, 'none' for general
- search_query: Extract ONLY the topic, remove location words (e.g., "Napoleon in Downloads" → "Napoleon")
- doc_path: Convert shortcuts (Desktop→~/Desktop, Documents→~/Documents, Downloads→~/Downloads), keep absolute paths, empty if not mentioned
- version: RAG version if mentioned (v1/v2/v3), default 'v2'

User query: "{user_input}"

Return ONLY valid JSON, nothing else:"""
    
    try:
        response = ollama.chat(
            model='qwen3:8b',
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        
        json_str = response['message']['content']
        data = json.loads(json_str)
        return QueryIntent(**data)
        
    except Exception as e:
        print(f"[QueryParser] Error: {e}, using defaults")
        # Fallback to safe defaults
        return QueryIntent(
            tool='none',
            search_query=user_input,
            doc_path='',
            version='v2'
        )
