import json

from tools.web_acquisition import DuckDuckGoSearchProvider, search_result_payload


def perform_web_search(query: str, limit: int = 5) -> str:
    try:
        provider = DuckDuckGoSearchProvider()
        items = provider.search(query=query, limit=limit)
        return search_result_payload(query=query, items=items)
    except Exception as e:
        return json.dumps({"error": f"Error during web search: {e}"}, ensure_ascii=False)
