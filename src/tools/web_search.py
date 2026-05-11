# tools/web_search.py
import json
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup


def _decode_ddg_url(raw_href: str) -> str:
    if not raw_href:
        return ""
    parsed = urlparse(raw_href)
    # DDG often wraps outbound URLs as /l/?uddg=<encoded_url>
    if parsed.path.startswith("/l/"):
        q = parse_qs(parsed.query)
        uddg = q.get("uddg", [])
        if uddg:
            return unquote(uddg[0])
    if raw_href.startswith("//"):
        return "https:" + raw_href
    if raw_href.startswith("/"):
        return urljoin("https://duckduckgo.com", raw_href)
    return raw_href


def _parse_results(html: str, limit: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for node in soup.select("div.result"):
        a = node.select_one("a.result__a")
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        raw_href = (a.get("href") or "").strip()
        url = _decode_ddg_url(raw_href)
        snippet_el = node.select_one(".result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        if not url:
            continue
        rows.append({"title": title, "url": url, "snippet": snippet})
        if len(rows) >= limit:
            break
    return rows


def perform_web_search(query: str, limit: int = 5) -> str:
    import httpx  # Lazy import to avoid proxy conflicts at module level

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        }
        with httpx.Client(headers=headers, follow_redirects=True, timeout=20) as client:
            response = client.get("https://duckduckgo.com/html/", params={"q": query})
            response.raise_for_status()
            items = _parse_results(response.text, limit=max(1, min(int(limit), 10)))
            if not items:
                return json.dumps(
                    {
                        "query": query,
                        "count": 0,
                        "items": [],
                        "results": "No search results found.",
                    },
                    ensure_ascii=False,
                )
            lines = []
            for idx, item in enumerate(items, 1):
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("snippet", "")
                lines.append(f"{idx}. {title}\n{url}\n{snippet}")
            return json.dumps(
                {
                    "query": query,
                    "count": len(items),
                    "items": items,
                    "results": "\n\n".join(lines),
                },
                ensure_ascii=False,
            )
    except Exception as e:
        return json.dumps({"error": f"Error during web search: {e}"}, ensure_ascii=False)
