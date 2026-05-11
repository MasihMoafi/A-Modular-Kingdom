import json
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class SearchItem:
    title: str
    url: str
    snippet: str


class SearchProvider:
    """Seam for query->URL discovery used by multiple modules."""

    def search(self, query: str, limit: int) -> list[SearchItem]:
        raise NotImplementedError


def create_http_client(timeout: int = 20):
    import httpx  # Lazy import to keep module import cheap/offline friendly.

    return httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout)


def _decode_ddg_url(raw_href: str) -> str:
    if not raw_href:
        return ""
    parsed = urlparse(raw_href)
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


def parse_ddg_results(html: str, limit: int) -> list[SearchItem]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[SearchItem] = []
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
        rows.append(SearchItem(title=title, url=url, snippet=snippet))
        if len(rows) >= limit:
            break
    return rows


class DuckDuckGoSearchProvider(SearchProvider):
    def search(self, query: str, limit: int) -> list[SearchItem]:
        clamped = max(1, min(int(limit), 10))
        with create_http_client(timeout=20) as client:
            response = client.get("https://duckduckgo.com/html/", params={"q": query})
            response.raise_for_status()
            return parse_ddg_results(response.text, limit=clamped)


def search_result_payload(query: str, items: list[SearchItem]) -> str:
    if not items:
        return json.dumps(
            {"query": query, "count": 0, "items": [], "results": "No search results found."},
            ensure_ascii=False,
        )

    rows = []
    for idx, item in enumerate(items, 1):
        rows.append(f"{idx}. {item.title}\n{item.url}\n{item.snippet}")
    return json.dumps(
        {
            "query": query,
            "count": len(items),
            "items": [{"title": i.title, "url": i.url, "snippet": i.snippet} for i in items],
            "results": "\n\n".join(rows),
        },
        ensure_ascii=False,
    )
