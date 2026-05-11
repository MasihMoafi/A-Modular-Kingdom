import hashlib
import json
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from tools.web_acquisition import DuckDuckGoSearchProvider, create_http_client


def _safe_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_page_text(soup: BeautifulSoup) -> str:
    for el in soup.select("script, style, noscript, svg, iframe, form, nav, header, footer, aside"):
        el.decompose()

    # Prefer semantic content regions when present.
    region = soup.select_one("main, article")
    base = region if region is not None else soup.body
    if base is None:
        return ""

    text = base.get_text("\n", strip=True)
    return _safe_text(text)


def _extract_links(url: str, soup: BeautifulSoup, same_domain_only: bool) -> list[str]:
    out = []
    base_domain = urlparse(url).netloc.lower()
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        absolute = urljoin(url, href)
        p = urlparse(absolute)
        if p.scheme not in ("http", "https"):
            continue
        clean = f"{p.scheme}://{p.netloc}{p.path}"
        if p.query:
            clean += "?" + p.query
        if same_domain_only and p.netloc.lower() != base_domain:
            continue
        out.append(clean)
    # Preserve order while de-duplicating.
    seen = set()
    uniq = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _slug_for_url(url: str) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).strip("/").replace("/", "_")
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)[:80] or "page"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{digest}"


def _write_page_markdown(path: Path, url: str, title: str, text: str) -> None:
    lines = [
        f"# {title or 'Untitled'}",
        "",
        f"Source: {url}",
        "",
        text,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _seed_urls_from_query(query: str, limit: int) -> list[str]:
    provider = DuckDuckGoSearchProvider()
    return [item.url for item in provider.search(query=query, limit=limit) if item.url]


def crawl_webpages(
    query: str = "",
    urls: list[str] | None = None,
    max_pages: int = 5,
    max_depth: int = 1,
    same_domain_only: bool = True,
    output_dir: str = "/tmp/web_crawl",
) -> str:
    try:
        max_pages = max(1, min(int(max_pages), 30))
        max_depth = max(0, min(int(max_depth), 3))
        urls = urls or []

        seeds = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        if not seeds and query.strip():
            seeds = _seed_urls_from_query(query=query.strip(), limit=max_pages)
        if not seeds:
            return json.dumps({"error": "No seed URLs available. Provide query or urls."}, ensure_ascii=False)

        run_dir = Path(output_dir).expanduser().resolve() / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        visited = set()
        queue = deque((seed, 0) for seed in seeds)
        pages = []

        with create_http_client(timeout=20) as client:
            while queue and len(pages) < max_pages:
                url, depth = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)

                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    ctype = (resp.headers.get("content-type") or "").lower()
                    if "text/html" not in ctype:
                        continue

                    soup = BeautifulSoup(resp.text, "html.parser")
                    title = _safe_text((soup.title.string if soup.title and soup.title.string else "")).strip()
                    text = _clean_page_text(soup)
                    if not text:
                        continue

                    md_path = run_dir / f"{len(pages)+1:02d}_{_slug_for_url(resp.url.__str__())}.md"
                    _write_page_markdown(md_path, resp.url.__str__(), title, text)

                    words = len(text.split())
                    pages.append(
                        {
                            "url": resp.url.__str__(),
                            "title": title,
                            "word_count": words,
                            "file_path": str(md_path),
                            "preview": text[:400],
                        }
                    )

                    if depth < max_depth:
                        links = _extract_links(resp.url.__str__(), soup, same_domain_only=same_domain_only)
                        for link in links:
                            if link not in visited:
                                queue.append((link, depth + 1))
                except Exception:
                    continue

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "seed_urls": seeds,
                "crawled_count": len(pages),
                "pages": pages,
                "files": [p["file_path"] for p in pages],
                "output_dir": str(run_dir),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": f"crawl_webpages failed: {e}"}, ensure_ascii=False)
