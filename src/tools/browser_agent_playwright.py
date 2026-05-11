"""
Task-oriented Playwright browser automation.

Supports multi-step tasks in one prompt using lightweight parsing:
- open/go to/navigate to URL
- search for <query>
- click <link/button text>
- wait <N>s
- scroll
"""
import json
import re

from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright


def _normalize_url(candidate: str) -> str:
    text = (candidate or "").strip().strip(".,)")
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if "." in text and " " not in text:
        return "https://" + text
    return ""


def _extract_url(task: str) -> str:
    m = re.search(r"https?://[^\s]+", task, re.IGNORECASE)
    if m:
        return _normalize_url(m.group(0))

    m = re.search(
        r"(?:go to|open|navigate to|visit)\s+([a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
        task,
        re.IGNORECASE,
    )
    if m:
        return _normalize_url(m.group(1))
    return ""


def _extract_search_query(clause: str) -> str:
    m = re.search(r"(?:search for|look up|find)\s+\"([^\"]+)\"", clause, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:search for|look up|find)\s+'([^']+)'", clause, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:search for|look up|find)\s+(.+)$", clause, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_click_target(clause: str) -> str:
    m = re.search(r"(?:click|open)\s+\"([^\"]+)\"", clause, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:click|open)\s+'([^']+)'", clause, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:click|open)\s+(.+)$", clause, re.IGNORECASE)
    return m.group(1).strip() if m else ""


async def _search_on_page(page, query: str) -> bool:
    selectors = [
        "input[type='search']",
        "input[name='q']",
        "textarea[name='q']",
        "input[aria-label*='Search']",
        "input[type='text']",
    ]
    for sel in selectors:
        try:
            locator = page.locator(sel).first
            if await locator.count() == 0:
                continue
            await locator.click(timeout=3000)
            await locator.fill(query, timeout=4000)
            await locator.press("Enter")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except PlaywrightTimeout:
                pass
            return True
        except Exception:
            continue
    return False


async def _click_by_text(page, target: str) -> bool:
    target = target.strip().strip(".")
    if not target:
        return False
    try:
        await page.get_by_role("link", name=re.compile(re.escape(target), re.IGNORECASE)).first.click(timeout=6000)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except PlaywrightTimeout:
            pass
        return True
    except Exception:
        pass
    try:
        await page.get_by_role("button", name=re.compile(re.escape(target), re.IGNORECASE)).first.click(timeout=6000)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except PlaywrightTimeout:
            pass
        return True
    except Exception:
        pass
    try:
        await page.locator(f"text={target}").first.click(timeout=6000)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except PlaywrightTimeout:
            pass
        return True
    except Exception:
        return False


def _split_clauses(task: str) -> list[str]:
    raw = re.split(r"\bthen\b|;", task, flags=re.IGNORECASE)
    return [c.strip() for c in raw if c and c.strip()]


async def browse_web_playwright(task: str, headless: bool = True) -> str:
    try:
        task = (task or "").strip()
        if not task:
            return json.dumps(
                {"status": "error", "task": task, "error": "Empty task", "message": "No task provided."},
                ensure_ascii=False,
                indent=2,
            )

        url = _extract_url(task)
        clauses = _split_clauses(task)
        performed_actions = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            if url:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                performed_actions.append({"action": "navigate", "url": page.url})
            else:
                # If task asks for search but no URL, use DuckDuckGo as default launch page.
                if re.search(r"\b(search for|look up|find)\b", task, re.IGNORECASE):
                    await page.goto("https://duckduckgo.com/", wait_until="domcontentloaded", timeout=30000)
                    performed_actions.append({"action": "navigate", "url": page.url})
                else:
                    return json.dumps(
                        {
                            "status": "error",
                            "task": task,
                            "error": "No URL found in task.",
                            "message": "Provide a URL/domain or an explicit search task.",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

            for clause in clauses:
                lower = clause.lower()

                search_q = _extract_search_query(clause)
                if search_q and re.search(r"\b(search for|look up|find)\b", lower):
                    ok = await _search_on_page(page, search_q)
                    performed_actions.append({"action": "search", "query": search_q, "ok": ok})
                    continue

                if "click" in lower or lower.startswith("open "):
                    click_target = _extract_click_target(clause)
                    # Avoid treating URLs/domains as click targets.
                    if click_target and not _normalize_url(click_target):
                        ok = await _click_by_text(page, click_target)
                        performed_actions.append({"action": "click", "target": click_target, "ok": ok})
                        continue

                wait_match = re.search(r"\bwait\s+(\d+)\s*(?:s|sec|secs|seconds)?\b", lower)
                if wait_match:
                    seconds = max(1, min(int(wait_match.group(1)), 30))
                    await page.wait_for_timeout(seconds * 1000)
                    performed_actions.append({"action": "wait", "seconds": seconds})
                    continue

                if "scroll" in lower:
                    await page.evaluate("window.scrollBy(0, Math.max(window.innerHeight, 800));")
                    await page.wait_for_timeout(600)
                    performed_actions.append({"action": "scroll"})
                    continue

            try:
                await page.wait_for_load_state("networkidle", timeout=6000)
            except PlaywrightTimeout:
                pass

            title = await page.title()
            current_url = page.url
            text_content = await page.evaluate(
                """() => {
                    const clone = document.body ? document.body.cloneNode(true) : null;
                    if (!clone) return '';
                    const unwanted = clone.querySelectorAll('script, style, nav, header, footer, aside, noscript, svg');
                    unwanted.forEach(el => el.remove());
                    let text = clone.innerText || clone.textContent || '';
                    text = text.replace(/\\t+/g, ' ');
                    text = text.replace(/ +/g, ' ');
                    text = text.replace(/\\n\\s*\\n\\s*\\n/g, '\\n\\n');
                    return text.trim().slice(0, 4000);
                }"""
            )

            screenshot_path = "/tmp/browser_screenshot.png"
            await page.screenshot(path=screenshot_path, full_page=False)
            await browser.close()

            result = {
                "status": "success",
                "task": task,
                "result": {
                    "title": title,
                    "url": current_url,
                    "text_content": text_content,
                    "screenshot": screenshot_path,
                    "performed_actions": performed_actions,
                },
                "message": f"Browser task executed with {len(performed_actions)} action(s).",
            }
            return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "task": task,
                "result": None,
                "error": str(e),
                "message": f"Browser task failed: {str(e)}",
            },
            indent=2,
            ensure_ascii=False,
        )


async def browse_web(task: str, headless: bool = True) -> str:
    """Backward-compatible wrapper used by MCP host."""
    return await browse_web_playwright(task, headless)
