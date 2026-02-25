#!/usr/bin/env python
"""Integration-only tool tests.

These tests rely on external services/tools and are skipped by default.
Set `AMK_RUN_INTEGRATION=1` to enable.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def _require_integration() -> None:
    if os.getenv("AMK_RUN_INTEGRATION", "0") != "1":
        pytest.skip("Integration test disabled. Set AMK_RUN_INTEGRATION=1 to run.")


def test_browser_automation():
    """Smoke test for Playwright browser automation."""
    _require_integration()

    from tools.browser_agent_playwright import browse_web_playwright

    result_json = asyncio.run(
        browse_web_playwright("Go to https://example.com", headless=True)
    )
    result = json.loads(result_json)
    assert result.get("status") == "success", result


def test_rag_system():
    """Smoke test for RAG query path."""
    _require_integration()

    from rag.fetch_2 import fetchExternalKnowledge

    result = fetchExternalKnowledge("web search duckduckgo", doc_path="tools")
    assert not result.startswith("Error:"), result


if __name__ == "__main__":
    os.environ.setdefault("AMK_RUN_INTEGRATION", "1")
    test_browser_automation()
    test_rag_system()
    print("Integration tool tests completed.")
