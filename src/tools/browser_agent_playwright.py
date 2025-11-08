"""
Direct Playwright browser automation - fast, simple, no LLM needed
Works reliably on Ubuntu without model dependencies
"""
import asyncio
import json
import re
from typing import Dict, Any, Optional
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout


async def browse_web_playwright(task: str, headless: bool = True) -> str:
    """
    Simple, fast Playwright automation. Extracts URLs from task and navigates.
    No LLM dependency - just direct browser control.
    
    Examples:
    - "Go to example.com and get the title"
    - "Navigate to https://github.com and screenshot"
    - "Open google.com and search for python"
    """
    try:
        # Extract URL from task
        url_match = re.search(r'https?://[^\s]+', task)
        if url_match:
            url = url_match.group(0)
        else:
            # Try to find domain names
            domain_match = re.search(r'(?:go to|open|navigate to|visit)\s+([a-zA-Z0-9-]+\.[a-zA-Z]{2,})', task, re.IGNORECASE)
            if domain_match:
                url = f"https://{domain_match.group(1)}"
            else:
                # Default fallback
                words = task.lower().split()
                for word in words:
                    if '.' in word and not word.startswith('.'):
                        url = f"https://{word}"
                        break
                else:
                    return json.dumps({
                        "status": "error",
                        "task": task,
                        "error": "No URL found in task. Please specify a URL or domain.",
                        "message": "Could not extract URL from task"
                    }, indent=2, ensure_ascii=False)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            # Navigate to URL
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait a bit for dynamic content
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeout:
                pass  # Continue anyway
            
            # Extract information
            title = await page.title()
            current_url = page.url
            
            # Get clean text content
            text_content = await page.evaluate("""() => {
                // Remove script, style, nav, header, footer elements
                const clone = document.body.cloneNode(true);
                const unwanted = clone.querySelectorAll('script, style, nav, header, footer, .navigation, .menu');
                unwanted.forEach(el => el.remove());
                
                // Get text and clean it up
                let text = clone.innerText || clone.textContent || '';
                
                // Remove excessive whitespace and newlines
                text = text.replace(/\\t+/g, ' ');  // Replace tabs with single space
                text = text.replace(/ +/g, ' ');   // Replace multiple spaces with single space
                text = text.replace(/\\n\\s*\\n\\s*\\n/g, '\\n\\n');  // Max 2 consecutive newlines
                text = text.trim();
                
                return text.substring(0, 3000);
            }""")
            
            # Take screenshot
            screenshot_path = "/tmp/browser_screenshot.png"
            await page.screenshot(path=screenshot_path, full_page=False)
            
            await browser.close()
            
            # Return clean JSON with ensure_ascii=False to preserve Unicode properly
            result = {
                "status": "success",
                "task": task,
                "result": {
                    "title": title,
                    "url": current_url,
                    "text_content": text_content,
                    "screenshot": screenshot_path
                },
                "message": f"Successfully navigated to {current_url}"
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "task": task,
            "result": None,
            "error": str(e),
            "message": f"Browser task failed: {str(e)}"
        }, indent=2, ensure_ascii=False)


# Keep the old function name for compatibility
async def browse_web(task: str, headless: bool = True) -> str:
    """Wrapper for backward compatibility"""
    return await browse_web_playwright(task, headless)
