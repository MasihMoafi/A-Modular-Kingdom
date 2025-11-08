"""
URL Fetcher for RAG - Fetch content from URLs for indexing
Supports:
- GitHub repos via gitingest.com (machine-readable format)
- Regular websites via web scraping
- Direct file URLs
"""
import os
import tempfile
import httpx
from bs4 import BeautifulSoup
import re


def is_github_url(url: str) -> bool:
    """Check if URL is a GitHub repository"""
    return 'github.com' in url and not url.endswith(('.md', '.py', '.txt', '.json'))


def convert_to_gitingest(github_url: str) -> str:
    """
    Convert GitHub URL to gitingest URL for machine-readable format
    
    Examples:
    https://github.com/microsoft/playwright-python
    -> https://gitingest.com/microsoft/playwright-python
    
    https://github.com/microsoft/playwright-python/tree/main/examples
    -> https://gitingest.com/microsoft/playwright-python/tree/main/examples
    """
    # Remove github.com and replace with gitingest.com
    gitingest_url = github_url.replace('github.com', 'gitingest.com')
    return gitingest_url


def fetch_github_repo(github_url: str, temp_dir: str) -> str:
    """
    Fetch GitHub repo content via gitingest.com using Playwright
    (gitingest loads content dynamically with JavaScript)
    Returns path to temporary file with content
    """
    gitingest_url = convert_to_gitingest(github_url)
    
    print(f"[URL Fetcher] Fetching GitHub repo via gitingest: {gitingest_url}")
    print(f"[URL Fetcher] Using Playwright to handle dynamic content...")
    
    try:
        import asyncio
        from playwright.async_api import async_playwright
        
        async def fetch_with_playwright():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to gitingest
                await page.goto(gitingest_url, wait_until="domcontentloaded", timeout=60000)
                
                # Wait for the digest content to load
                try:
                    await page.wait_for_selector('#digest-content', timeout=30000)
                    await page.wait_for_timeout(2000)  # Extra wait for content to populate
                except Exception as e:
                    print(f"[URL Fetcher] Warning: digest-content selector not found ({e}), trying alternative...")
                
                # Try to get content from textarea
                content = await page.evaluate("""() => {
                    const textarea = document.getElementById('digest-content');
                    if (textarea && textarea.value) {
                        return textarea.value;
                    }
                    // Fallback: try to get from pre tag
                    const pre = document.querySelector('pre');
                    if (pre) {
                        return pre.innerText;
                    }
                    return '';
                }""")
                
                await browser.close()
                return content
        
        # Run async function - handle event loop properly
        # Try to get running loop
        try:
            loop = asyncio.get_running_loop()
            in_event_loop = True
        except RuntimeError:
            in_event_loop = False
        
        if in_event_loop:
            # We're in an event loop - run in separate process
            import subprocess
            import sys
            
            script = f"""
import asyncio
from playwright.async_api import async_playwright

async def fetch():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto({repr(gitingest_url)}, wait_until="domcontentloaded", timeout=60000)
        
        try:
            await page.wait_for_selector('#digest-content', timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception:
            pass
        
        content = await page.evaluate('''() => {{
            const textarea = document.getElementById('digest-content');
            if (textarea && textarea.value) return textarea.value;
            const pre = document.querySelector('pre');
            if (pre) return pre.innerText;
            return '';
        }}''')
        
        await browser.close()
        return content

print(asyncio.run(fetch()))
"""
            
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=90,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                raise ValueError(f"Subprocess failed: {result.stderr}")
            
            content = result.stdout.strip()
        else:
            # No event loop, safe to use asyncio.run directly
            content = asyncio.run(fetch_with_playwright())
        
        if not content or len(content) < 100:
            raise ValueError(f"Content too short ({len(content)} chars), gitingest might have failed")
        
        # Save to temp file
        repo_name = github_url.split('/')[-1].replace('.git', '')
        temp_file = os.path.join(temp_dir, f"{repo_name}_gitingest.txt")
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"# Source: {github_url}\n")
            f.write(f"# Fetched via: {gitingest_url}\n\n")
            f.write(content)
        
        print(f"[URL Fetcher] ✓ Fetched {len(content)} chars from {repo_name}")
        return temp_file
        
    except Exception as e:
        print(f"[URL Fetcher] Error fetching from gitingest: {e}")
        raise


def fetch_website_content(url: str, temp_dir: str) -> str:
    """
    Fetch content from regular website
    Returns path to temporary file with content
    """
    print(f"[URL Fetcher] Fetching website: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
        
        with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Save to temp file
            domain = url.split('/')[2].replace('.', '_')
            temp_file = os.path.join(temp_dir, f"{domain}_content.txt")
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {url}\n\n")
                f.write(text)
            
            print(f"[URL Fetcher] ✓ Fetched {len(text)} chars from {domain}")
            return temp_file
            
    except Exception as e:
        print(f"[URL Fetcher] Error fetching website: {e}")
        raise


def fetch_url_content(url: str) -> str:
    """
    Main function to fetch content from URL
    Returns path to temporary directory containing fetched content
    """
    # Create temp directory for this URL
    temp_dir = tempfile.mkdtemp(prefix="rag_url_")
    
    try:
        if is_github_url(url):
            fetch_github_repo(url, temp_dir)
        else:
            fetch_website_content(url, temp_dir)
        
        return temp_dir
        
    except Exception as e:
        print(f"[URL Fetcher] Failed to fetch {url}: {e}")
        # Clean up temp dir on failure
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def is_url(path: str) -> bool:
    """Check if string is a URL"""
    return path.startswith(('http://', 'https://'))
