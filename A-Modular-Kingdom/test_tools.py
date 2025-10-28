#!/usr/bin/env python
"""
Comprehensive test suite for browser automation and RAG tools
"""
import asyncio
import json
from tools.browser_agent_playwright import browse_web_playwright
from rag.fetch_2 import fetchExternalKnowledge

async def test_browser_automation():
    """Test browser automation with multiple scenarios"""
    print("=" * 60)
    print("TESTING BROWSER AUTOMATION")
    print("=" * 60)
    
    test_cases = [
        ("https://example.com", "Basic website"),
        ("https://github.com/MasihMoafi", "GitHub profile"),
        ("https://playwright.dev", "Documentation site"),
    ]
    
    for url, description in test_cases:
        print(f"\n📍 Testing: {description} ({url})")
        print("-" * 60)
        
        result_json = await browse_web_playwright(f"Go to {url}", headless=True)
        result = json.loads(result_json)
        
        if result["status"] == "success":
            print(f"✅ SUCCESS")
            print(f"   Title: {result['result']['title']}")
            print(f"   URL: {result['result']['url']}")
            print(f"   Text preview (first 200 chars):")
            print(f"   {result['result']['text_content'][:200]}...")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")

def test_rag_system():
    """Test RAG system with code and documentation queries"""
    print("\n" + "=" * 60)
    print("TESTING RAG SYSTEM")
    print("=" * 60)
    
    test_queries = [
        ("tools", "playwright browser automation implementation", "Code search"),
        ("tools", "web search duckduckgo", "Code search"),
        (".", "RAG system architecture", "Documentation search"),
    ]
    
    for doc_path, query, description in test_queries:
        print(f"\n📍 Testing: {description}")
        print(f"   Path: {doc_path}")
        print(f"   Query: {query}")
        print("-" * 60)
        
        try:
            result = fetchExternalKnowledge(query, doc_path=doc_path)
            
            if "error" in result.lower() or "no relevant" in result.lower():
                print(f"❌ FAILED: {result[:200]}")
            else:
                print(f"✅ SUCCESS")
                print(f"   Result preview (first 300 chars):")
                print(f"   {result[:300]}...")
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")

async def main():
    """Run all tests"""
    print("\n🚀 Starting comprehensive tool tests...\n")
    
    # Test browser automation
    await test_browser_automation()
    
    # Test RAG system
    test_rag_system()
    
    print("\n" + "=" * 60)
    print("✨ ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
