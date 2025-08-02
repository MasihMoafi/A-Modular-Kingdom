#!/usr/bin/env python3
"""
Quick test to isolate RAG V3 hanging issue
"""
import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_ollama_connection():
    """Test if Ollama is responsive"""
    print("Testing Ollama connection...")
    try:
        import ollama
        start_time = time.time()
        response = ollama.chat(
            model="qwen3:8b",
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
        elapsed = time.time() - start_time
        print(f"✅ Ollama responsive ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return False

def test_rag_initialization():
    """Test RAG initialization with timing"""
    print("Testing RAG V3 initialization...")
    try:
        start_time = time.time()
        from rag.fetch_3 import get_rag_pipeline_v3
        pipeline = get_rag_pipeline_v3()
        elapsed = time.time() - start_time
        print(f"✅ RAG initialized ({elapsed:.2f}s)")
        return pipeline
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        return None

def test_rag_search_no_rerank(pipeline):
    """Test RAG search without LLM reranking"""
    if not pipeline:
        return
    
    print("Testing RAG search without reranking...")
    try:
        # Backup original config
        original_rerank = pipeline.config.get("rerank_top_k", 3)
        pipeline.config["rerank_top_k"] = 0  # Disable reranking
        
        start_time = time.time()
        result = pipeline.search("What is the main topic?")
        elapsed = time.time() - start_time
        
        # Restore config
        pipeline.config["rerank_top_k"] = original_rerank
        
        print(f"✅ Search without reranking completed ({elapsed:.2f}s)")
        print(f"Result length: {len(result)}")
        return True
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

if __name__ == "__main__":
    print("=== RAG V3 Debug Test ===")
    
    # Test 1: Ollama connection
    ollama_ok = test_ollama_connection()
    
    # Test 2: RAG initialization  
    pipeline = test_rag_initialization()
    
    # Test 3: Search without reranking
    if pipeline:
        test_rag_search_no_rerank(pipeline)
    
    print("=== Debug Test Complete ===")