import time
import json
from fetch import fetchExternalKnowledge # Change to fetch_v2 to test V2

def run_retrieval_test():
    print(f"--- Testing RAG System: '{fetchExternalKnowledge.__module__}' ---")
    
    with open("evaluation_data.json", 'r', encoding='utf-8') as f:
        queries = [data["query"] for data in json.load(f).values()]

    for i, query in enumerate(queries):
        start_time = time.time()
        fetchExternalKnowledge(query)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Query {i+1} took: {duration:.4f} seconds")

if __name__ == "__main__":
    run_retrieval_test()
