import time
import json
from fetch import fetchExternalKnowledge # Change to fetch_v2 to test V2

def run_retrieval_test():
    print(f"--- Testing RAG System with 'next(iter(...))' syntax ---")
    
    with open("evaluation_data.json", 'r', encoding='utf-8') as f:
        # Load the entire dictionary of queries
        evaluation_data = json.load(f)

    # The next(iter(...)) logic grabs the key of the VERY FIRST query
    first_query_key = next(iter(evaluation_data))
    first_query = evaluation_data[first_query_key]["query"]

    print(f"Testing only the first query: '{first_query[:60]}...'")
    
    start_time = time.time()
    fetchExternalKnowledge(first_query)
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nQuery took: {duration:.4f} seconds")
    print("\nNOTE: This syntax is designed to process only one item.")

if __name__ == "__main__":
    run_retrieval_test()
