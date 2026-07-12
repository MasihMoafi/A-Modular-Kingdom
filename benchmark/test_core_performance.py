import time
import timeit

def original_implementation(final_docs):
    unique_content = []
    for doc in final_docs:
        content = doc.get('original_content', doc.get('content', ''))
        if content not in unique_content:
            unique_content.append(content)
    return unique_content

def optimized_implementation(final_docs):
    unique_content = []
    seen = set()
    for doc in final_docs:
        content = doc.get('original_content', doc.get('content', ''))
        if content not in seen:
            seen.add(content)
            unique_content.append(content)
    return unique_content

def benchmark():
    # Generate some dummy data.
    # In search scenarios we might have e.g. up to 1000 items, and some repeats.
    # Let's say 1000 docs, with 500 unique ones repeated.
    final_docs = [{'content': f'content_{i % 500}'} for i in range(1000)]

    # Alternatively, 10000 docs
    final_docs_large = [{'content': f'content_{i % 1000}'} for i in range(10000)]

    print("Testing with 1,000 docs:")
    t_orig = timeit.timeit(lambda: original_implementation(final_docs), number=1000)
    t_opt = timeit.timeit(lambda: optimized_implementation(final_docs), number=1000)

    print(f"Original: {t_orig:.4f}s")
    print(f"Optimized: {t_opt:.4f}s")
    print(f"Speedup: {t_orig / t_opt:.2f}x\n")

    print("Testing with 10,000 docs:")
    t_orig_large = timeit.timeit(lambda: original_implementation(final_docs_large), number=100)
    t_opt_large = timeit.timeit(lambda: optimized_implementation(final_docs_large), number=100)

    print(f"Original: {t_orig_large:.4f}s")
    print(f"Optimized: {t_opt_large:.4f}s")
    print(f"Speedup: {t_orig_large / t_opt_large:.2f}x\n")

if __name__ == '__main__':
    benchmark()
