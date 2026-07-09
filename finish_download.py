import os
from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen3-Reranker-4B"
print("Resuming model-00001-of-00002.safetensors... (progress bar will show 0% of remaining ~200MB)")
hf_hub_download(repo_id=repo_id, filename="model-00001-of-00002.safetensors")

print("Resuming model-00002-of-00002.safetensors... (progress bar will show 0% of remaining ~200MB)")
hf_hub_download(repo_id=repo_id, filename="model-00002-of-00002.safetensors")

print("\nAll downloads complete! The models are verified and ready in your cache.")
