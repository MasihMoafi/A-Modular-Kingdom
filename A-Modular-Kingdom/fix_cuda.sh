#!/bin/bash
# Fix CUDA initialization for PyTorch
# Run this before using the project: source fix_cuda.sh

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Test if it works
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ CUDA fixed! GPU is now available."
else
    echo "⚠️  CUDA still not working. You may need to:"
    echo "   1. Restart your system"
    echo "   2. Check nvidia-smi shows your GPU"
    echo "   3. Reinstall NVIDIA drivers"
fi
