#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 设置使用的 GPU
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1
#export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TRITON_CACHE_DIR="/tmp/triton_cache"

echo "🔧 启动 VLLM 服务器..."
#启动Tensor并行
vllm serve /share-data/wzk-1/model/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len  8192\
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128 