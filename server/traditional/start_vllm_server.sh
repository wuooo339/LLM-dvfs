#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 设置使用的 GPU
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TRITON_CACHE_DIR="/tmp/triton_cache"

echo "🔧 启动 VLLM 服务器..."


#启动Tensor并行
vllm serve /share-data/wzk-1/model/deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs 256 \
    --block-size 128
    
#启动卸载执行
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve /share-data/wzk-1/model/deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 20 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 16 \
    --block-size 128 \
    --enforce-eager