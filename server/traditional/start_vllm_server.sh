#!/bin/bash

# VLLM 服务器启动脚本
# 使用您提供的参数配置

echo "启动 VLLM 服务器..."
echo "模型路径: /share-data/wzk-1/model/deepseek-v2-lite"
echo "端口: 8000"
echo "配置参数:"
echo "  - CPU offload: 20GB"
echo "  - 强制 eager 模式"
echo "  - GPU 内存利用率: 95%"
echo "  - 最大模型长度: 512"
echo "  - 信任远程代码"
echo ""

# 设置环境变量避免多进程问题
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn

# 启动 VLLM 服务器
vllm serve /share-data/wzk-1/model/deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 20 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-model-len 512 \
    --disable-log-stats
