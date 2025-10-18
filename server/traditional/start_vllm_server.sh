#!/bin/bash

# VLLM 服务器启动脚本
# 使用您提供的参数配置

set -e

echo "🚀 启动 VLLM 传统服务器..."
echo "模型路径: /share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B"
echo "端口: 8000"
echo "配置参数:"
echo "  - CPU offload: 56GB"
echo "  - 强制 eager 模式"
echo "  - GPU 内存利用率: 80% (降低以提高稳定性)"
echo "  - 最大模型长度: 8192 (支持长输入BBH测试)"
echo "  - 信任远程代码"
echo "  - 批处理优化: 支持长序列"
echo "  - 最大并发序列数: ${MAX_SEQS:-8}"
echo ""

# 预检查函数
precheck() {
    echo "🔍 执行预检查..."
    
    # 检查模型路径
    if [ ! -d "/share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B" ]; then
        echo "❌ 模型路径不存在: /share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B"
        exit 1
    fi
    
    # 检查GPU状态
    if ! nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi 不可用，请检查CUDA环境"
        exit 1
    fi
    
    # 检查端口占用
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then 
        echo "⚠️  端口 8000 已被占用，尝试清理..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    echo "✅ 预检查完成"
}

# 清理函数
cleanup() {
    echo "🧹 清理服务器进程..."
    pgrep -f "vllm serve" | xargs kill -9 2>/dev/null || true
    echo "✅ 清理完成"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup INT

# 执行预检查
precheck

# 设置环境变量避免多进程问题
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TRITON_CACHE_DIR="/tmp/triton_cache"

echo "🔧 启动 VLLM 服务器..."

# 启动 VLLM 服务器
vllm serve /share-data/wzk-1/model/deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 20 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs ${MAX_SEQS:-8} \
    --disable-log-stats
