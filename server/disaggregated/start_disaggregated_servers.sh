#!/bin/bash
# 启动 VLLM 分离式 prefill+decode 服务器
# 基于官方 disaggregated_prefill.sh 修改

set -xe

echo "🚀 启动 VLLM 分离式 Prefill+Decode 服务器"
echo "模型: /share-data/wzk-1/model/deepseek-v2-lite"
sleep 1

# 模型路径
MODEL_NAME="/share-data/wzk-1/model/deepseek-v2-lite"

# 清理函数
cleanup() {
    echo "清理服务器进程..."
    pgrep -f "vllm serve" | xargs kill -9 2>/dev/null || true
    pkill -f "disagg_prefill_proxy_server.py" 2>/dev/null || true
    echo "清理完成"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup INT

# 获取主机IP
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# 安装 quart（代理服务器需要）
if python3 -c "import quart" &> /dev/null; then
    echo "✅ Quart 已安装"
else
    echo "📦 安装 Quart..."
    python3 -m pip install quart
fi

# 等待服务器启动的函数
wait_for_server() {
    local port=$1
    local name=$2
    echo "⏳ 等待 $name 服务器启动 (端口 $port)..."
    timeout 300 bash -c "
        until curl -s localhost:${port}/v1/completions > /dev/null 2>&1; do
            sleep 1
        done" && echo "✅ $name 服务器已启动" || {
        echo "❌ $name 服务器启动失败"
        return 1
    }
}

echo "🔧 启动 Prefill 实例 (GPU 0, 端口 8100)..."
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_connector_extra_config":{"http_port":8101}}' &

PREFILL_PID=$!

echo "🔧 启动 Decode 实例 (GPU 1, 端口 8200)..."
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_connector_extra_config":{"http_port":8201}}' &

DECODE_PID=$!

# 等待两个实例启动
wait_for_server 8100 "Prefill"
wait_for_server 8200 "Decode"

echo "🔧 启动代理服务器 (端口 8000)..."
# 注意：这里需要代理服务器脚本，我们先创建一个简化版本
python3 disagg_prefill_proxy_server.py &

PROXY_PID=$!

# 等待代理服务器启动
wait_for_server 8000 "Proxy"

echo ""
echo "🎉 所有服务器已启动！"
echo "📊 服务端点："
echo "  - Prefill 实例: http://localhost:8100"
echo "  - Decode 实例: http://localhost:8200" 
echo "  - 代理服务器: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 保持脚本运行
wait
