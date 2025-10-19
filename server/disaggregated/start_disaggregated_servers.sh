#!/bin/bash
# 启动 VLLM 分离式 prefill+decode 服务器
# 基于官方 disaggregated_prefill.sh 修改

set -xe

echo "🚀 启动 VLLM 分离式 Prefill+Decode 服务器"
echo "模型: /share-data/wzk-1/model/opt-1.3b"
sleep 1

# 模型路径
MODEL_NAME="/share-data/wzk-1/model/opt-1.3b"

# 预检查函数
precheck() {
    echo "🔍 执行预检查..."
    
    # 检查模型路径
    if [ ! -d "$MODEL_NAME" ]; then
        echo "❌ 模型路径不存在: $MODEL_NAME"
        exit 1
    fi
    
    # 检查GPU状态
    if ! nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi 不可用，请检查CUDA环境"
        exit 1
    fi
    
    # 检查端口占用
    for port in 8100 8200 8000 29800 29801; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "⚠️  端口 $port 已被占用，尝试清理..."
            lsof -ti:$port | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
    done
    
    echo "✅ 预检查完成"
}

# 清理函数
cleanup() {
    echo "🧹 清理服务器进程..."
    pgrep -f "vllm serve" | xargs kill -9 2>/dev/null || true
    pkill -f "disagg_prefill_proxy_server.py" 2>/dev/null || true
    echo "✅ 清理完成"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup INT

# 设置环境变量
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TRITON_CACHE_DIR="/tmp/triton_cache"

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
    local max_wait=${3:-300}
    echo "⏳ 等待 $name 服务器启动 (端口 $port, 最多等待 ${max_wait}秒)..."
    
    local count=0
    while [ $count -lt $max_wait ]; do
        if curl -s localhost:${port}/v1/completions > /dev/null 2>&1; then
            echo "✅ $name 服务器已启动"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        if [ $((count % 30)) -eq 0 ]; then
            echo "⏳ 仍在等待 $name 服务器启动... (${count}/${max_wait}秒)"
        fi
    done
    
    echo "❌ $name 服务器启动失败 (超时 ${max_wait}秒)"
    return 1
}

# 检查进程是否正在运行
check_process() {
    local pid=$1
    local name=$2
    if kill -0 $pid 2>/dev/null; then
        echo "✅ $name 进程正在运行 (PID: $pid)"
        return 0
    else
        echo "❌ $name 进程已停止 (PID: $pid)"
        return 1
    fi
}

# 执行预检查
precheck

echo "🔧 启动 Prefill 实例 (GPU 0, 端口 8100)..."
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --enforce-eager \
    --verbose \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_connector_extra_config":{"http_port":29800,"local_node_id":"producer","peer_nodes":["consumer:29801"],"mem_pool_size_gb":2}}' &

PREFILL_PID=$!
echo "Prefill 进程 PID: $PREFILL_PID"

# 等待 Prefill 实例启动
if ! wait_for_server 8100 "Prefill" 180; then
    echo "❌ Prefill 实例启动失败，检查进程状态..."
    check_process $PREFILL_PID "Prefill"
    echo "查看 Prefill 日志..."
    ps aux | grep "vllm serve" | grep -v grep
    exit 1
fi

echo "🔧 启动 Decode 实例 (GPU 1, 端口 8200)..."
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --enforce-eager \
    --verbose \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_connector_extra_config":{"http_port":29801,"local_node_id":"consumer","peer_nodes":["producer:29800"],"mem_pool_size_gb":2}}' &

DECODE_PID=$!
echo "Decode 进程 PID: $DECODE_PID"

# 等待 Decode 实例启动
if ! wait_for_server 8200 "Decode" 180; then
    echo "❌ Decode 实例启动失败，检查进程状态..."
    check_process $DECODE_PID "Decode"
    echo "查看 Decode 日志..."
    ps aux | grep "vllm serve" | grep -v grep
    exit 1
fi

echo "🔧 启动代理服务器 (端口 8000)..."
# 检查代理服务器脚本是否存在
if [ -f "disagg_prefill_proxy_server.py" ]; then
    python3 disagg_prefill_proxy_server.py &
    PROXY_PID=$!
    echo "代理服务器进程 PID: $PROXY_PID"
    
    # 等待代理服务器启动
    if ! wait_for_server 8000 "Proxy" 60; then
        echo "⚠️  代理服务器启动失败，但核心服务已就绪"
    fi
else
    echo "⚠️  代理服务器脚本不存在，跳过代理服务器启动"
    PROXY_PID=""
fi

echo ""
echo "🎉 核心服务器已启动！"
echo "📊 服务端点："
echo "  - Prefill 实例: http://localhost:8100"
echo "  - Decode 实例: http://localhost:8200"
if [ -n "$PROXY_PID" ]; then
    echo "  - 代理服务器: http://localhost:8000"
fi
echo ""
echo "🔍 进程状态检查："
check_process $PREFILL_PID "Prefill"
check_process $DECODE_PID "Decode"
if [ -n "$PROXY_PID" ]; then
    check_process $PROXY_PID "Proxy"
fi
echo ""
echo "按 Ctrl+C 停止所有服务"

# 保持脚本运行，定期检查进程状态
while true; do
    sleep 30
    if ! check_process $PREFILL_PID "Prefill" >/dev/null 2>&1; then
        echo "❌ Prefill 进程意外停止，退出脚本"
        cleanup
    fi
    if ! check_process $DECODE_PID "Decode" >/dev/null 2>&1; then
        echo "❌ Decode 进程意外停止，退出脚本"
        cleanup
    fi
done
