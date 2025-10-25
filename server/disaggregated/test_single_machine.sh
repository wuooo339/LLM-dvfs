#!/bin/bash

# =============================================================================
# VLLM 单机多卡 P2P NCCL 分离式服务管理脚本
# =============================================================================
# 本脚本用于管理 Prefill+Decode 分离架构服务
# 
# 使用方法:
#   ./test_single_machine.sh [proxy|prefill|decode|test|all]
#
# 选项:
#   proxy   - 启动代理服务器
#   prefill - 启动 Prefill 实例
#   decode  - 启动 Decode 实例
#   test    - 检测并测试所有服务
#   all     - 启动所有服务（代理+Prefill+Decode）
# =============================================================================

set -e

# 配置参数
MODEL=${MODEL:-"/share-data/wzk-1/model/Qwen3-4B"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-600}
PROXY_PORT=${PROXY_PORT:-30001}
HTTP_PROXY_PORT=${HTTP_PROXY_PORT:-8000}

# 单机多卡配置 (1 Prefill + 1 Decode)
PREFILL_GPU=${PREFILL_GPU:-2}
DECODE_GPU=${DECODE_GPU:-3}
PREFILL_HTTP_PORT=${PREFILL_HTTP_PORT:-8100}
DECODE_HTTP_PORT=${DECODE_HTTP_PORT:-8200}
PREFILL_KV_PORT=${PREFILL_KV_PORT:-21001}
DECODE_KV_PORT=${DECODE_KV_PORT:-22001}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 打印配置信息
print_config() {
    echo "📋 当前配置:"
    echo "   模型: $MODEL"
    echo "   Prefill: GPU $PREFILL_GPU, 端口 $PREFILL_HTTP_PORT"
    echo "   Decode:  GPU $DECODE_GPU, 端口 $DECODE_HTTP_PORT"
    echo "   Proxy:   端口 $HTTP_PROXY_PORT"
    echo "   ⚠️  注意: 已禁用 GPU P2P 和 SHM (NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1)"
    echo "   📡 将使用网络/系统内存传输（适合不支持 GPU peer access 的硬件）"
}

# 打印详细配置
print_config_detailed() {
    echo "============================================================================"
    echo "🔧 详细配置信息"
    echo "============================================================================"
    echo "模型路径: $MODEL"
    echo ""
    echo "Prefill 实例:"
    echo "  GPU ID: $PREFILL_GPU"
    echo "  HTTP 端口: $PREFILL_HTTP_PORT"
    echo "  KV 端口: $PREFILL_KV_PORT"
    echo ""
    echo "Decode 实例:"
    echo "  GPU ID: $DECODE_GPU"
    echo "  HTTP 端口: $DECODE_HTTP_PORT"
    echo "  KV 端口: $DECODE_KV_PORT"
    echo ""
    echo "代理服务器:"
    echo "  HTTP 端口: $HTTP_PROXY_PORT"
    echo "  ZMQ 端口: $PROXY_PORT"
    echo "============================================================================"
    echo ""
}

# 等待服务器启动
wait_for_server() {
    local port=$1
    local name=$2
    local timeout=$TIMEOUT_SECONDS
    local start_time=$(date +%s)
    
    echo "⏳ 等待 $name 在端口 $port 启动..."
    
    while true; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "✅ $name 已就绪"
            return 0
        fi
        
        local now=$(date +%s)
        if (( now - start_time >= timeout )); then
            echo "❌ 超时: $name 未能在 $timeout 秒内启动"
            return 1
        fi
        
        sleep 2
    done
}

# 启动代理服务器
start_proxy() {
    echo "============================================================================"
    echo "📡 启动代理服务器"
    echo "============================================================================"
    print_config_detailed
    
    echo "▶️  启动中..."
    cd "$SCRIPT_DIR"
    # 注意：不需要设置 PREFILL_ADDRS 和 DECODE_ADDRS，因为实例会通过动态注册自动发现
    HTTP_PROXY_PORT=$HTTP_PROXY_PORT \
        PROXY_PORT=$PROXY_PORT \
        python3 disagg_prefill_proxy_server.py
}

# 启动 Prefill 实例
start_prefill() {
    echo "============================================================================"
    echo "🔵 启动 Prefill 实例"
    echo "============================================================================"
    print_config_detailed
    
    echo "▶️  在 GPU $PREFILL_GPU 上启动 Prefill 实例..."
    # 强制禁用所有 GPU 直接通信，使用网络/系统内存传输
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO \
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $PREFILL_HTTP_PORT \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-hybrid-kv-cache-manager \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_port":'$PREFILL_KV_PORT',"kv_connector_extra_config":{"http_port":'$PREFILL_HTTP_PORT',"local_node_id":"producer","peer_nodes":["localhost:'$DECODE_HTTP_PORT'"],"mem_pool_size_gb":2,"proxy_ip":"localhost","proxy_port":'$PROXY_PORT'}}'
}

# 启动 Decode 实例
start_decode() {
    echo "============================================================================"
    echo "🔵 启动 Decode 实例"
    echo "============================================================================"
    print_config_detailed
    
    echo "▶️  在 GPU $DECODE_GPU 上启动 Decode 实例..."
    # 强制禁用所有 GPU 直接通信，使用网络/系统内存传输
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO \
    CUDA_VISIBLE_DEVICES=$DECODE_GPU vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $DECODE_HTTP_PORT \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-hybrid-kv-cache-manager \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_port":'$DECODE_KV_PORT',"kv_connector_extra_config":{"http_port":'$DECODE_HTTP_PORT',"local_node_id":"consumer","peer_nodes":["localhost:'$PREFILL_HTTP_PORT'"],"mem_pool_size_gb":2,"proxy_ip":"localhost","proxy_port":'$PROXY_PORT'}}'
}

# 启动所有服务
start_all() {
    echo "============================================================================"
    echo "🚀 启动所有服务（后台模式）"
    echo "============================================================================"
    print_config_detailed
    
    PIDS=()
    
    # 1. 启动代理服务器
    echo "📡 [1/3] 启动代理服务器..."
    # 注意：不需要设置 PREFILL_ADDRS 和 DECODE_ADDRS，因为实例会通过动态注册自动发现
    HTTP_PROXY_PORT=$HTTP_PROXY_PORT \
        PROXY_PORT=$PROXY_PORT \
        python3 "$SCRIPT_DIR/disagg_prefill_proxy_server.py" > "$SCRIPT_DIR/proxy.log" 2>&1 &
    PIDS+=($!)
    echo "   PID: ${PIDS[-1]}, 日志: $SCRIPT_DIR/proxy.log"
    sleep 3
    
    # 2. 启动 Prefill 实例
    echo "🔵 [2/3] 启动 Prefill 实例..."
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO \
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $PREFILL_HTTP_PORT \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-hybrid-kv-cache-manager \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_port":'$PREFILL_KV_PORT',"kv_connector_extra_config":{"http_port":'$PREFILL_HTTP_PORT',"local_node_id":"producer","peer_nodes":["localhost:'$DECODE_HTTP_PORT'"],"mem_pool_size_gb":2,"proxy_ip":"localhost","proxy_port":'$PROXY_PORT'}}' \
        > "$SCRIPT_DIR/prefill.log" 2>&1 &
    PIDS+=($!)
    echo "   PID: ${PIDS[-1]}, 日志: $SCRIPT_DIR/prefill.log"
    
    # 3. 启动 Decode 实例
    echo "🔵 [3/3] 启动 Decode 实例..."
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO \
    CUDA_VISIBLE_DEVICES=$DECODE_GPU vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $DECODE_HTTP_PORT \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-hybrid-kv-cache-manager \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_port":'$DECODE_KV_PORT',"kv_connector_extra_config":{"http_port":'$DECODE_HTTP_PORT',"local_node_id":"consumer","peer_nodes":["localhost:'$PREFILL_HTTP_PORT'"],"mem_pool_size_gb":2,"proxy_ip":"localhost","proxy_port":'$PROXY_PORT'}}' \
        > "$SCRIPT_DIR/decode.log" 2>&1 &
    PIDS+=($!)
    echo "   PID: ${PIDS[-1]}, 日志: $SCRIPT_DIR/decode.log"
    
    echo ""
    echo "✅ 所有服务已在后台启动"
    echo "📝 进程 ID: ${PIDS[@]}"
    echo ""
    echo "⏳ 等待服务就绪（这可能需要几分钟，取决于模型加载时间）..."
    
    # 等待服务就绪
    if ! wait_for_server $HTTP_PROXY_PORT "代理服务器"; then
        echo "❌ 代理服务器启动失败，查看日志: $SCRIPT_DIR/proxy.log"
        exit 1
    fi
    
    if ! wait_for_server $PREFILL_HTTP_PORT "Prefill 实例"; then
        echo "❌ Prefill 实例启动失败，查看日志: $SCRIPT_DIR/prefill.log"
        exit 1
    fi
    
    if ! wait_for_server $DECODE_HTTP_PORT "Decode 实例"; then
        echo "❌ Decode 实例启动失败，查看日志: $SCRIPT_DIR/decode.log"
        exit 1
    fi
    
    echo ""
    echo "============================================================================"
    echo "🎉 所有服务已就绪！"
    echo "============================================================================"
    echo ""
    echo "🛑 停止服务: kill ${PIDS[@]}"
    echo "📊 运行测试: $0 test"
    echo ""
}

# 测试服务
test_services() {
    echo "============================================================================"
    echo "🧪 测试服务"
    echo "============================================================================"
    print_config
    
    echo "🔍 检测服务状态..."
    echo ""
    
    # 检测代理服务器
    if ! wait_for_server $HTTP_PROXY_PORT "代理服务器"; then
        echo "❌ 代理服务器未就绪"
        echo "💡 请先启动: $0 proxy"
        exit 1
    fi
    
    # 检测 Prefill 实例
    if ! wait_for_server $PREFILL_HTTP_PORT "Prefill 实例"; then
        echo "❌ Prefill 实例未就绪"
        echo "💡 请先启动: $0 prefill"
        exit 1
    fi
    
    # 检测 Decode 实例
    if ! wait_for_server $DECODE_HTTP_PORT "Decode 实例"; then
        echo "❌ Decode 实例未就绪"
        echo "💡 请先启动: $0 decode"
        exit 1
    fi
    
    echo ""
    echo "============================================================================"
    echo "🎉 所有服务检测成功，已就绪!"
    echo "============================================================================"
    echo ""
    
    # 运行健康检查
    echo "【健康检查】"
    echo "curl http://localhost:$HTTP_PROXY_PORT/health"
    curl -s "http://localhost:$HTTP_PROXY_PORT/health" | jq 2>/dev/null || curl -s "http://localhost:$HTTP_PROXY_PORT/health"
    echo ""
    
    # 运行简单测试
    echo ""
    echo "【简单测试】发送一个请求..."
    TEST_RESULT=$(curl -s -X POST "http://localhost:$HTTP_PROXY_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"Hello\",\"max_tokens\":10,\"temperature\":0,\"stream\":false}")
    
    if echo "$TEST_RESULT" | grep -q "error"; then
        echo "❌ 测试失败:"
        echo "$TEST_RESULT" | jq 2>/dev/null || echo "$TEST_RESULT"
    else
        echo "✅ 测试成功！"
        echo "$TEST_RESULT" | jq '.choices[0].text' 2>/dev/null || echo "$TEST_RESULT"
    fi
    
    echo ""
    echo "============================================================================"
    echo "📊 更多测试命令"
    echo "============================================================================"
    echo ""
    echo "【单个请求测试】"
    echo "curl -X POST http://localhost:$HTTP_PROXY_PORT/v1/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"model\":\"$MODEL\",\"prompt\":\"San Francisco is a\",\"max_tokens\":50,\"temperature\":0}'"
    echo ""
    echo "【统计信息】"
    echo "curl http://localhost:$HTTP_PROXY_PORT/stats | jq"
    echo ""
    echo "【压力测试】"
    echo "vllm bench serve \\"
    echo "  --backend vllm \\"
    echo "  --model $MODEL \\"
    echo "  --host localhost \\"
    echo "  --port $HTTP_PROXY_PORT \\"
    echo "  --dataset-name random \\"
    echo "  --random-input-len 1024 \\"
    echo "  --random-output-len 128 \\"
    echo "  --num-prompts 100 \\"
    echo "  --request-rate 2"
    echo ""
}

# 显示交互式菜单
show_menu() {
    clear
    echo "============================================================================"
    echo "          VLLM 单机多卡 P2P NCCL 分离式服务管理脚本"
    echo "============================================================================"
    echo ""
    print_config
    echo ""
    echo "请选择操作："
    echo ""
    echo "  1) 启动代理服务器 (Proxy)"
    echo "  2) 启动 Prefill 实例"
    echo "  3) 启动 Decode 实例"
    echo "  4) 一键启动所有服务 (后台模式)"
    echo "  5) 检测并测试服务"
    echo "  6) 查看配置信息"
    echo "  0) 退出"
    echo ""
    echo "============================================================================"
}

# 交互式主循环
interactive_mode() {
    while true; do
        show_menu
        read -p "请输入选项 [0-6]: " choice
        echo ""
        
        case $choice in
            1)
                echo "即将启动代理服务器（前台运行，按 Ctrl+C 停止）..."
                sleep 2
                start_proxy
                ;;
            2)
                echo "即将启动 Prefill 实例（前台运行，按 Ctrl+C 停止）..."
                sleep 2
                start_prefill
                ;;
            3)
                echo "即将启动 Decode 实例（前台运行，按 Ctrl+C 停止）..."
                sleep 2
                start_decode
                ;;
            4)
                echo "即将启动所有服务（后台模式）..."
                sleep 2
                start_all
                echo ""
                read -p "按 Enter 键返回菜单..."
                ;;
            5)
                echo "开始检测和测试服务..."
                sleep 1
                test_services
                echo ""
                read -p "按 Enter 键返回菜单..."
                ;;
            6)
                clear
                print_config_detailed
                echo "环境变量说明："
                echo "  可通过设置环境变量来修改配置"
                echo ""
                echo "  MODEL              - 模型路径"
                echo "  PREFILL_GPU        - Prefill GPU ID"
                echo "  DECODE_GPU         - Decode GPU ID"
                echo "  PREFILL_HTTP_PORT  - Prefill HTTP 端口"
                echo "  DECODE_HTTP_PORT   - Decode HTTP 端口"
                echo "  HTTP_PROXY_PORT    - 代理 HTTP 端口"
                echo "  PREFILL_KV_PORT    - Prefill KV 端口"
                echo "  DECODE_KV_PORT     - Decode KV 端口"
                echo "  PROXY_PORT         - 代理 ZMQ 端口"
                echo ""
                echo "示例："
                echo "  MODEL=/path/to/model $0"
                echo ""
                read -p "按 Enter 键返回菜单..."
                ;;
            0)
                echo "退出程序..."
                exit 0
                ;;
            *)
                echo "❌ 无效选项，请输入 0-6"
                sleep 2
                ;;
        esac
    done
}

# 显示命令行使用方法
show_usage() {
    echo "============================================================================"
    echo "VLLM 单机多卡 P2P NCCL 分离式服务管理脚本"
    echo "============================================================================"
    echo ""
    echo "使用方法:"
    echo "  $0                  - 交互式菜单模式"
    echo "  $0 [选项]           - 命令行模式"
    echo ""
    echo "命令行选项:"
    echo "  proxy   - 启动代理服务器（前台运行）"
    echo "  prefill - 启动 Prefill 实例（前台运行）"
    echo "  decode  - 启动 Decode 实例（前台运行）"
    echo "  test    - 检测并测试所有服务"
    echo "  all     - 启动所有服务（后台运行）"
    echo ""
    echo "示例:"
    echo "  # 交互式模式"
    echo "  $0"
    echo ""
    echo "  # 命令行模式 - 在 3 个终端分别启动"
    echo "  终端1: $0 proxy"
    echo "  终端2: $0 prefill"
    echo "  终端3: $0 decode"
    echo ""
    echo "  # 命令行模式 - 一键启动所有服务"
    echo "  $0 all"
    echo ""
    echo "  # 命令行模式 - 测试服务"
    echo "  $0 test"
    echo ""
    echo "环境变量:"
    echo "  MODEL              - 模型路径 (默认: $MODEL)"
    echo "  PREFILL_GPU        - Prefill GPU ID (默认: $PREFILL_GPU)"
    echo "  DECODE_GPU         - Decode GPU ID (默认: $DECODE_GPU)"
    echo "  PREFILL_HTTP_PORT  - Prefill HTTP 端口 (默认: $PREFILL_HTTP_PORT)"
    echo "  DECODE_HTTP_PORT   - Decode HTTP 端口 (默认: $DECODE_HTTP_PORT)"
    echo "  HTTP_PROXY_PORT    - 代理 HTTP 端口 (默认: $HTTP_PROXY_PORT)"
    echo ""
}

# 主函数
main() {
    local action="${1:-}"
    
    # 如果没有参数，进入交互式模式
    if [ -z "$action" ]; then
        interactive_mode
        return
    fi
    
    # 命令行模式
    case "$action" in
        proxy)
            start_proxy
            ;;
        prefill)
            start_prefill
            ;;
        decode)
            start_decode
            ;;
        test)
            test_services
            ;;
        all)
            start_all
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "❌ 错误: 未知选项 '$action'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"

