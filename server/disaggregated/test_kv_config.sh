#!/bin/bash
# 测试KV传输配置修复

set -e

echo "🧪 测试KV传输配置修复"
echo "================================"

# 清理函数
cleanup() {
    echo "🧹 清理测试进程..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "VLLM::EngineC" 2>/dev/null || true
    sleep 2
    echo "✅ 清理完成"
}

# 捕获退出信号
trap cleanup EXIT

# 检查端口是否可用
check_ports() {
    echo "🔍 检查端口可用性..."
    local ports=(8100 8101 8200 8201 14579 14580)
    for port in "${ports[@]}"; do
        if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            echo "❌ 端口 $port 被占用"
            return 1
        else
            echo "✅ 端口 $port 可用"
        fi
    done
    return 0
}

# 测试Prefill实例启动
test_prefill() {
    echo "🚀 测试Prefill实例启动..."
    
    # 创建日志文件
    local log_file="/tmp/vllm_prefill_test.log"
    echo "📝 日志文件: $log_file"
    
    CUDA_VISIBLE_DEVICES=0 vllm serve /share-data/wzk-1/model/deepseek-v2-lite \
        --port 8100 \
        --max-model-len 512 \
        --gpu-memory-utilization 0.8 \
        --trust-remote-code \
        --enforce-eager \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_connector_extra_config":{"http_port":8101}}' \
        > "$log_file" 2>&1 &
    
    local prefill_pid=$!
    echo "📝 Prefill PID: $prefill_pid"
    
    # 等待启动
    echo "⏳ 等待Prefill实例启动..."
    local count=0
    while [ $count -lt 30 ]; do
        # 检查进程是否还在运行
        if ! kill -0 $prefill_pid 2>/dev/null; then
            echo "❌ Prefill进程已退出"
            echo "📋 最后10行日志:"
            tail -10 "$log_file" 2>/dev/null || echo "无法读取日志文件"
            return 1
        fi
        
        # 检查端口是否响应
        if curl -s localhost:8100/v1/completions > /dev/null 2>&1; then
            echo "✅ Prefill实例启动成功"
            return 0
        fi
        
        # 每5次检查显示一次进度
        if [ $((count % 5)) -eq 0 ]; then
            echo "⏳ 等待中... ($count/30)"
        fi
        
        sleep 2
        count=$((count + 1))
    done
    
    echo "❌ Prefill实例启动超时"
    echo "📋 最后20行日志:"
    tail -20 "$log_file" 2>/dev/null || echo "无法读取日志文件"
    kill $prefill_pid 2>/dev/null || true
    return 1
}

# 主测试流程
main() {
    echo "开始测试..."
    
    # 清理现有进程
    cleanup
    
    # 检查端口
    if ! check_ports; then
        echo "❌ 端口检查失败，请手动清理占用端口的进程"
        exit 1
    fi
    
    # 测试Prefill实例
    if test_prefill; then
        echo "🎉 测试成功！KV传输配置修复有效"
        return 0
    else
        echo "❌ 测试失败"
        return 1
    fi
}

# 运行测试
main "$@"
