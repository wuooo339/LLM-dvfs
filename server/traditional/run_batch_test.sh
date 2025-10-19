#!/bin/bash

# 简化的批量测试启动脚本

echo "简化的VLLM批量测试工具"
echo "======================"

# 检查VLLM服务器状态
echo "检查VLLM服务器状态..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ VLLM服务器运行正常"
else
    echo "❌ VLLM服务器未运行，请先启动服务器"
    echo "运行命令: ./start_vllm_server.sh"
    exit 1
fi

echo ""
echo "使用方法:"
echo "python3 simple_batch_test.py --batch-size <批次大小> [--requests <总请求数>] [--max-tokens <最大tokens>]"
echo ""
echo "示例:"
echo "  python3 simple_batch_test.py --batch-size 4                    # 4个请求同时发送"
echo "  python3 simple_batch_test.py --batch-size 8 --requests 16      # 16个请求，每批8个"
echo "  python3 simple_batch_test.py --batch-size 2 --requests 6       # 6个请求，每批2个"
echo ""

# 如果提供了参数，直接运行
if [ $# -gt 0 ]; then
    echo "运行命令: python3 simple_batch_test.py $@"
    python3 simple_batch_test.py "$@"
else
    # 交互式输入
    read -p "请输入批次大小 (同时发送的请求数): " batch_size
    read -p "请输入总请求数 (默认8): " requests
    requests=${requests:-8}
    read -p "请输入最大tokens (默认64): " max_tokens
    max_tokens=${max_tokens:-64}
    
    echo ""
    echo "开始测试..."
    python3 simple_batch_test.py --batch-size $batch_size --requests $requests --max-tokens $max_tokens
fi
