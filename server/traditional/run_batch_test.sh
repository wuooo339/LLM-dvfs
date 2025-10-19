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
echo "python3 server/traditional/simple_batch_test.py --batch-size <批次大小> [--requests <总请求数>] [--max-tokens <最大tokens>] [--test-length <测试长度>] [--preset <预设配置>]"
echo ""
echo "测试长度选项: 1024, 2048, 4096, 8192, 16384, 32768 (默认: 4096)"
echo "预设配置选项:"
echo "  short  - 1024 tokens, 64输出, 8请求"
echo "  medium - 4096 tokens, 128输出, 8请求 (默认)"
echo "  long   - 8192 tokens, 256输出, 4请求"
echo "  xlong  - 16384 tokens, 512输出, 2请求"
echo ""
echo "示例:"
echo "  python3 server/traditional/simple_batch_test.py --batch-size 4                                    # 4个请求同时发送，4096 tokens"
echo "  python3 server/traditional/simple_batch_test.py --batch-size 8 --requests 16                     # 16个请求，每批8个，4096 tokens"
echo "  python3 server/traditional/simple_batch_test.py --batch-size 2 --requests 6 --test-length 2048   # 6个请求，每批2个，2048 tokens"
echo "  python3 server/traditional/simple_batch_test.py --batch-size 4 --test-length 8192 --max-tokens 256 # 4个请求，8192 tokens输入，256 tokens输出"
echo "  python3 server/traditional/simple_batch_test.py --batch-size 2 --preset long                      # 使用long预设配置"
echo ""

# 如果提供了参数，直接运行
if [ $# -gt 0 ]; then
    echo "运行命令: python3 server/traditional/simple_batch_test.py $@"
    python3 server/traditional/simple_batch_test.py "$@"
else
    # 交互式输入
    read -p "请输入批次大小 (同时发送的请求数): " batch_size
    read -p "请输入总请求数 (默认8): " requests
    requests=${requests:-8}
    read -p "请输入最大tokens (默认128): " max_tokens
    max_tokens=${max_tokens:-128}
    
    echo ""
    echo "测试长度选项: 1024, 2048, 4096, 8192, 16384, 32768"
    read -p "请输入测试长度 (默认4096): " test_length
    test_length=${test_length:-4096}
    
    echo ""
    echo "开始测试..."
    python3 server/traditional/simple_batch_test.py --batch-size $batch_size --requests $requests --max-tokens $max_tokens --test-length $test_length
fi
