#!/bin/bash

# VLLM 分离式 Prefill+Decode 测试运行脚本

echo "=========================================="
echo "VLLM 分离式 Prefill+Decode 性能测试"
echo "=========================================="

# 检查分离式服务器是否运行
echo "检查分离式服务器状态..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 分离式服务器正在运行"
else
    echo "❌ 分离式服务器未运行"
    echo "请先启动分离式服务器:"
    echo "  ./start_disaggregated_servers.sh"
    echo ""
    echo "或者手动启动:"
    echo "  1. 启动 Prefill 实例 (GPU 0, 端口 8100)"
    echo "  2. 启动 Decode 实例 (GPU 1, 端口 8200)"
    echo "  3. 启动代理服务器 (端口 8000)"
    exit 1
fi

echo ""
echo "开始分离式性能测试..."
echo ""

# 运行分离式性能测试
python3 test_disaggregated_performance.py

echo ""
echo "=========================================="
echo "分离式性能测试完成！"
echo "结果保存在 disaggregated_results 目录中"
echo "=========================================="
