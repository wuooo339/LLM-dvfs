#!/bin/bash

# VLLM 服务器测试运行脚本

echo "=========================================="
echo "VLLM 服务器 Prefill/Decode 分离测试"
echo "=========================================="

# 检查服务器是否运行
echo "检查 VLLM 服务器状态..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ VLLM 服务器正在运行"
else
    echo "❌ VLLM 服务器未运行"
    echo "请先启动服务器:"
    echo "  ./start_vllm_server.sh"
    echo ""
    echo "或者手动运行:"
    echo "  vllm serve /share-data/wzk-1/model/deepseek-v2-lite \\"
    echo "    --host 0.0.0.0 \\"
    echo "    --port 8000 \\"
    echo "    --cpu-offload-gb 20 \\"
    echo "    --enforce-eager \\"
    echo "    --gpu-memory-utilization 0.95 \\"
    echo "    --trust-remote-code \\"
    echo "    --max-model-len 512"
    exit 1
fi

echo ""
echo "开始客户端测试..."
echo ""

# 运行客户端测试
cd "$(dirname "$0")"
python3 test_vllm_server.py

echo ""
echo "=========================================="
echo "测试完成！"
echo "结果保存在 vllm_server_results 目录中"
echo "=========================================="
