#!/bin/bash

# VLLM 统一测试运行脚本
# 执行 prefill 和 decode 一起生成的测试

echo "VLLM 统一生成测试"
echo "=================="
echo ""

# 检查服务器是否运行
echo "检查 VLLM 服务器状态..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ VLLM 服务器未运行"
    echo "请先运行: ./start_vllm_server.sh"
    exit 1
fi

echo "✅ VLLM 服务器运行正常"
echo ""

# 运行统一测试
echo "开始执行统一生成测试..."
echo "测试内容:"
echo "  - Prefill 和 Decode 一起生成"
echo "  - 测量 TTFT 和 TPOT"
echo "  - 100ms 间隔的功耗和频率数据"
echo "  - 自动标注 prefill/decode 阶段"
echo ""

python3 test_vllm_unified.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试完成！"
    echo ""
    
    # 运行结果分析
    echo "开始分析结果..."
    python3 analyze_unified_results.py
    
    echo ""
    echo "📊 结果文件:"
    echo "  - vllm_unified_results/unified_results.json (原始数据)"
    echo "  - vllm_unified_results/gpu_data.json (GPU监控数据)"
    echo "  - vllm_unified_results/vllm_unified_timeline.png (时间线图)"
    echo "  - vllm_unified_results/phase_comparison.png (阶段对比图)"
    echo ""
    echo "🎉 测试和分析完成！"
else
    echo "❌ 测试失败"
    exit 1
fi
