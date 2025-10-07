#!/bin/bash

# BBH (BIG-Bench Hard) 数据集测试运行脚本
# 使用长输入进行推理任务测试

echo "🧠 BIG-Bench Hard (BBH) 数据集测试"
echo "=================================="
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

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,memory.used,memory.total,power.draw --format=csv,noheader,nounits

echo ""
echo "开始执行BBH测试..."
echo "测试内容:"
echo "  - 单次实例测试10条数据 (流式打印token与时间)"
echo "  - 可配置并发以提升GPU负载"
echo "  - 最大输出长度: 512 tokens"
echo "  - 实时显示prompt和输出token信息"
echo "  - 测量功耗、频率和性能"
echo "  - 分析长输入对GPU功耗的影响"
echo ""

# 配置并发实例数量（可通过环境变量覆盖）
CONCURRENCY=${CONCURRENCY:-1}
echo "并发实例数量: ${CONCURRENCY}"

if [ "$CONCURRENCY" -gt 1 ]; then
    echo "启动 ${CONCURRENCY} 个并发实例以提升负载（后台运行）..."
    # 确保日志目录存在
    mkdir -p bbh_test_results
    # 启动 CONCURRENCY-1 个后台实例，提升GPU并行负载
    BG_COUNT=$((CONCURRENCY-1))
    for i in $(seq 1 ${BG_COUNT}); do
        echo "  启动实例 $i (后台)"
        python3 -u test_bbh_dataset.py > "bbh_test_results/run_${i}_$(date +%s).log" 2>&1 &
        # 轻微错峰
        sleep 0.2
    done
    echo "等待后台实例完成..."
    wait
    echo "后台实例已完成。"
fi

echo "运行前台实例用于生成汇总..."
python3 test_bbh_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ BBH测试完成！"
    echo ""
    
    # 显示结果摘要
    if [ -f "bbh_test_results/bbh_analysis.json" ]; then
        echo "📊 测试结果摘要:"
        python3 -c "
import json
with open('bbh_test_results/bbh_analysis.json', 'r') as f:
    analysis = json.load(f)
    
print(f'  总任务数: {analysis[\"summary\"][\"total_tasks\"]}')
print(f'  总请求数: {analysis[\"summary\"][\"total_requests\"]}')
print(f'  成功请求数: {analysis[\"summary\"][\"successful_requests\"]}')
print(f'  成功率: {analysis[\"summary\"][\"success_rate\"]:.2%}')

if 'gpu_analysis' in analysis and analysis['gpu_analysis']:
    gpu = analysis['gpu_analysis']
    print(f'  平均功耗: {gpu[\"avg_power\"]:.1f}W')
    print(f'  最大功耗: {gpu[\"max_power\"]:.1f}W')
    print(f'  平均利用率: {gpu[\"avg_utilization\"]:.1f}%')
    print(f'  最大利用率: {gpu[\"max_utilization\"]:.1f}%')

print('')
print('📁 详细结果文件:')
print('  - bbh_test_results/bbh_test_results.json (原始数据)')
print('  - bbh_test_results/bbh_analysis.json (分析结果)')
"
    fi
    
    echo ""
    echo "🎉 BBH测试和分析完成！"
else
    echo "❌ BBH测试失败"
    exit 1
fi
