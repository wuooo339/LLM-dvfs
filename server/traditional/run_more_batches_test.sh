#!/bin/bash

# 多批次自动化测试脚本（集成GPU监控）
# 自动运行多个批次大小的测试，并为每个批次独立监控GPU功耗

echo "======================================"
echo "  VLLM 多批次自动化测试工具"
echo "  (集成GPU功耗监控)"
echo "======================================"
echo ""

# 检查VLLM服务器状态
echo "检查VLLM服务器状态..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ VLLM服务器运行正常"
else
    echo "❌ VLLM服务器未运行，请先启动服务器"
    echo "运行命令: ./start_vllm_server.sh"
    exit 1
fi

# 检查GPU监控脚本
if [ ! -f "batch_gpu_monitor.py" ]; then
    echo "❌ 错误: 未找到 batch_gpu_monitor.py"
    echo "请确保GPU监控脚本在当前目录下"
    exit 1
fi

# 检查nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  警告: nvidia-smi 未找到，GPU监控将被禁用"
    ENABLE_GPU_MONITOR=false
else
    ENABLE_GPU_MONITOR=true
fi

# 测试配置
BATCH_SIZES=(8 16 32 64 96 128)
REQUESTS=128
MAX_TOKENS=1
TEST_LENGTH=1024
GPU_IDS="0,1,2,3"
GPU_INTERVAL=0.1

# 创建时间戳目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="batch_results_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

echo ""
echo "======================================"
echo "测试配置:"
echo "======================================"
echo "批次大小: ${BATCH_SIZES[@]}"
echo "总请求数: $REQUESTS"
echo "最大tokens: $MAX_TOKENS"
echo "测试长度: $TEST_LENGTH"
if [ "$ENABLE_GPU_MONITOR" = true ]; then
    echo "GPU监控: 已启用 (GPU ${GPU_IDS}, 采样间隔 ${GPU_INTERVAL}s)"
else
    echo "GPU监控: 已禁用"
fi
echo "结果目录: $RESULT_DIR"
echo "======================================"
echo ""

# 询问是否继续
read -p "是否开始测试? (Y/n): " confirm
confirm=${confirm:-Y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消测试"
    exit 0
fi

echo ""
echo "======================================"
echo "开始多批次测试..."
echo "======================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
TOTAL_TESTS=${#BATCH_SIZES[@]}
CURRENT_TEST=0
FAILED_TESTS=()

# GPU监控控制脚本
GPU_MONITOR_SCRIPT=$(cat <<'GPUEOF'
import sys
import time
import signal

# 确保能找到batch_gpu_monitor模块
sys.path.insert(0, '.')

from batch_gpu_monitor import BatchGPUMonitor, monitor_loop
import threading

# 解析参数
batch_size = int(sys.argv[1])
result_dir = sys.argv[2]
gpu_ids_str = sys.argv[3]
interval = float(sys.argv[4])

# 解析GPU IDs
gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]

# 创建监控器
monitor = BatchGPUMonitor(gpu_ids=gpu_ids, interval=interval)
monitor.start_monitoring(batch_size, result_dir)

# 启动监控循环线程
monitor_thread = threading.Thread(target=monitor_loop, args=(monitor,))
monitor_thread.daemon = False
monitor_thread.start()

# 信号处理
def signal_handler(sig, frame):
    monitor.stop_monitoring(result_dir)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 等待直到被终止
try:
    monitor_thread.join()
except KeyboardInterrupt:
    monitor.stop_monitoring(result_dir)
GPUEOF
)

# 循环运行每个批次大小的测试
for batch_size in "${BATCH_SIZES[@]}"; do
    CURRENT_TEST=$((CURRENT_TEST + 1))
    
    echo ""
    echo "======================================"
    echo "测试进度: [$CURRENT_TEST/$TOTAL_TESTS]"
    echo "======================================"
    echo "批次大小: $batch_size"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"
    echo ""
    
    # 启动GPU监控（后台进程）
    GPU_MONITOR_PID=""
    if [ "$ENABLE_GPU_MONITOR" = true ]; then
        echo "启动GPU监控..."
        echo "$GPU_MONITOR_SCRIPT" | python3 - $batch_size "$RESULT_DIR" "$GPU_IDS" $GPU_INTERVAL &
        GPU_MONITOR_PID=$!
        
        # 等待GPU监控初始化
        sleep 2
        
        # 检查GPU监控是否成功启动
        if kill -0 $GPU_MONITOR_PID 2>/dev/null; then
            echo "✓ GPU监控已启动 (PID: $GPU_MONITOR_PID)"
        else
            echo "⚠️  GPU监控启动失败，继续测试..."
            GPU_MONITOR_PID=""
        fi
    fi
    
    # 运行测试
    echo "开始批次测试..."
    python3 simple_batch_test.py \
        --batch-size $batch_size \
        --requests $REQUESTS \
        --max-tokens $MAX_TOKENS \
        --test-length $TEST_LENGTH
    
    TEST_EXIT_CODE=$?
    
    # 停止GPU监控
    if [ -n "$GPU_MONITOR_PID" ] && [ "$ENABLE_GPU_MONITOR" = true ]; then
        echo ""
        echo "停止GPU监控..."
        kill -SIGINT $GPU_MONITOR_PID 2>/dev/null
        
        # 等待GPU监控进程结束（最多5秒）
        for i in {1..10}; do
            if ! kill -0 $GPU_MONITOR_PID 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        
        # 如果还没结束，强制终止
        if kill -0 $GPU_MONITOR_PID 2>/dev/null; then
            kill -9 $GPU_MONITOR_PID 2>/dev/null
        fi
        
        wait $GPU_MONITOR_PID 2>/dev/null
        echo "✓ GPU监控已停止"
    fi
    
    # 检查测试结果
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ 批次 $batch_size 测试完成"
        
        # 移动并重命名结果文件
        if [ -f "simple_batch_results/batch_results.json" ]; then
            mv "simple_batch_results/batch_results.json" "$RESULT_DIR/batch_results_${batch_size}.json"
            echo "  ✓ 测试结果: batch_results_${batch_size}.json"
        else
            echo "  ⚠️  警告: 未找到测试结果文件"
            FAILED_TESTS+=($batch_size)
        fi
        
        # 检查GPU监控文件
        if [ -f "$RESULT_DIR/gpu_monitor_${batch_size}.csv" ]; then
            echo "  ✓ GPU数据: gpu_monitor_${batch_size}.csv"
        fi
        
        if [ -f "$RESULT_DIR/gpu_stats_${batch_size}.json" ]; then
            echo "  ✓ GPU统计: gpu_stats_${batch_size}.json"
        fi
        
        # 清理临时目录
        rm -rf "simple_batch_results"
    else
        echo ""
        echo "❌ 批次 $batch_size 测试失败 (退出码: $TEST_EXIT_CODE)"
        FAILED_TESTS+=($batch_size)
    fi
    
    # 在测试之间稍作延迟，让系统稳定
    if [ $CURRENT_TEST -lt $TOTAL_TESTS ]; then
        echo ""
        echo "等待5秒后继续下一个测试..."
        sleep 5
    fi
done

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================"
echo "所有批次测试完成！"
echo "======================================"
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo "成功: $((TOTAL_TESTS - ${#FAILED_TESTS[@]}))/$TOTAL_TESTS"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "失败批次: ${FAILED_TESTS[@]}"
fi

echo "======================================"
echo ""

# 统一分析所有结果
echo "======================================"
echo "开始统一分析所有测试结果..."
echo "======================================"
echo ""

# 检查分析脚本是否存在
if [ ! -f "analyze_batch_results.py" ]; then
    echo "⚠️  警告: 未找到 analyze_batch_results.py 分析脚本"
    echo "请确保脚本在当前目录下"
    echo ""
    echo "📁 测试结果已保存到: $RESULT_DIR"
    exit 0
fi

# 运行分析脚本
python3 analyze_batch_results.py --dir "$RESULT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "📊 完整结果文件:"
    echo "======================================"
    echo "结果目录: $RESULT_DIR/"
    echo ""
    echo "各批次测试数据:"
    for batch_size in "${BATCH_SIZES[@]}"; do
        if [ -f "$RESULT_DIR/batch_results_${batch_size}.json" ]; then
            echo "  ✓ batch_results_${batch_size}.json (测试数据)"
        fi
        if [ -f "$RESULT_DIR/gpu_monitor_${batch_size}.csv" ]; then
            echo "  ✓ gpu_monitor_${batch_size}.csv (GPU监控数据)"
        fi
        if [ -f "$RESULT_DIR/gpu_stats_${batch_size}.json" ]; then
            echo "  ✓ gpu_stats_${batch_size}.json (GPU统计)"
        fi
    done
    echo ""
    echo "综合分析报告:"
    echo "  - batch_analysis.png (性能分析图)"
    echo "  - batch_analysis_summary.csv (详细CSV报告)"
    echo ""
    echo "======================================"
    echo "🎉 测试和分析完成！"
    echo "======================================"
    echo ""
    echo "💡 快速查看:"
    echo "  - 查看图表: xdg-open $RESULT_DIR/batch_analysis.png"
    echo "  - 查看CSV: cat $RESULT_DIR/batch_analysis_summary.csv"
    echo "  - 查看GPU数据: cat $RESULT_DIR/gpu_stats_8.json"
    echo "  - 查看详情: ls -lh $RESULT_DIR/"
else
    echo ""
    echo "⚠️  分析失败，但测试数据已保存"
    echo "可以手动运行: python3 analyze_batch_results.py --dir $RESULT_DIR"
fi

echo ""
echo "======================================"

# 生成测试总结报告
SUMMARY_FILE="${RESULT_DIR}/test_summary.txt"
cat > "$SUMMARY_FILE" << EOF
====================================
VLLM 多批次测试总结报告
(包含GPU功耗监控)
====================================

测试时间: $(date '+%Y-%m-%d %H:%M:%S')
总耗时: ${MINUTES}分${SECONDS}秒

测试配置:
- 批次大小: ${BATCH_SIZES[@]}
- 总请求数: $REQUESTS
- 最大tokens: $MAX_TOKENS  
- 测试长度: $TEST_LENGTH
EOF

if [ "$ENABLE_GPU_MONITOR" = true ]; then
    echo "- GPU监控: GPU ${GPU_IDS}, 采样间隔 ${GPU_INTERVAL}s" >> "$SUMMARY_FILE"
else
    echo "- GPU监控: 已禁用" >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

测试结果:
- 总测试数: $TOTAL_TESTS
- 成功数: $((TOTAL_TESTS - ${#FAILED_TESTS[@]}))
- 失败数: ${#FAILED_TESTS[@]}
EOF

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "- 失败批次: ${FAILED_TESTS[@]}" >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

结果文件:
测试数据:
EOF

for batch_size in "${BATCH_SIZES[@]}"; do
    if [ -f "$RESULT_DIR/batch_results_${batch_size}.json" ]; then
        echo "- batch_results_${batch_size}.json" >> "$SUMMARY_FILE"
    fi
done

if [ "$ENABLE_GPU_MONITOR" = true ]; then
    cat >> "$SUMMARY_FILE" << EOF

GPU监控数据:
EOF

    for batch_size in "${BATCH_SIZES[@]}"; do
        if [ -f "$RESULT_DIR/gpu_monitor_${batch_size}.csv" ]; then
            echo "- gpu_monitor_${batch_size}.csv" >> "$SUMMARY_FILE"
            echo "- gpu_stats_${batch_size}.json" >> "$SUMMARY_FILE"
        fi
    done
fi

cat >> "$SUMMARY_FILE" << EOF

综合分析:
- batch_analysis.png
- batch_analysis_summary.csv

====================================
EOF

echo ""
echo "📝 测试总结已保存到: $SUMMARY_FILE"
echo ""
