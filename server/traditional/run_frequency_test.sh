#!/bin/bash

# 多频率自动化测试脚本
# 固定显存频率，遍历不同核心频率，每个频率进行完整的多批次测试

echo "=========================================="
echo "  多频率VLLM性能测试工具"
echo "  (集成GPU频率控制与功耗监控)"
echo "=========================================="
echo ""

# ==================== 配置区 ====================

# 固定显存频率（MHz）
MEMORY_FREQ=5001

# 要测试的核心频率列表（MHz）
CORE_FREQUENCIES=(1200 1800 2400 2805 3105)

# 每个频率下的批次大小列表
BATCH_SIZES=(8 16 32 64 96 128 256 512 1024)

# 每个批次的测试参数
REQUESTS=1024
MAX_TOKENS=128
TEST_LENGTH=1024

# GPU配置
GPU_IDS="0,1,2,3"
GPU_INTERVAL=0.1

# ==================== 函数定义 ====================

# 清理函数 - 确保退出时恢复GPU设置
cleanup() {
    echo ""
    echo "=========================================="
    echo "捕获到退出信号，正在清理..."
    echo "=========================================="
    
    # 恢复GPU频率设置
    echo "恢复GPU频率到默认设置..."
    sudo nvidia-smi -rgc
    sudo nvidia-smi -rmc
    sudo nvidia-smi -pm 0
    
    echo "清理完成！"
    exit 1
}

# 注册信号处理
trap cleanup INT TERM

# 锁定GPU频率函数
lock_gpu_frequency() {
    local core_freq=$1
    local is_first_lock=$2  # 是否是第一次锁定
    
    echo "=========================================="
    echo "设置GPU频率"
    echo "=========================================="
    echo "显存频率: ${MEMORY_FREQ} MHz (固定)"
    echo "核心频率: ${core_freq} MHz"
    echo "=========================================="
    
    # 只在第一次锁定时启用持久模式和锁定显存
    if [ "$is_first_lock" = "true" ]; then
        # 启用持久模式
        sudo nvidia-smi -pm 1
        if [ $? -ne 0 ]; then
            echo "❌ 错误: 无法启用持久模式"
            return 1
        fi
        
        # 锁定显存频率
        sudo nvidia-smi -lmc ${MEMORY_FREQ}
        if [ $? -ne 0 ]; then
            echo "❌ 错误: 无法锁定显存频率"
            return 1
        fi
        
        echo "✅ 持久模式已启用"
        echo "✅ 显存频率已锁定到 ${MEMORY_FREQ} MHz"
    fi
    
    # 锁定核心频率（每次都需要）
    sudo nvidia-smi -lgc ${core_freq}
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 无法锁定核心频率"
        return 1
    fi
    
    # 等待频率稳定
    sleep 3
    
    # 验证频率设置
    echo ""
    echo "验证GPU频率设置:"
    nvidia-smi --query-gpu=index,clocks.gr,clocks.mem --format=csv
    
    echo ""
    echo "✅ GPU频率设置成功"
    echo "=========================================="
    
    return 0
}

# 解锁GPU频率函数
unlock_gpu_frequency() {
    echo ""
    echo "=========================================="
    echo "恢复GPU频率到默认设置"
    echo "=========================================="
    
    sudo nvidia-smi -rgc
    sudo nvidia-smi -rmc
    sudo nvidia-smi -pm 0
    
    # 验证恢复
    echo ""
    echo "验证GPU频率恢复:"
    nvidia-smi --query-gpu=index,clocks.gr,clocks.mem --format=csv
    
    echo ""
    echo "✅ GPU频率已恢复"
    echo "=========================================="
}

# ==================== 主程序 ====================

echo "测试配置:"
echo "=========================================="
echo "显存频率: ${MEMORY_FREQ} MHz (固定)"
echo "核心频率: ${CORE_FREQUENCIES[@]} MHz"
echo "批次大小: ${BATCH_SIZES[@]}"
echo "每批次请求数: ${REQUESTS}"
echo "最大tokens: ${MAX_TOKENS}"
echo "测试长度: ${TEST_LENGTH}"
echo "GPU监控: GPU ${GPU_IDS}, 采样间隔 ${GPU_INTERVAL}s"
echo "=========================================="
echo ""

# 检查依赖
echo "检查依赖..."

# 检查VLLM服务器
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ VLLM服务器未运行"
    echo "请先启动服务器: ./start_vllm_server.sh"
    exit 1
fi
echo "✅ VLLM服务器运行正常"

# 检查GPU监控脚本
if [ ! -f "batch_gpu_monitor.py" ]; then
    echo "❌ 未找到 batch_gpu_monitor.py"
    exit 1
fi
echo "✅ GPU监控脚本已找到"

# 检查批次测试脚本
if [ ! -f "simple_batch_test.py" ]; then
    echo "❌ 未找到 simple_batch_test.py"
    exit 1
fi
echo "✅ 批次测试脚本已找到"

# 检查nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到"
    exit 1
fi
echo "✅ nvidia-smi 可用"

# 检查sudo权限
if ! sudo -n nvidia-smi -pm 1 2>/dev/null; then
    echo ""
    echo "⚠️  需要sudo权限来控制GPU频率"
    echo "请输入sudo密码:"
    sudo nvidia-smi -pm 0
fi

echo ""
echo "=========================================="
read -p "确认开始多频率测试? (Y/n): " confirm
confirm=${confirm:-Y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消测试"
    exit 0
fi

# 创建总结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_RESULT_DIR="frequency_test_${TIMESTAMP}"
mkdir -p "$MASTER_RESULT_DIR"

echo ""
echo "=========================================="
echo "开始多频率测试"
echo "总结果目录: ${MASTER_RESULT_DIR}"
echo "=========================================="

# 记录开始时间
TEST_START_TIME=$(date +%s)
TOTAL_FREQUENCIES=${#CORE_FREQUENCIES[@]}
CURRENT_FREQ_INDEX=0
FAILED_FREQUENCIES=()
IS_FIRST_FREQUENCY=true

# GPU监控控制脚本（内嵌）
GPU_MONITOR_SCRIPT=$(cat <<'GPUEOF'
import sys
import time
import signal
sys.path.insert(0, '.')
from batch_gpu_monitor import BatchGPUMonitor, monitor_loop
import threading

batch_size = int(sys.argv[1])
result_dir = sys.argv[2]
gpu_ids_str = sys.argv[3]
interval = float(sys.argv[4])
gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]

monitor = BatchGPUMonitor(gpu_ids=gpu_ids, interval=interval)
monitor.start_monitoring(batch_size, result_dir)

monitor_thread = threading.Thread(target=monitor_loop, args=(monitor,))
monitor_thread.daemon = False
monitor_thread.start()

def signal_handler(sig, frame):
    monitor.stop_monitoring(result_dir)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    monitor_thread.join()
except KeyboardInterrupt:
    monitor.stop_monitoring(result_dir)
GPUEOF
)

# 遍历每个核心频率
for core_freq in "${CORE_FREQUENCIES[@]}"; do
    CURRENT_FREQ_INDEX=$((CURRENT_FREQ_INDEX + 1))
    
    echo ""
    echo "=========================================="
    echo "频率测试进度: [$CURRENT_FREQ_INDEX/$TOTAL_FREQUENCIES]"
    echo "=========================================="
    echo "当前核心频率: ${core_freq} MHz"
    echo "显存频率: ${MEMORY_FREQ} MHz"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 锁定GPU频率（第一次需要设置持久模式和显存，后续只切换核心频率）
    if ! lock_gpu_frequency ${core_freq} ${IS_FIRST_FREQUENCY}; then
        echo "❌ 频率 ${core_freq} MHz 设置失败，跳过此频率"
        FAILED_FREQUENCIES+=(${core_freq})
        continue
    fi
    
    # 标记已经不是第一次了
    IS_FIRST_FREQUENCY=false
    
    # 为当前频率创建子目录
    FREQ_RESULT_DIR="${MASTER_RESULT_DIR}/freq_${core_freq}MHz"
    mkdir -p "$FREQ_RESULT_DIR"
    
    echo ""
    echo "=========================================="
    echo "开始批次测试 (核心频率: ${core_freq} MHz)"
    echo "结果将保存到: ${FREQ_RESULT_DIR}"
    echo "=========================================="
    echo ""
    
    # 批次测试统计
    BATCH_SUCCESS_COUNT=0
    BATCH_FAIL_COUNT=0
    FAILED_BATCHES=()
    
    # 遍历每个批次大小
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "======================================"
        echo "批次大小: ${batch_size}"
        echo "频率: ${core_freq} MHz | 批次: ${batch_size}"
        echo "======================================"
        
        # 启动GPU监控
        GPU_MONITOR_PID=""
        echo "启动GPU监控..."
        echo "$GPU_MONITOR_SCRIPT" | python3 - $batch_size "$FREQ_RESULT_DIR" "$GPU_IDS" $GPU_INTERVAL &
        GPU_MONITOR_PID=$!
        sleep 2
        
        if kill -0 $GPU_MONITOR_PID 2>/dev/null; then
            echo "✓ GPU监控已启动 (PID: $GPU_MONITOR_PID)"
        else
            echo "⚠️  GPU监控启动失败，继续测试..."
            GPU_MONITOR_PID=""
        fi
        
        # 运行批次测试
        echo "开始批次测试..."
        python3 simple_batch_test.py \
            --batch-size $batch_size \
            --requests $REQUESTS \
            --max-tokens $MAX_TOKENS \
            --test-length $TEST_LENGTH
        
        TEST_EXIT_CODE=$?
        
        # 停止GPU监控
        if [ -n "$GPU_MONITOR_PID" ]; then
            echo ""
            echo "停止GPU监控..."
            kill -SIGINT $GPU_MONITOR_PID 2>/dev/null
            
            for i in {1..10}; do
                if ! kill -0 $GPU_MONITOR_PID 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            
            if kill -0 $GPU_MONITOR_PID 2>/dev/null; then
                kill -9 $GPU_MONITOR_PID 2>/dev/null
            fi
            
            wait $GPU_MONITOR_PID 2>/dev/null
            echo "✓ GPU监控已停止"
        fi
        
        # 检查测试结果
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo ""
            echo "✅ 批次 ${batch_size} 测试完成"
            BATCH_SUCCESS_COUNT=$((BATCH_SUCCESS_COUNT + 1))
            
            # 移动结果文件
            if [ -f "simple_batch_results/batch_results.json" ]; then
                mv "simple_batch_results/batch_results.json" "$FREQ_RESULT_DIR/batch_results_${batch_size}.json"
                echo "  ✓ 测试结果: batch_results_${batch_size}.json"
            else
                echo "  ⚠️  警告: 未找到测试结果文件"
                FAILED_BATCHES+=(${batch_size})
            fi
            
            # 检查GPU监控文件
            if [ -f "$FREQ_RESULT_DIR/gpu_monitor_${batch_size}.csv" ]; then
                echo "  ✓ GPU数据: gpu_monitor_${batch_size}.csv"
            fi
            
            if [ -f "$FREQ_RESULT_DIR/gpu_stats_${batch_size}.json" ]; then
                echo "  ✓ GPU统计: gpu_stats_${batch_size}.json"
            fi
            
            rm -rf "simple_batch_results"
        else
            echo ""
            echo "❌ 批次 ${batch_size} 测试失败"
            BATCH_FAIL_COUNT=$((BATCH_FAIL_COUNT + 1))
            FAILED_BATCHES+=(${batch_size})
        fi
        
        # 批次间延迟
        if [ $batch_size -ne ${BATCH_SIZES[-1]} ]; then
            echo ""
            echo "等待5秒后继续下一个批次..."
            sleep 5
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "频率 ${core_freq} MHz 的批次测试完成"
    echo "=========================================="
    echo "成功批次: ${BATCH_SUCCESS_COUNT}/${#BATCH_SIZES[@]}"
    echo "失败批次: ${BATCH_FAIL_COUNT}"
    if [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
        echo "失败的批次大小: ${FAILED_BATCHES[@]}"
    fi
    echo "=========================================="
    
    # 生成当前频率的测试总结
    FREQ_SUMMARY_FILE="${FREQ_RESULT_DIR}/frequency_summary.txt"
    cat > "$FREQ_SUMMARY_FILE" << EOF
====================================
频率测试总结
====================================

核心频率: ${core_freq} MHz
显存频率: ${MEMORY_FREQ} MHz
测试时间: $(date '+%Y-%m-%d %H:%M:%S')

批次配置:
- 批次大小: ${BATCH_SIZES[@]}
- 每批次请求数: ${REQUESTS}
- 最大tokens: ${MAX_TOKENS}
- 测试长度: ${TEST_LENGTH}

测试结果:
- 总批次数: ${#BATCH_SIZES[@]}
- 成功批次: ${BATCH_SUCCESS_COUNT}
- 失败批次: ${BATCH_FAIL_COUNT}
EOF

    if [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
        echo "- 失败的批次: ${FAILED_BATCHES[@]}" >> "$FREQ_SUMMARY_FILE"
    fi

    echo "" >> "$FREQ_SUMMARY_FILE"
    echo "结果文件:" >> "$FREQ_SUMMARY_FILE"
    for batch_size in "${BATCH_SIZES[@]}"; do
        if [ -f "$FREQ_RESULT_DIR/batch_results_${batch_size}.json" ]; then
            echo "- batch_results_${batch_size}.json" >> "$FREQ_SUMMARY_FILE"
            echo "- gpu_monitor_${batch_size}.csv" >> "$FREQ_SUMMARY_FILE"
            echo "- gpu_stats_${batch_size}.json" >> "$FREQ_SUMMARY_FILE"
        fi
    done
    
    echo "=====================================" >> "$FREQ_SUMMARY_FILE"
    
    echo "📝 频率总结已保存: ${FREQ_SUMMARY_FILE}"
    
    # 频率间冷却时间（不解锁频率，保持持久模式）
    if [ $CURRENT_FREQ_INDEX -lt $TOTAL_FREQUENCIES ]; then
        echo ""
        echo "等待10秒后切换到下一个频率..."
        echo "（保持GPU持久模式，将直接切换频率）"
        sleep 10
    fi
done

# 最终清理 - 确保GPU频率恢复
echo ""
echo "=========================================="
echo "所有频率测试完成，恢复GPU设置..."
echo "=========================================="
unlock_gpu_frequency

# 计算总耗时
TEST_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((TEST_END_TIME - TEST_START_TIME))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=========================================="
echo "多频率测试完成！"
echo "=========================================="
echo "总耗时: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒"
echo "成功频率: $((TOTAL_FREQUENCIES - ${#FAILED_FREQUENCIES[@]}))/$TOTAL_FREQUENCIES"

if [ ${#FAILED_FREQUENCIES[@]} -gt 0 ]; then
    echo "失败频率: ${FAILED_FREQUENCIES[@]} MHz"
fi

echo "=========================================="
echo ""

# 生成总测试报告
MASTER_SUMMARY_FILE="${MASTER_RESULT_DIR}/master_summary.txt"
cat > "$MASTER_SUMMARY_FILE" << EOF
====================================
多频率VLLM性能测试总报告
====================================

测试时间: $(date '+%Y-%m-%d %H:%M:%S')
总耗时: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒

频率配置:
- 显存频率: ${MEMORY_FREQ} MHz (固定)
- 核心频率: ${CORE_FREQUENCIES[@]} MHz

批次配置:
- 批次大小: ${BATCH_SIZES[@]}
- 每批次请求数: ${REQUESTS}
- 最大tokens: ${MAX_TOKENS}
- 测试长度: ${TEST_LENGTH}

测试结果:
- 总频率数: ${TOTAL_FREQUENCIES}
- 成功频率: $((TOTAL_FREQUENCIES - ${#FAILED_FREQUENCIES[@]}))
- 失败频率: ${#FAILED_FREQUENCIES[@]}
EOF

if [ ${#FAILED_FREQUENCIES[@]} -gt 0 ]; then
    echo "- 失败的频率: ${FAILED_FREQUENCIES[@]} MHz" >> "$MASTER_SUMMARY_FILE"
fi

echo "" >> "$MASTER_SUMMARY_FILE"
echo "结果目录结构:" >> "$MASTER_SUMMARY_FILE"
echo "${MASTER_RESULT_DIR}/" >> "$MASTER_SUMMARY_FILE"
for core_freq in "${CORE_FREQUENCIES[@]}"; do
    if [ -d "${MASTER_RESULT_DIR}/freq_${core_freq}MHz" ]; then
        echo "├── freq_${core_freq}MHz/" >> "$MASTER_SUMMARY_FILE"
        echo "│   ├── frequency_summary.txt" >> "$MASTER_SUMMARY_FILE"
        echo "│   ├── batch_results_*.json" >> "$MASTER_SUMMARY_FILE"
        echo "│   ├── gpu_monitor_*.csv" >> "$MASTER_SUMMARY_FILE"
        echo "│   └── gpu_stats_*.json" >> "$MASTER_SUMMARY_FILE"
    fi
done

echo "" >> "$MASTER_SUMMARY_FILE"
echo "====================================" >> "$MASTER_SUMMARY_FILE"

echo "📝 总测试报告已保存: ${MASTER_SUMMARY_FILE}"

echo ""
echo "=========================================="
echo "📊 完整结果文件:"
echo "=========================================="
echo "总结果目录: ${MASTER_RESULT_DIR}/"
echo ""
echo "各频率子目录:"
for core_freq in "${CORE_FREQUENCIES[@]}"; do
    if [ -d "${MASTER_RESULT_DIR}/freq_${core_freq}MHz" ]; then
        echo "  ✓ freq_${core_freq}MHz/ ($(ls ${MASTER_RESULT_DIR}/freq_${core_freq}MHz/ | wc -l) 个文件)"
    fi
done

# ==================== 自动分析部分 ====================

echo ""
echo "=========================================="
echo "开始自动分析所有频率数据..."
echo "=========================================="
echo ""

# 检查分析脚本是否存在
if [ ! -f "analyze_frequency_results.py" ]; then
    echo "⚠️  警告: 未找到 analyze_frequency_results.py 分析脚本"
    echo "请确保脚本在当前目录下"
    echo ""
    echo "📁 测试结果已保存到: ${MASTER_RESULT_DIR}"
    echo ""
    echo "建议手动运行分析:"
    echo "  python3 analyze_frequency_results.py --dir ${MASTER_RESULT_DIR}"
else
    # 运行分析脚本
    echo "正在运行分析脚本..."
    python3 analyze_frequency_results.py --dir "${MASTER_RESULT_DIR}"
    
    ANALYSIS_EXIT_CODE=$?
    
    if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "📊 完整分析报告:"
        echo "=========================================="
        echo "主目录: ${MASTER_RESULT_DIR}/"
        echo ""
        echo "📈 生成的图表:"
        echo "  - cross_frequency_comparison.png (跨频率对比)"
        for core_freq in "${CORE_FREQUENCIES[@]}"; do
            if [ -f "${MASTER_RESULT_DIR}/analysis_freq_${core_freq}MHz.png" ]; then
                echo "  - analysis_freq_${core_freq}MHz.png (${core_freq}MHz详细分析)"
            fi
        done
        echo ""
        echo "📋 数据报告:"
        echo "  - frequency_analysis_summary.csv (完整数据表)"
        echo "  - master_summary.txt (测试总结)"
        echo ""
        echo "📁 各频率子目录:"
        for core_freq in "${CORE_FREQUENCIES[@]}"; do
            if [ -d "${MASTER_RESULT_DIR}/freq_${core_freq}MHz" ]; then
                echo "  - freq_${core_freq}MHz/"
                echo "    ├── frequency_summary.txt"
                echo "    ├── batch_results_*.json"
                echo "    ├── gpu_monitor_*.csv"
                echo "    └── gpu_stats_*.json"
            fi
        done
        echo ""
        echo "=========================================="
        echo "💡 快速查看方式:"
        echo "=========================================="
        echo "查看跨频率对比图:"
        echo "  xdg-open ${MASTER_RESULT_DIR}/cross_frequency_comparison.png"
        echo ""
        echo "查看某个频率的详细分析:"
        echo "  xdg-open ${MASTER_RESULT_DIR}/analysis_freq_2400MHz.png"
        echo ""
        echo "查看完整数据表:"
        echo "  cat ${MASTER_RESULT_DIR}/frequency_analysis_summary.csv"
        echo ""
        echo "查看测试总结:"
        echo "  cat ${MASTER_RESULT_DIR}/master_summary.txt"
        echo ""
        echo "=========================================="
        echo "🎉 多频率测试和分析全部完成！"
        echo "=========================================="
        echo ""
        echo "关键发现（需要查看图表）:"
        echo "1. 最高吞吐量频率: 查看 cross_frequency_comparison.png"
        echo "2. 最优能效比频率: 查看 Energy Efficiency 曲线"
        echo "3. 功耗-性能平衡点: 对比 Power 和 Throughput 曲线"
        echo ""
    else
        echo ""
        echo "⚠️  分析脚本执行失败（退出码: ${ANALYSIS_EXIT_CODE}）"
        echo "但测试数据已完整保存"
        echo ""
        echo "可以手动重新运行分析:"
        echo "  python3 analyze_frequency_results.py --dir ${MASTER_RESULT_DIR}"
    fi
fi

echo ""
echo "=========================================="
echo "📂 所有结果保存在: ${MASTER_RESULT_DIR}"
echo "=========================================="
echo ""
