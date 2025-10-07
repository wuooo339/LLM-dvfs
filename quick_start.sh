#!/bin/bash

# GPU功耗监控快速启动脚本

echo "GPU功耗监控工具 - 快速启动"
echo "=============================="

# 检查依赖
echo "检查依赖..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: nvidia-smi 未找到，请确保NVIDIA驱动已正确安装"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: python3 未找到"
    exit 1
fi

echo "✓ 基本依赖检查通过"

# 检查Python包
echo "检查Python包..."
python3 -c "import pandas, matplotlib, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要的Python包，正在安装..."
    pip install pandas matplotlib numpy
    if [ $? -ne 0 ]; then
        echo "❌ Python包安装失败"
        exit 1
    fi
fi

echo "✓ Python包检查通过"

# 显示菜单
while true; do
    echo ""
    echo "请选择操作:"
    echo "1. 运行测试 (验证工具是否正常工作)"
    echo "2. 开始监控 (监控4个GPU，100ms间隔)"
    echo "3. 定时监控 (监控指定时间)"
    echo "4. 生成图表 (从现有数据生成图表)"
    echo "5. 查看帮助"
    echo "6. 退出"
    
    read -p "请输入选择 (1-6): " choice
    
    case $choice in
        1)
            echo "运行测试..."
            python3 test_gpu_monitor.py
            ;;
        2)
            echo "开始持续监控..."
            echo "按 Ctrl+C 停止监控"
            output_file="gpu_power_$(date +%Y%m%d_%H%M%S).csv"
            python3 multi_gpu_monitor.py --output "$output_file" &
            monitor_pid=$!
            
            echo "监控进程PID: $monitor_pid"
            echo "数据保存到: $output_file"
            echo ""
            echo "按任意键停止监控..."
            read -n 1 -s
            echo ""
            echo "正在停止监控..."
            kill $monitor_pid 2>/dev/null
            wait $monitor_pid 2>/dev/null
            echo "监控已停止"
            ;;
        3)
            read -p "请输入监控时间（秒）: " duration
            if [[ "$duration" =~ ^[0-9]+$ ]]; then
                output_file="gpu_power_${duration}s_$(date +%Y%m%d_%H%M%S).csv"
                echo "开始监控 ${duration} 秒..."
                python3 multi_gpu_monitor.py --duration $duration --output $output_file
                echo "监控完成，数据保存到: $output_file"
            else
                echo "❌ 请输入有效的数字"
            fi
            ;;
        4)
            read -p "请输入CSV文件路径: " csv_file
            if [ -f "$csv_file" ]; then
                output_dir="plots_$(date +%Y%m%d_%H%M%S)"
                echo "生成图表到目录: $output_dir"
                python3 plot_gpu_power.py "$csv_file" --output-dir "$output_dir"
                echo "图表生成完成！"
            else
                echo "❌ 文件不存在: $csv_file"
            fi
            ;;
        5)
            echo ""
            echo "使用说明:"
            echo "1. 多GPU监控脚本 (multi_gpu_monitor.py):"
            echo "   - 实时监控GPU功耗、温度、利用率等数据"
            echo "   - 支持100ms高精度采样"
            echo "   - 数据保存为CSV和JSON格式"
            echo ""
            echo "2. 数据可视化脚本 (plot_gpu_power.py):"
            echo "   - 生成功耗变化图、对比图、关系图等"
            echo "   - 支持多种图表类型"
            echo "   - 生成统计报告"
            echo ""
            echo "3. 测试脚本 (test_gpu_monitor.py):"
            echo "   - 验证工具是否正常工作"
            echo "   - 检查依赖和环境"
            echo ""
            echo "详细文档请查看 README.md"
            ;;
        6)
            echo "退出程序"
            break
            ;;
        *)
            echo "❌ 无效选择，请重新输入"
            ;;
    esac
done
