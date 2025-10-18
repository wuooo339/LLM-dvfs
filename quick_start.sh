#!/bin/bash

# CPU功耗监控快速启动脚本

echo "CPU功耗监控工具 - 快速启动"
echo "=========================="

# 检查依赖
echo "检查依赖..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: python3 未找到"
    exit 1
fi

if ! command -v perf &> /dev/null; then
    echo "❌ 错误: perf 未找到，请安装perf工具:"
    echo "  Ubuntu/Debian: sudo apt install linux-tools-common linux-tools-generic"
    echo "  CentOS/RHEL: sudo yum install perf"
    exit 1
fi

echo "✓ 基本依赖检查通过"

# 显示菜单
while true; do
    echo ""
    echo "请选择操作:"
    echo "1. 开始监控 (监控4个GPU和CPU，100ms间隔，基于perf工具)"
    echo "2. 只监控GPU (监控4个GPU，不监控CPU)"
    echo "3. 只监控CPU (监控CPU性能，基于perf工具)"
    echo "4. 定时监控 (监控指定时间)"
    echo "5. 生成图表 (从现有数据生成图表)"
    echo "6. 查看帮助"
    echo "7. 退出"
    
    read -p "请输入选择 (1-7): " choice
    
    case $choice in
        1)
            echo "开始持续监控GPU和CPU..."
            echo "按 Ctrl+C 停止监控"
            output_file="gpu_cpu_power_$(date +%Y%m%d_%H%M%S).csv"
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
        2)
            echo "开始持续监控GPU（不监控CPU）..."
            echo "按 Ctrl+C 停止监控"
            output_file="gpu_power_$(date +%Y%m%d_%H%M%S).csv"
            python3 multi_gpu_monitor.py --no-cpu --output "$output_file" &
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
            echo "开始持续监控CPU（基于perf工具）..."
            echo "按 Ctrl+C 停止监控"
            output_file="cpu_power_$(date +%Y%m%d_%H%M%S).json"
            python3 cpu_monitor.py &
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
        4)
            read -p "请输入监控时间（秒）: " duration
            if [[ "$duration" =~ ^[0-9]+$ ]]; then
                output_file="gpu_cpu_power_${duration}s_$(date +%Y%m%d_%H%M%S).csv"
                echo "开始监控 ${duration} 秒..."
                python3 multi_gpu_monitor.py --duration $duration --output $output_file
                echo "监控完成，数据保存到: $output_file"
            else
                echo "❌ 请输入有效的数字"
            fi
            ;;
        5)
            read -p "请输入CSV文件路径: " csv_file
            if [ -f "$csv_file" ]; then
                output_dir="plots_$(date +%Y%m%d_%H%M%S)"
                echo "生成GPU和CPU功耗/温度图到目录: $output_dir"
                python3 plot_gpu_power.py "$csv_file" --output-dir "$output_dir"
                echo "图表生成完成！包含："
                echo "  - gpu_power_consumption.png（每GPU功耗，叠加CPU功耗）"
                echo "  - gpu_power_comparison.png（GPU与CPU总功耗对比）"
                echo "  - gpu_temperature.png（各GPU温度）"
                echo "  - cpu_power_monitoring.png（CPU功耗/能量/温度/PkgWatt）"
                echo "  - gpu_utilization_vs_power.png（利用率与功耗关系）"
            else
                echo "❌ 文件不存在: $csv_file"
            fi
            ;;
        6)
            echo ""
            echo "GPU和CPU功耗监控工具使用说明:"
            echo "================================"
            echo ""
            echo "1. 多GPU和CPU监控脚本 (multi_gpu_monitor.py):"
            echo "   - 实时监控GPU和CPU功耗、温度、利用率等数据"
            echo "   - 支持100ms高精度采样"
            echo "   - 数据保存为CSV和JSON格式"
            echo "   - 使用 --no-cpu 参数可禁用CPU监控"
            echo ""
            echo "2. 数据可视化脚本 (plot_gpu_power.py):"
            echo "   - 生成功耗变化图、对比图、关系图等"
            echo "   - 支持GPU和CPU数据的可视化"
            echo "   - 支持多种图表类型"
            echo "   - 生成统计报告"
            echo ""
            echo "3. CPU监控模块 (cpu_monitor.py):"
            echo "   - 使用 sudo perf stat -e power/energy-pkg/ -a sleep 0.1 获取CPU功耗数据"
            echo "   - 通过能量增量计算实时功率"
            echo "   - 支持0.1秒高精度采样"
            echo "   - 数据保存为JSON格式"
            echo "   - 目前仅支持功耗监控"
            echo ""
            echo "4. 监控数据包含:"
            echo "   - GPU: 功耗、温度、利用率、显存使用等"
            echo "   - CPU: 功耗（基于perf stat能量测量）"
            echo "   - 时间戳和可读时间"
            echo ""
            echo "5. 注意事项:"
            echo "   - 需要sudo权限运行perf命令"
            echo "   - 确保系统支持power/energy-pkg/事件"
            echo "   - 建议在负载较高时进行监控以获得更准确的数据"
            echo ""
            echo "详细文档请查看 README.md"
            ;;
        7)
            echo "退出程序"
            break
            ;;
        *)
            echo "❌ 无效选择，请重新输入"
            ;;
    esac
done
