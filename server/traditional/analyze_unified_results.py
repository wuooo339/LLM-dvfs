#!/usr/bin/env python3
"""
分析统一测试结果
绘制 prefill 和 decode 阶段的功耗频率折线图
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_true_peak_power(analyzed_intervals):
    """获取真正的峰值功耗（从原始数据中）"""
    peak_power = 0
    for interval in analyzed_intervals:
        if 'power_data' in interval and interval['power_data']:
            interval_peak = max(interval['power_data'])
            peak_power = max(peak_power, interval_peak)
    return peak_power

def get_gpu_max_power():
    """获取GPU最大功耗限制"""
    try:
        import subprocess
        # 使用 nvidia-smi -q -d POWER 获取详细信息
        result = subprocess.run([
            "nvidia-smi", 
            "-q", 
            "-d", 
            "POWER",
            "--id=0"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # 解析输出查找 "Current Power Limit"
            lines = result.stdout.split('\n')
            for line in lines:
                if "Current Power Limit" in line:
                    # 提取功率值，格式通常是 "Current Power Limit          : 320.00 W"
                    parts = line.split(':')
                    if len(parts) > 1:
                        power_str = parts[1].strip().replace('W', '').strip()
                        power_limit = float(power_str)
                        return power_limit
    except Exception as e:
        pass
    
    # 如果无法获取，尝试简单的查询
    try:
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
            "--id=0"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            power_limit = float(result.stdout.strip())
            return power_limit
    except Exception as e:
        pass
    
    # 默认值
    return 320

def load_results(storage_dir="vllm_unified_results"):
    """加载测试结果"""
    storage_path = Path(storage_dir)
    
    results_file = storage_path / "unified_results.json"
    if not results_file.exists():
        print(f"错误: 未找到结果文件 {results_file}")
        return None
    
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)

def create_phase_timeline_plot(analyzed_intervals, storage_dir):
    """创建阶段时间线折线图"""
    
    if not analyzed_intervals:
        print("没有数据可以绘制")
        return
    
    # 准备数据 - 使用相对时间（从0开始）
    start_time = analyzed_intervals[0]['start_time']
    times = [(interval['start_time'] - start_time) for interval in analyzed_intervals]
    powers = [interval['avg_power'] for interval in analyzed_intervals]
    graphics_clocks = [interval['avg_graphics_clock'] for interval in analyzed_intervals]
    memory_clocks = [interval['avg_memory_clock'] for interval in analyzed_intervals]
    phases = [interval['phase'] for interval in analyzed_intervals]
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('VLLM Unified Generation Test - Power and Frequency Timeline', fontsize=16, fontweight='bold')
    
    # 功耗图
    ax1.plot(times, powers, 'b-', linewidth=2, label='Average Power')
    
    # 添加GPU最大功耗参考线
    max_power = max(powers) if powers else 0
    ax1.axhline(y=max_power, color='red', linestyle='--', alpha=0.7, label=f'Peak Power (100ms avg): {max_power:.1f}W')
    
    # 获取真正的峰值功耗（从原始GPU数据中）
    true_peak_power = get_true_peak_power(analyzed_intervals)
    if true_peak_power > max_power:
        ax1.axhline(y=true_peak_power, color='purple', linestyle='-', alpha=0.8, label=f'True Peak Power: {true_peak_power:.1f}W')
    
    # 获取GPU最大可用功耗
    gpu_max_power = get_gpu_max_power()
    ax1.axhline(y=gpu_max_power, color='orange', linestyle=':', alpha=0.7, label=f'GPU Max: {gpu_max_power:.0f}W')
    
    ax1.set_ylabel('Power (W)')
    ax1.set_title('GPU Power Timeline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加阶段背景色
    add_phase_backgrounds(ax1, times, phases)
    
    # 图形时钟频率图
    ax2.plot(times, graphics_clocks, 'g-', linewidth=2, label='Graphics Clock')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title('GPU Graphics Clock Frequency Timeline')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    add_phase_backgrounds(ax2, times, phases)
    
    # 内存时钟频率图
    ax3.plot(times, memory_clocks, 'r-', linewidth=2, label='Memory Clock')
    ax3.set_ylabel('Frequency (MHz)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('GPU Memory Clock Frequency Timeline')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    add_phase_backgrounds(ax3, times, phases)
    
    # 格式化x轴 - 使用相对时间
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, max(times) if times else 1)
        ax.set_xticks(range(0, int(max(times)) + 1, 1))
    
    plt.tight_layout()
    plt.savefig(storage_dir / "vllm_unified_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"时间线图表已保存到: {storage_dir / 'vllm_unified_timeline.png'}")

def add_phase_backgrounds(ax, times, phases):
    """为图表添加阶段背景色"""
    current_phase = None
    phase_start = None
    
    for i, (time, phase) in enumerate(zip(times, phases)):
        if phase != current_phase:
            if current_phase and phase_start is not None:
                # 添加阶段背景色
                if current_phase == 'prefill':
                    ax.axvspan(phase_start, time, alpha=0.3, color='red', label='Prefill' if i == 1 else "")
                elif current_phase == 'decode':
                    ax.axvspan(phase_start, time, alpha=0.3, color='green', label='Decode' if i == 1 else "")
            current_phase = phase
            phase_start = time
    
    # 处理最后一个阶段
    if current_phase and phase_start is not None:
        if current_phase == 'prefill':
            ax.axvspan(phase_start, times[-1], alpha=0.3, color='red')
        elif current_phase == 'decode':
            ax.axvspan(phase_start, times[-1], alpha=0.3, color='green')

def create_phase_comparison_plot(analyzed_intervals, storage_dir):
    """创建阶段对比图"""
    
    # 分离不同阶段的数据
    prefill_data = [i for i in analyzed_intervals if i['phase'] == 'prefill']
    decode_data = [i for i in analyzed_intervals if i['phase'] == 'decode']
    idle_data = [i for i in analyzed_intervals if i['phase'] == 'idle']
    
    if not prefill_data and not decode_data:
        print("没有prefill或decode数据可以绘制")
        return
    
    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VLLM Phase Performance Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 功耗对比
    phases = []
    avg_powers = []
    max_powers = []
    
    if prefill_data:
        phases.append('Prefill')
        avg_powers.append(sum(i['avg_power'] for i in prefill_data) / len(prefill_data))
        max_powers.append(max(i['max_power'] for i in prefill_data))
    
    if decode_data:
        phases.append('Decode')
        avg_powers.append(sum(i['avg_power'] for i in decode_data) / len(decode_data))
        max_powers.append(max(i['max_power'] for i in decode_data))
    
    if idle_data:
        phases.append('Idle')
        avg_powers.append(sum(i['avg_power'] for i in idle_data) / len(idle_data))
        max_powers.append(max(i['max_power'] for i in idle_data))
    
    x = np.arange(len(phases))
    width = 0.35
    
    ax1.bar(x - width/2, avg_powers, width, label='Average Power', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, max_powers, width, label='Max Power', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 频率对比 - 图形时钟
    graphics_avg = []
    graphics_max = []
    
    for phase_data in [prefill_data, decode_data, idle_data]:
        if phase_data:
            graphics_avg.append(sum(i['avg_graphics_clock'] for i in phase_data) / len(phase_data))
            graphics_max.append(max(i['avg_graphics_clock'] for i in phase_data))
        else:
            graphics_avg.append(0)
            graphics_max.append(0)
    
    ax2.bar(x - width/2, graphics_avg, width, label='Average Frequency', color='lightgreen', alpha=0.8)
    ax2.bar(x + width/2, graphics_max, width, label='Max Frequency', color='orange', alpha=0.8)
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Graphics Clock Frequency (MHz)')
    ax2.set_title('Graphics Clock Frequency Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 频率对比 - 内存时钟
    memory_avg = []
    memory_max = []
    
    for phase_data in [prefill_data, decode_data, idle_data]:
        if phase_data:
            memory_avg.append(sum(i['avg_memory_clock'] for i in phase_data) / len(phase_data))
            memory_max.append(max(i['avg_memory_clock'] for i in phase_data))
        else:
            memory_avg.append(0)
            memory_max.append(0)
    
    ax3.bar(x - width/2, memory_avg, width, label='Average Frequency', color='purple', alpha=0.8)
    ax3.bar(x + width/2, memory_max, width, label='Max Frequency', color='brown', alpha=0.8)
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Memory Clock Frequency (MHz)')
    ax3.set_title('Memory Clock Frequency Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phases)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 数据点数量对比
    data_counts = []
    for phase_data in [prefill_data, decode_data, idle_data]:
        data_counts.append(len(phase_data))
    
    ax4.bar(phases, data_counts, color=['red', 'green', 'blue'], alpha=0.7)
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Data Points Count')
    ax4.set_title('Data Points Count by Phase')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(storage_dir / "phase_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"阶段对比图已保存到: {storage_dir / 'phase_comparison.png'}")

def print_summary(results):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("VLLM 统一生成测试结果总结")
    print("=" * 60)
    
    # 基本统计
    test_info = results.get('test_info', {})
    print(f"\n测试信息:")
    print(f"  总请求数: {test_info.get('total_requests', 0)}")
    print(f"  GPU监控间隔: {test_info.get('gpu_monitor_interval', 0)}s")
    
    # 性能指标
    test_results = results.get('results', [])
    if test_results:
        ttft_values = [r['ttft'] for r in test_results]
        tpot_values = [r['tpot'] for r in test_results if r['tpot'] > 0]
        token_counts = [r['token_count'] for r in test_results]
        
        print(f"\n性能指标:")
        print(f"  平均 TTFT: {np.mean(ttft_values):.4f}s")
        print(f"  平均 TPOT: {np.mean(tpot_values):.4f}s" if tpot_values else "  平均 TPOT: N/A")
        print(f"  总生成tokens: {sum(token_counts)}")
        print(f"  平均每请求tokens: {np.mean(token_counts):.1f}")
    
    # GPU统计
    gpu_stats = results.get('gpu_statistics', {})
    if gpu_stats:
        print(f"\nGPU 统计:")
        print(f"  平均功耗: {gpu_stats.get('power_draw', {}).get('avg', 0):.1f}W")
        print(f"  最大功耗: {gpu_stats.get('power_draw', {}).get('max', 0):.1f}W")
        print(f"  平均GPU利用率: {gpu_stats.get('gpu_utilization', {}).get('avg', 0):.1f}%")
        print(f"  总能耗: {gpu_stats.get('power_draw', {}).get('total_energy', 0):.4f}kWh")
    
    # 阶段分析
    analyzed_intervals = results.get('analyzed_intervals', [])
    if analyzed_intervals:
        prefill_intervals = [i for i in analyzed_intervals if i['phase'] == 'prefill']
        decode_intervals = [i for i in analyzed_intervals if i['phase'] == 'decode']
        idle_intervals = [i for i in analyzed_intervals if i['phase'] == 'idle']
        
        print(f"\n阶段分析:")
        print(f"  Prefill 间隔数: {len(prefill_intervals)}")
        print(f"  Decode 间隔数: {len(decode_intervals)}")
        print(f"  Idle 间隔数: {len(idle_intervals)}")
        
        if prefill_intervals:
            prefill_avg_power = sum(i['avg_power'] for i in prefill_intervals) / len(prefill_intervals)
            print(f"  Prefill 平均功耗: {prefill_avg_power:.1f}W")
        
        if decode_intervals:
            decode_avg_power = sum(i['avg_power'] for i in decode_intervals) / len(decode_intervals)
            print(f"  Decode 平均功耗: {decode_avg_power:.1f}W")
        
        if idle_intervals:
            idle_avg_power = sum(i['avg_power'] for i in idle_intervals) / len(idle_intervals)
            print(f"  Idle 平均功耗: {idle_avg_power:.1f}W")

def main():
    """主分析函数"""
    storage_dir = Path("vllm_unified_results")
    
    # 加载结果
    results = load_results(storage_dir)
    if not results:
        return
    
    # 打印总结
    print_summary(results)
    
    # 创建图表
    analyzed_intervals = results.get('analyzed_intervals', [])
    if analyzed_intervals:
        print(f"\n正在生成图表...")
        create_phase_timeline_plot(analyzed_intervals, storage_dir)
        create_phase_comparison_plot(analyzed_intervals, storage_dir)
        print(f"图表已保存到 {storage_dir} 目录")
    else:
        print("没有分析间隔数据可以绘制")

if __name__ == "__main__":
    main()
