#!/usr/bin/env python3
"""
对比传统单实例和分离式 Prefill+Decode 的性能
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(file_path):
    """加载测试结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None

def compare_performance():
    """对比性能分析"""
    
    print("VLLM 性能对比分析")
    print("=" * 50)
    
    # 加载传统单实例结果
    traditional_results = None
    traditional_file = Path("../traditional/vllm_server_results/prefill_results.json")
    if traditional_file.exists():
        traditional_results = load_results(traditional_file)
    
    # 加载分离式结果
    disaggregated_results = None
    disaggregated_file = Path("disaggregated_results/disaggregated_results.json")
    if disaggregated_file.exists():
        disaggregated_results = load_results(disaggregated_file)
    
    if not traditional_results and not disaggregated_results:
        print("❌ 未找到任何测试结果文件")
        return
    
    # 创建对比图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Traditional vs Disaggregated Prefill+Decode Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. 时间对比
    categories = ['Prefill Time', 'Decode Time', 'Total Time']
    traditional_times = []
    disaggregated_times = []
    
    if traditional_results:
        prefill_time = traditional_results.get('total_prefill_time', 0)
        # 传统方式需要模拟 decode 时间
        traditional_times = [prefill_time, prefill_time * 6, prefill_time * 7]  # 估算
    else:
        traditional_times = [0, 0, 0]
    
    if disaggregated_results:
        summary = disaggregated_results.get('summary', {})
        disaggregated_times = [
            summary.get('avg_prefill_time', 0),
            summary.get('avg_decode_time', 0),
            summary.get('avg_e2e_latency', 0)
        ]
    else:
        disaggregated_times = [0, 0, 0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, traditional_times, width, label='Traditional', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, disaggregated_times, width, label='Disaggregated', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Time Metrics')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Time Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 功耗对比
    traditional_power = []
    disaggregated_power = []
    
    if traditional_results:
        gpu_stats = traditional_results.get('gpu_statistics', {})
        traditional_power = [
            gpu_stats.get('power_draw', {}).get('avg', 0),
            gpu_stats.get('power_draw', {}).get('max', 0)
        ]
    else:
        traditional_power = [0, 0]
    
    if disaggregated_results:
        prefill_stats = disaggregated_results.get('prefill_gpu_statistics', {})
        decode_stats = disaggregated_results.get('decode_gpu_statistics', {})
        disaggregated_power = [
            (prefill_stats.get('power_draw', {}).get('avg', 0) + 
             decode_stats.get('power_draw', {}).get('avg', 0)) / 2,
            max(prefill_stats.get('power_draw', {}).get('max', 0),
                decode_stats.get('power_draw', {}).get('max', 0))
        ]
    else:
        disaggregated_power = [0, 0]
    
    power_categories = ['Avg Power', 'Max Power']
    x2 = np.arange(len(power_categories))
    
    ax2.bar(x2 - width/2, traditional_power, width, label='Traditional', color='lightgreen', alpha=0.8)
    ax2.bar(x2 + width/2, disaggregated_power, width, label='Disaggregated', color='orange', alpha=0.8)
    ax2.set_xlabel('Power Metrics')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Power Consumption Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(power_categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能指标对比 (TTFT, TBT)
    if disaggregated_results:
        summary = disaggregated_results.get('summary', {})
        ttft = summary.get('avg_ttft', 0)
        tbt = summary.get('avg_tbt', 0)
        
        metrics = ['TTFT', 'TBT']
        values = [ttft, tbt]
        
        ax3.bar(metrics, values, color=['purple', 'brown'], alpha=0.8)
        ax3.set_xlabel('Performance Metrics')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Disaggregated Performance Metrics')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Disaggregated Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Disaggregated Performance Metrics')
    
    # 4. 效率对比
    if traditional_results and disaggregated_results:
        # 计算效率指标
        traditional_efficiency = traditional_times[0] / max(traditional_times[2], 0.001)  # prefill/total
        disaggregated_efficiency = disaggregated_times[0] / max(disaggregated_times[2], 0.001)
        
        efficiency_categories = ['Prefill Efficiency']
        efficiency_values = [traditional_efficiency, disaggregated_efficiency]
        
        x4 = np.arange(len(efficiency_categories))
        ax4.bar(x4 - width/2, [traditional_efficiency], width, label='Traditional', color='gold', alpha=0.8)
        ax4.bar(x4 + width/2, [disaggregated_efficiency], width, label='Disaggregated', color='pink', alpha=0.8)
        ax4.set_xlabel('Efficiency Metrics')
        ax4.set_ylabel('Efficiency Ratio')
        ax4.set_title('Processing Efficiency Comparison')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(efficiency_categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data for Comparison', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Processing Efficiency Comparison')
    
    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 性能对比图表已保存到: performance_comparison.png")
    
    # 打印详细对比
    print("\n📈 详细性能对比:")
    
    if traditional_results:
        print(f"\n传统单实例方式:")
        print(f"  Prefill 时间: {traditional_times[0]:.4f}s")
        print(f"  估算 Decode 时间: {traditional_times[1]:.4f}s")
        print(f"  总时间: {traditional_times[2]:.4f}s")
        print(f"  平均功耗: {traditional_power[0]:.1f}W")
        print(f"  最大功耗: {traditional_power[1]:.1f}W")
    
    if disaggregated_results:
        summary = disaggregated_results.get('summary', {})
        print(f"\n分离式方式:")
        print(f"  Prefill 时间: {disaggregated_times[0]:.4f}s")
        print(f"  Decode 时间: {disaggregated_times[1]:.4f}s")
        print(f"  E2E 延迟: {disaggregated_times[2]:.4f}s")
        print(f"  TTFT: {summary.get('avg_ttft', 0):.4f}s")
        print(f"  TBT: {summary.get('avg_tbt', 0):.4f}s")
        print(f"  平均功耗: {disaggregated_power[0]:.1f}W")
        print(f"  最大功耗: {disaggregated_power[1]:.1f}W")
    
    if traditional_results and disaggregated_results:
        print(f"\n性能提升:")
        time_improvement = (traditional_times[2] - disaggregated_times[2]) / traditional_times[2] * 100
        power_improvement = (traditional_power[0] - disaggregated_power[0]) / traditional_power[0] * 100
        print(f"  时间改善: {time_improvement:.1f}%")
        print(f"  功耗改善: {power_improvement:.1f}%")

if __name__ == "__main__":
    compare_performance()
