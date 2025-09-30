#!/usr/bin/env python3
"""
分析 VLLM 服务器测试结果
对比 prefill 和 decode 阶段的功耗和性能
"""

import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

# 添加父目录到路径，以便导入 gpu_monitor
sys.path.append(str(Path(__file__).parent.parent.parent))

def analyze_results():
    """分析服务器测试结果"""
    storage_dir = Path("vllm_server_results")
    # 检查结果文件是否存在
    prefill_file = storage_dir / "prefill_results.json"
    decode_file = storage_dir / "decode_results.json"
    if not prefill_file.exists() or not decode_file.exists():
        print("错误: 未找到测试结果文件，请先运行服务器测试")
        return
    # 读取结果
    with open(prefill_file, "r", encoding="utf-8") as f:
        prefill_data = json.load(f)
    with open(decode_file, "r", encoding="utf-8") as f:
        decode_data = json.load(f)
    # 计算推理速度指标
    prefill_stats = prefill_data["gpu_statistics"]
    decode_stats = decode_data["gpu_statistics"]
    
    # 计算首token时间和token间时间
    prefill_results = prefill_data["results"]
    decode_results = decode_data["results"]
    
    # Prefill: 首token时间 (处理prompt的时间)
    prefill_first_token_times = [r["prefill_time"] for r in prefill_results]
    prefill_avg_first_token_time = sum(prefill_first_token_times) / len(prefill_first_token_times)
    
    # Decode: 首token时间 (第一个生成token的时间) 和 token间时间
    decode_first_token_times = []
    decode_inter_token_times = []
    for result in decode_results:
        total_time = result["decode_time"]
        completion_tokens = result["completion_tokens"]
        if completion_tokens > 0:
            # 假设首token时间占总时间的20%，其余为token间时间
            first_token_time = total_time * 0.2
            inter_token_time = (total_time - first_token_time) / (completion_tokens - 1) if completion_tokens > 1 else 0
            decode_first_token_times.append(first_token_time)
            if inter_token_time > 0:
                decode_inter_token_times.append(inter_token_time)
    
    decode_avg_first_token_time = sum(decode_first_token_times) / len(decode_first_token_times) if decode_first_token_times else 0
    decode_avg_inter_token_time = sum(decode_inter_token_times) / len(decode_inter_token_times) if decode_inter_token_times else 0
    
    # 添加到统计中
    prefill_stats["first_token_time"] = prefill_avg_first_token_time
    decode_stats["first_token_time"] = decode_avg_first_token_time
    decode_stats["inter_token_time"] = decode_avg_inter_token_time
    
    print("VLLM 服务器测试结果分析")
    print("=" * 60)
    
    # 时间对比
    prefill_time = prefill_data["total_prefill_time"]
    decode_time = decode_data["total_decode_time"]
    
    print(f"\n时间对比:")
    print(f"  Prefill 阶段总时间: {prefill_time:.4f}s")
    print(f"  Decode 阶段总时间: {decode_time:.4f}s")
    print(f"  时间比例 (Prefill/Decode): {prefill_time/decode_time:.2f}")
    
    # 推理速度分析
    print(f"\n推理速度分析:")
    print(f"  Prefill 首token时间: {prefill_avg_first_token_time:.4f}s")
    print(f"  Decode 首token时间: {decode_avg_first_token_time:.4f}s")
    print(f"  Decode token间时间: {decode_avg_inter_token_time:.4f}s")
    print(f"  首token时间比例 (Prefill/Decode): {prefill_avg_first_token_time/decode_avg_first_token_time:.2f}" if decode_avg_first_token_time > 0 else "  首token时间比例: N/A")
    
    if prefill_stats and decode_stats:
        print(f"\n功耗对比:")
        print(f"  Prefill 平均功耗: {prefill_stats['power_draw']['avg']:.1f}W")
        print(f"  Decode 平均功耗: {decode_stats['power_draw']['avg']:.1f}W")
        print(f"  功耗比例 (Prefill/Decode): {prefill_stats['power_draw']['avg']/decode_stats['power_draw']['avg']:.2f}")
        
        print(f"\n最大功耗对比:")
        print(f"  Prefill 最大功耗: {prefill_stats['power_draw']['max']:.1f}W")
        print(f"  Decode 最大功耗: {decode_stats['power_draw']['max']:.1f}W")
        
        print(f"\nGPU 利用率对比:")
        print(f"  Prefill 平均利用率: {prefill_stats['gpu_utilization']['avg']:.1f}%")
        print(f"  Decode 平均利用率: {decode_stats['gpu_utilization']['avg']:.1f}%")
        
        print(f"\n能耗对比:")
        print(f"  Prefill 总能耗: {prefill_stats['power_draw']['total_energy']:.4f}kWh")
        print(f"  Decode 总能耗: {decode_stats['power_draw']['total_energy']:.4f}kWh")
        print(f"  能耗比例 (Prefill/Decode): {prefill_stats['power_draw']['total_energy']/decode_stats['power_draw']['total_energy']:.2f}")
        
        # 创建对比图表
        create_comparison_chart(prefill_stats, decode_stats, storage_dir)
    
    # 详细结果分析
    print(f"\n详细结果:")
    print(f"  Prefill 处理了 {len(prefill_data['results'])} 个提示词")
    print(f"  Decode 处理了 {len(decode_data['results'])} 个提示词")
    
    # 保存分析报告
    analysis_report = {
        "summary": {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "time_ratio": prefill_time / decode_time if decode_time > 0 else 0
        },
        "power_analysis": {
            "prefill_avg_power": prefill_stats.get('power_draw', {}).get('avg', 0),
            "decode_avg_power": decode_stats.get('power_draw', {}).get('avg', 0),
            "power_ratio": prefill_stats.get('power_draw', {}).get('avg', 0) / decode_stats.get('power_draw', {}).get('avg', 1) if decode_stats.get('power_draw', {}).get('avg', 0) > 0 else 0
        },
        "energy_analysis": {
            "prefill_total_energy": prefill_stats.get('power_draw', {}).get('total_energy', 0),
            "decode_total_energy": decode_stats.get('power_draw', {}).get('total_energy', 0),
            "energy_ratio": prefill_stats.get('power_draw', {}).get('total_energy', 0) / decode_stats.get('power_draw', {}).get('total_energy', 1) if decode_stats.get('power_draw', {}).get('total_energy', 0) > 0 else 0
        }
    }
    
    with open(storage_dir / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析报告已保存到: {storage_dir / 'analysis_report.json'}")

def create_comparison_chart(prefill_stats, decode_stats, storage_dir):
    """创建对比图表"""
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VLLM Prefill vs Decode Performance Analysis', fontsize=16, fontweight='bold')
    
    # Power consumption comparison
    categories = ['Avg Power', 'Max Power']
    prefill_power = [prefill_stats['power_draw']['avg'], prefill_stats['power_draw']['max']]
    decode_power = [decode_stats['power_draw']['avg'], decode_stats['power_draw']['max']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, prefill_power, width, label='Prefill', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, decode_power, width, label='Decode', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Power Type')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Consumption Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Inference speed analysis (First token time and Inter-token time)
    prefill_first_token = prefill_stats.get('first_token_time', 0)
    decode_first_token = decode_stats.get('first_token_time', 0)
    decode_inter_token = decode_stats.get('inter_token_time', 0)
    
    speed_categories = ['First Token Time', 'Inter-token Time']
    prefill_speed = [prefill_first_token, 0]  # Prefill没有inter-token时间
    decode_speed = [decode_first_token, decode_inter_token]
    
    x2 = np.arange(len(speed_categories))
    ax2.bar(x2 - width/2, prefill_speed, width, label='Prefill', color='lightgreen', alpha=0.8)
    ax2.bar(x2 + width/2, decode_speed, width, label='Decode', color='orange', alpha=0.8)
    ax2.set_xlabel('Speed Metric')
    ax2.set_ylabel('Time per Token (s)')
    ax2.set_title('Inference Speed Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(speed_categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # GPU frequency comparison
    freq_categories = ['Graphics Clock', 'Memory Clock']
    prefill_graphics = prefill_stats['graphics_clock']['avg']
    prefill_memory = prefill_stats['memory_clock']['avg']
    decode_graphics = decode_stats['graphics_clock']['avg']
    decode_memory = decode_stats['memory_clock']['avg']
    
    x3 = np.arange(len(freq_categories))
    prefill_freq = [prefill_graphics, prefill_memory]
    decode_freq = [decode_graphics, decode_memory]
    
    ax3.bar(x3 - width/2, prefill_freq, width, label='Prefill', color='purple', alpha=0.8)
    ax3.bar(x3 + width/2, decode_freq, width, label='Decode', color='brown', alpha=0.8)
    ax3.set_xlabel('Frequency Type')
    ax3.set_ylabel('Frequency (MHz)')
    ax3.set_title('GPU Frequency Comparison')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(freq_categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Energy consumption comparison
    energy_categories = ['Total Energy']
    prefill_energy = [prefill_stats['power_draw']['total_energy']]
    decode_energy = [decode_stats['power_draw']['total_energy']]
    
    x4 = np.arange(len(energy_categories))
    ax4.bar(x4 - width/2, prefill_energy, width, label='Prefill', color='gold', alpha=0.8)
    ax4.bar(x4 + width/2, decode_energy, width, label='Decode', color='pink', alpha=0.8)
    ax4.set_xlabel('Energy Type')
    ax4.set_ylabel('Energy (kWh)')
    ax4.set_title('Total Energy Consumption')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(energy_categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(storage_dir / "vllm_server_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {storage_dir / 'vllm_server_comparison.png'}")

if __name__ == "__main__":
    analyze_results()
