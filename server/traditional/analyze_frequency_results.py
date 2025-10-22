#!/usr/bin/env python3
"""
多频率测试结果综合分析工具
分析不同GPU核心频率下的性能和功耗数据，生成对比分析图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import re
import csv

# 关闭matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning)

# 使用默认英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FrequencyAnalyzer:
    """多频率测试结果分析器"""
    
    def __init__(self, master_result_dir):
        """
        初始化分析器
        Args:
            master_result_dir: 包含多个频率子目录的主结果目录
        """
        self.master_dir = Path(master_result_dir)
        self.frequency_data = {}  # {freq: {batch_size: data}}
        self.frequencies = []
        
    def load_all_frequencies(self):
        """加载所有频率的测试数据"""
        print("=" * 100)
        print("扫描频率测试结果目录...")
        print("=" * 100)
        
        # 查找所有频率子目录
        freq_dirs = list(self.master_dir.glob("freq_*MHz"))
        
        if not freq_dirs:
            print(f"错误: 在 {self.master_dir} 中未找到频率子目录 (freq_*MHz)")
            return False
        
        print(f"找到 {len(freq_dirs)} 个频率目录:")
        for freq_dir in sorted(freq_dirs):
            print(f"  - {freq_dir.name}")
        
        # 遍历每个频率目录
        for freq_dir in sorted(freq_dirs):
            # 从目录名提取频率值
            match = re.search(r'freq_(\d+)MHz', freq_dir.name)
            if not match:
                print(f"警告: 无法解析频率值: {freq_dir.name}")
                continue
            
            freq = int(match.group(1))
            self.frequencies.append(freq)
            
            print(f"\n正在加载频率 {freq} MHz 的数据...")
            
            # 加载该频率下的所有批次数据
            batch_data = self._load_frequency_batches(freq_dir)
            
            if batch_data:
                self.frequency_data[freq] = batch_data
                print(f"  ✓ 成功加载 {len(batch_data)} 个批次")
            else:
                print(f"  ✗ 未找到有效数据")
        
        self.frequencies = sorted(self.frequencies)
        
        if not self.frequency_data:
            print("\n错误: 没有成功加载任何频率数据")
            return False
        
        print("\n" + "=" * 100)
        print(f"✅ 成功加载 {len(self.frequency_data)} 个频率的数据")
        print(f"频率列表: {self.frequencies} MHz")
        print("=" * 100)
        return True
    
    def _load_frequency_batches(self, freq_dir):
        """加载单个频率目录下的所有批次数据"""
        batch_data = {}
        
        # 查找所有 batch_results_*.json 文件
        json_files = list(freq_dir.glob("batch_results_*.json"))
        
        for json_file in json_files:
            # 提取批次大小
            match = re.search(r'batch_results_(\d+)\.json', json_file.name)
            if not match:
                continue
            
            batch_size = int(match.group(1))
            
            try:
                # 加载测试结果
                with open(json_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                
                # 加载对应的GPU统计数据
                gpu_stats_file = freq_dir / f"gpu_stats_{batch_size}.json"
                gpu_stats = {}
                if gpu_stats_file.exists():
                    with open(gpu_stats_file, 'r', encoding='utf-8') as f:
                        gpu_stats = json.load(f)
                
                # 处理并合并数据
                processed_data = self._process_batch_data(test_data, gpu_stats)
                batch_data[batch_size] = processed_data
                
            except Exception as e:
                print(f"  警告: 加载批次 {batch_size} 失败: {e}")
        
        return batch_data
    
    def _process_batch_data(self, test_data, gpu_stats):
        """处理单个批次的测试数据和GPU数据"""
        results = test_data.get('results', [])
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            return None
        
        # 测试指标
        ttft_values = [r['ttft'] for r in successful if 'ttft' in r]
        token_counts = [r['token_count'] for r in successful if 'token_count' in r]
        response_times = [r['response_time'] for r in successful]
        
        # TBT统计
        tbt_values = []
        for r in successful:
            if 'token_times' in r and len(r['token_times']) > 1:
                times = r['token_times']
                tbts = [(times[i] - times[i-1]) * 1000 for i in range(1, len(times))]
                tbt_values.extend(tbts)
        
        # GPU指标
        gpu_metrics = {
            'avg_power': 0,
            'max_power': 0,
            'total_energy_wh': 0,
            'avg_utilization': 0,
            'avg_temperature': 0,
            'avg_memory_used': 0
        }
        
        if gpu_stats:
            stats = gpu_stats.get('statistics', {})
            total_stats = stats.get('total', {})
            gpu_metrics['avg_power'] = total_stats.get('avg_power', 0)
            gpu_metrics['max_power'] = total_stats.get('max_power', 0)
            gpu_metrics['total_energy_wh'] = total_stats.get('total_energy_wh', 0)
            
            # 计算所有GPU的平均利用率和温度
            util_list = []
            temp_list = []
            mem_list = []
            for gpu_key, gpu_data in stats.items():
                if gpu_key.startswith('gpu_'):
                    util_list.append(gpu_data.get('utilization', {}).get('avg', 0))
                    temp_list.append(gpu_data.get('temperature', {}).get('avg', 0))
                    mem_list.append(gpu_data.get('memory', {}).get('avg_used', 0))
            
            if util_list:
                gpu_metrics['avg_utilization'] = np.mean(util_list)
            if temp_list:
                gpu_metrics['avg_temperature'] = np.mean(temp_list)
            if mem_list:
                gpu_metrics['avg_memory_used'] = np.mean(mem_list)
        
        return {
            'total_requests': test_data.get('total_requests', 0),
            'successful_requests': len(successful),
            'success_rate': len(successful) / test_data.get('total_requests', 1) * 100,
            'ttft_mean': np.mean(ttft_values) if ttft_values else 0,
            'ttft_std': np.std(ttft_values) if ttft_values else 0,
            'tbt_mean': np.mean(tbt_values) if tbt_values else 0,
            'tbt_std': np.std(tbt_values) if tbt_values else 0,
            'total_tokens': sum(token_counts),
            'avg_tokens': np.mean(token_counts) if token_counts else 0,
            'response_mean': np.mean(response_times) if response_times else 0,
            'response_std': np.std(response_times) if response_times else 0,
            'throughput': test_data.get('throughput', 0),
            **gpu_metrics
        }
    
    def plot_single_frequency(self, freq, output_dir):
        """为单个频率生成完整的分析图表"""
        if freq not in self.frequency_data:
            print(f"警告: 频率 {freq} MHz 没有数据")
            return
        
        batch_data = self.frequency_data[freq]
        batch_sizes = sorted(batch_data.keys())
        
        # 准备数据
        ttft_means = [batch_data[bs]['ttft_mean'] for bs in batch_sizes]
        ttft_stds = [batch_data[bs]['ttft_std'] for bs in batch_sizes]
        tbt_means = [batch_data[bs]['tbt_mean'] for bs in batch_sizes]
        throughput = [batch_data[bs]['throughput'] for bs in batch_sizes]
        gpu_avg_power = [batch_data[bs]['avg_power'] for bs in batch_sizes]
        gpu_total_energy = [batch_data[bs]['total_energy_wh'] for bs in batch_sizes]
        
        # 创建2x3布局的图表
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. TTFT vs Batch Size
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(batch_sizes, ttft_means, yerr=ttft_stds,
                     marker='o', linewidth=2.5, markersize=10,
                     capsize=5, capthick=2, color='#2E86AB')
        ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('TTFT (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title(f'TTFT @ {freq} MHz', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(batch_sizes)
        ax1.set_xticklabels(batch_sizes, rotation=45)
        
        # 2. TBT vs Batch Size
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(batch_sizes, tbt_means, marker='s', linewidth=2.5,
                 markersize=10, color='#A23B72')
        ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('TBT (milliseconds)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Time Between Tokens @ {freq} MHz', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(batch_sizes)
        ax2.set_xticklabels(batch_sizes, rotation=45)
        
        # 3. Throughput vs Batch Size
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(batch_sizes, throughput, marker='^', linewidth=2.5,
                 markersize=10, color='#F18F01')
        for bs, tp in zip(batch_sizes, throughput):
            ax3.text(bs, tp, f'{tp:.1f}', ha='center', va='bottom', fontsize=9)
        ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Throughput (req/s)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Throughput @ {freq} MHz', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xticks(batch_sizes)
        ax3.set_xticklabels(batch_sizes, rotation=45)
        
        # 4. GPU Average Power
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(batch_sizes, gpu_avg_power, marker='o', linewidth=2.5,
                 markersize=10, color='#06A77D')
        for bs, power in zip(batch_sizes, gpu_avg_power):
            if power > 0:
                ax4.text(bs, power, f'{power:.0f}W', ha='center', va='bottom', fontsize=9)
        ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax4.set_ylabel('GPU Power (Watts)', fontsize=12, fontweight='bold')
        ax4.set_title(f'Average GPU Power @ {freq} MHz', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xticks(batch_sizes)
        ax4.set_xticklabels(batch_sizes, rotation=45)
        
        # 5. GPU Total Energy
        ax5 = fig.add_subplot(gs[1, 1])
        bars = ax5.bar(batch_sizes, gpu_total_energy, color='#6A4C93',
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        ax5.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Total Energy (Wh)', fontsize=12, fontweight='bold')
        ax5.set_title(f'GPU Energy Consumption @ {freq} MHz', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax5.set_xticks(batch_sizes)
        ax5.set_xticklabels(batch_sizes, rotation=45)
        
        # 6. Energy Efficiency (Requests per Wh)
        ax6 = fig.add_subplot(gs[1, 2])
        efficiency = []
        for bs in batch_sizes:
            energy = batch_data[bs]['total_energy_wh']
            requests = batch_data[bs]['successful_requests']
            eff = requests / energy if energy > 0 else 0
            efficiency.append(eff)
        
        ax6.plot(batch_sizes, efficiency, marker='d', linewidth=2.5,
                 markersize=10, color='#C73E1D')
        for bs, eff in zip(batch_sizes, efficiency):
            if eff > 0:
                ax6.text(bs, eff, f'{eff:.1f}', ha='center', va='bottom', fontsize=9)
        ax6.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Requests per Wh', fontsize=12, fontweight='bold')
        ax6.set_title(f'Energy Efficiency @ {freq} MHz', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.set_xticks(batch_sizes)
        ax6.set_xticklabels(batch_sizes, rotation=45)
        
        # 总标题
        fig.suptitle(f'Performance and Power Analysis @ {freq} MHz Core Frequency',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 保存图表
        output_file = output_dir / f"analysis_freq_{freq}MHz.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 图表已保存: {output_file.name}")
    
    def plot_cross_frequency_comparison(self, output_dir):
        """生成跨频率对比分析图表"""
        print("\n生成跨频率对比分析...")
        
        # 选择一个代表性的批次大小进行对比（选择中间值）
        # 找到所有频率共有的批次大小
        common_batches = set(self.frequency_data[self.frequencies[0]].keys())
        for freq in self.frequencies[1:]:
            common_batches &= set(self.frequency_data[freq].keys())
        
        if not common_batches:
            print("  警告: 没有找到所有频率都包含的批次大小")
            return
        
        common_batches = sorted(common_batches)
        representative_batch = common_batches[len(common_batches)//2]
        
        print(f"  使用批次大小 {representative_batch} 进行跨频率对比")
        
        # 准备数据
        ttft_by_freq = []
        throughput_by_freq = []
        power_by_freq = []
        energy_by_freq = []
        efficiency_by_freq = []
        
        for freq in self.frequencies:
            if representative_batch in self.frequency_data[freq]:
                data = self.frequency_data[freq][representative_batch]
                ttft_by_freq.append(data['ttft_mean'])
                throughput_by_freq.append(data['throughput'])
                power_by_freq.append(data['avg_power'])
                energy_by_freq.append(data['total_energy_wh'])
                
                # 计算能效
                eff = data['successful_requests'] / data['total_energy_wh'] if data['total_energy_wh'] > 0 else 0
                efficiency_by_freq.append(eff)
            else:
                ttft_by_freq.append(0)
                throughput_by_freq.append(0)
                power_by_freq.append(0)
                energy_by_freq.append(0)
                efficiency_by_freq.append(0)
        
        # 创建2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Throughput vs Frequency
        ax1 = axes[0, 0]
        ax1.plot(self.frequencies, throughput_by_freq, marker='o',
                 linewidth=3, markersize=12, color='#F18F01')
        for freq, tp in zip(self.frequencies, throughput_by_freq):
            if tp > 0:
                ax1.text(freq, tp, f'{tp:.1f}', ha='center', va='bottom', fontsize=10)
        ax1.set_xlabel('Core Frequency (MHz)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Throughput (req/s)', fontsize=13, fontweight='bold')
        ax1.set_title(f'Throughput vs Core Frequency\n(Batch Size={representative_batch})',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(self.frequencies)
        
        # 2. Average GPU Power vs Frequency
        ax2 = axes[0, 1]
        ax2.plot(self.frequencies, power_by_freq, marker='s',
                 linewidth=3, markersize=12, color='#06A77D')
        for freq, power in zip(self.frequencies, power_by_freq):
            if power > 0:
                ax2.text(freq, power, f'{power:.0f}W', ha='center', va='bottom', fontsize=10)
        ax2.set_xlabel('Core Frequency (MHz)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Average GPU Power (Watts)', fontsize=13, fontweight='bold')
        ax2.set_title(f'GPU Power vs Core Frequency\n(Batch Size={representative_batch})',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(self.frequencies)
        
        # 3. Energy Efficiency vs Frequency
        ax3 = axes[1, 0]
        ax3.plot(self.frequencies, efficiency_by_freq, marker='^',
                 linewidth=3, markersize=12, color='#C73E1D')
        for freq, eff in zip(self.frequencies, efficiency_by_freq):
            if eff > 0:
                ax3.text(freq, eff, f'{eff:.1f}', ha='center', va='bottom', fontsize=10)
        ax3.set_xlabel('Core Frequency (MHz)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Energy Efficiency (req/Wh)', fontsize=13, fontweight='bold')
        ax3.set_title(f'Energy Efficiency vs Core Frequency\n(Batch Size={representative_batch})',
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xticks(self.frequencies)
        
        # 4. TTFT vs Frequency
        ax4 = axes[1, 1]
        ax4.plot(self.frequencies, ttft_by_freq, marker='d',
                 linewidth=3, markersize=12, color='#2E86AB')
        for freq, ttft in zip(self.frequencies, ttft_by_freq):
            if ttft > 0:
                ax4.text(freq, ttft, f'{ttft:.3f}', ha='center', va='bottom', fontsize=10)
        ax4.set_xlabel('Core Frequency (MHz)', fontsize=13, fontweight='bold')
        ax4.set_ylabel('TTFT (seconds)', fontsize=13, fontweight='bold')
        ax4.set_title(f'Time To First Token vs Core Frequency\n(Batch Size={representative_batch})',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xticks(self.frequencies)
        
        # 总标题
        fig.suptitle('Cross-Frequency Performance and Power Comparison',
                    fontsize=18, fontweight='bold', y=0.995)
        
        # 保存图表
        output_file = output_dir / "cross_frequency_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 跨频率对比图已保存: {output_file.name}")
    
    def export_comprehensive_csv(self, output_dir):
        """导出完整的CSV报告"""
        print("\n导出CSV报告...")
        
        csv_file = output_dir / "frequency_analysis_summary.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'frequency_mhz', 'batch_size', 'successful_requests', 'success_rate',
                'ttft_mean', 'ttft_std', 'tbt_mean', 'tbt_std',
                'total_tokens', 'throughput',
                'avg_power', 'max_power', 'total_energy_wh',
                'energy_efficiency_req_per_wh',
                'avg_utilization', 'avg_temperature', 'avg_memory_used_mb'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for freq in sorted(self.frequencies):
                batch_data = self.frequency_data[freq]
                for batch_size in sorted(batch_data.keys()):
                    data = batch_data[batch_size]
                    
                    # 计算能效
                    energy_eff = (data['successful_requests'] / data['total_energy_wh']
                                 if data['total_energy_wh'] > 0 else 0)
                    
                    row = {
                        'frequency_mhz': freq,
                        'batch_size': batch_size,
                        'successful_requests': data['successful_requests'],
                        'success_rate': data['success_rate'],
                        'ttft_mean': data['ttft_mean'],
                        'ttft_std': data['ttft_std'],
                        'tbt_mean': data['tbt_mean'],
                        'tbt_std': data['tbt_std'],
                        'total_tokens': data['total_tokens'],
                        'throughput': data['throughput'],
                        'avg_power': data['avg_power'],
                        'max_power': data['max_power'],
                        'total_energy_wh': data['total_energy_wh'],
                        'energy_efficiency_req_per_wh': energy_eff,
                        'avg_utilization': data['avg_utilization'],
                        'avg_temperature': data['avg_temperature'],
                        'avg_memory_used_mb': data['avg_memory_used']
                    }
                    
                    writer.writerow(row)
        
        print(f"  ✓ CSV报告已保存: {csv_file.name}")
    
    def print_summary(self):
        """打印综合摘要"""
        print("\n" + "=" * 120)
        print("多频率测试结果综合摘要")
        print("=" * 120)
        
        for freq in sorted(self.frequencies):
            print(f"\n频率: {freq} MHz")
            print("-" * 120)
            
            batch_data = self.frequency_data[freq]
            batch_sizes = sorted(batch_data.keys())
            
            print(f"{'Batch':<8} {'Success%':<10} {'TTFT(s)':<12} {'TBT(ms)':<12} "
                  f"{'Throughput':<15} {'Power(W)':<12} {'Energy(Wh)':<12} {'Efficiency':<15}")
            print("-" * 120)
            
            for bs in batch_sizes:
                data = batch_data[bs]
                eff = (data['successful_requests'] / data['total_energy_wh']
                      if data['total_energy_wh'] > 0 else 0)
                
                print(f"{bs:<8} "
                      f"{data['success_rate']:<10.1f} "
                      f"{data['ttft_mean']:<12.4f} "
                      f"{data['tbt_mean']:<12.4f} "
                      f"{data['throughput']:<15.2f} "
                      f"{data['avg_power']:<12.1f} "
                      f"{data['total_energy_wh']:<12.4f} "
                      f"{eff:<15.2f}")
        
        print("=" * 120)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多频率测试结果综合分析工具')
    parser.add_argument('--dir', required=True,
                       help='包含多个频率子目录的主结果目录 (例如: frequency_test_20251022_140530)')
    
    args = parser.parse_args()
    
    print("=" * 120)
    print("多频率测试结果综合分析工具")
    print("=" * 120)
    
    # 创建分析器
    analyzer = FrequencyAnalyzer(master_result_dir=args.dir)
    
    # 加载所有频率数据
    if not analyzer.load_all_frequencies():
        print("\n❌ 错误: 无法加载频率测试数据")
        return 1
    
    # 打印综合摘要
    analyzer.print_summary()
    
    # 为每个频率生成独立的分析图表
    print("\n" + "=" * 120)
    print("生成各频率的独立分析图表...")
    print("=" * 120)
    
    output_dir = Path(args.dir)
    for freq in analyzer.frequencies:
        print(f"\n生成频率 {freq} MHz 的分析图...")
        analyzer.plot_single_frequency(freq, output_dir)
    
    # 生成跨频率对比分析
    analyzer.plot_cross_frequency_comparison(output_dir)
    
    # 导出CSV报告
    analyzer.export_comprehensive_csv(output_dir)
    
    print("\n" + "=" * 120)
    print("✅ 所有分析完成！")
    print("=" * 120)
    print(f"\n生成的文件列表:")
    print(f"  - frequency_analysis_summary.csv (完整数据表)")
    print(f"  - cross_frequency_comparison.png (跨频率对比图)")
    for freq in analyzer.frequencies:
        print(f"  - analysis_freq_{freq}MHz.png (频率{freq}MHz的详细分析)")
    print("=" * 120)
    
    return 0

if __name__ == "__main__":
    exit(main())
