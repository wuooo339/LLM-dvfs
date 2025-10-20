#!/usr/bin/env python3
"""
批量测试结果综合分析工具
聚合多个批次的测试结果（包含GPU功耗数据），生成对比分析图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import re

# 关闭matplotlib字体警告
warnings.filterwarnings('ignore', category=UserWarning)

# 使用默认英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MultiBatchAnalyzer:
    """多批次测试结果综合分析器（包含GPU功耗）"""
    
    def __init__(self, result_dir):
        """
        初始化分析器
        Args:
            result_dir: 包含多个批次结果文件的目录
        """
        self.result_dir = Path(result_dir)
        self.batch_data = {}  # {batch_size: data}
        self.gpu_data = {}    # {batch_size: gpu_stats}
        
    def load_all_batches(self):
        """加载所有批次的测试结果"""
        print("正在扫描测试结果文件...")
        
        # 查找所有 batch_results_*.json 文件
        json_files = list(self.result_dir.glob("batch_results_*.json"))
        
        if not json_files:
            print(f"错误: 在 {self.result_dir} 中未找到 batch_results_*.json 文件")
            return False
        
        print(f"找到 {len(json_files)} 个批次结果文件")
        
        for json_file in json_files:
            # 从文件名提取批次大小
            match = re.search(r'batch_results_(\d+)\.json', json_file.name)
            if not match:
                print(f"警告: 无法解析批次大小: {json_file.name}")
                continue
            
            batch_size = int(match.group(1))
            
            # 加载并处理该批次的数据
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._process_batch(batch_size, data)
                    print(f"  ✓ 加载批次 {batch_size}")
            except Exception as e:
                print(f"警告: 加载批次 {batch_size} 失败: {e}")
        
        # 加载GPU功耗数据
        print("\n正在加载GPU功耗数据...")
        self._load_gpu_data()
        
        if not self.batch_data:
            print("错误: 没有成功加载任何批次数据")
            return False
        
        print(f"\n成功加载 {len(self.batch_data)} 个批次的数据")
        print(f"批次大小: {sorted(self.batch_data.keys())}")
        print(f"GPU数据: {len(self.gpu_data)} 个批次")
        return True
    
    def _load_gpu_data(self):
        """加载GPU功耗数据"""
        gpu_stats_files = list(self.result_dir.glob("gpu_stats_*.json"))
        
        for gpu_file in gpu_stats_files:
            match = re.search(r'gpu_stats_(\d+)\.json', gpu_file.name)
            if not match:
                continue
            
            batch_size = int(match.group(1))
            
            try:
                with open(gpu_file, 'r', encoding='utf-8') as f:
                    gpu_stats = json.load(f)
                    self.gpu_data[batch_size] = gpu_stats
                    print(f"  ✓ 加载GPU数据 批次 {batch_size}")
            except Exception as e:
                print(f"警告: 加载GPU数据批次 {batch_size} 失败: {e}")
    
    def _process_batch(self, batch_size, data):
        """处理单个批次的数据"""
        results = data.get('results', [])
        
        if not results:
            print(f"警告: 批次 {batch_size} 没有测试结果")
            return
        
        # 筛选成功的请求
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            print(f"警告: 批次 {batch_size} 没有成功的请求")
            return
        
        # 计算各项指标
        # TTFT统计
        ttft_values = [r['ttft'] for r in successful if 'ttft' in r]
        
        # Token统计
        token_counts = [r['token_count'] for r in successful if 'token_count' in r]
        
        # TBT统计 (Token间延迟)
        tbt_values = []
        for r in successful:
            if 'token_times' in r and len(r['token_times']) > 1:
                times = r['token_times']
                tbts = [(times[i] - times[i-1]) * 1000 for i in range(1, len(times))]
                tbt_values.extend(tbts)
        
        # 响应时间统计
        response_times = [r['response_time'] for r in successful]
        
        # 保存该批次的统计数据
        self.batch_data[batch_size] = {
            'total_requests': data.get('total_requests', 0),
            'successful_requests': len(successful),
            'success_rate': len(successful) / data.get('total_requests', 1) * 100,
            
            # TTFT
            'ttft_mean': np.mean(ttft_values) if ttft_values else 0,
            'ttft_std': np.std(ttft_values) if ttft_values else 0,
            'ttft_min': np.min(ttft_values) if ttft_values else 0,
            'ttft_max': np.max(ttft_values) if ttft_values else 0,
            
            # TBT
            'tbt_mean': np.mean(tbt_values) if tbt_values else 0,
            'tbt_std': np.std(tbt_values) if tbt_values else 0,
            'tbt_min': np.min(tbt_values) if tbt_values else 0,
            'tbt_max': np.max(tbt_values) if tbt_values else 0,
            
            # Tokens
            'total_tokens': sum(token_counts),
            'avg_tokens': np.mean(token_counts) if token_counts else 0,
            
            # Response time
            'response_mean': np.mean(response_times) if response_times else 0,
            'response_std': np.std(response_times) if response_times else 0,
            
            # Throughput
            'throughput': data.get('throughput', 0),
        }
    
    def print_summary(self):
        """打印综合分析摘要"""
        if not self.batch_data:
            print("没有数据可以显示")
            return
        
        print("\n" + "=" * 140)
        print("多批次测试结果综合分析（包含GPU功耗）")
        print("=" * 140)
        
        batch_sizes = sorted(self.batch_data.keys())
        
        print(f"\n批次大小: {batch_sizes}")
        print(f"批次数量: {len(batch_sizes)}")
        
        print("\n" + "-" * 140)
        print(f"{'Batch':<8} {'Success%':<10} {'TTFT(s)':<15} {'TBT(ms)':<15} "
              f"{'Tokens':<12} {'Response(s)':<15} {'Throughput':<15} {'GPU Power(W)':<15}")
        print("-" * 140)
        
        for batch_size in batch_sizes:
            data = self.batch_data[batch_size]
            gpu_power = "N/A"
            if batch_size in self.gpu_data:
                gpu_stats = self.gpu_data[batch_size].get('statistics', {})
                total_stats = gpu_stats.get('total', {})
                avg_power = total_stats.get('avg_power', 0)
                if avg_power > 0:
                    gpu_power = f"{avg_power:.1f}"
            
            print(f"{batch_size:<8} "
                  f"{data['success_rate']:<10.1f} "
                  f"{data['ttft_mean']:<15.4f} "
                  f"{data['tbt_mean']:<15.4f} "
                  f"{data['total_tokens']:<12} "
                  f"{data['response_mean']:<15.3f} "
                  f"{data['throughput']:<15.2f} "
                  f"{gpu_power:<15}")
        
        print("-" * 140)
    
    def plot_comprehensive_analysis(self, output_dir=None):
        """绘制综合对比分析图（包含GPU功耗）"""
        if not self.batch_data:
            print("没有数据可以绘制")
            return
        
        output_dir = Path(output_dir or self.result_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 准备数据
        batch_sizes = sorted(self.batch_data.keys())
        
        ttft_means = [self.batch_data[bs]['ttft_mean'] for bs in batch_sizes]
        ttft_stds = [self.batch_data[bs]['ttft_std'] for bs in batch_sizes]
        
        tbt_means = [self.batch_data[bs]['tbt_mean'] for bs in batch_sizes]
        tbt_stds = [self.batch_data[bs]['tbt_std'] for bs in batch_sizes]
        
        throughput_means = [self.batch_data[bs]['throughput'] for bs in batch_sizes]
        
        response_means = [self.batch_data[bs]['response_mean'] for bs in batch_sizes]
        response_stds = [self.batch_data[bs]['response_std'] for bs in batch_sizes]
        
        # GPU功耗数据
        gpu_avg_power = []
        gpu_max_power = []
        gpu_total_energy = []
        
        for bs in batch_sizes:
            if bs in self.gpu_data:
                gpu_stats = self.gpu_data[bs].get('statistics', {})
                total_stats = gpu_stats.get('total', {})
                gpu_avg_power.append(total_stats.get('avg_power', 0))
                gpu_max_power.append(total_stats.get('max_power', 0))
                gpu_total_energy.append(total_stats.get('total_energy_wh', 0))
            else:
                gpu_avg_power.append(0)
                gpu_max_power.append(0)
                gpu_total_energy.append(0)
        
        # 创建图表 - 2x3布局
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. TTFT vs Batch Size
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(batch_sizes, ttft_means, yerr=ttft_stds, 
                     marker='o', linewidth=2.5, markersize=10, 
                     capsize=5, capthick=2, color='#2E86AB', ecolor='#2E86AB')
        ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('TTFT (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Time To First Token vs Batch Size', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(batch_sizes)
        
        # 2. TBT vs Batch Size
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.errorbar(batch_sizes, tbt_means, yerr=tbt_stds, 
                     marker='s', linewidth=2.5, markersize=10, 
                     capsize=5, capthick=2, color='#A23B72', ecolor='#A23B72')
        ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('TBT (milliseconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Time Between Tokens vs Batch Size', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(batch_sizes)
        
        # 3. Throughput vs Batch Size
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(batch_sizes, throughput_means, 
                 marker='^', linewidth=2.5, markersize=10, color='#F18F01')
        for i, (bs, tp) in enumerate(zip(batch_sizes, throughput_means)):
            ax3.text(bs, tp, f'{tp:.2f}', ha='center', va='bottom', fontsize=9)
        ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Throughput (req/sec)', fontsize=12, fontweight='bold')
        ax3.set_title('Throughput vs Batch Size', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xticks(batch_sizes)
        
        # 4. Average Response Time vs Batch Size
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.errorbar(batch_sizes, response_means, yerr=response_stds,
                     marker='d', linewidth=2.5, markersize=10, 
                     capsize=5, capthick=2, color='#C73E1D', ecolor='#C73E1D')
        ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Average Response Time vs Batch Size', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xticks(batch_sizes)
        
        # 5. GPU Average Power vs Batch Size
        ax5 = fig.add_subplot(gs[1, 1])
        if any(gpu_avg_power):
            ax5.plot(batch_sizes, gpu_avg_power, 
                     marker='o', linewidth=2.5, markersize=10, color='#06A77D')
            ax5.fill_between(batch_sizes, 
                            [avg - (max_p - avg) * 0.1 for avg, max_p in zip(gpu_avg_power, gpu_max_power)],
                            gpu_max_power, alpha=0.2, color='#06A77D')
            for i, (bs, power) in enumerate(zip(batch_sizes, gpu_avg_power)):
                if power > 0:
                    ax5.text(bs, power, f'{power:.0f}W', ha='center', va='bottom', fontsize=9)
            ax5.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
            ax5.set_ylabel('GPU Power (Watts)', fontsize=12, fontweight='bold')
            ax5.set_title('Average GPU Power vs Batch Size', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3, linestyle='--')
            ax5.set_xticks(batch_sizes)
        else:
            ax5.text(0.5, 0.5, 'No GPU Data Available', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=14)
            ax5.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
            ax5.set_ylabel('GPU Power (Watts)', fontsize=12, fontweight='bold')
            ax5.set_title('Average GPU Power vs Batch Size', fontsize=13, fontweight='bold')
        
        # 6. GPU Total Energy vs Batch Size
        ax6 = fig.add_subplot(gs[1, 2])
        if any(gpu_total_energy):
            bars = ax6.bar(batch_sizes, gpu_total_energy, color='#6A4C93', alpha=0.7, 
                          edgecolor='black', linewidth=1.5)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            ax6.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Total Energy (Wh)', fontsize=12, fontweight='bold')
            ax6.set_title('GPU Total Energy Consumption vs Batch Size', fontsize=13, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax6.set_xticks(batch_sizes)
        else:
            ax6.text(0.5, 0.5, 'No GPU Data Available', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=14)
            ax6.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Total Energy (Wh)', fontsize=12, fontweight='bold')
            ax6.set_title('GPU Total Energy Consumption vs Batch Size', fontsize=13, fontweight='bold')
        
        # 总标题
        fig.suptitle('Multi-Batch Performance and Power Comparison Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 保存图表
        output_file = output_dir / "batch_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n综合分析图表已保存到: {output_file}")
    
    def export_csv_report(self, output_dir=None):
        """导出CSV格式的详细报告（包含GPU功耗）"""
        if not self.batch_data:
            print("没有数据可以导出")
            return
        
        output_dir = Path(output_dir or self.result_dir)
        output_dir.mkdir(exist_ok=True)
        
        import csv
        
        batch_sizes = sorted(self.batch_data.keys())
        
        csv_file = output_dir / "batch_analysis_summary.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'batch_size', 'total_requests', 'successful_requests', 'success_rate',
                'ttft_mean', 'ttft_std', 'ttft_min', 'ttft_max',
                'tbt_mean', 'tbt_std', 'tbt_min', 'tbt_max',
                'total_tokens', 'avg_tokens', 
                'response_mean', 'response_std',
                'throughput',
                'gpu_avg_power', 'gpu_max_power', 'gpu_total_energy_wh'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for batch_size in batch_sizes:
                data = self.batch_data[batch_size].copy()
                
                # 添加GPU数据
                if batch_size in self.gpu_data:
                    gpu_stats = self.gpu_data[batch_size].get('statistics', {})
                    total_stats = gpu_stats.get('total', {})
                    data['gpu_avg_power'] = total_stats.get('avg_power', 0)
                    data['gpu_max_power'] = total_stats.get('max_power', 0)
                    data['gpu_total_energy_wh'] = total_stats.get('total_energy_wh', 0)
                else:
                    data['gpu_avg_power'] = 0
                    data['gpu_max_power'] = 0
                    data['gpu_total_energy_wh'] = 0
                
                row = {'batch_size': batch_size, **data}
                writer.writerow(row)
        
        print(f"CSV报告已保存到: {csv_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多批次测试结果综合分析工具（包含GPU功耗）')
    parser.add_argument('--dir', required=True, help='包含批次结果文件的目录')
    
    args = parser.parse_args()
    
    print("=" * 140)
    print("多批次测试结果综合分析工具（包含GPU功耗）")
    print("=" * 140)
    
    # 创建分析器
    analyzer = MultiBatchAnalyzer(result_dir=args.dir)
    
    # 加载所有批次数据
    if not analyzer.load_all_batches():
        print("\n错误: 无法加载测试数据")
        return
    
    # 打印综合摘要
    analyzer.print_summary()
    
    # 生成综合对比图
    print("\n正在生成综合对比分析图...")
    analyzer.plot_comprehensive_analysis(args.dir)
    
    # 导出CSV报告
    analyzer.export_csv_report(args.dir)
    
    print("\n" + "=" * 140)
    print("✅ 综合分析完成!")
    print("=" * 140)

if __name__ == "__main__":
    main()
