#!/usr/bin/env python3
"""
CPU功耗数据可视化脚本
读取CPU监控JSON数据并生成可视化图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import os
import sys
import numpy as np
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class CPUPowerPlotter:
    """CPU功耗数据可视化工具"""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = None
        self.timestamps = []
        self.power_data = []
        self.energy_data = []
        
    def load_data(self):
        """加载JSON数据"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.data = data
            raw_data = data.get('raw_data', [])
            print(f"成功加载数据: {len(raw_data)} 行")
            
            # 提取时间戳和功耗数据
            for point in raw_data:
                if 'timestamp' in point:
                    self.timestamps.append(point['timestamp'])
                if 'cpu_power_w' in point:
                    self.power_data.append(point['cpu_power_w'])
                else:
                    self.power_data.append(0.0)
                if 'cpu_energy_j' in point:
                    self.energy_data.append(point['cpu_energy_j'])
                else:
                    self.energy_data.append(0.0)
            
            # 转换时间戳为datetime对象
            self.timestamps = [datetime.fromtimestamp(ts) for ts in self.timestamps]
            
            print(f"时间范围: {self.timestamps[0]} 到 {self.timestamps[-1]}")
            print(f"功耗范围: {min(self.power_data):.1f}W 到 {max(self.power_data):.1f}W")
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            sys.exit(1)
    
    def plot_power_over_time(self, output_file: str = None, show_plot: bool = True):
        """绘制功耗随时间变化图"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.timestamps, self.power_data, 'b-', linewidth=1.5, label='CPU Power')
        plt.fill_between(self.timestamps, self.power_data, alpha=0.3, color='blue')
        
        plt.title('CPU Power Consumption Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (Watts)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 格式化x轴时间显示
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, len(self.timestamps)//10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"功耗图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_energy_accumulation(self, output_file: str = None, show_plot: bool = True):
        """绘制能量累积图"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.timestamps, self.energy_data, 'g-', linewidth=1.5, label='Cumulative Energy')
        
        plt.title('CPU Energy Accumulation Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Energy (Joules)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 格式化x轴时间显示
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, len(self.timestamps)//10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"能量图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_power_distribution(self, output_file: str = None, show_plot: bool = True):
        """绘制功耗分布直方图"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(self.power_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(self.power_data), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.power_data):.1f}W')
        plt.axvline(np.median(self.power_data), color='orange', linestyle='--', 
                   label=f'Median: {np.median(self.power_data):.1f}W')
        
        plt.title('CPU Power Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Power (Watts)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"分布图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_statistics_report(self, output_file: str = None):
        """生成统计报告"""
        if not self.power_data:
            return
        
        # 计算统计信息
        avg_power = np.mean(self.power_data)
        max_power = np.max(self.power_data)
        min_power = np.min(self.power_data)
        std_power = np.std(self.power_data)
        
        # 计算总能量消耗
        if len(self.energy_data) > 1:
            total_energy = self.energy_data[-1] - self.energy_data[0]
        else:
            total_energy = 0
        
        # 计算监控时长
        if len(self.timestamps) > 1:
            duration = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        else:
            duration = 0
        
        # 生成报告
        report = []
        report.append("CPU Power Monitoring Statistics Report")
        report.append("=" * 50)
        report.append(f"Monitoring Duration: {duration:.1f} seconds")
        report.append(f"Sample Count: {len(self.power_data)}")
        report.append(f"Method: {self.data.get('method', 'Unknown')}")
        report.append("")
        report.append("Power Statistics:")
        report.append(f"  Average Power: {avg_power:.2f}W")
        report.append(f"  Maximum Power: {max_power:.2f}W")
        report.append(f"  Minimum Power: {min_power:.2f}W")
        report.append(f"  Standard Deviation: {std_power:.2f}W")
        report.append("")
        report.append("Energy Statistics:")
        report.append(f"  Total Energy Consumed: {total_energy:.2f}J")
        report.append(f"  Average Power (from energy): {total_energy/duration:.2f}W" if duration > 0 else "  Average Power: N/A")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"统计报告已保存到: {output_file}")
    
    def generate_all_plots(self, output_dir: str):
        """生成所有图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成功耗随时间变化图
        self.plot_power_over_time(
            os.path.join(output_dir, "cpu_power_over_time.png"),
            show_plot=False
        )
        
        # 生成能量累积图
        self.plot_energy_accumulation(
            os.path.join(output_dir, "cpu_energy_accumulation.png"),
            show_plot=False
        )
        
        # 生成功耗分布图
        self.plot_power_distribution(
            os.path.join(output_dir, "cpu_power_distribution.png"),
            show_plot=False
        )
        
        # 生成统计报告
        self.generate_statistics_report(
            os.path.join(output_dir, "cpu_statistics_report.txt")
        )
        
        print(f"所有CPU图表已生成到目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="CPU功耗数据可视化工具")
    parser.add_argument("json_file", help="CPU监控JSON文件路径")
    parser.add_argument("--output-dir", default="cpu_plots", 
                       help="输出目录 (默认: cpu_plots)")
    parser.add_argument("--show", action="store_true", 
                       help="显示图表（默认不显示）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"错误: 文件不存在 {args.json_file}")
        sys.exit(1)
    
    plotter = CPUPowerPlotter(args.json_file)
    plotter.load_data()
    
    if args.show:
        # 显示图表
        plotter.plot_power_over_time()
        plotter.plot_energy_accumulation()
        plotter.plot_power_distribution()
    else:
        # 生成所有图表到文件
        plotter.generate_all_plots(args.output_dir)

if __name__ == "__main__":
    main()
