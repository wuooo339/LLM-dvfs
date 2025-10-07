#!/usr/bin/env python3
"""
GPU功耗变化图绘制脚本
读取多GPU监控数据并生成可视化图表
"""

import pandas as pd
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

class GPUPowerPlotter:
    """GPU功耗数据可视化工具"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data = None
        self.gpu_ids = []
        
    def load_data(self):
        """加载CSV数据"""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"成功加载数据: {len(self.data)} 行")
            
            # 提取GPU ID列表
            power_columns = [col for col in self.data.columns if col.endswith('_power')]
            self.gpu_ids = [col.replace('gpu_', '').replace('_power', '') for col in power_columns]
            print(f"检测到GPU: {self.gpu_ids}")
            
            # 转换时间戳
            if 'timestamp' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='s')
            elif 'datetime' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            else:
                # 如果没有时间列，创建索引时间
                self.data['datetime'] = pd.date_range(start='2024-01-01', periods=len(self.data), freq='100ms')
            
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def plot_power_consumption(self, output_file: str = None, show_plot: bool = True):
        """绘制功耗变化图"""
        if self.data is None:
            print("请先加载数据")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Power Monitoring Data', fontsize=16, fontweight='bold')
        
        # 为每个GPU绘制功耗图
        for i, gpu_id in enumerate(self.gpu_ids):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            power_col = f'gpu_{gpu_id}_power'
            util_col = f'gpu_{gpu_id}_utilization'
            temp_col = f'gpu_{gpu_id}_temperature'
            
            if power_col in self.data.columns:
                # 绘制功耗曲线
                ax.plot(self.data['datetime'], self.data[power_col], 
                       label=f'GPU {gpu_id} Power', color=f'C{i}', linewidth=1.5)
                
                # 添加利用率作为背景
                if util_col in self.data.columns:
                    ax2 = ax.twinx()
                    ax2.fill_between(self.data['datetime'], 0, self.data[util_col], 
                                   alpha=0.3, color=f'C{i}', label=f'GPU {gpu_id} Utilization')
                    ax2.set_ylabel('GPU Utilization (%)', fontsize=10)
                    ax2.set_ylim(0, 100)
                
                ax.set_title(f'GPU {gpu_id} Power Consumption', fontsize=12, fontweight='bold')
                ax.set_ylabel('Power (W)', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # 设置x轴格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 隐藏多余的子图
        for i in range(len(self.gpu_ids), 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_combined_power(self, output_file: str = None, show_plot: bool = True):
        """绘制所有GPU的功耗对比图"""
        if self.data is None:
            print("请先加载数据")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 绘制所有GPU的功耗曲线
        for i, gpu_id in enumerate(self.gpu_ids):
            power_col = f'gpu_{gpu_id}_power'
            if power_col in self.data.columns:
                plt.plot(self.data['datetime'], self.data[power_col], 
                        label=f'GPU {gpu_id}', linewidth=2, color=f'C{i}')
        
        # 绘制总功耗
        total_power = 0
        for gpu_id in self.gpu_ids:
            power_col = f'gpu_{gpu_id}_power'
            if power_col in self.data.columns:
                total_power += self.data[power_col]
        
        plt.plot(self.data['datetime'], total_power, 
                label='Total Power', linewidth=3, color='red', linestyle='--')
        
        plt.title('Multi-GPU Power Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 设置x轴格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_utilization_and_power(self, output_file: str = None, show_plot: bool = True):
        """绘制利用率和功耗的关系图"""
        if self.data is None:
            print("请先加载数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Utilization vs Power Relationship', fontsize=16, fontweight='bold')
        
        for i, gpu_id in enumerate(self.gpu_ids):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            power_col = f'gpu_{gpu_id}_power'
            util_col = f'gpu_{gpu_id}_utilization'
            
            if power_col in self.data.columns and util_col in self.data.columns:
                # 散点图显示利用率和功耗的关系
                ax.scatter(self.data[util_col], self.data[power_col], 
                          alpha=0.6, s=10, color=f'C{i}')
                
                # 添加趋势线（处理数值稳定性问题）
                try:
                    # 过滤有效数据点
                    valid_mask = (self.data[util_col] > 0) & (self.data[power_col] > 0)
                    if valid_mask.sum() > 1:  # 至少需要2个有效点
                        util_valid = self.data[util_col][valid_mask]
                        power_valid = self.data[power_col][valid_mask]
                        z = np.polyfit(util_valid, power_valid, 1)
                        p = np.poly1d(z)
                        ax.plot(util_valid, p(util_valid), 
                               "r--", alpha=0.8, linewidth=2)
                except (np.linalg.LinAlgError, ValueError):
                    # 如果拟合失败，跳过趋势线
                    pass
                
                ax.set_title(f'GPU {gpu_id} Utilization vs Power', fontsize=12, fontweight='bold')
                ax.set_xlabel('GPU Utilization (%)', fontsize=10)
                ax.set_ylabel('Power (W)', fontsize=10)
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.gpu_ids), 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"关系图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_temperature(self, output_file: str = None, show_plot: bool = True):
        """绘制温度变化图"""
        if self.data is None:
            print("请先加载数据")
            return
        
        plt.figure(figsize=(12, 8))
        
        for i, gpu_id in enumerate(self.gpu_ids):
            temp_col = f'gpu_{gpu_id}_temperature'
            if temp_col in self.data.columns:
                plt.plot(self.data['datetime'], self.data[temp_col], 
                        label=f'GPU {gpu_id}', linewidth=2, color=f'C{i}')
        
        plt.title('GPU Temperature Changes', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 设置x轴格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"温度图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_summary_report(self, output_file: str = None):
        """生成统计报告"""
        if self.data is None:
            print("请先加载数据")
            return
        
        report = []
        report.append("=" * 60)
        report.append("GPU Power Monitoring Statistics Report")
        report.append("=" * 60)
        
        duration = (self.data['datetime'].iloc[-1] - self.data['datetime'].iloc[0]).total_seconds()
        report.append(f"Monitoring Duration: {duration:.1f} seconds")
        report.append(f"Data Points: {len(self.data)}")
        report.append("")
        
        for gpu_id in self.gpu_ids:
            power_col = f'gpu_{gpu_id}_power'
            util_col = f'gpu_{gpu_id}_utilization'
            temp_col = f'gpu_{gpu_id}_temperature'
            
            if power_col in self.data.columns:
                power_data = self.data[power_col]
                util_data = self.data[util_col] if util_col in self.data.columns else None
                temp_data = self.data[temp_col] if temp_col in self.data.columns else None
                
                report.append(f"GPU {gpu_id}:")
                report.append(f"  Power - Average: {power_data.mean():.1f}W, Max: {power_data.max():.1f}W, Min: {power_data.min():.1f}W")
                
                if util_data is not None:
                    report.append(f"  Utilization - Average: {util_data.mean():.1f}%, Max: {util_data.max():.1f}%")
                
                if temp_data is not None:
                    report.append(f"  Temperature - Average: {temp_data.mean():.1f}°C, Max: {temp_data.max():.1f}°C")
                
                # 计算总能耗
                total_energy = power_data.sum() * 0.1 / 3600  # 假设100ms间隔
                report.append(f"  Total Energy: {total_energy:.3f} Wh")
                report.append("")
        
        # 计算总功耗
        total_power = 0
        for gpu_id in self.gpu_ids:
            power_col = f'gpu_{gpu_id}_power'
            if power_col in self.data.columns:
                total_power += self.data[power_col]
        
        if not total_power.empty:
            report.append("Overall Statistics:")
            report.append(f"  Total Power - Average: {total_power.mean():.1f}W, Max: {total_power.max():.1f}W")
            total_energy = total_power.sum() * 0.1 / 3600
            report.append(f"  Total Energy: {total_energy:.3f} Wh")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="GPU功耗数据可视化工具")
    parser.add_argument("csv_file", help="CSV数据文件路径")
    parser.add_argument("--output-dir", type=str, default="./plots", 
                       help="图表输出目录 (默认: ./plots)")
    parser.add_argument("--no-show", action="store_true", 
                       help="不显示图表，只保存文件")
    parser.add_argument("--report", type=str, 
                       help="生成统计报告文件")
    
    args = parser.parse_args()
    
    # 检查CSV文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"错误: CSV文件不存在: {args.csv_file}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建绘图器
    plotter = GPUPowerPlotter(args.csv_file)
    
    # 加载数据
    if not plotter.load_data():
        sys.exit(1)
    
    show_plot = not args.no_show
    
    # 生成各种图表
    print("正在生成图表...")
    
    # 功耗变化图
    plotter.plot_power_consumption(
        os.path.join(args.output_dir, "gpu_power_consumption.png"), 
        show_plot
    )
    
    # 功耗对比图
    plotter.plot_combined_power(
        os.path.join(args.output_dir, "gpu_power_comparison.png"), 
        show_plot
    )
    
    # 利用率和功耗关系图
    plotter.plot_utilization_and_power(
        os.path.join(args.output_dir, "gpu_utilization_vs_power.png"), 
        show_plot
    )
    
    # 温度变化图
    plotter.plot_temperature(
        os.path.join(args.output_dir, "gpu_temperature.png"), 
        show_plot
    )
    
    # 生成统计报告
    if args.report:
        plotter.generate_summary_report(args.report)
    else:
        plotter.generate_summary_report()
    
    print(f"所有图表已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()
