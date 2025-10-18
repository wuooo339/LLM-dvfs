#!/usr/bin/env python3
"""
GPU和CPU功耗变化图绘制脚本
读取多GPU和CPU监控数据并生成可视化图表
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

# 设置matplotlib避免刻度警告（某些版本不支持该rc参数，做兼容处理）
try:
    plt.rcParams['axes.locator.max_ticks'] = 20
except Exception:
    pass

class GPUPowerPlotter:
    """GPU和CPU功耗数据可视化工具"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data = None
        self.gpu_ids = []
        self.has_cpu_data = False
        
    def load_data(self):
        """加载CSV数据"""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"成功加载数据: {len(self.data)} 行")
            
            # 提取GPU ID列表
            power_columns = [col for col in self.data.columns if col.endswith('_power') and col.startswith('gpu_')]
            self.gpu_ids = [col.replace('gpu_', '').replace('_power', '') for col in power_columns]
            print(f"检测到GPU: {self.gpu_ids}")
            
            # 检查是否有CPU数据
            self.has_cpu_data = 'cpu_power' in self.data.columns
            if self.has_cpu_data:
                print("检测到CPU数据")
            
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
        """绘制功耗变化图（按GPU分面+CPU曲线叠加）。"""
        if self.data is None:
            print("请先加载数据")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU/CPU Power Monitoring', fontsize=16, fontweight='bold')
        
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
                
                # 若存在CPU功耗，叠加在同图便于对比
                if 'cpu_power' in self.data.columns:
                    ax.plot(self.data['datetime'], self.data['cpu_power'],
                            label='CPU Power', color='C4', linewidth=1.2, alpha=0.8)

                ax.set_title(f'GPU {gpu_id} Power', fontsize=12, fontweight='bold')
                ax.set_ylabel('Power (W)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                
                # 设置x轴格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                # 根据数据长度调整刻度间隔
                data_length = len(self.data)
                if data_length > 1000:
                    interval = max(60, data_length // 20)
                elif data_length > 100:
                    interval = max(10, data_length // 20)
                else:
                    interval = 1
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
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
        """绘制所有GPU和CPU的功耗对比图"""
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
        
        # 绘制CPU功耗曲线
        if self.has_cpu_data and 'cpu_power' in self.data.columns:
            plt.plot(self.data['datetime'], self.data['cpu_power'], 
                    label='CPU', linewidth=2, color='C4', linestyle='-')
        
        # 绘制总功耗
        total_power = 0
        for gpu_id in self.gpu_ids:
            power_col = f'gpu_{gpu_id}_power'
            if power_col in self.data.columns:
                total_power += self.data[power_col]
        
        if self.has_cpu_data and 'cpu_power' in self.data.columns:
            total_power += self.data['cpu_power']
            plt.plot(self.data['datetime'], total_power, 
                    label='Total Power (GPU+CPU)', linewidth=3, color='red', linestyle='--')
        else:
            plt.plot(self.data['datetime'], total_power, 
                    label='Total GPU Power', linewidth=3, color='red', linestyle='--')
        
        title = 'Multi-GPU and CPU Power Comparison' if self.has_cpu_data else 'Multi-GPU Power Comparison'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        
        # 设置x轴格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        # 根据数据长度调整刻度间隔
        data_length = len(self.data)
        if data_length > 1000:
            interval = max(60, data_length // 20)
        elif data_length > 100:
            interval = max(10, data_length // 20)
        else:
            interval = 1
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_cpu_power(self, output_file: str = None, show_plot: bool = True):
        """绘制CPU功耗/能量与温度图表"""
        if self.data is None:
            print("请先加载数据")
            return
        
        if not self.has_cpu_data:
            print("没有CPU数据")
            return
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CPU Power / Energy / Temperature', fontsize=16, fontweight='bold')
        
        # CPU功耗图
        ax1 = axes[0, 0]
        ax1.plot(self.data['datetime'], self.data['cpu_power'], 
                label='CPU Power', color='blue', linewidth=2)
        ax1.set_title('CPU Power Consumption', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Power (W)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(bottom=0)
        
        # CPU能量图（如果存在）
        ax2 = axes[0, 1]
        if 'cpu_energy' in self.data.columns:
            ax2.plot(self.data['datetime'], self.data['cpu_energy'], 
                    label='CPU Energy', color='green', linewidth=2)
            ax2.set_title('CPU Energy Accumulation', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Energy (J)', fontsize=10)
            ax2.set_ylim(bottom=0)
        else:
            ax2.text(0.5, 0.5, 'CPU Energy data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('CPU Energy (N/A)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # CPU温度（平均与插槽）
        ax3 = axes[1, 0]
        has_any = False
        if 'cpu_temperature' in self.data.columns:
            ax3.plot(self.data['datetime'], self.data['cpu_temperature'],
                     label='CPU Avg Temp', color='C4', linewidth=2)
            has_any = True
        colors = ['C5', 'C6']
        for idx in range(2):
            col = f'cpu_socket{idx}_temperature'
            if col in self.data.columns:
                ax3.plot(self.data['datetime'], self.data[col],
                         label=f'Socket{idx} Temp', linewidth=1.5, color=colors[idx % len(colors)])
                has_any = True
        if not has_any:
            ax3.text(0.5, 0.5, 'CPU Temperature data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('CPU Temperature', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Temperature (°C)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        ax3.set_ylim(bottom=0)
        
        # CPU每插槽PkgWatt（如有）
        ax4 = axes[1, 1]
        has_pkg = False
        for idx in range(2):
            col = f'cpu_socket{idx}_pkg_watt'
            if col in self.data.columns:
                ax4.plot(self.data['datetime'], self.data[col], label=f'Socket{idx} PkgWatt', linewidth=1.5)
                has_pkg = True
        if not has_pkg:
            ax4.text(0.5, 0.5, 'CPU PkgWatt data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('CPU Socket PkgWatt', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Power (W)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        ax4.set_ylim(bottom=0)
        
        # 设置x轴格式
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            # 根据数据长度调整刻度间隔
            data_length = len(self.data)
            if data_length > 1000:
                interval = max(60, data_length // 20)  # 至少60秒间隔
            elif data_length > 100:
                interval = max(10, data_length // 20)  # 至少10秒间隔
            else:
                interval = 1
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"CPU图表已保存到: {output_file}")
        
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
        plt.ylim(bottom=0)
        plt.ylim(bottom=0)
        
        # 设置x轴格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        # 根据数据长度调整刻度间隔
        data_length = len(self.data)
        if data_length > 1000:
            interval = max(60, data_length // 20)
        elif data_length > 100:
            interval = max(10, data_length // 20)
        else:
            interval = 1
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"温度图已保存到: {output_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_cpu_temperature(self, output_file: str = None, show_plot: bool = True):
        """绘制CPU温度变化图（平均与每插槽）。"""
        if self.data is None:
            print("请先加载数据")
            return
        
        has_any = False
        plt.figure(figsize=(12, 8))
        
        # 平均温度
        if 'cpu_temperature' in self.data.columns:
            plt.plot(self.data['datetime'], self.data['cpu_temperature'],
                    label='CPU Avg Temp', linewidth=2, color='C4')
            has_any = True
        
        # 每插槽温度（只绘制Socket 0和1）
        colors = ['C5', 'C6']
        for idx in range(2):
            col = f'cpu_socket{idx}_temperature'
            if col in self.data.columns:
                plt.plot(self.data['datetime'], self.data[col],
                        label=f'CPU Socket{idx} Temp', linewidth=1.8, color=colors[idx % len(colors)])
                has_any = True
        
        if not has_any:
            plt.text(0.5, 0.5, 'CPU Temperature data not available', 
                     ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.title('CPU Temperature Changes', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 设置x轴格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        data_length = len(self.data)
        if data_length > 1000:
            interval = max(60, data_length // 20)
        elif data_length > 100:
            interval = max(10, data_length // 20)
        else:
            interval = 1
        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"CPU温度图已保存到: {output_file}")
        
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
        
        # CPU统计信息
        if self.has_cpu_data:
            cpu_power_data = self.data['cpu_power'] if 'cpu_power' in self.data.columns else None
            cpu_util_data = self.data['cpu_utilization'] if 'cpu_utilization' in self.data.columns else None
            cpu_temp_data = self.data['cpu_temperature'] if 'cpu_temperature' in self.data.columns else None
            cpu_freq_data = self.data['cpu_frequency'] if 'cpu_frequency' in self.data.columns else None
            
            report.append("CPU:")
            if cpu_power_data is not None:
                report.append(f"  Power - Average: {cpu_power_data.mean():.1f}W, Max: {cpu_power_data.max():.1f}W, Min: {cpu_power_data.min():.1f}W")
                total_energy = cpu_power_data.sum() * 0.1 / 3600  # 假设100ms间隔
                report.append(f"  Total Energy: {total_energy:.3f} Wh")
            
            if cpu_util_data is not None:
                report.append(f"  Utilization - Average: {cpu_util_data.mean():.1f}%, Max: {cpu_util_data.max():.1f}%")
            
            if cpu_temp_data is not None:
                report.append(f"  Temperature - Average: {cpu_temp_data.mean():.1f}°C, Max: {cpu_temp_data.max():.1f}°C")
            
            if cpu_freq_data is not None:
                report.append(f"  Frequency - Average: {cpu_freq_data.mean():.0f}MHz, Max: {cpu_freq_data.max():.0f}MHz")
            
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
        total_gpu_power = 0
        for gpu_id in self.gpu_ids:
            power_col = f'gpu_{gpu_id}_power'
            if power_col in self.data.columns:
                total_gpu_power += self.data[power_col]
        
        total_cpu_power = 0
        if self.has_cpu_data and 'cpu_power' in self.data.columns:
            total_cpu_power = self.data['cpu_power']
        
        total_power = total_gpu_power + total_cpu_power
        
        if not total_power.empty:
            report.append("Overall Statistics:")
            if not total_gpu_power.empty:
                report.append(f"  Total GPU Power - Average: {total_gpu_power.mean():.1f}W, Max: {total_gpu_power.max():.1f}W")
            if not total_cpu_power.empty:
                report.append(f"  Total CPU Power - Average: {total_cpu_power.mean():.1f}W, Max: {total_cpu_power.max():.1f}W")
            report.append(f"  Total System Power - Average: {total_power.mean():.1f}W, Max: {total_power.max():.1f}W")
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
    
    # CPU功耗图（如果有CPU数据）
    if plotter.has_cpu_data:
        plotter.plot_cpu_power(
            os.path.join(args.output_dir, "cpu_power_monitoring.png"), 
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
