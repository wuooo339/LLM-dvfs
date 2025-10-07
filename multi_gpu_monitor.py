#!/usr/bin/env python3
"""
多GPU功耗监控脚本
以100ms间隔监控4个GPU的功耗、温度、利用率等数据
"""

import subprocess
import time
import threading
import json
import os
import signal
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

class MultiGPUMonitor:
    """多GPU功耗监控器"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1, 2, 3], interval: float = 0.1):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.monitor_thread = None
        self.start_time = None
        self.running = True
        self.output_file = None
        self.csv_writer = None
        self.csv_file = None
        self.display_counter = 0  # 用于控制显示频率
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n\n收到信号 {signum}，正在停止监控...")
        self.running = False
        self.stop_monitoring()
        sys.exit(0)
    
    def get_gpu_info(self, gpu_id: int) -> Dict[str, Any]:
        """获取指定GPU的信息"""
        try:
            # 使用更高效的查询，减少字段数量
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                return None
            
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 6:
                return {
                    "gpu_id": gpu_id,
                    "timestamp": time.time(),
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "graphics_clock": 0,  # 暂时不查询，减少延迟
                    "memory_clock": 0,    # 暂时不查询，减少延迟
                    "power_draw": float(parts[1]) if parts[1] != 'N/A' else 0.0,
                    "temperature": int(parts[2]) if parts[2] != 'N/A' else 0,
                    "gpu_utilization": int(parts[3]) if parts[3] != 'N/A' else 0,
                    "memory_used": int(parts[4]) if parts[4] != 'N/A' else 0,
                    "memory_total": int(parts[5]) if parts[5] != 'N/A' else 0
                }
        except Exception as e:
            return None
        
        return None
    
    def get_all_gpus_info(self) -> Dict[str, Any]:
        """获取所有GPU的信息 - 使用单次查询优化性能"""
        timestamp = time.time()
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
            # 单次查询所有GPU，提高效率
            gpu_list = ','.join(map(str, self.gpu_ids))
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_list}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                raise Exception("nvidia-smi command failed")
            
            gpu_data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.strip().split(', ')
                    if len(parts) >= 6:
                        gpu_id = int(parts[0])
                        gpu_data[f"gpu_{gpu_id}"] = {
                            "gpu_id": gpu_id,
                            "timestamp": timestamp,
                            "datetime": datetime_str,
                            "graphics_clock": 0,  # 暂时不查询
                            "memory_clock": 0,    # 暂时不查询
                            "power_draw": float(parts[1]) if parts[1] != 'N/A' else 0.0,
                            "temperature": int(parts[2]) if parts[2] != 'N/A' else 0,
                            "gpu_utilization": int(parts[3]) if parts[3] != 'N/A' else 0,
                            "memory_used": int(parts[4]) if parts[4] != 'N/A' else 0,
                            "memory_total": int(parts[5]) if parts[5] != 'N/A' else 0
                        }
            
            # 确保所有请求的GPU都有数据
            for gpu_id in self.gpu_ids:
                if f"gpu_{gpu_id}" not in gpu_data:
                    gpu_data[f"gpu_{gpu_id}"] = {
                        "gpu_id": gpu_id,
                        "timestamp": timestamp,
                        "datetime": datetime_str,
                        "graphics_clock": 0,
                        "memory_clock": 0,
                        "power_draw": 0.0,
                        "temperature": 0,
                        "gpu_utilization": 0,
                        "memory_used": 0,
                        "memory_total": 0
                    }
            
        except Exception as e:
            # 如果查询失败，返回默认值
            gpu_data = {}
            for gpu_id in self.gpu_ids:
                gpu_data[f"gpu_{gpu_id}"] = {
                    "gpu_id": gpu_id,
                    "timestamp": timestamp,
                    "datetime": datetime_str,
                    "graphics_clock": 0,
                    "memory_clock": 0,
                    "power_draw": 0.0,
                    "temperature": 0,
                    "gpu_utilization": 0,
                    "memory_used": 0,
                    "memory_total": 0
                }
        
        return {
            "timestamp": timestamp,
            "datetime": datetime_str,
            "gpus": gpu_data
        }
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring and self.running:
            start_time = time.time()
            
            all_gpu_data = self.get_all_gpus_info()
            if all_gpu_data:
                self.data.append(all_gpu_data)
                
                # 写入CSV文件
                if self.csv_writer:
                    self._write_csv_row(all_gpu_data)
                
                # 控制显示频率，每10次采样显示一次（约1秒显示一次）
                self.display_counter += 1
                if self.display_counter >= 10:
                    self._display_current_status(all_gpu_data)
                    self.display_counter = 0
            
            # 计算剩余时间，确保精确的采样间隔
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _write_csv_row(self, data: Dict[str, Any]):
        """写入CSV行数据"""
        try:
            row = {
                'timestamp': data['timestamp'],
                'datetime': data['datetime']
            }
            
            for gpu_id in self.gpu_ids:
                gpu_key = f"gpu_{gpu_id}"
                if gpu_key in data['gpus']:
                    gpu_data = data['gpus'][gpu_key]
                    row.update({
                        f'gpu_{gpu_id}_power': gpu_data['power_draw'],
                        f'gpu_{gpu_id}_utilization': gpu_data['gpu_utilization'],
                        f'gpu_{gpu_id}_temperature': gpu_data['temperature'],
                        f'gpu_{gpu_id}_memory_used': gpu_data['memory_used'],
                        f'gpu_{gpu_id}_memory_total': gpu_data['memory_total'],
                        f'gpu_{gpu_id}_graphics_clock': gpu_data['graphics_clock'],
                        f'gpu_{gpu_id}_memory_clock': gpu_data['memory_clock']
                    })
                else:
                    # 如果GPU数据不可用，填入0
                    row.update({
                        f'gpu_{gpu_id}_power': 0.0,
                        f'gpu_{gpu_id}_utilization': 0,
                        f'gpu_{gpu_id}_temperature': 0,
                        f'gpu_{gpu_id}_memory_used': 0,
                        f'gpu_{gpu_id}_memory_total': 0,
                        f'gpu_{gpu_id}_graphics_clock': 0,
                        f'gpu_{gpu_id}_memory_clock': 0
                    })
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            print(f"写入CSV数据失败: {e}")
    
    def _display_current_status(self, data: Dict[str, Any]):
        """显示当前所有GPU状态"""
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 100)
        print(f"多GPU实时功耗监控 - {data['datetime']}")
        print("=" * 100)
        
        # 显示每个GPU的状态
        for gpu_id in self.gpu_ids:
            gpu_key = f"gpu_{gpu_id}"
            if gpu_key in data['gpus']:
                gpu_data = data['gpus'][gpu_key]
                print(f"GPU {gpu_id:2d}: 功耗 {gpu_data['power_draw']:6.1f}W | "
                      f"利用率 {gpu_data['gpu_utilization']:3d}% | "
                      f"温度 {gpu_data['temperature']:3d}°C | "
                      f"显存 {gpu_data['memory_used']:6d}/{gpu_data['memory_total']:6d}MB | "
                      f"频率 {gpu_data['graphics_clock']:4d}MHz")
            else:
                print(f"GPU {gpu_id:2d}: 数据不可用")
        
        # 显示统计信息
        if len(self.data) > 1:
            duration = self.data[-1]['timestamp'] - self.data[0]['timestamp']
            print(f"\n监控时长: {duration:.1f}秒 | 采样次数: {len(self.data)}")
            
            # 计算总功耗
            total_power = 0.0
            for gpu_id in self.gpu_ids:
                gpu_key = f"gpu_{gpu_id}"
                if gpu_key in data['gpus']:
                    total_power += data['gpus'][gpu_key]['power_draw']
            print(f"总功耗: {total_power:.1f}W")
        
        print("\n" + "-" * 100)
        print("按 Ctrl+C 停止监控")
        print("=" * 100)
    
    def start_monitoring(self, output_file: str = None):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.data = []
        self.start_time = time.time()
        
        # 设置输出文件
        if output_file:
            self.output_file = output_file
            # 创建CSV文件
            self.csv_file = open(output_file, 'w', newline='', encoding='utf-8')
            fieldnames = ['timestamp', 'datetime']
            for gpu_id in self.gpu_ids:
                fieldnames.extend([
                    f'gpu_{gpu_id}_power',
                    f'gpu_{gpu_id}_utilization', 
                    f'gpu_{gpu_id}_temperature',
                    f'gpu_{gpu_id}_memory_used',
                    f'gpu_{gpu_id}_memory_total',
                    f'gpu_{gpu_id}_graphics_clock',
                    f'gpu_{gpu_id}_memory_clock'
                ])
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            print(f"数据将保存到: {output_file}")
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"开始监控GPU {self.gpu_ids}，采样间隔: {self.interval}s")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # 关闭CSV文件
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        # 显示最终统计
        self._display_final_statistics()
    
    def _display_final_statistics(self):
        """显示最终统计信息"""
        if not self.data:
            print("没有收集到数据")
            return
        
        print("\n" + "=" * 100)
        print("最终统计报告")
        print("=" * 100)
        
        duration = self.data[-1]['timestamp'] - self.data[0]['timestamp'] if len(self.data) > 1 else 0
        print(f"监控时长: {duration:.1f}秒")
        print(f"采样次数: {len(self.data)}")
        
        # 为每个GPU计算统计信息
        for gpu_id in self.gpu_ids:
            gpu_key = f"gpu_{gpu_id}"
            power_values = []
            util_values = []
            temp_values = []
            
            for data_point in self.data:
                if gpu_key in data_point['gpus']:
                    gpu_data = data_point['gpus'][gpu_key]
                    if gpu_data['power_draw'] > 0:
                        power_values.append(gpu_data['power_draw'])
                    if gpu_data['gpu_utilization'] > 0:
                        util_values.append(gpu_data['gpu_utilization'])
                    if gpu_data['temperature'] > 0:
                        temp_values.append(gpu_data['temperature'])
            
            print(f"\nGPU {gpu_id}:")
            if power_values:
                avg_power = sum(power_values) / len(power_values)
                max_power = max(power_values)
                min_power = min(power_values)
                total_energy = sum(power_values) * self.interval / 3600  # Wh
                print(f"  功耗 - 平均: {avg_power:.1f}W, 最大: {max_power:.1f}W, 最小: {min_power:.1f}W, 总能耗: {total_energy:.3f}Wh")
            
            if util_values:
                avg_util = sum(util_values) / len(util_values)
                max_util = max(util_values)
                print(f"  利用率 - 平均: {avg_util:.1f}%, 最大: {max_util:.1f}%")
            
            if temp_values:
                avg_temp = sum(temp_values) / len(temp_values)
                max_temp = max(temp_values)
                print(f"  温度 - 平均: {avg_temp:.1f}°C, 最高: {max_temp:.1f}°C")
        
        print("=" * 100)
    
    def save_json_data(self, filename: str):
        """保存JSON格式的详细数据"""
        data_to_save = {
            "gpu_ids": self.gpu_ids,
            "interval": self.interval,
            "start_time": self.start_time,
            "end_time": time.time(),
            "raw_data": self.data
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"详细数据已保存到: {filename}")
    
    def run_monitoring(self, output_file: str = None, json_file: str = None):
        """运行监控"""
        try:
            self.start_monitoring(output_file)
            
            # 保持监控直到用户中断
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n用户中断监控")
        finally:
            self.stop_monitoring()
            if json_file:
                self.save_json_data(json_file)

def main():
    parser = argparse.ArgumentParser(description="多GPU功耗监控工具")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3", 
                       help="要监控的GPU ID列表，用逗号分隔 (默认: 0,1,2,3)")
    parser.add_argument("--interval", type=float, default=0.1, 
                       help="采样间隔秒数 (默认: 0.1)")
    parser.add_argument("--output", type=str, 
                       help="CSV输出文件路径")
    parser.add_argument("--json", type=str,
                       help="JSON详细数据输出文件路径")
    parser.add_argument("--duration", type=int,
                       help="监控持续时间（秒），不指定则持续监控直到手动停止")
    
    args = parser.parse_args()
    
    # 解析GPU ID列表
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    # 检查nvidia-smi是否可用
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: nvidia-smi 未找到，请确保NVIDIA驱动已正确安装")
        sys.exit(1)
    
    monitor = MultiGPUMonitor(gpu_ids=gpu_ids, interval=args.interval)
    
    try:
        if args.duration:
            # 定时监控
            monitor.start_monitoring(args.output)
            time.sleep(args.duration)
            monitor.stop_monitoring()
        else:
            # 持续监控
            monitor.run_monitoring(args.output, args.json)
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()
