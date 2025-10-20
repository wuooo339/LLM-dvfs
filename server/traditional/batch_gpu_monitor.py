#!/usr/bin/env python3
"""
批次GPU监控脚本
为每个批次测试独立监控和记录GPU功耗数据
"""

import subprocess
import time
import json
import csv
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

class BatchGPUMonitor:
    """批次GPU监控器 - 为每个批次独立监控GPU功耗"""
    
    def __init__(self, gpu_ids: List[int] = [0, 1, 2, 3], interval: float = 0.1):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.start_time = None
        self.batch_size = None
        self.csv_file = None
        self.csv_writer = None
        
    def get_all_gpus_info(self) -> Dict[str, Any]:
        """获取所有GPU的信息 - 单次查询优化"""
        timestamp = time.time()
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
            gpu_list = ','.join(map(str, self.gpu_ids))
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total,clocks.gr,clocks.mem",
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
                    if len(parts) >= 8:
                        gpu_id = int(parts[0])
                        gpu_data[f"gpu_{gpu_id}"] = {
                            "gpu_id": gpu_id,
                            "power_draw": float(parts[1]) if parts[1] != 'N/A' else 0.0,
                            "temperature": int(parts[2]) if parts[2] != 'N/A' else 0,
                            "utilization": int(parts[3]) if parts[3] != 'N/A' else 0,
                            "memory_used": int(parts[4]) if parts[4] != 'N/A' else 0,
                            "memory_total": int(parts[5]) if parts[5] != 'N/A' else 0,
                            "graphics_clock": int(parts[6]) if parts[6] != 'N/A' else 0,
                            "memory_clock": int(parts[7]) if parts[7] != 'N/A' else 0
                        }
            
            # 确保所有GPU都有数据
            for gpu_id in self.gpu_ids:
                if f"gpu_{gpu_id}" not in gpu_data:
                    gpu_data[f"gpu_{gpu_id}"] = {
                        "gpu_id": gpu_id,
                        "power_draw": 0.0,
                        "temperature": 0,
                        "utilization": 0,
                        "memory_used": 0,
                        "memory_total": 0,
                        "graphics_clock": 0,
                        "memory_clock": 0
                    }
            
            return {
                "timestamp": timestamp,
                "datetime": datetime_str,
                "gpus": gpu_data
            }
            
        except Exception as e:
            # 失败时返回默认值
            gpu_data = {}
            for gpu_id in self.gpu_ids:
                gpu_data[f"gpu_{gpu_id}"] = {
                    "gpu_id": gpu_id,
                    "power_draw": 0.0,
                    "temperature": 0,
                    "utilization": 0,
                    "memory_used": 0,
                    "memory_total": 0,
                    "graphics_clock": 0,
                    "memory_clock": 0
                }
            return {
                "timestamp": timestamp,
                "datetime": datetime_str,
                "gpus": gpu_data
            }
    
    def start_monitoring(self, batch_size: int, output_dir: str):
        """开始监控指定批次"""
        if self.monitoring:
            return False
        
        self.batch_size = batch_size
        self.monitoring = True
        self.data = []
        self.start_time = time.time()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建CSV文件
        csv_filename = output_path / f"gpu_monitor_{batch_size}.csv"
        self.csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        
        # CSV字段
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
        
        print(f"✓ GPU监控已启动 - 批次大小: {batch_size}")
        return True
    
    def record_sample(self):
        """记录一次采样"""
        if not self.monitoring:
            return
        
        gpu_data = self.get_all_gpus_info()
        self.data.append(gpu_data)
        
        # 写入CSV
        if self.csv_writer:
            row = {
                'timestamp': gpu_data['timestamp'],
                'datetime': gpu_data['datetime']
            }
            
            for gpu_id in self.gpu_ids:
                gpu_key = f"gpu_{gpu_id}"
                if gpu_key in gpu_data['gpus']:
                    gdata = gpu_data['gpus'][gpu_key]
                    row.update({
                        f'gpu_{gpu_id}_power': gdata['power_draw'],
                        f'gpu_{gpu_id}_utilization': gdata['utilization'],
                        f'gpu_{gpu_id}_temperature': gdata['temperature'],
                        f'gpu_{gpu_id}_memory_used': gdata['memory_used'],
                        f'gpu_{gpu_id}_memory_total': gdata['memory_total'],
                        f'gpu_{gpu_id}_graphics_clock': gdata['graphics_clock'],
                        f'gpu_{gpu_id}_memory_clock': gdata['memory_clock']
                    })
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()
    
    def stop_monitoring(self, output_dir: str):
        """停止监控并保存数据"""
        if not self.monitoring:
            return None
        
        self.monitoring = False
        
        # 关闭CSV文件
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        # 计算统计信息
        stats = self._calculate_statistics()
        
        # 保存JSON统计文件
        output_path = Path(output_dir)
        json_filename = output_path / f"gpu_stats_{self.batch_size}.json"
        
        result = {
            "batch_size": self.batch_size,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.start_time,
            "sample_count": len(self.data),
            "interval": self.interval,
            "statistics": stats
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ GPU监控已停止 - 批次 {self.batch_size}")
        print(f"  - 采样次数: {len(self.data)}")
        print(f"  - 监控时长: {result['duration']:.1f}秒")
        print(f"  - 数据已保存: gpu_monitor_{self.batch_size}.csv")
        print(f"  - 统计已保存: gpu_stats_{self.batch_size}.json")
        
        return result
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算统计信息"""
        if not self.data:
            return {}
        
        stats = {}
        
        # 为每个GPU计算统计
        for gpu_id in self.gpu_ids:
            gpu_key = f"gpu_{gpu_id}"
            
            power_values = []
            util_values = []
            temp_values = []
            mem_values = []
            
            for sample in self.data:
                if gpu_key in sample['gpus']:
                    gdata = sample['gpus'][gpu_key]
                    if gdata['power_draw'] > 0:
                        power_values.append(gdata['power_draw'])
                    util_values.append(gdata['utilization'])
                    temp_values.append(gdata['temperature'])
                    mem_values.append(gdata['memory_used'])
            
            stats[f"gpu_{gpu_id}"] = {
                "power": {
                    "avg": sum(power_values) / len(power_values) if power_values else 0,
                    "max": max(power_values) if power_values else 0,
                    "min": min(power_values) if power_values else 0,
                    "total_energy_wh": sum(power_values) * self.interval / 3600 if power_values else 0
                },
                "utilization": {
                    "avg": sum(util_values) / len(util_values) if util_values else 0,
                    "max": max(util_values) if util_values else 0
                },
                "temperature": {
                    "avg": sum(temp_values) / len(temp_values) if temp_values else 0,
                    "max": max(temp_values) if temp_values else 0
                },
                "memory": {
                    "avg_used": sum(mem_values) / len(mem_values) if mem_values else 0,
                    "max_used": max(mem_values) if mem_values else 0
                }
            }
        
        # 计算总体统计
        total_power_values = []
        for sample in self.data:
            total_power = sum(sample['gpus'][f"gpu_{gid}"]['power_draw'] 
                            for gid in self.gpu_ids 
                            if f"gpu_{gid}" in sample['gpus'])
            total_power_values.append(total_power)
        
        stats["total"] = {
            "avg_power": sum(total_power_values) / len(total_power_values) if total_power_values else 0,
            "max_power": max(total_power_values) if total_power_values else 0,
            "total_energy_wh": sum(total_power_values) * self.interval / 3600 if total_power_values else 0
        }
        
        return stats

def monitor_loop(monitor: BatchGPUMonitor):
    """监控循环 - 在后台持续采样"""
    while monitor.monitoring:
        start = time.time()
        monitor.record_sample()
        elapsed = time.time() - start
        sleep_time = max(0, monitor.interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

def main():
    """命令行接口 - 用于独立测试"""
    parser = argparse.ArgumentParser(description="批次GPU监控工具")
    parser.add_argument("--batch-size", type=int, required=True, help="批次大小")
    parser.add_argument("--duration", type=int, required=True, help="监控时长（秒）")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3", help="GPU ID列表")
    parser.add_argument("--interval", type=float, default=0.1, help="采样间隔（秒）")
    
    args = parser.parse_args()
    
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    monitor = BatchGPUMonitor(gpu_ids=gpu_ids, interval=args.interval)
    
    # 启动监控
    monitor.start_monitoring(args.batch_size, args.output_dir)
    
    # 监控指定时长
    import threading
    monitor_thread = threading.Thread(target=monitor_loop, args=(monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n用户中断")
    
    # 停止监控
    monitor.stop_monitoring(args.output_dir)

if __name__ == "__main__":
    main()

