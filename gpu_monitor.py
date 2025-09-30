import subprocess
import time
import threading
import json
from pathlib import Path
from typing import List, Dict, Any

class GPUMonitor:
    """GPU频率和功耗监控器"""
    
    def __init__(self, gpu_id: int = 0, interval: float = 0.1):
        self.gpu_id = gpu_id
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.monitor_thread = None
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取当前GPU信息"""
        try:
            # 获取GPU频率、功耗、温度等信息
            cmd = [
                "nvidia-smi", 
                "--query-gpu=index,timestamp,clocks.gr,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={self.gpu_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return {}
            
            # 解析输出
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 9:
                return {
                    "timestamp": time.time(),
                    "gpu_index": int(parts[0]),
                    "graphics_clock": int(parts[2]) if parts[2] != 'N/A' else 0,
                    "memory_clock": int(parts[3]) if parts[3] != 'N/A' else 0,
                    "power_draw": float(parts[4]) if parts[4] != 'N/A' else 0.0,
                    "temperature": int(parts[5]) if parts[5] != 'N/A' else 0,
                    "gpu_utilization": int(parts[6]) if parts[6] != 'N/A' else 0,
                    "memory_used": int(parts[7]) if parts[7] != 'N/A' else 0,
                    "memory_total": int(parts[8]) if parts[8] != 'N/A' else 0
                }
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
            return {}
        
        return {}
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                self.data.append(gpu_info)
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"开始监控GPU {self.gpu_id}，采样间隔: {self.interval}s")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print(f"停止监控GPU {self.gpu_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        if not self.data:
            return {}
        
        graphics_clocks = [d["graphics_clock"] for d in self.data if d["graphics_clock"] > 0]
        memory_clocks = [d["memory_clock"] for d in self.data if d["memory_clock"] > 0]
        power_draws = [d["power_draw"] for d in self.data if d["power_draw"] > 0]
        temperatures = [d["temperature"] for d in self.data if d["temperature"] > 0]
        gpu_utils = [d["gpu_utilization"] for d in self.data if d["gpu_utilization"] > 0]
        
        stats = {
            "monitoring_duration": self.data[-1]["timestamp"] - self.data[0]["timestamp"] if len(self.data) > 1 else 0,
            "sample_count": len(self.data),
            "graphics_clock": {
                "avg": sum(graphics_clocks) / len(graphics_clocks) if graphics_clocks else 0,
                "max": max(graphics_clocks) if graphics_clocks else 0,
                "min": min(graphics_clocks) if graphics_clocks else 0
            },
            "memory_clock": {
                "avg": sum(memory_clocks) / len(memory_clocks) if memory_clocks else 0,
                "max": max(memory_clocks) if memory_clocks else 0,
                "min": min(memory_clocks) if memory_clocks else 0
            },
            "power_draw": {
                "avg": sum(power_draws) / len(power_draws) if power_draws else 0,
                "max": max(power_draws) if power_draws else 0,
                "min": min(power_draws) if power_draws else 0,
                "total_energy": sum(power_draws) * self.interval / 1000  # 转换为kWh
            },
            "temperature": {
                "avg": sum(temperatures) / len(temperatures) if temperatures else 0,
                "max": max(temperatures) if temperatures else 0,
                "min": min(temperatures) if temperatures else 0
            },
            "gpu_utilization": {
                "avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                "max": max(gpu_utils) if gpu_utils else 0,
                "min": min(gpu_utils) if gpu_utils else 0
            }
        }
        
        return stats
    
    def save_data(self, filename: str):
        """保存监控数据"""
        data_to_save = {
            "gpu_id": self.gpu_id,
            "interval": self.interval,
            "statistics": self.get_statistics(),
            "raw_data": self.data
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"GPU监控数据已保存到: {filename}")

def monitor_gpu_during_execution(func, gpu_id: int = 0, interval: float = 0.1, save_file: str = None):
    """在函数执行期间监控GPU"""
    monitor = GPUMonitor(gpu_id=gpu_id, interval=interval)
    
    try:
        monitor.start_monitoring()
        result = func()
        return result, monitor.get_statistics()
    finally:
        monitor.stop_monitoring()
        if save_file:
            monitor.save_data(save_file)
