#!/usr/bin/env python3
"""
测试GPU监控的采样频率
"""

import time
import subprocess
import sys

def test_sampling_frequency():
    """测试采样频率"""
    print("测试GPU监控采样频率...")
    
    # 运行5秒的监控
    start_time = time.time()
    result = subprocess.run([
        "python3", "multi_gpu_monitor.py",
        "--duration", "5",
        "--output", "freq_test.csv",
        "--gpu-ids", "0"
    ], capture_output=True, text=True)
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # 读取CSV文件计算采样次数
    try:
        with open("freq_test.csv", "r") as f:
            lines = f.readlines()
            # 减去header行
            sample_count = len(lines) - 1
    except FileNotFoundError:
        print("CSV文件未找到")
        return
    
    # 计算实际采样频率
    if sample_count > 0:
        avg_interval = actual_duration / sample_count
        expected_interval = 0.1  # 100ms
        frequency_error = abs(avg_interval - expected_interval) / expected_interval * 100
        
        print(f"实际监控时长: {actual_duration:.2f}秒")
        print(f"采样次数: {sample_count}")
        print(f"平均采样间隔: {avg_interval:.3f}秒 ({avg_interval*1000:.1f}ms)")
        print(f"期望采样间隔: {expected_interval:.3f}秒 ({expected_interval*1000:.1f}ms)")
        print(f"频率误差: {frequency_error:.1f}%")
        
        if frequency_error < 10:  # 误差小于10%认为合格
            print("✅ 采样频率符合要求")
        else:
            print("❌ 采样频率误差较大")
    else:
        print("❌ 没有采集到数据")
    
    # 清理测试文件
    import os
    if os.path.exists("freq_test.csv"):
        os.remove("freq_test.csv")

if __name__ == "__main__":
    test_sampling_frequency()
