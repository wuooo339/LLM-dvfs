#!/usr/bin/env python3
"""
GPU监控工具测试脚本
用于验证multi_gpu_monitor.py和plot_gpu_power.py是否正常工作
"""

import subprocess
import os
import sys
import time
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖...")
    
    # 检查nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_count = len([line.strip() for line in result.stdout.strip().split('\n') if line.strip()])
            print(f"✓ nvidia-smi 可用，检测到 {gpu_count} 个GPU")
        else:
            print("✗ nvidia-smi 不可用")
            return False
    except Exception as e:
        print(f"✗ nvidia-smi 检查失败: {e}")
        return False
    
    # 检查Python包
    required_packages = ['pandas', 'matplotlib', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_multi_gpu_monitor():
    """测试多GPU监控脚本"""
    print("\n测试多GPU监控脚本...")
    
    # 测试短时间监控
    test_duration = 5  # 5秒
    output_file = "test_gpu_data.csv"
    
    try:
        cmd = [
            "python3", "multi_gpu_monitor.py",
            "--duration", str(test_duration),
            "--output", output_file,
            "--gpu-ids", "0"  # 只测试第一个GPU
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=test_duration + 10)
        
        if result.returncode == 0:
            print("✓ 多GPU监控脚本运行成功")
            
            # 检查输出文件
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    print(f"✓ 生成了 {len(lines)} 行数据")
                    if len(lines) > 1:  # 包含header
                        print("✓ 数据格式正确")
                    else:
                        print("✗ 数据文件为空")
                        return False
            else:
                print("✗ 未生成输出文件")
                return False
        else:
            print(f"✗ 脚本运行失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 脚本运行超时")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    
    return True

def test_plot_script():
    """测试绘图脚本"""
    print("\n测试绘图脚本...")
    
    csv_file = "test_gpu_data.csv"
    output_dir = "test_plots"
    
    if not os.path.exists(csv_file):
        print("✗ 找不到测试数据文件")
        return False
    
    try:
        cmd = [
            "python3", "plot_gpu_power.py",
            csv_file,
            "--output-dir", output_dir,
            "--no-show"  # 不显示图表
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ 绘图脚本运行成功")
            
            # 检查生成的图表文件
            plot_files = [
                "gpu_power_consumption.png",
                "gpu_power_comparison.png", 
                "gpu_utilization_vs_power.png",
                "gpu_temperature.png"
            ]
            
            for plot_file in plot_files:
                plot_path = os.path.join(output_dir, plot_file)
                if os.path.exists(plot_path):
                    print(f"✓ 生成了 {plot_file}")
                else:
                    print(f"✗ 未生成 {plot_file}")
                    return False
            
            print("✓ 所有图表生成成功")
        else:
            print(f"✗ 绘图脚本运行失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 绘图脚本运行超时")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    
    return True

def cleanup_test_files():
    """清理测试文件"""
    print("\n清理测试文件...")
    
    test_files = [
        "test_gpu_data.csv",
        "test_gpu_data.json",
        "test_plots"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
                print(f"✓ 删除目录: {file_path}")
            else:
                os.remove(file_path)
                print(f"✓ 删除文件: {file_path}")

def main():
    """主测试函数"""
    print("GPU监控工具测试")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装必要的依赖")
        sys.exit(1)
    
    # 测试多GPU监控
    if not test_multi_gpu_monitor():
        print("\n❌ 多GPU监控测试失败")
        sys.exit(1)
    
    # 测试绘图脚本
    if not test_plot_script():
        print("\n❌ 绘图脚本测试失败")
        sys.exit(1)
    
    print("\n✅ 所有测试通过！")
    print("\n工具使用示例:")
    print("1. 监控GPU功耗: python3 multi_gpu_monitor.py --duration 60 --output gpu_data.csv")
    print("2. 生成图表: python3 plot_gpu_power.py gpu_data.csv --output-dir ./plots")
    
    # 询问是否清理测试文件
    try:
        cleanup = input("\n是否清理测试文件？(y/N): ").strip().lower()
        if cleanup in ['y', 'yes']:
            cleanup_test_files()
        else:
            print("保留测试文件")
    except KeyboardInterrupt:
        print("\n保留测试文件")

if __name__ == "__main__":
    main()
