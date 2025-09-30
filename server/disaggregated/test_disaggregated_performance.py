#!/usr/bin/env python3
"""
VLLM 分离式 Prefill+Decode 性能测试脚本
测量 TTFT、TBT、E2E Latency 和详细功耗变化
"""

import requests
import json
import time
import threading
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加父目录到路径，以便导入 gpu_monitor
sys.path.append(str(Path(__file__).parent.parent.parent))
from gpu_monitor import GPUMonitor

class DisaggregatedClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    def generate(self, prompt, max_tokens=16, temperature=0.7):
        """发送生成请求到分离式服务器"""
        payload = {
            "model": "/share-data/wzk-1/model/deepseek-v2-lite",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"生成请求失败: {e}")
            return None

def test_disaggregated_performance(client, prompts, gpu_monitor_prefill, gpu_monitor_decode):
    """测试分离式 prefill+decode 性能"""
    print("\n=== 分离式 Prefill+Decode 性能测试 ===")
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n处理请求 {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # 启动 GPU 监控
        gpu_monitor_prefill.start_monitoring()
        gpu_monitor_decode.start_monitoring()
        
        # 记录 E2E 开始时间
        e2e_start = time.time()
        
        # 发送请求
        result = client.generate(prompt, max_tokens=16)
        
        # 记录 E2E 结束时间
        e2e_end = time.time()
        e2e_latency = e2e_end - e2e_start
        
        # 停止 GPU 监控
        gpu_monitor_prefill.stop_monitoring()
        gpu_monitor_decode.stop_monitoring()
        
        if result:
            # 提取时间指标
            timing = result.get("timing", {})
            usage = result.get("usage", {})
            choices = result.get("choices", [])
            generated_text = choices[0].get("text", "") if choices else ""
            
            # 计算指标
            ttft = timing.get("prefill_time", 0)  # Time to First Token
            tbt = timing.get("tbt", 0)  # Time Between Tokens
            prefill_time = timing.get("prefill_time", 0)
            decode_time = timing.get("decode_time", 0)
            
            result_data = {
                "request_id": i,
                "prompt": prompt,
                "generated_text": generated_text,
                "e2e_latency": e2e_latency,
                "ttft": ttft,
                "tbt": tbt,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "result": result
            }
            
            results.append(result_data)
            
            print(f"  ✅ 请求完成")
            print(f"     E2E Latency: {e2e_latency:.4f}s")
            print(f"     TTFT: {ttft:.4f}s")
            print(f"     TBT: {tbt:.4f}s")
            print(f"     Prefill: {prefill_time:.4f}s")
            print(f"     Decode: {decode_time:.4f}s")
            print(f"     Tokens: {usage.get('completion_tokens', 0)}")
            print(f"     生成内容: {generated_text[:50]}...")
        else:
            print(f"  ❌ 请求失败")
    
    return results

def create_detailed_power_visualization(prefill_gpu_data, decode_gpu_data, results, storage_dir):
    """创建详细的功耗变化图表"""
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Disaggregated Prefill+Decode Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. 功耗时间序列图
    if prefill_gpu_data and 'raw_data' in prefill_gpu_data:
        prefill_timestamps = [d['timestamp'] for d in prefill_gpu_data['raw_data']]
        prefill_power = [d['power_draw'] for d in prefill_gpu_data['raw_data']]
        prefill_times = [(t - prefill_timestamps[0]) for t in prefill_timestamps]
        
        ax1.plot(prefill_times, prefill_power, 'b-', label='Prefill GPU Power', linewidth=2)
    
    if decode_gpu_data and 'raw_data' in decode_gpu_data:
        decode_timestamps = [d['timestamp'] for d in decode_gpu_data['raw_data']]
        decode_power = [d['power_draw'] for d in decode_gpu_data['raw_data']]
        decode_times = [(t - decode_timestamps[0]) for t in decode_timestamps]
        
        ax1.plot(decode_times, decode_power, 'r-', label='Decode GPU Power', linewidth=2)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Consumption Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 频率变化图
    if prefill_gpu_data and 'raw_data' in prefill_gpu_data:
        prefill_freq = [d['graphics_clock'] for d in prefill_gpu_data['raw_data']]
        ax2.plot(prefill_times, prefill_freq, 'b-', label='Prefill GPU Freq', linewidth=2)
    
    if decode_gpu_data and 'raw_data' in decode_gpu_data:
        decode_freq = [d['graphics_clock'] for d in decode_gpu_data['raw_data']]
        ax2.plot(decode_times, decode_freq, 'r-', label='Decode GPU Freq', linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title('GPU Frequency Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能指标对比
    if results:
        ttft_values = [r['ttft'] for r in results]
        tbt_values = [r['tbt'] for r in results]
        e2e_values = [r['e2e_latency'] for r in results]
        
        x = np.arange(len(results))
        width = 0.25
        
        ax3.bar(x - width, ttft_values, width, label='TTFT', color='skyblue', alpha=0.8)
        ax3.bar(x, tbt_values, width, label='TBT', color='lightcoral', alpha=0.8)
        ax3.bar(x + width, e2e_values, width, label='E2E Latency', color='lightgreen', alpha=0.8)
        
        ax3.set_xlabel('Request ID')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Performance Metrics by Request')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Req {i+1}' for i in range(len(results))])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 功耗统计对比
    if prefill_gpu_data and decode_gpu_data:
        prefill_stats = prefill_gpu_data.get('statistics', {})
        decode_stats = decode_gpu_data.get('statistics', {})
        
        categories = ['Avg Power', 'Max Power', 'Avg Freq', 'Max Freq']
        prefill_values = [
            prefill_stats.get('power_draw', {}).get('avg', 0),
            prefill_stats.get('power_draw', {}).get('max', 0),
            prefill_stats.get('graphics_clock', {}).get('avg', 0),
            prefill_stats.get('graphics_clock', {}).get('max', 0)
        ]
        decode_values = [
            decode_stats.get('power_draw', {}).get('avg', 0),
            decode_stats.get('power_draw', {}).get('max', 0),
            decode_stats.get('graphics_clock', {}).get('avg', 0),
            decode_stats.get('graphics_clock', {}).get('max', 0)
        ]
        
        x4 = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x4 - width/2, prefill_values, width, label='Prefill GPU', color='blue', alpha=0.8)
        ax4.bar(x4 + width/2, decode_values, width, label='Decode GPU', color='red', alpha=0.8)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('GPU Performance Comparison')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(categories, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(storage_dir / "disaggregated_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 详细性能分析图表已保存到: {storage_dir / 'disaggregated_performance_analysis.png'}")

def main():
    """主测试函数"""
    print("VLLM 分离式 Prefill+Decode 性能测试")
    print("=" * 60)
    
    # 创建客户端
    client = DisaggregatedClient("http://localhost:8000")
    
    # 检查服务器状态
    print("检查分离式服务器状态...")
    if not client.health_check():
        print("❌ 分离式服务器未运行，请先启动服务器")
        print("运行命令: ./start_disaggregated_servers.sh")
        return
    
    print("✅ 分离式服务器运行正常")
    
    # 测试提示词
    test_prompts = [
        "1. 若3台机器5小时生产180个零件，7台机器8小时可生产多少零件？",
        "2. 甲比乙大6岁，5年前甲年龄是乙的2倍，求两人现在年龄。",
        "3. 编写一个Python函数计算斐波那契数列的第n项",
        "4. 解释什么是机器学习中的过拟合现象"
    ]
    
    # 创建存储目录
    storage_dir = Path("disaggregated_results")
    storage_dir.mkdir(exist_ok=True)
    
    # 初始化 GPU 监控器（分别监控两个 GPU）
    gpu_monitor_prefill = GPUMonitor(gpu_id=0, interval=0.1)  # GPU 0 (Prefill)
    gpu_monitor_decode = GPUMonitor(gpu_id=1, interval=0.1)   # GPU 1 (Decode)
    
    # 执行分离式性能测试
    print(f"\n开始分离式性能测试，处理 {len(test_prompts)} 个请求...")
    results = test_disaggregated_performance(client, test_prompts, gpu_monitor_prefill, gpu_monitor_decode)
    
    # 获取 GPU 监控数据
    prefill_gpu_stats = gpu_monitor_prefill.get_statistics()
    decode_gpu_stats = gpu_monitor_decode.get_statistics()
    
    # 保存结果
    test_results = {
        "test_type": "disaggregated_prefill_decode",
        "total_requests": len(results),
        "results": results,
        "prefill_gpu_statistics": prefill_gpu_stats,
        "decode_gpu_statistics": decode_gpu_stats,
        "summary": {
            "avg_e2e_latency": np.mean([r['e2e_latency'] for r in results]) if results else 0,
            "avg_ttft": np.mean([r['ttft'] for r in results]) if results else 0,
            "avg_tbt": np.mean([r['tbt'] for r in results]) if results else 0,
            "avg_prefill_time": np.mean([r['prefill_time'] for r in results]) if results else 0,
            "avg_decode_time": np.mean([r['decode_time'] for r in results]) if results else 0
        }
    }
    
    # 保存到文件
    with open(storage_dir / "disaggregated_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    # 保存 GPU 监控数据
    gpu_monitor_prefill.save_data(storage_dir / "prefill_gpu_data.json")
    gpu_monitor_decode.save_data(storage_dir / "decode_gpu_data.json")
    
    # 创建详细的可视化图表
    prefill_gpu_data = json.load(open(storage_dir / "prefill_gpu_data.json", "r"))
    decode_gpu_data = json.load(open(storage_dir / "decode_gpu_data.json", "r"))
    create_detailed_power_visualization(prefill_gpu_data, decode_gpu_data, results, storage_dir)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("分离式性能测试完成！")
    print(f"总请求数: {len(results)}")
    
    if results:
        summary = test_results["summary"]
        print(f"\n📊 性能指标总结:")
        print(f"  平均 E2E Latency: {summary['avg_e2e_latency']:.4f}s")
        print(f"  平均 TTFT: {summary['avg_ttft']:.4f}s")
        print(f"  平均 TBT: {summary['avg_tbt']:.4f}s")
        print(f"  平均 Prefill 时间: {summary['avg_prefill_time']:.4f}s")
        print(f"  平均 Decode 时间: {summary['avg_decode_time']:.4f}s")
        
        print(f"\n⚡ GPU 功耗总结:")
        if prefill_gpu_stats:
            print(f"  Prefill GPU 平均功耗: {prefill_gpu_stats['power_draw']['avg']:.1f}W")
            print(f"  Prefill GPU 最大功耗: {prefill_gpu_stats['power_draw']['max']:.1f}W")
        if decode_gpu_stats:
            print(f"  Decode GPU 平均功耗: {decode_gpu_stats['power_draw']['avg']:.1f}W")
            print(f"  Decode GPU 最大功耗: {decode_gpu_stats['power_draw']['max']:.1f}W")
    
    print(f"\n📁 结果已保存到 {storage_dir} 目录")

if __name__ == "__main__":
    main()
