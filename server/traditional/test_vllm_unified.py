#!/usr/bin/env python3
"""
VLLM 服务器统一测试脚本
Prefill 和 Decode 一起生成，测量 TTFT、TPOT 和 100ms 间隔的功耗频率数据
"""

import requests
import json
import time
import threading
import sys
from pathlib import Path
import numpy as np

# 添加父目录到路径，以便导入 gpu_monitor
sys.path.append(str(Path(__file__).parent.parent.parent))
from gpu_monitor import GPUMonitor

class VLLMClient:
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
    
    def generate_stream(self, prompt, max_tokens=16, temperature=0.7):
        """发送流式生成请求"""
        payload = {
            "model": "/share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"流式生成请求失败: {e}")
            return None

def test_unified_generation(client, prompts, gpu_monitor, max_tokens=32):
    """统一测试 prefill 和 decode 阶段"""
    print("\n=== 统一生成测试 (Prefill + Decode) ===")
    
    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"处理提示词 {i+1}/{len(prompts)}: {prompt[:50]}...")
        # 记录开始时间
        request_start = time.time()
        first_token_time = None
        last_token_time = None
        token_times = []
        # 发送流式请求
        response = client.generate_stream(prompt, max_tokens=max_tokens)
        if not response:
            print(f"  请求失败")
            continue   
        # 处理流式响应
        generated_text = ""
        token_count = 0
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get('choices', [])
                            if choices:
                                choice = choices[0]
                                text_content = choice.get('text', '')
                                if text_content:
                                    current_time = time.time()
                                    token_count += 1
                                    # 记录首token时间 (TTFT)
                                    if first_token_time is None:
                                        first_token_time = current_time - request_start
                                    # 记录每个token的时间
                                    token_times.append(current_time - request_start)
                                    last_token_time = current_time
                                    
                                    generated_text += text_content
                                    
                                    # 显示token内容，过滤不可见字符
                                    display_text = text_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                    if len(display_text) > 20:
                                        display_text = display_text[:20] + "..."
                                    print(f"    Token {token_count}: '{display_text}' (时间: {current_time - request_start:.4f}s)")
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            print(f"  处理流式响应时出错: {e}")
        
        # 计算时间指标
        total_time = time.time() - request_start
        ttft = first_token_time if first_token_time else 0
        
        # 修复TPOT计算 - 使用token_times中的时间
        if token_count > 1 and len(token_times) > 1:
            # token_times[0] 是首token时间，token_times[-1] 是末token时间
            tpot = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
        else:
            tpot = 0
        
        result = {
            "prompt_id": i,
            "prompt": prompt,
            "generated_text": generated_text,
            "total_time": total_time,
            "ttft": ttft,  # Time To First Token
            "tpot": tpot,  # Time Per Other Token
            "token_count": token_count,
            "token_times": token_times,
            "first_token_time": first_token_time,
            "last_token_time": last_token_time
        }
        
        all_results.append(result)
        
        print(f"\n  完成，总时间: {total_time:.4f}s")
        print(f"  TTFT: {ttft:.4f}s")
        print(f"  TPOT: {tpot:.4f}s")
        print(f"  生成tokens: {token_count}")
        print(f"  生成内容: {generated_text[:100]}...")
    
    return all_results

def analyze_gpu_data_with_phases(gpu_data, results, interval_ms=50):
    """分析GPU数据，标注prefill和decode阶段"""
    
    # 将GPU数据按时间间隔分组
    interval_seconds = interval_ms / 1000.0
    gpu_timestamps = [entry['timestamp'] for entry in gpu_data]
    
    if not gpu_timestamps:
        return []
    
    start_time = min(gpu_timestamps)
    end_time = max(gpu_timestamps)
    
    # 创建时间间隔
    intervals = []
    current_time = start_time
    
    while current_time <= end_time:
        intervals.append({
            'start_time': current_time,
            'end_time': current_time + interval_seconds,
            'phase': 'idle',  # 默认空闲
            'power_data': [],
            'frequency_data': []
        })
        current_time += interval_seconds
    
    # 为每个间隔分配GPU数据
    for entry in gpu_data:
        timestamp = entry['timestamp']
        for interval in intervals:
            if interval['start_time'] <= timestamp < interval['end_time']:
                interval['power_data'].append(entry.get('power_draw', 0))
                interval['frequency_data'].append({
                    'graphics_clock': entry.get('graphics_clock', 0),
                    'memory_clock': entry.get('memory_clock', 0)
                })
                break
    
    # 根据结果标注阶段
    # 需要找到每个请求的实际开始时间（不是first_token_time）
    request_times = []
    for i, result in enumerate(results):
        # 从token_times中找到第一个时间点作为请求开始时间
        token_times = result.get('token_times', [])
        if token_times:
            # 请求开始时间 = 第一个token时间 - TTFT
            first_token_time = token_times[0]
            ttft = result.get('ttft', 0)
            request_start_time = first_token_time - ttft
            request_end_time = token_times[-1] if token_times else first_token_time
            
            request_times.append({
                'request_start': request_start_time,
                'prefill_start': request_start_time,
                'prefill_end': first_token_time,
                'decode_start': first_token_time,
                'decode_end': request_end_time
            })
            
            print(f"请求 {i+1} 时间分析:")
            print(f"  首token时间: {first_token_time:.4f}s")
            print(f"  TTFT: {ttft:.4f}s")
            print(f"  请求开始时间: {request_start_time:.4f}s")
            print(f"  Prefill阶段: {request_start_time:.4f}s - {first_token_time:.4f}s")
            print(f"  Decode阶段: {first_token_time:.4f}s - {request_end_time:.4f}s")
    
    print(f"\nGPU数据时间范围: {start_time:.4f}s - {end_time:.4f}s")
    print(f"间隔数量: {len(intervals)}")
    print(f"请求数量: {len(request_times)}")
    
    # 标注阶段
    for request in request_times:
        # 标注prefill阶段 (从请求开始到首token)
        for interval in intervals:
            if (request['prefill_start'] <= interval['start_time'] < request['prefill_end'] or
                request['prefill_start'] < interval['end_time'] <= request['prefill_end']):
                interval['phase'] = 'prefill'
        
        # 标注decode阶段 (从首token到末token)
        for interval in intervals:
            if (request['decode_start'] <= interval['start_time'] < request['decode_end'] or
                request['decode_start'] < interval['end_time'] <= request['decode_end']):
                interval['phase'] = 'decode'
    
    # 计算每个间隔的统计值
    analyzed_intervals = []
    for i, interval in enumerate(intervals):
        if interval['power_data']:
            avg_power = sum(interval['power_data']) / len(interval['power_data'])
            max_power = max(interval['power_data'])
            
            if interval['frequency_data']:
                avg_graphics_clock = sum(f['graphics_clock'] for f in interval['frequency_data']) / len(interval['frequency_data'])
                avg_memory_clock = sum(f['memory_clock'] for f in interval['frequency_data']) / len(interval['frequency_data'])
            else:
                avg_graphics_clock = 0
                avg_memory_clock = 0
            
            analyzed_intervals.append({
                'interval_id': i,
                'start_time': interval['start_time'],
                'end_time': interval['end_time'],
                'phase': interval['phase'],
                'avg_power': avg_power,
                'max_power': max_power,
                'avg_graphics_clock': avg_graphics_clock,
                'avg_memory_clock': avg_memory_clock,
                'data_points': len(interval['power_data']),
                'power_data': interval['power_data']  # 保存原始数据用于峰值计算
            })
    
    return analyzed_intervals

def get_gpu_max_power():
    """获取GPU最大功耗限制"""
    try:
        import subprocess
        # 使用 nvidia-smi -q -d POWER 获取详细信息
        result = subprocess.run([
            "nvidia-smi", 
            "-q", 
            "-d", 
            "POWER",
            "--id=0"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # 解析输出查找 "Current Power Limit"
            lines = result.stdout.split('\n')
            for line in lines:
                if "Current Power Limit" in line:
                    # 提取功率值，格式通常是 "Current Power Limit          : 320.00 W"
                    parts = line.split(':')
                    if len(parts) > 1:
                        power_str = parts[1].strip().replace('W', '').strip()
                        power_limit = float(power_str)
                        print(f"检测到GPU功率限制: {power_limit}W")
                        return power_limit
    except Exception as e:
        print(f"获取GPU功率限制失败: {e}")
    
    # 如果无法获取，尝试简单的查询
    try:
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
            "--id=0"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            power_limit = float(result.stdout.strip())
            print(f"通过简单查询获取GPU功率限制: {power_limit}W")
            return power_limit
    except Exception as e:
        print(f"简单查询也失败: {e}")
    
    # 默认值
    print("使用默认GPU功率限制: 320W")
    return 320

def get_true_peak_power(analyzed_intervals):
    """获取真正的峰值功耗（从原始数据中）"""
    peak_power = 0
    for interval in analyzed_intervals:
        if 'power_data' in interval and interval['power_data']:
            interval_peak = max(interval['power_data'])
            peak_power = max(peak_power, interval_peak)
    return peak_power


def create_timeline_plot(analyzed_intervals, results, storage_dir):
    """创建时间线折线图"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    if not analyzed_intervals:
        print("没有数据可以绘制")
        return
    
    # 准备数据 - 使用相对时间（从0开始）
    start_time = analyzed_intervals[0]['start_time']
    times = [(interval['start_time'] - start_time) for interval in analyzed_intervals]
    powers = [interval['avg_power'] for interval in analyzed_intervals]
    graphics_clocks = [interval['avg_graphics_clock'] for interval in analyzed_intervals]
    memory_clocks = [interval['avg_memory_clock'] for interval in analyzed_intervals]
    phases = [interval['phase'] for interval in analyzed_intervals]
    
    # 调试输出
    print(f"频率数据调试:")
    print(f"  图形时钟范围: {min(graphics_clocks):.0f} - {max(graphics_clocks):.0f} MHz")
    print(f"  内存时钟范围: {min(memory_clocks):.0f} - {max(memory_clocks):.0f} MHz")
    print(f"  功耗范围: {min(powers):.1f} - {max(powers):.1f} W")
    
    # 创建图表 - 3个子图排版
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('VLLM Unified Generation Test - Power and Frequency Timeline', fontsize=16, fontweight='bold')
    
    # 子图1: GPU功耗图
    ax1.plot(times, powers, 'b-', linewidth=2, label='Average Power')
    
    # 添加GPU最大功耗参考线
    max_power = max(powers) if powers else 0
    ax1.axhline(y=max_power, color='red', linestyle='--', alpha=0.7, label=f'Peak Power (100ms avg): {max_power:.1f}W')
    
    # 获取真正的峰值功耗（从原始GPU数据中）
    true_peak_power = get_true_peak_power(analyzed_intervals)
    if true_peak_power > max_power:
        ax1.axhline(y=true_peak_power, color='purple', linestyle='-', alpha=0.8, label=f'True Peak Power: {true_peak_power:.1f}W')
    
    # 获取GPU最大可用功耗
    gpu_max_power = get_gpu_max_power()
    ax1.axhline(y=gpu_max_power, color='orange', linestyle=':', alpha=0.7, label=f'GPU Max: {gpu_max_power:.0f}W')
    
    ax1.set_ylabel('Power (W)')
    ax1.set_title('GPU Power Timeline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: 图形时钟频率图
    ax2.plot(times, graphics_clocks, 'g-', linewidth=2, label='Graphics Clock')
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title('GPU Graphics Clock Frequency Timeline')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3: 内存时钟频率图
    ax3.plot(times, memory_clocks, 'r-', linewidth=2, label='Memory Clock')
    ax3.set_ylabel('Frequency (MHz)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('GPU Memory Clock Frequency Timeline')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 格式化x轴 - 使用相对时间
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, max(times) if times else 1)
        ax.set_xticks(range(0, int(max(times)) + 1, 1))
    
    plt.tight_layout()
    plt.savefig(storage_dir / "vllm_unified_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"时间线图表已保存到: {storage_dir / 'vllm_unified_timeline.png'}")

def main():
    """主测试函数"""
    print("VLLM 统一生成测试")
    print("=" * 50)
    
    # 创建客户端
    client = VLLMClient("http://localhost:8000")
    
    # 检查服务器状态
    print("检查服务器状态...")
    if not client.health_check():
        print("❌ 服务器未运行，请先启动 VLLM 服务器")
        print("运行命令: ./start_vllm_server.sh")
        return
    
    print("✅ 服务器运行正常")
    
    # 测试提示词
    test_prompts = [
        "1. 若3台机器5小时生产180个零件，7台机器8小时可生产多少零件？",
        "2. 甲比乙大6岁，5年前甲年龄是乙的2倍，求两人现在年龄。",
        "3. 编写一个Python函数计算斐波那契数列的第n项",
        "4. 解释什么是机器学习中的过拟合现象"
    ]
    
    # 创建存储目录
    storage_dir = Path("vllm_unified_results")
    storage_dir.mkdir(exist_ok=True)
    
    # 初始化 GPU 监控器 (100ms间隔)
    gpu_monitor = GPUMonitor(gpu_id=0, interval=0.1)
    
    # 启动 GPU 监控
    gpu_monitor.start_monitoring()
    
    # 执行统一测试
    print(f"\n开始统一生成测试，处理 {len(test_prompts)} 个提示词...")
    results = test_unified_generation(client, test_prompts, gpu_monitor, max_tokens=32)
    
    # 停止 GPU 监控
    gpu_monitor.stop_monitoring()
    
    # 获取GPU数据
    gpu_data = gpu_monitor.data
    gpu_stats = gpu_monitor.get_statistics()
    
    # 分析GPU数据并标注阶段
    analyzed_intervals = analyze_gpu_data_with_phases(gpu_data, results, interval_ms=100)
    
    # 保存结果
    unified_results = {
        "test_info": {
            "total_requests": len(results),
            "test_timestamp": time.time(),
            "gpu_monitor_interval": 0.1
        },
        "results": results,
        "gpu_statistics": gpu_stats,
        "analyzed_intervals": analyzed_intervals
    }
    
    with open(storage_dir / "unified_results.json", "w", encoding="utf-8") as f:
        json.dump(unified_results, f, ensure_ascii=False, indent=2)
    
    gpu_monitor.save_data(storage_dir / "gpu_data.json")
    
    # 创建时间线图表
    create_timeline_plot(analyzed_intervals, results, storage_dir)
    
    # 打印总结
    print("\n" + "=" * 50)
    print("统一测试完成！")
    
    if results:
        # 计算总体统计
        total_ttft = sum(r['ttft'] for r in results)
        total_tpot = sum(r['tpot'] for r in results)
        total_tokens = sum(r['token_count'] for r in results)
        total_time = sum(r['total_time'] for r in results)
        avg_ttft = total_ttft / len(results)
        avg_tpot = total_tpot / len(results)
        
        print(f"\n性能指标:")
        print(f"  平均 TTFT: {avg_ttft:.4f}s")
        print(f"  平均 TPOT: {avg_tpot:.4f}s")
        print(f"  总生成tokens: {total_tokens}")
        print(f"  总生成时间: {total_time:.4f}s")
        print(f"  平均生成速度: {total_tokens/total_time:.2f} tokens/s")
    
    if gpu_stats:
        print(f"\nGPU 统计:")
        print(f"  平均功耗: {gpu_stats['power_draw']['avg']:.1f}W")
        print(f"  最大功耗: {gpu_stats['power_draw']['max']:.1f}W")
        print(f"  平均GPU利用率: {gpu_stats['gpu_utilization']['avg']:.1f}%")
        print(f"  总能耗: {gpu_stats['power_draw']['total_energy']:.4f}kWh")
    
    if analyzed_intervals:
        prefill_intervals = [i for i in analyzed_intervals if i['phase'] == 'prefill']
        decode_intervals = [i for i in analyzed_intervals if i['phase'] == 'decode']
        
        print(f"\n阶段分析:")
        print(f"  Prefill 间隔数: {len(prefill_intervals)}")
        print(f"  Decode 间隔数: {len(decode_intervals)}")
        
        if prefill_intervals:
            prefill_power = sum(i['avg_power'] for i in prefill_intervals) / len(prefill_intervals)
            print(f"  Prefill 平均功耗: {prefill_power:.1f}W")
        
        if decode_intervals:
            decode_power = sum(i['avg_power'] for i in decode_intervals) / len(decode_intervals)
            print(f"  Decode 平均功耗: {decode_power:.1f}W")
    
    print(f"\n结果已保存到 {storage_dir} 目录")

if __name__ == "__main__":
    main()
