#!/usr/bin/env python3
"""
VLLM 服务器客户端测试脚本
分离 prefill 和 decode 阶段进行功耗测试
"""

import requests
import json
import time
import threading
import sys
from pathlib import Path

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
    
    def generate(self, prompt, max_tokens=0, temperature=0.7, stream=False):
        """发送生成请求"""
        payload = {
            "model": "/share-data/wzk-1/model/deepseek-v2-lite",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"生成请求失败: {e}")
            return None

def test_prefill_only(client, prompts, gpu_monitor):
    """测试 prefill 阶段（使用最小 tokens 来模拟 prefill）"""
    print("\n=== Prefill 阶段测试 ===")
    
    gpu_monitor.start_monitoring()
    start_time = time.time()
    
    prefill_results = []
    for i, prompt in enumerate(prompts):
        print(f"处理提示词 {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # 为每个请求单独计时
        request_start = time.time()
        
        # 使用 max_tokens=1 来模拟 prefill，然后只统计 prompt_tokens
        result = client.generate(prompt, max_tokens=1)
        
        request_time = time.time() - request_start
        
        if result:
            # 从 VLLM API 响应中提取信息
            usage = result.get("usage", {})
            prefill_results.append({
                "prompt_id": i,
                "prompt": prompt,
                "prefill_time": request_time,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "result": result
            })
            print(f"  Prefill 完成，耗时: {request_time:.4f}s")
            print(f"    提示词tokens: {usage.get('prompt_tokens', 0)}")
            print(f"    生成tokens: {usage.get('completion_tokens', 0)} (仅用于模拟prefill)")
        else:
            print(f"  Prefill 失败")
    
    gpu_monitor.stop_monitoring()
    prefill_time_total = time.time() - start_time
    
    return prefill_results, prefill_time_total

def test_decode_only(client, prompts, gpu_monitor, max_tokens=16):
    """测试 decode 阶段"""
    print("\n=== Decode 阶段测试 ===")
    
    gpu_monitor.start_monitoring()
    start_time = time.time()
    
    decode_results = []
    for i, prompt in enumerate(prompts):
        print(f"处理提示词 {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # 为每个请求单独计时
        request_start = time.time()
        
        # 生成 token
        result = client.generate(prompt, max_tokens=max_tokens)
        
        request_time = time.time() - request_start
        
        if result:
            # 从 VLLM API 响应中提取信息
            choices = result.get("choices", [])
            generated_text = choices[0].get("text", "") if choices else ""
            usage = result.get("usage", {})
            decode_results.append({
                "prompt_id": i,
                "prompt": prompt,
                "generated_text": generated_text,
                "decode_time": request_time,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "result": result
            })
            print(f"  Decode 完成，耗时: {request_time:.4f}s")
            print(f"    生成内容: {generated_text[:50]}...")
            print(f"    提示词tokens: {usage.get('prompt_tokens', 0)}")
            print(f"    生成tokens: {usage.get('completion_tokens', 0)}")
            print(f"    总tokens: {usage.get('total_tokens', 0)}")
        else:
            print(f"  Decode 失败")
    
    gpu_monitor.stop_monitoring()
    decode_time_total = time.time() - start_time
    
    return decode_results, decode_time_total

def main():
    """主测试函数"""
    print("VLLM 服务器客户端测试")
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
    storage_dir = Path("vllm_server_results")
    storage_dir.mkdir(exist_ok=True)
    
    # 初始化 GPU 监控器
    gpu_monitor = GPUMonitor(gpu_id=0, interval=0.1)
    
    # 测试 1: Prefill 阶段
    print(f"\n开始 Prefill 阶段测试，处理 {len(test_prompts)} 个提示词...")
    prefill_results, prefill_time_total = test_prefill_only(client, test_prompts, gpu_monitor)
    
    # 保存 prefill 结果
    prefill_stats = gpu_monitor.get_statistics()
    with open(storage_dir / "prefill_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_prefill_time": prefill_time_total,
            "gpu_statistics": prefill_stats,
            "results": prefill_results
        }, f, ensure_ascii=False, indent=2)
    
    gpu_monitor.save_data(storage_dir / "prefill_gpu_data.json")
    
    # 等待一下
    time.sleep(2)
    
    # 测试 2: Decode 阶段
    print(f"\n开始 Decode 阶段测试，处理 {len(test_prompts)} 个提示词...")
    decode_results, decode_time_total = test_decode_only(client, test_prompts, gpu_monitor, max_tokens=16)
    
    # 保存 decode 结果
    decode_stats = gpu_monitor.get_statistics()
    with open(storage_dir / "decode_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_decode_time": decode_time_total,
            "gpu_statistics": decode_stats,
            "results": decode_results
        }, f, ensure_ascii=False, indent=2)
    
    gpu_monitor.save_data(storage_dir / "decode_gpu_data.json")
    
    # 打印总结
    print("\n" + "=" * 50)
    print("测试完成！")
    print(f"Prefill 阶段总时间: {prefill_time_total:.4f}s")
    print(f"Decode 阶段总时间: {decode_time_total:.4f}s")
    
    if prefill_stats:
        print(f"\nPrefill 阶段 GPU 统计:")
        print(f"  平均功耗: {prefill_stats['power_draw']['avg']:.1f}W")
        print(f"  最大功耗: {prefill_stats['power_draw']['max']:.1f}W")
        print(f"  平均GPU利用率: {prefill_stats['gpu_utilization']['avg']:.1f}%")
        print(f"  总能耗: {prefill_stats['power_draw']['total_energy']:.4f}kWh")
        
        # 计算 token 统计（只统计 prompt_tokens，因为这是真正的 prefill 工作）
        total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in prefill_results)
        total_completion_tokens = sum(r.get('completion_tokens', 0) for r in prefill_results)
        individual_times = [r.get('prefill_time', 0) for r in prefill_results]
        avg_request_time = sum(individual_times) / len(individual_times) if individual_times else 0
        
        print(f"  总提示词tokens: {total_prompt_tokens} (真正的prefill工作)")
        print(f"  总生成tokens: {total_completion_tokens} (仅用于模拟，不计入prefill)")
        print(f"  平均每个请求耗时: {avg_request_time:.4f}s")
        print(f"  总批次时间: {prefill_time_total:.4f}s (包含网络延迟等)")
        if prefill_time_total > 0:
            print(f"  Prefill速度: {total_prompt_tokens / prefill_time_total:.2f} prompt_tokens/s")
    
    if decode_stats:
        print(f"\nDecode 阶段 GPU 统计:")
        print(f"  平均功耗: {decode_stats['power_draw']['avg']:.1f}W")
        print(f"  最大功耗: {decode_stats['power_draw']['max']:.1f}W")
        print(f"  平均GPU利用率: {decode_stats['gpu_utilization']['avg']:.1f}%")
        print(f"  总能耗: {decode_stats['power_draw']['total_energy']:.4f}kWh")
        
        # 计算 token 统计
        total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in decode_results)
        total_completion_tokens = sum(r.get('completion_tokens', 0) for r in decode_results)
        individual_times = [r.get('decode_time', 0) for r in decode_results]
        avg_request_time = sum(individual_times) / len(individual_times) if individual_times else 0
        
        print(f"  总提示词tokens: {total_prompt_tokens}")
        print(f"  总生成tokens: {total_completion_tokens} (真正的decode工作)")
        print(f"  平均每个请求耗时: {avg_request_time:.4f}s")
        print(f"  总批次时间: {decode_time_total:.4f}s (包含网络延迟等)")
        if decode_time_total > 0:
            print(f"  Decode速度: {total_completion_tokens / decode_time_total:.2f} completion_tokens/s")
    
    print(f"\n结果已保存到 {storage_dir} 目录")

if __name__ == "__main__":
    main()
