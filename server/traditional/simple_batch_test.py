#!/usr/bin/env python3
"""
简化的VLLM批量测试脚本
只保留批次调节功能，批次数就是同时发送的请求数
"""

import requests
import json
import time
import threading
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加父目录到路径，以便导入 gpu_monitor
sys.path.append(str(Path(__file__).parent.parent.parent))
from gpu_monitor import GPUMonitor

class SimpleBatchClient:
    """简化的批量测试客户端"""
    
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
    
    def generate_single(self, prompt, max_tokens=64, temperature=0.7, request_id=None):
        """发送单个流式生成请求"""
        payload = {
            "model": "/home/wzk/deepseek-v2-lite",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        # 立即显示请求开始
        print(f"\n[请求 {request_id+1}] 开始处理...")
        print(f"  提示词: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"  生成内容: ", end="")
        sys.stdout.flush()
        
        try:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            generated_text = ""
            token_times = []
            
            # 根据max_tokens动态调整超时时间
            timeout = max(30, max_tokens * 2)
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()
            
            # 处理流式响应
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
                                    
                                    # 记录首token时间
                                    if first_token_time is None:
                                        first_token_time = current_time - start_time
                                    
                                    # 记录每个token的时间
                                    token_times.append(current_time - start_time)
                                    
                                    generated_text += text_content
                                    
                                    # 实时显示token内容和时间，格式类似test_vllm_unified.py
                                    display_text = text_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                    if len(display_text) > 20:
                                        display_text = display_text[:20] + "..."
                                    print(f"    [请求 {request_id+1}] Token {token_count}: '{display_text}' (时间: {current_time - start_time:.4f}s)")
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            
            # 计算时间指标
            total_time = end_time - start_time
            ttft = first_token_time if first_token_time else 0
            
            print(f"\n[请求 {request_id+1}] ✅ 完成")
            print(f"  总时间: {total_time:.3f}s")
            print(f"  TTFT: {ttft:.3f}s")
            print(f"  生成tokens: {token_count}")
            print("-" * 60)
            sys.stdout.flush()
            
            return {
                'request_id': request_id,
                'prompt': prompt,
                'generated_text': generated_text,
                'response_time': total_time,
                'ttft': ttft,
                'token_count': token_count,
                'token_times': token_times,
                'success': True,
                'error': None
            }
        except Exception as e:
            error_msg = str(e)
            print(f"\n[请求 {request_id+1}] ❌ 失败")
            print(f"  错误: {error_msg}")
            print("-" * 60)
            sys.stdout.flush()
            return {
                'request_id': request_id,
                'prompt': prompt,
                'generated_text': '',
                'response_time': 0,
                'ttft': 0,
                'token_count': 0,
                'token_times': [],
                'success': False,
                'error': error_msg
            }
    
    def generate_batch(self, prompts, max_tokens=64, temperature=0.7, batch_size=None):
        """批量生成请求 - 批次数就是同时发送的请求数"""
        if batch_size is None:
            batch_size = len(prompts)
        
        print(f"开始批量生成测试")
        print(f"总请求数: {len(prompts)}")
        print(f"批次大小: {batch_size} (同时发送的请求数)")
        print(f"最大tokens: {max_tokens}")
        print("=" * 60)
        
        # 打印所有提示词
        for i, prompt in enumerate(prompts):
            print(f"请求 {i+1}: {prompt}")
        print("=" * 60)
        
        all_results = []
        start_time = time.time()
        
        # 按批次处理
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            print(f"\n处理批次 {batch_start//batch_size + 1}: 请求 {batch_start+1}-{batch_end}")
            print("=" * 60)
            
            # 使用ThreadPoolExecutor处理当前批次
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # 提交当前批次的所有任务
                future_to_prompt = {
                    executor.submit(self.generate_single, prompt, max_tokens, temperature, batch_start + i): prompt
                    for i, prompt in enumerate(batch_prompts)
                }
                
                # 实时等待并显示结果
                batch_results = []
                completed_count = 0
                for future in as_completed(future_to_prompt):
                    result = future.result()
                    batch_results.append(result)
                    all_results.append(result)
                    completed_count += 1
                    
                    # 实时显示进度
                    print(f"📊 批次进度: {completed_count}/{len(batch_prompts)} 完成")
                    sys.stdout.flush()
                
                # 打印批次统计
                successful = sum(1 for r in batch_results if r['success'])
                failed = len(batch_results) - successful
                avg_time = sum(r['response_time'] for r in batch_results if r['success']) / max(successful, 1)
                
                print(f"\n批次 {batch_start//batch_size + 1} 完成:")
                print(f"  成功: {successful}/{len(batch_prompts)}")
                print(f"  失败: {failed}")
                print(f"  平均响应时间: {avg_time:.3f}s")
                print("=" * 60)
        
        total_time = time.time() - start_time
        
        # 计算总体统计
        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]
        
        if successful_results:
            response_times = [r['response_time'] for r in successful_results]
            ttft_times = [r['ttft'] for r in successful_results if 'ttft' in r]
            token_counts = [r['token_count'] for r in successful_results if 'token_count' in r]
            
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            throughput = len(successful_results) / total_time
            
            if ttft_times:
                avg_ttft = sum(ttft_times) / len(ttft_times)
                min_ttft = min(ttft_times)
                max_ttft = max(ttft_times)
            else:
                avg_ttft = min_ttft = max_ttft = 0
                
            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                total_tokens = sum(token_counts)
            else:
                avg_tokens = total_tokens = 0
        else:
            avg_response_time = min_response_time = max_response_time = 0
            avg_ttft = min_ttft = max_ttft = 0
            avg_tokens = total_tokens = 0
            throughput = 0
        
        # 打印最终统计
        print(f"\n最终统计:")
        print(f"  总请求数: {len(prompts)}")
        print(f"  成功请求: {len(successful_results)}")
        print(f"  失败请求: {len(failed_results)}")
        print(f"  成功率: {len(successful_results)/len(prompts)*100:.1f}%")
        print(f"  总时间: {total_time:.3f}s")
        print(f"  平均响应时间: {avg_response_time:.3f}s")
        print(f"  最小响应时间: {min_response_time:.3f}s")
        print(f"  最大响应时间: {max_response_time:.3f}s")
        print(f"  平均TTFT: {avg_ttft:.3f}s")
        print(f"  最小TTFT: {min_ttft:.3f}s")
        print(f"  最大TTFT: {max_ttft:.3f}s")
        print(f"  总生成tokens: {total_tokens}")
        print(f"  平均tokens/请求: {avg_tokens:.1f}")
        print(f"  吞吐量: {throughput:.2f} 请求/秒")
        
        return {
            'total_requests': len(prompts),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'total_time': total_time,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'throughput': throughput,
            'results': all_results
        }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='简化的VLLM批量测试工具')
    parser.add_argument('--batch-size', type=int, required=True, help='批次大小（同时发送的请求数）')
    parser.add_argument('--requests', type=int, default=8, help='总请求数')
    parser.add_argument('--max-tokens', type=int, default=64, help='最大生成token数')
    parser.add_argument('--output-dir', type=str, default='simple_batch_results', help='输出目录')
    
    args = parser.parse_args()
    
    print("简化的VLLM批量测试工具")
    print("=" * 50)
    print(f"批次大小: {args.batch_size}")
    print(f"总请求数: {args.requests}")
    print(f"最大tokens: {args.max_tokens}")
    
    # 创建客户端
    client = SimpleBatchClient("http://localhost:8000")
    
    # 检查服务器状态
    print("\n检查服务器状态...")
    if not client.health_check():
        print("❌ 服务器未运行，请先启动 VLLM 服务器")
        print("运行命令: ./start_vllm_server.sh")
        return
    
    print("✅ 服务器运行正常")
    
    # 生成测试提示词
    base_prompts = [
        "1. 若3台机器5小时生产180个零件，7台机器8小时可生产多少零件？",
        "2. 甲比乙大6岁，5年前甲年龄是乙的2倍，求两人现在年龄。",
        "3. 编写一个Python函数计算斐波那契数列的第n项",
        "4. 解释什么是机器学习中的过拟合现象",
        "5. 什么是深度学习中的注意力机制？",
        "6. 如何优化数据库查询性能？",
        "7. 解释什么是微服务架构",
        "8. 什么是容器化技术？"
    ]
    
    # 根据请求数量生成提示词
    test_prompts = (base_prompts * ((args.requests // len(base_prompts)) + 1))[:args.requests]
    
    # 创建存储目录
    storage_dir = Path(args.output_dir)
    storage_dir.mkdir(exist_ok=True)
    
    # 执行批量测试
    print(f"\n开始批量测试...")
    batch_stats = client.generate_batch(test_prompts, max_tokens=args.max_tokens, batch_size=args.batch_size)
    
    # 保存结果
    with open(storage_dir / "batch_results.json", "w", encoding="utf-8") as f:
        json.dump(batch_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {storage_dir} 目录")

if __name__ == "__main__":
    main()
