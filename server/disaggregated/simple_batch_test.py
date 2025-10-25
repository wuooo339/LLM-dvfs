#!/usr/bin/env python3
"""
Disaggregated Prefill+Decode 批量测试脚本
测试分离式架构（Prefill 和 Decode 分离）的性能
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
try:
    from gpu_monitor import GPUMonitor
except ImportError:
    print("警告: 无法导入 GPUMonitor，GPU监控功能将不可用")
    GPUMonitor = None

class DisaggBatchClient:
    """Disaggregated 架构批量测试客户端"""
    def __init__(self, proxy_url="http://localhost:8000", prefill_url="http://localhost:8100", decode_url="http://localhost:8200"):
        self.proxy_url = proxy_url
        self.prefill_url = prefill_url
        self.decode_url = decode_url
        self.session = requests.Session()
        
    def estimate_tokens(self, text):
        """估算文本的token数量（中文字符按2计算，英文按1计算）"""
        return sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in text)
    
    def health_check(self):
        """检查所有服务器健康状态"""
        print("检查服务健康状态...")
        all_healthy = True
        
        # 检查代理服务器
        try:
            response = self.session.get(f"{self.proxy_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"  ✅ 代理服务器: 正常")
                print(f"     Prefill实例数: {health_data.get('prefill_instances', 0)}")
                print(f"     Decode实例数: {health_data.get('decode_instances', 0)}")
                print(f"     Prefill地址: {health_data.get('prefill_addrs', [])}")
                print(f"     Decode地址: {health_data.get('decode_addrs', [])}")
            else:
                print(f"  ❌ 代理服务器: 异常 (状态码 {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"  ❌ 代理服务器: 无法连接 ({e})")
            all_healthy = False
        
        # 检查 Prefill 服务器
        try:
            response = self.session.get(f"{self.prefill_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ Prefill服务器: 正常")
            else:
                print(f"  ⚠️  Prefill服务器: 状态码 {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Prefill服务器: 无法直接连接 ({e})")
        
        # 检查 Decode 服务器
        try:
            response = self.session.get(f"{self.decode_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ Decode服务器: 正常")
            else:
                print(f"  ⚠️  Decode服务器: 状态码 {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Decode服务器: 无法直接连接 ({e})")
        
        return all_healthy
    
    def generate_single(self, prompt, max_tokens=128, temperature=0.7, request_id=None):
        """发送单个流式生成请求（通过代理服务器，自动分离 prefill 和 decode）"""
        payload = {
            "model": "/share-data/wzk-1/model/Qwen3-4B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        # 估算token数量
        estimated_tokens = self.estimate_tokens(prompt)
        # 立即显示请求开始
        print(f"\n[请求 {request_id+1}] 开始处理（Disaggregated模式）...")
        print(f"  提示词长度: {len(prompt)} 字符")
        print(f"  估算tokens: {estimated_tokens}")
        # 只显示原始问题部分，不显示填充内容
        original_prompt = prompt.split('\n\n')[0] if '\n\n' in prompt else prompt[:100]
        print(f"  问题: {original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}")
        print(f"  → 流程: Prefill阶段 → KV cache传输 → Decode阶段")
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
                f"{self.proxy_url}/v1/completions",
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
                                    
                                    # 实时显示token内容和时间
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
            print(f"  TTFT (首token延迟): {ttft:.3f}s")
            print(f"  生成tokens: {token_count}")
            if token_count > 1:
                tpot = (total_time - ttft) / (token_count - 1) if token_count > 1 else 0
                print(f"  TPOT (每token时间): {tpot*1000:.2f}ms")
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
            
        print(f"\n" + "="*80)
        print(f"开始 Disaggregated Prefill+Decode 批量测试")
        print(f"="*80)
        print(f"架构: Prefill 和 Decode 分离")
        print(f"总请求数: {len(prompts)}")
        print(f"批次大小: {batch_size} (同时发送的请求数)")
        print(f"最大tokens: {max_tokens}")
        print("=" * 80)
        
        # 打印提示词概览
        print("提示词概览:")
        for i, prompt in enumerate(prompts):
            estimated_tokens = self.estimate_tokens(prompt)
            print(f"  请求 {i+1}: {len(prompt)} 字符, ~{estimated_tokens} tokens")
            # 只显示原始问题部分，不显示填充内容
            original_prompt = prompt.split('\n\n')[0] if '\n\n' in prompt else prompt[:100]
            print(f"    问题: {original_prompt[:60]}{'...' if len(original_prompt) > 60 else ''}")
        print("=" * 80)
        
        all_results = []
        start_time = time.time()
        
        # 按批次处理
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            print(f"\n处理批次 {batch_start//batch_size + 1}: 请求 {batch_start+1}-{batch_end}")
            print("=" * 80)
            
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
                avg_ttft = sum(r['ttft'] for r in batch_results if r['success']) / max(successful, 1)
                
                print(f"\n批次 {batch_start//batch_size + 1} 完成:")
                print(f"  成功: {successful}/{len(batch_prompts)}")
                print(f"  失败: {failed}")
                print(f"  平均响应时间: {avg_time:.3f}s")
                print(f"  平均TTFT: {avg_ttft:.3f}s")
                print("=" * 80)
        
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
                token_throughput = total_tokens / total_time
            else:
                avg_tokens = total_tokens = token_throughput = 0
        else:
            avg_response_time = min_response_time = max_response_time = 0
            avg_ttft = min_ttft = max_ttft = 0
            avg_tokens = total_tokens = token_throughput = 0
            throughput = 0
        
        # 打印最终统计
        print(f"\n" + "="*80)
        print(f"最终统计 (Disaggregated Prefill+Decode 架构)")
        print(f"="*80)
        print(f"📊 请求统计:")
        print(f"  总请求数: {len(prompts)}")
        print(f"  成功请求: {len(successful_results)}")
        print(f"  失败请求: {len(failed_results)}")
        print(f"  成功率: {len(successful_results)/len(prompts)*100:.1f}%")
        print(f"\n⏱️  延迟统计:")
        print(f"  总时间: {total_time:.3f}s")
        print(f"  平均响应时间: {avg_response_time:.3f}s")
        print(f"  最小响应时间: {min_response_time:.3f}s")
        print(f"  最大响应时间: {max_response_time:.3f}s")
        print(f"  平均TTFT: {avg_ttft:.3f}s")
        print(f"  最小TTFT: {min_ttft:.3f}s")
        print(f"  最大TTFT: {max_ttft:.3f}s")
        print(f"\n🚀 吞吐量统计:")
        print(f"  总生成tokens: {total_tokens}")
        print(f"  平均tokens/请求: {avg_tokens:.1f}")
        print(f"  请求吞吐量: {throughput:.2f} 请求/秒")
        print(f"  Token吞吐量: {token_throughput:.2f} tokens/秒")
        print(f"="*80)
        
        return {
            'architecture': 'disaggregated',
            'total_requests': len(prompts),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'total_time': total_time,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'avg_ttft': avg_ttft,
            'min_ttft': min_ttft,
            'max_ttft': max_ttft,
            'total_tokens': total_tokens,
            'avg_tokens': avg_tokens,
            'throughput': throughput,
            'token_throughput': token_throughput,
            'results': all_results
        }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Disaggregated Prefill+Decode 批量测试工具')
    parser.add_argument('--batch-size', type=int, required=True, help='批次大小（同时发送的请求数）')
    parser.add_argument('--requests', type=int, default=8, help='总请求数')
    parser.add_argument('--max-tokens', type=int, default=128, help='最大生成token数')
    parser.add_argument('--test-length', type=str, default='2048', 
                       choices=['1024', '2048', '4096', '8192'],
                       help='测试长度（token数）：1024, 2048, 4096, 8192')
    parser.add_argument('--preset', type=str, choices=['short', 'medium', 'long'],
                       help='预设配置: short(1024), medium(2048), long(4096)')
    parser.add_argument('--proxy-url', type=str, default='http://localhost:8000', 
                       help='代理服务器URL')
    parser.add_argument('--prefill-url', type=str, default='http://localhost:8100',
                       help='Prefill服务器URL（用于健康检查）')
    parser.add_argument('--decode-url', type=str, default='http://localhost:8200',
                       help='Decode服务器URL（用于健康检查）')
    parser.add_argument('--output-dir', type=str, default='disagg_batch_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 处理预设配置
    if args.preset:
        preset_configs = {
            'short': {'test_length': '1024', 'max_tokens': 64, 'requests': 8},
            'medium': {'test_length': '2048', 'max_tokens': 128, 'requests': 8},
            'long': {'test_length': '4096', 'max_tokens': 256, 'requests': 4}
        }
        
        config = preset_configs[args.preset]
        args.test_length = config['test_length']
        args.max_tokens = config['max_tokens']
        args.requests = config['requests']
        
        print(f"使用预设配置: {args.preset}")
        print(f"  测试长度: {args.test_length} tokens")
        print(f"  最大tokens: {args.max_tokens}")
        print(f"  总请求数: {args.requests}")
    
    print("\n" + "="*80)
    print("Disaggregated Prefill+Decode 批量测试工具")
    print("="*80)
    print(f"架构模式: Prefill 和 Decode 分离")
    print(f"代理服务器: {args.proxy_url}")
    print(f"批次大小: {args.batch_size}")
    print(f"总请求数: {args.requests}")
    print(f"最大tokens: {args.max_tokens}")
    print(f"测试长度: {args.test_length} tokens")
    print("="*80)
    
    # 创建客户端
    client = DisaggBatchClient(args.proxy_url, args.prefill_url, args.decode_url)
    
    # 健康检查
    if not client.health_check():
        print("\n⚠️  警告: 部分服务未就绪，但继续测试...")
    
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
    
    def pad_prompt_to_length(prompt, target_length=2048):
        """将提示词填充到指定长度（以字符数估算，约4个字符=1个token）"""
        # 估算当前长度（中文字符按2计算，英文按1计算）
        current_length = sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in prompt)
        target_chars = target_length * 4  # 假设1个token约等于4个字符
        
        if current_length >= target_chars:
            return prompt
        
        # 填充内容
        padding_text = """

这是一个用于测试长序列处理能力的填充文本。在disaggregated prefill+decode架构中，prefill阶段负责处理输入序列并生成KV cache，然后将KV cache传输到decode阶段进行token生成。

这种分离架构的优势在于：1）prefill和decode可以使用不同的GPU资源；2）可以优化各自的批处理策略；3）提高整体资源利用率。

在测试中，我们会关注：prefill延迟、KV cache传输时间、decode吞吐量等关键指标。这些指标对于评估disaggregated架构的性能非常重要。

填充文本继续：在实际应用中，disaggregated架构特别适合处理长序列输入的场景。因为prefill阶段可以专注于并行处理长输入序列，而decode阶段可以优化连续token生成的效率。

对于不同的序列长度，系统的行为可能会有所不同。较短的序列可能不会充分体现分离架构的优势，而较长的序列则能更好地展示prefill和decode分离带来的性能提升。

在性能测试中，我们需要观察：1）TTFT（首token延迟）是否受prefill和KV传输影响；2）后续token生成速度是否稳定；3）多请求并发时的资源调度效率。

测试数据填充：为确保测试的全面性，我们使用不同长度和复杂度的输入序列，以评估disaggregated架构在各种场景下的性能表现。
"""
        
        # 计算需要添加的填充文本长度
        remaining_chars = target_chars - current_length
        if remaining_chars > 0:
            # 重复填充文本直到达到目标长度
            padded_prompt = prompt
            while len(padded_prompt) < target_chars:
                padded_prompt += padding_text
            return padded_prompt[:target_chars]
        
        return prompt
    
    # 将每个提示词填充到指定长度
    test_length = int(args.test_length)
    padded_prompts = [pad_prompt_to_length(prompt, test_length) for prompt in base_prompts]
    
    # 根据请求数量生成提示词
    test_prompts = (padded_prompts * ((args.requests // len(padded_prompts)) + 1))[:args.requests]
    
    # 创建存储目录
    storage_dir = Path(args.output_dir)
    storage_dir.mkdir(exist_ok=True)
    
    # 执行批量测试
    print(f"\n开始批量测试...")
    batch_stats = client.generate_batch(test_prompts, max_tokens=args.max_tokens, batch_size=args.batch_size)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = storage_dir / f"batch_results_{timestamp}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(batch_stats, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ 结果已保存到 {result_file}")

if __name__ == "__main__":
    main()

