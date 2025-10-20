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
    def estimate_tokens(self, text):
        """估算文本的token数量（中文字符按2计算，英文按1计算）"""
        return sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in text)
    
    def health_check(self):
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    def generate_single(self, prompt, max_tokens=128, temperature=0.7, request_id=None):
        """发送单个流式生成请求"""
        payload = {
            "model": "/share-data/wzk-1/model/Qwen3-8B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        # 估算token数量
        estimated_tokens = self.estimate_tokens(prompt)
        # 立即显示请求开始
        print(f"\n[请求 {request_id+1}] 开始处理...")
        print(f"  提示词长度: {len(prompt)} 字符")
        print(f"  估算tokens: {estimated_tokens}")
        # 只显示原始问题部分，不显示填充内容
        original_prompt = prompt.split('\n\n')[0] if '\n\n' in prompt else prompt[:100]
        print(f"  问题: {original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}")
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
        
        # 打印提示词概览
        print("提示词概览:")
        for i, prompt in enumerate(prompts):
            estimated_tokens = self.estimate_tokens(prompt)
            print(f"  请求 {i+1}: {len(prompt)} 字符, ~{estimated_tokens} tokens")
            # 只显示原始问题部分，不显示填充内容
            original_prompt = prompt.split('\n\n')[0] if '\n\n' in prompt else prompt[:100]
            print(f"    问题: {original_prompt[:60]}{'...' if len(original_prompt) > 60 else ''}")
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
    parser.add_argument('--max-tokens', type=int, default=128, help='最大生成token数')
    parser.add_argument('--test-length', type=str, default='4096', 
                       choices=['1024', '2048', '4096', '8192', '16384', '32768'],
                       help='测试长度（token数）：1024, 2048, 4096, 8192, 16384, 32768')
    parser.add_argument('--preset', type=str, choices=['short', 'medium', 'long', 'xlong'],
                       help='预设配置: short(1024), medium(4096), long(8192), xlong(16384)')
    parser.add_argument('--output-dir', type=str, default='simple_batch_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 处理预设配置
    if args.preset:
        preset_configs = {
            'short': {'test_length': '1024', 'max_tokens': 64, 'requests': 8},
            'medium': {'test_length': '4096', 'max_tokens': 128, 'requests': 8},
            'long': {'test_length': '8192', 'max_tokens': 256, 'requests': 4},
            'xlong': {'test_length': '16384', 'max_tokens': 512, 'requests': 2}
        }
        
        config = preset_configs[args.preset]
        args.test_length = config['test_length']
        args.max_tokens = config['max_tokens']
        args.requests = config['requests']
        
        print(f"使用预设配置: {args.preset}")
        print(f"  测试长度: {args.test_length} tokens")
        print(f"  最大tokens: {args.max_tokens}")
        print(f"  总请求数: {args.requests}")
    
    print("简化的VLLM批量测试工具")
    print("=" * 50)
    print(f"批次大小: {args.batch_size}")
    print(f"总请求数: {args.requests}")
    print(f"最大tokens: {args.max_tokens}")
    print(f"测试长度: {args.test_length} tokens")
    
    # 创建客户端
    client = SimpleBatchClient("http://localhost:8000")
      
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
    
    def pad_prompt_to_length(prompt, target_length=4096):
        """将提示词填充到指定长度（以字符数估算，约4个字符=1个token）"""
        # 估算当前长度（中文字符按2计算，英文按1计算）
        current_length = sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in prompt)
        target_chars = target_length * 4  # 假设1个token约等于4个字符
        
        if current_length >= target_chars:
            return prompt
        
        # 填充内容
        padding_text = """
        
        这是一个用于测试长序列处理能力的填充文本。在自然语言处理中，序列长度是一个重要的参数，它直接影响模型的性能和计算资源消耗。较长的序列可以包含更多的上下文信息，但同时也需要更多的计算资源和内存。

        在深度学习模型中，特别是Transformer架构中，序列长度的平方关系使得长序列处理变得计算密集。因此，在实际应用中需要平衡序列长度和计算效率。

        对于语言模型来说，长序列处理能力是评估模型性能的重要指标之一。模型需要能够理解和处理长距离的依赖关系，这对于许多复杂的自然语言理解任务至关重要。

        在测试过程中，我们会监控GPU的功耗、频率和利用率，以了解不同序列长度对硬件资源消耗的影响。这对于优化模型部署和资源分配具有重要意义。

        填充文本继续：在机器学习和深度学习的实际应用中，序列长度是一个需要仔细考虑的超参数。不同的任务可能需要不同的序列长度，而模型的最大序列长度也受到硬件限制的约束。

        对于文本生成任务，较长的输入序列可以提供更多的上下文信息，有助于生成更相关和连贯的输出。然而，这也意味着需要更多的计算资源和更长的处理时间。

        在分布式训练和推理中，序列长度的选择还会影响通信开销和内存使用模式。因此，在实际部署中需要根据具体的硬件配置和性能要求来选择合适的序列长度。

        测试数据填充：为了确保测试的全面性，我们需要使用不同长度和复杂度的输入序列。这包括短序列、中等长度序列和长序列，以全面评估模型在各种情况下的性能表现。

        在性能测试中，我们关注的主要指标包括：延迟（latency）、吞吐量（throughput）、内存使用量、GPU利用率等。这些指标可以帮助我们了解模型在不同负载下的表现。

        序列长度的影响：较长的序列通常需要更多的注意力计算，这会导致二次方的时间复杂度。因此，在实际应用中，我们需要在模型性能和计算效率之间找到平衡点。

        对于预填充（prefill）阶段，长序列意味着需要处理更多的输入token，这通常需要更多的计算资源。而对于解码（decode）阶段，序列长度主要影响KV缓存的存储需求。

        在实际的LLM应用中，序列长度的选择往往受到以下因素的限制：1）模型的最大序列长度限制；2）可用GPU内存大小；3）推理延迟要求；4）批处理大小等。

        为了优化长序列处理，研究人员提出了多种技术，包括：注意力机制的优化、内存高效的注意力计算、序列并行化等。这些技术有助于在保持模型性能的同时提高计算效率。

        在测试过程中，我们会记录详细的性能指标，包括每个token的生成时间、GPU功耗变化、内存使用情况等。这些数据对于理解模型的行为和优化部署策略非常重要。

        填充文本继续：在实际的LLM部署中，序列长度的选择是一个需要综合考虑多个因素的决策过程。不同的应用场景可能有不同的序列长度需求。

        对于对话系统，通常需要较长的序列来维持对话的上下文。对于代码生成任务，可能需要处理较长的代码文件。对于文档摘要任务，输入序列可能包含整个文档的内容。

        因此，在设计和测试LLM系统时，我们需要考虑各种可能的序列长度，并确保系统能够在这些情况下稳定运行。

        测试数据填充结束：这是填充文本的最后部分，用于确保提示词达到指定的长度要求。通过这种方式，我们可以测试模型在处理长序列时的性能表现。
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
    with open(storage_dir / "batch_results.json", "w", encoding="utf-8") as f:
        json.dump(batch_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {storage_dir} 目录")

if __name__ == "__main__":
    main()
