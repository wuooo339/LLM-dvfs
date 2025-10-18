#!/usr/bin/env python3
"""
BIG-Bench Hard (BBH) 数据集测试脚本
使用BBH数据集进行长输入测试，测量功耗和性能
"""

import requests
import time
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加上级目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from gpu_monitor import GPUMonitor

# 尝试导入datasets库
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️  datasets库未安装，将使用内置示例数据")
    print("   安装命令: pip install datasets")
    DATASETS_AVAILABLE = False

class BBHTestClient:
    """BBH数据集测试客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "BBH-Test-Client/1.0"
        })
    
    def health_check(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[Dict]:
        """发送生成请求"""
        payload = {
            "model": "/share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=120
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # 实时打印prompt和输出信息
                prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)
                completion_tokens = result.get('usage', {}).get('completion_tokens', 0)
                total_tokens = result.get('usage', {}).get('total_tokens', 0)
                response_text = result.get('choices', [{}])[0].get('text', '')
                
                print(f"      📝 Prompt tokens: {prompt_tokens}")
                print(f"      📤 Completion tokens: {completion_tokens}")
                print(f"      📊 Total tokens: {total_tokens}")
                print(f"      ⏱️  Request time: {end_time - start_time:.2f}s")
                print(f"      💬 Response preview: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                
                return {
                    "response": result,
                    "request_time": end_time - start_time,
                    "timestamp": start_time,
                    "prompt_length": len(prompt),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            else:
                print(f"请求失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"生成请求失败: {e}")
            return None

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """发送流式生成请求"""
        payload = {
            "model": "/share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        # 简单重试：在并发高时提高稳健性
        last_err = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=300,
                    stream=True
                )
                response.raise_for_status()
                return response
            except Exception as e:
                last_err = e
                print(f"流式生成请求失败(第{attempt+1}次): {e}")
                time.sleep(1.0 * (attempt + 1))
        print(f"流式生成请求最终失败: {last_err}")
        return None

def load_bbh_tasks() -> List[Dict[str, Any]]:
    """加载BBH任务数据"""
    if DATASETS_AVAILABLE:
        return load_bbh_tasks_from_official()
    else:
        return load_bbh_tasks_fallback()

def load_bbh_tasks_from_official() -> List[Dict[str, Any]]:
    """从官方数据集加载BBH任务"""
    print("📚 从官方BIG-Bench Hard数据集加载任务...")
    
    # BBH官方数据集中的主要任务
    bbh_task_names = [
        "logical_deduction_three_objects",
        "date_understanding", 
        "causal_judgment",
        "disambiguation_qa",
        "geometric_shapes",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "multistep_arithmetic_two",
        "reasoning_about_colored_objects",
        "navigate",
        "ruin_names",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects"
    ]
    
    bbh_tasks = []
    
    for task_name in bbh_task_names:
        try:
            print(f"  加载任务: {task_name}")
            # 加载特定任务
            dataset = load_dataset("bigbenchhard", task_name)
            
            # 获取训练数据
            train_data = dataset['train']
            
            # 随机选择一些示例（最多5个）
            examples = []
            if len(train_data) > 0:
                # 随机选择示例
                selected_indices = random.sample(range(len(train_data)), min(5, len(train_data)))
                for idx in selected_indices:
                    example = train_data[idx]
                    examples.append({
                        "input": example['input'],
                        "target": example['target']
                    })
            
            if examples:
                bbh_tasks.append({
                    "task_name": task_name,
                    "description": f"BIG-Bench Hard: {task_name}",
                    "examples": examples,
                    "total_examples": len(train_data)
                })
                print(f"    ✅ 加载了 {len(examples)} 个示例 (总共 {len(train_data)} 个)")
            else:
                print(f"    ⚠️  任务 {task_name} 没有可用示例")
                
        except Exception as e:
            print(f"    ❌ 加载任务 {task_name} 失败: {e}")
            continue
    
    print(f"📊 成功加载 {len(bbh_tasks)} 个BBH任务")
    return bbh_tasks

def load_bbh_tasks_fallback() -> List[Dict[str, Any]]:
    """加载内置的BBH任务示例数据（当官方数据集不可用时）"""
    print("📚 使用内置BBH任务示例数据...")
    bbh_tasks = [
        {
            "task_name": "causal_judgment",
            "description": "因果判断任务",
            "examples": [
                {
                    "input": "A man named John goes to a restaurant and orders a hamburger. The hamburger is poisoned. John eats the hamburger and dies. What was the cause of John's death?",
                    "target": "The poisoned hamburger"
                },
                {
                    "input": "A woman named Sarah is driving her car on a rainy day. She hits a patch of ice and crashes into a tree. What was the cause of the crash?",
                    "target": "The ice on the road"
                }
            ]
        },
        {
            "task_name": "date_understanding",
            "description": "日期理解任务",
            "examples": [
                {
                    "input": "Today is the 1st of January, 2023. What will be the date 3 days from now?",
                    "target": "January 4th, 2023"
                },
                {
                    "input": "If today is March 15th, 2023, what was the date 2 weeks ago?",
                    "target": "March 1st, 2023"
                }
            ]
        },
        {
            "task_name": "disambiguation_qa",
            "description": "消歧问答任务",
            "examples": [
                {
                    "input": "The word 'bank' can refer to a financial institution or the side of a river. In the sentence 'I went to the bank to deposit money', what does 'bank' mean?",
                    "target": "A financial institution"
                },
                {
                    "input": "The word 'bark' can refer to the sound a dog makes or the outer covering of a tree. In the sentence 'The dog's bark woke me up', what does 'bark' mean?",
                    "target": "The sound a dog makes"
                }
            ]
        },
        {
            "task_name": "geometric_shapes",
            "description": "几何形状任务",
            "examples": [
                {
                    "input": "A square has 4 sides of equal length. If one side is 5 cm long, what is the perimeter of the square?",
                    "target": "20 cm"
                },
                {
                    "input": "A circle has a radius of 3 cm. What is the area of the circle? (Use π ≈ 3.14)",
                    "target": "28.26 cm²"
                }
            ]
        },
        {
            "task_name": "logical_deduction",
            "description": "逻辑推理任务",
            "examples": [
                {
                    "input": "All birds can fly. Penguins are birds. Can penguins fly?",
                    "target": "No, penguins cannot fly (this is a logical paradox)"
                },
                {
                    "input": "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                    "target": "Not necessarily - the ground could be wet for other reasons"
                }
            ]
        },
        {
            "task_name": "multistep_arithmetic",
            "description": "多步算术任务",
            "examples": [
                {
                    "input": "Sarah has 12 apples. She gives 3 apples to her friend and buys 7 more apples. How many apples does Sarah have now?",
                    "target": "16 apples"
                },
                {
                    "input": "A store has 50 items. They sell 15 items in the morning and 20 items in the afternoon. How many items are left?",
                    "target": "15 items"
                }
            ]
        },
        {
            "task_name": "navigate",
            "description": "导航任务",
            "examples": [
                {
                    "input": "You are at point A. To get to point B, you need to go 3 blocks north, then 2 blocks east, then 1 block south. What direction should you start walking?",
                    "target": "North"
                },
                {
                    "input": "Starting from home, you walk 2 blocks west, then 3 blocks north, then 1 block east. How many blocks are you from home?",
                    "target": "3 blocks (using Pythagorean theorem: √(1² + 3²) = √10 ≈ 3.16)"
                }
            ]
        },
        {
            "task_name": "reasoning_about_colored_objects",
            "description": "颜色对象推理任务",
            "examples": [
                {
                    "input": "There are 3 boxes: a red box, a blue box, and a green box. The red box contains a ball. The blue box is empty. What color is the box that contains the ball?",
                    "target": "Red"
                },
                {
                    "input": "A red car and a blue car are parked. The red car is faster than the blue car. Which car is faster?",
                    "target": "The red car"
                }
            ]
        },
        {
            "task_name": "ruin_names",
            "description": "名字推理任务",
            "examples": [
                {
                    "input": "If you rearrange the letters in 'LISTEN', what word do you get?",
                    "target": "SILENT"
                },
                {
                    "input": "What word can be formed by rearranging the letters in 'HEART'?",
                    "target": "EARTH"
                }
            ]
        },
        {
            "task_name": "snarks",
            "description": "讽刺理解任务",
            "examples": [
                {
                    "input": "John said 'Great weather we're having' while it was pouring rain. What did John really mean?",
                    "target": "He was being sarcastic - the weather is terrible"
                },
                {
                    "input": "Sarah said 'Oh, that's just perfect' when her computer crashed. What did Sarah really mean?",
                    "target": "She was being sarcastic - this is the worst possible timing"
                }
            ]
        }
    ]
    
    return bbh_tasks

def create_long_prompt(task: Dict[str, Any], example: Dict[str, str], context_length: int = 2000) -> str:
    """创建长输入提示"""
    base_prompt = f"""
Task: {task['description']}
Task Name: {task['task_name']}

Instructions: Please solve the following problem step by step. Think carefully and provide a clear, logical answer.

Problem: {example['input']}

Please provide your answer and explain your reasoning process.
"""
    
    # 添加额外的上下文来增加输入长度
    context_padding = f"""

Additional Context:
This is a complex reasoning task that requires careful analysis. You should:
1. Read the problem carefully
2. Identify the key information
3. Apply logical reasoning
4. Consider alternative interpretations
5. Provide a well-reasoned answer

The problem involves multiple steps of reasoning and may require you to consider various factors before arriving at the correct answer. Take your time to think through each step carefully.

Remember to:
- Break down complex problems into smaller parts
- Consider all given information
- Apply logical rules consistently
- Check your reasoning for errors
- Provide clear explanations for your answer

Now, please solve the problem:
"""
    
    # 如果上下文还不够长，添加更多内容
    while len(base_prompt + context_padding) < context_length:
        context_padding += f"""

Additional reasoning guidelines:
- Pay attention to details in the problem statement
- Consider edge cases and exceptions
- Use systematic approaches to problem-solving
- Verify your answer makes logical sense
- Consider alternative interpretations of the problem

The key to solving this type of problem is to be methodical and thorough in your analysis.
"""
    
    return base_prompt + context_padding

def run_bbh_test(client: BBHTestClient, gpu_monitor: GPUMonitor, limit: int = 10, context_length: int = 2000, max_tokens: int = 512) -> Dict[str, Any]:
    """运行BBH测试（仅选取指定数量的数据，流式打印token与时间）"""
    print("🧠 开始BBH数据集测试...")

    # 加载BBH任务
    bbh_tasks = load_bbh_tasks()
    print(f"📚 加载了 {len(bbh_tasks)} 个BBH任务")

    # 预先选取不超过 limit 个示例，跨任务随机采样
    selected_items: List[Dict[str, Any]] = []
    task_pool = bbh_tasks.copy()
    random.shuffle(task_pool)
    for task in task_pool:
        examples = task.get('examples', [])
        random.shuffle(examples)
        for example in examples:
            long_prompt = create_long_prompt(task, example, context_length)
            selected_items.append({
                "task_name": task['task_name'],
                "description": task['description'],
                "prompt": long_prompt
            })
            if len(selected_items) >= limit:
                break
        if len(selected_items) >= limit:
            break

    print(f"🔢 将测试 {len(selected_items)} 条数据（上限 {limit}）")

    results: Dict[str, Any] = {
        "test_info": {
            "dataset": "BIG-Bench Hard (BBH)",
            "context_length": context_length,
            "limit": limit,
            "timestamp": time.time()
        },
        "requests": [],
        "gpu_data": []
    }

    # 开始GPU监控
    gpu_monitor.start_monitoring()

    for idx, item in enumerate(selected_items, start=1):
        prompt = item["prompt"]
        print(f"\n=== 请求 {idx}/{len(selected_items)} - 任务: {item['task_name']} ===")
        print(f"📝 Prompt (length={len(prompt)}):\n{prompt}")

        request_start = time.time()
        first_token_time = None
        token_times: List[float] = []
        token_count = 0
        generated_text = ""

        # 流式请求
        response = client.generate_stream(prompt=prompt, max_tokens=max_tokens, temperature=0.7)
        if not response:
            print("❌ 请求失败")
            results["requests"].append({
                "task_name": item['task_name'],
                "description": item['description'],
                "prompt_length": len(prompt),
                "error": "request_failed"
            })
            continue

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get('choices', [])
                if not choices:
                    continue
                text_content = choices[0].get('text', '')
                if not text_content:
                    continue
                now = time.time()
                token_count += 1
                if first_token_time is None:
                    first_token_time = now - request_start
                token_times.append(now - request_start)
                generated_text += text_content
                display_text = text_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                if len(display_text) > 40:
                    display_text = display_text[:40] + "..."
                print(f"    Token {token_count}: '{display_text}' @ {now - request_start:.4f}s")
        except Exception as e:
            print(f"  处理流式响应时出错: {e}")

        total_time = time.time() - request_start
        ttft = first_token_time or 0.0
        if token_count > 1 and len(token_times) > 1:
            tpot = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
        else:
            tpot = 0.0

        results["requests"].append({
            "task_name": item['task_name'],
            "description": item['description'],
            "prompt_length": len(prompt),
            "response_preview": generated_text[:200],
            "token_count": token_count,
            "ttft": ttft,
            "tpot": tpot,
            "total_time": total_time,
            "token_times": token_times
        })

        print(f"  完成: 总时间={total_time:.4f}s, TTFT={ttft:.4f}s, TPOT={tpot:.4f}s, Tokens={token_count}")

        # 轻微间隔，避免过于密集
        time.sleep(0.2)

    # 停止GPU监控
    gpu_monitor.stop_monitoring()
    results['gpu_data'] = gpu_monitor.get_data()

    return results

def analyze_bbh_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """分析BBH测试结果"""
    analysis = {
        "summary": {},
        "context_length_analysis": {},
        "task_performance": {},
        "gpu_analysis": {}
    }
    
    # 基本统计
    total_tasks = len(results['task_results'])
    total_requests = sum(len(task['context_length_results']) for task in results['task_results'])
    successful_requests = sum(
        len([r for r in task['context_length_results'] if 'error' not in r])
        for task in results['task_results']
    )
    
    analysis['summary'] = {
        "total_tasks": total_tasks,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0
    }
    
    # 按上下文长度分析
    context_lengths = results['test_info']['context_lengths']
    for context_length in context_lengths:
        context_results = []
        for task in results['task_results']:
            for result in task['context_length_results']:
                if result['context_length'] == context_length and 'error' not in result:
                    context_results.append(result)
        
        if context_results:
            analysis['context_length_analysis'][str(context_length)] = {
                "count": len(context_results),
                "avg_request_time": sum(r['request_time'] for r in context_results) / len(context_results),
                "avg_prompt_tokens": sum(r['prompt_tokens'] for r in context_results) / len(context_results),
                "avg_completion_tokens": sum(r['completion_tokens'] for r in context_results) / len(context_results),
                "avg_total_tokens": sum(r['total_tokens'] for r in context_results) / len(context_results)
            }
    
    # GPU分析
    if results['gpu_data']:
        gpu_data = results['gpu_data']
        powers = [entry.get('power_draw', 0) for entry in gpu_data if entry.get('power_draw', 0) > 0]
        utilizations = [entry.get('gpu_utilization', 0) for entry in gpu_data if entry.get('gpu_utilization', 0) > 0]
        
        analysis['gpu_analysis'] = {
            "avg_power": sum(powers) / len(powers) if powers else 0,
            "max_power": max(powers) if powers else 0,
            "min_power": min(powers) if powers else 0,
            "avg_utilization": sum(utilizations) / len(utilizations) if utilizations else 0,
            "max_utilization": max(utilizations) if utilizations else 0,
            "data_points": len(gpu_data)
        }
    
    return analysis

def save_results(results: Dict[str, Any], analysis: Dict[str, Any], storage_dir: Path):
    """保存测试结果"""
    storage_dir.mkdir(exist_ok=True)
    
    # 保存原始结果
    with open(storage_dir / 'bbh_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存分析结果
    with open(storage_dir / 'bbh_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"📁 结果已保存到 {storage_dir} 目录")

def main():
    """主函数"""
    print("🧠 BIG-Bench Hard (BBH) 数据集测试")
    print("=" * 50)
    
    # 创建客户端
    client = BBHTestClient()
    
    # 检查服务器状态
    if not client.health_check():
        print("❌ VLLM 服务器未运行")
        print("请先运行: ./start_vllm_server.sh")
        return
    
    print("✅ VLLM 服务器运行正常")
    
    # 创建GPU监控器
    gpu_monitor = GPUMonitor(interval=0.1)
    
    # 运行测试（仅取10条，流式打印token与时间）
    results = run_bbh_test(client, gpu_monitor, limit=10, context_length=2000, max_tokens=512)
    
    # 分析结果
    print("\n📊 分析测试结果...")
    analysis = analyze_bbh_results(results)
    
    # 保存结果
    storage_dir = Path("bbh_test_results")
    save_results(results, analysis, storage_dir)
    
    # 打印摘要
    print(f"\n📈 测试摘要:")
    print(f"  总任务数: {analysis['summary']['total_tasks']}")
    print(f"  总请求数: {analysis['summary']['total_requests']}")
    print(f"  成功请求数: {analysis['summary']['successful_requests']}")
    print(f"  成功率: {analysis['summary']['success_rate']:.2%}")
    
    if analysis['gpu_analysis']:
        print(f"\n⚡ GPU 性能:")
        print(f"  平均功耗: {analysis['gpu_analysis']['avg_power']:.1f}W")
        print(f"  最大功耗: {analysis['gpu_analysis']['max_power']:.1f}W")
        print(f"  平均利用率: {analysis['gpu_analysis']['avg_utilization']:.1f}%")
        print(f"  最大利用率: {analysis['gpu_analysis']['max_utilization']:.1f}%")
    
    print(f"\n📁 详细结果保存在: {storage_dir}")
    print("🎉 BBH测试完成！")

if __name__ == "__main__":
    main()
