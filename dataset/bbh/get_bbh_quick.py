#!/usr/bin/env python3
"""
BBH数据集快速获取脚本
直接从命令行参数获取随机prompt并输出到终端
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加上级目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# 尝试导入datasets库
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("❌ datasets库未安装，请运行: pip install datasets")
    sys.exit(1)

class BBHQuickGetter:
    """BBH数据集快速获取器"""
    
    def __init__(self):
        self.bbh_task_names = [
            "logical_deduction_three_objects",
            "date_understanding", 
            "causal_judgment",
            "disambiguation_qa",
            "geometric_shapes",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "multistep_arithmetic_two",
            "navigate",
            "reasoning_about_colored_objects",
            "ruin_names",
            "snarks",
            "sports_understanding",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects"
        ]
        
        # 任务描述
        self.task_descriptions = {
            "logical_deduction_three_objects": "逻辑推理（三个对象）",
            "date_understanding": "日期理解",
            "causal_judgment": "因果判断",
            "disambiguation_qa": "歧义问答",
            "geometric_shapes": "几何形状",
            "logical_deduction_five_objects": "逻辑推理（五个对象）",
            "logical_deduction_seven_objects": "逻辑推理（七个对象）",
            "multistep_arithmetic_two": "多步算术（两个步骤）",
            "navigate": "导航",
            "reasoning_about_colored_objects": "彩色对象推理",
            "ruin_names": "名字推理",
            "snarks": "Snarks推理",
            "sports_understanding": "体育理解",
            "temporal_sequences": "时间序列",
            "tracking_shuffled_objects_five_objects": "跟踪洗牌对象（五个）",
            "tracking_shuffled_objects_seven_objects": "跟踪洗牌对象（七个）",
            "tracking_shuffled_objects_three_objects": "跟踪洗牌对象（三个）"
        }
    
    def list_tasks(self):
        """列出所有可用任务"""
        print("📚 可用的BBH任务:")
        print("-" * 50)
        for i, task_name in enumerate(self.bbh_task_names, 1):
            description = self.task_descriptions.get(task_name, "无描述")
            print(f"{i:2d}. {task_name}")
            print(f"    {description}")
        print("-" * 50)
    
    def load_task_data(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """加载指定任务的数据"""
        # 首先尝试从本地JSON文件加载
        try:
            print(f"📥 正在从本地JSON文件加载任务: {task_name}", file=sys.stderr)
            local_file_path = Path(__file__).parent / "BIG-Bench-Hard" / "bbh" / f"{task_name}.json"
            
            if local_file_path.exists():
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    examples = data.get('examples', [])
                    print(f"✅ 成功从本地文件加载 {len(examples)} 条数据", file=sys.stderr)
                    return examples
            else:
                print(f"⚠️  本地文件不存在: {local_file_path}", file=sys.stderr)
                print("🔄 尝试使用内置示例数据...", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  从本地文件加载失败: {e}", file=sys.stderr)
            print("🔄 尝试使用内置示例数据...", file=sys.stderr)
        
        # 如果本地文件加载失败，使用内置示例数据
        return self.load_fallback_data(task_name)
    
    def load_fallback_data(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """加载内置的BBH任务示例数据"""
        print(f"📚 使用内置示例数据: {task_name}", file=sys.stderr)
        
        # 内置示例数据（与get_bbh_random.py相同）
        fallback_data = {
            "causal_judgment": [
                {
                    "input": "A man named John goes to a restaurant and orders a hamburger. The hamburger is poisoned. John eats the hamburger and dies. What was the cause of John's death?",
                    "target": "The poisoned hamburger"
                },
                {
                    "input": "A woman named Sarah is driving her car on a rainy day. She hits a patch of ice and crashes into a tree. What was the cause of the crash?",
                    "target": "The ice on the road"
                },
                {
                    "input": "The power went out in the building. All the computers shut down. What caused the computers to shut down?",
                    "target": "The power outage"
                }
            ],
            "date_understanding": [
                {
                    "input": "Today is the 1st of January, 2023. What will be the date 3 days from now?",
                    "target": "January 4th, 2023"
                },
                {
                    "input": "If today is March 15th, 2023, what was the date 2 weeks ago?",
                    "target": "March 1st, 2023"
                },
                {
                    "input": "What is the date 5 days after December 28th, 2022?",
                    "target": "January 2nd, 2023"
                }
            ],
            "disambiguation_qa": [
                {
                    "input": "The word 'bank' can refer to a financial institution or the side of a river. In the sentence 'I went to the bank to deposit money', what does 'bank' mean?",
                    "target": "A financial institution"
                },
                {
                    "input": "The word 'bark' can refer to the sound a dog makes or the outer covering of a tree. In the sentence 'The dog's bark woke me up', what does 'bark' mean?",
                    "target": "The sound a dog makes"
                },
                {
                    "input": "The word 'bat' can refer to a flying mammal or a sports equipment. In the sentence 'The baseball player swung the bat', what does 'bat' mean?",
                    "target": "Sports equipment"
                }
            ],
            "geometric_shapes": [
                {
                    "input": "A square has 4 sides of equal length. If one side is 5 cm long, what is the perimeter of the square?",
                    "target": "20 cm"
                },
                {
                    "input": "A circle has a radius of 3 cm. What is the area of the circle? (Use π ≈ 3.14)",
                    "target": "28.26 cm²"
                },
                {
                    "input": "A rectangle has a length of 8 cm and width of 6 cm. What is the area of the rectangle?",
                    "target": "48 cm²"
                }
            ],
            "logical_deduction_three_objects": [
                {
                    "input": "All birds can fly. Penguins are birds. Can penguins fly?",
                    "target": "No, penguins cannot fly (this is a logical paradox)"
                },
                {
                    "input": "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                    "target": "Not necessarily - the ground could be wet for other reasons"
                },
                {
                    "input": "All cats are animals. Fluffy is a cat. Is Fluffy an animal?",
                    "target": "Yes, Fluffy is an animal"
                }
            ],
            "multistep_arithmetic_two": [
                {
                    "input": "Sarah has 12 apples. She gives 3 apples to her friend and buys 7 more apples. How many apples does Sarah have now?",
                    "target": "16 apples"
                },
                {
                    "input": "A store has 50 items. They sell 15 items in the morning and 20 items in the afternoon. How many items are left?",
                    "target": "15 items"
                },
                {
                    "input": "Tom has $20. He spends $8 on lunch and $5 on a movie ticket. How much money does he have left?",
                    "target": "$7"
                }
            ],
            "navigate": [
                {
                    "input": "You are at point A. To get to point B, you need to go 3 blocks north, then 2 blocks east, then 1 block south. What direction should you start walking?",
                    "target": "North"
                },
                {
                    "input": "Starting from home, you walk 2 blocks west, then 3 blocks north, then 1 block east. How many blocks are you from home?",
                    "target": "3 blocks (using Pythagorean theorem: √(1² + 3²) = √10 ≈ 3.16)"
                },
                {
                    "input": "To get from the library to the school, you need to go 4 blocks south, then 2 blocks east. What is the total distance?",
                    "target": "6 blocks"
                }
            ],
            "reasoning_about_colored_objects": [
                {
                    "input": "There are 3 boxes: a red box, a blue box, and a green box. The red box contains a ball. The blue box is empty. What color is the box that contains the ball?",
                    "target": "Red"
                },
                {
                    "input": "A red car and a blue car are parked. The red car is faster than the blue car. Which car is faster?",
                    "target": "The red car"
                },
                {
                    "input": "There are three flowers: a red rose, a yellow daisy, and a blue violet. The red rose is the most fragrant. Which flower is most fragrant?",
                    "target": "The red rose"
                }
            ],
            "ruin_names": [
                {
                    "input": "If you rearrange the letters in 'LISTEN', what word do you get?",
                    "target": "SILENT"
                },
                {
                    "input": "Rearrange the letters in 'EARTH' to form a word meaning 'heart'.",
                    "target": "HEART"
                },
                {
                    "input": "What word can you make by rearranging the letters in 'STUDY'?",
                    "target": "DUSTY"
                }
            ],
            "sports_understanding": [
                {
                    "input": "In basketball, how many points is a three-pointer worth?",
                    "target": "3 points"
                },
                {
                    "input": "In soccer, how many players are on the field for each team?",
                    "target": "11 players"
                },
                {
                    "input": "In tennis, what is the score called when both players have 40 points?",
                    "target": "Deuce"
                }
            ],
            "temporal_sequences": [
                {
                    "input": "If event A happens before event B, and event B happens before event C, what is the order of events?",
                    "target": "A, then B, then C"
                },
                {
                    "input": "Sarah wakes up at 7 AM, has breakfast at 8 AM, and goes to work at 9 AM. What does she do first?",
                    "target": "Wakes up"
                },
                {
                    "input": "The meeting is scheduled for 2 PM, lunch is at 12 PM, and the presentation is at 3 PM. What happens at 2 PM?",
                    "target": "The meeting"
                }
            ],
            "tracking_shuffled_objects_three_objects": [
                {
                    "input": "Three objects are placed in a line: a red ball, a blue cube, and a green triangle. The red ball is moved to the end. What is the new order?",
                    "target": "Blue cube, green triangle, red ball"
                },
                {
                    "input": "A book, a pen, and a notebook are on a desk. The pen is moved between the book and notebook. What is the new order?",
                    "target": "Book, pen, notebook"
                }
            ]
        }
        
        if task_name in fallback_data:
            examples = fallback_data[task_name]
            print(f"✅ 成功加载 {len(examples)} 条内置示例数据", file=sys.stderr)
            return examples
        else:
            print(f"❌ 任务 '{task_name}' 没有可用的内置示例数据", file=sys.stderr)
            return None
    
    def get_random_prompts(self, task_data: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """获取随机prompt"""
        if not task_data:
            return []
        
        # 随机选择指定数量的prompt
        selected_count = min(count, len(task_data))
        selected_prompts = random.sample(task_data, selected_count)
        
        return selected_prompts
    
    def output_prompts(self, prompts: List[Dict[str, Any]], task_name: str, output_format: str, output_file: Optional[str] = None):
        """输出prompt"""
        output_stream = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout
        
        try:
            if output_format == 'json':
                # JSON格式输出
                output_data = {
                    "task_name": task_name,
                    "task_description": self.task_descriptions.get(task_name, "无描述"),
                    "count": len(prompts),
                    "prompts": prompts
                }
                json.dump(output_data, output_stream, ensure_ascii=False, indent=2)
                
            elif output_format == 'text':
                # 纯文本格式输出
                print(f"任务: {task_name}", file=output_stream)
                print(f"描述: {self.task_descriptions.get(task_name, '无描述')}", file=output_stream)
                print(f"数量: {len(prompts)} 条", file=output_stream)
                print("=" * 60, file=output_stream)
                
                for i, prompt in enumerate(prompts, 1):
                    print(f"\nPrompt {i}:", file=output_stream)
                    print(f"输入: {prompt['input']}", file=output_stream)
                    print(f"目标答案: {prompt['target']}", file=output_stream)
                    print("-" * 40, file=output_stream)
                    
            elif output_format == 'prompt-only':
                # 只输出prompt内容
                for prompt in prompts:
                    print(prompt['input'], file=output_stream)
                    
            elif output_format == 'csv':
                # CSV格式输出
                import csv
                writer = csv.writer(output_stream)
                writer.writerow(['task_name', 'input', 'target'])
                for prompt in prompts:
                    writer.writerow([task_name, prompt['input'], prompt['target']])
            
        finally:
            if output_file and output_stream != sys.stdout:
                output_stream.close()
    
    def run(self, args):
        """运行主程序"""
        # 如果请求列出任务
        if args.list:
            self.list_tasks()
            return
        
        # 验证任务名称
        if args.task not in self.bbh_task_names:
            print(f"❌ 无效的任务名称: {args.task}", file=sys.stderr)
            print("使用 --list 查看可用任务", file=sys.stderr)
            sys.exit(1)
        
        # 加载任务数据
        task_data = self.load_task_data(args.task)
        if not task_data:
            sys.exit(1)
        
        # 获取随机prompt
        prompts = self.get_random_prompts(task_data, args.count)
        
        # 输出结果
        self.output_prompts(prompts, args.task, args.format, args.output)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="BBH数据集快速获取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --list                                    # 列出所有任务
  %(prog)s --task date_understanding --count 3       # 获取3条日期理解任务
  %(prog)s --task logical_deduction_three_objects --count 1 --format prompt-only  # 只输出prompt
  %(prog)s --task causal_judgment --count 5 --output results.json --format json   # 保存为JSON
        """
    )
    
    parser.add_argument('--list', action='store_true', help='列出所有可用任务')
    parser.add_argument('--task', type=str, help='任务名称')
    parser.add_argument('--count', type=int, default=1, help='获取的prompt数量 (默认: 1)')
    parser.add_argument('--format', choices=['json', 'text', 'prompt-only', 'csv'], 
                       default='text', help='输出格式 (默认: text)')
    parser.add_argument('--output', type=str, help='输出文件路径 (默认: 标准输出)')
    
    args = parser.parse_args()
    
    # 如果没有指定任务且没有列出任务，显示帮助
    if not args.task and not args.list:
        parser.print_help()
        return
    
    try:
        getter = BBHQuickGetter()
        getter.run(args)
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
