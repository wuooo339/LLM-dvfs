#!/usr/bin/env python3
"""
BBH数据集随机获取脚本
交互式选择BBH任务并获取随机prompt
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict

# 添加上级目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# 尝试导入datasets库
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️  datasets库未安装，将使用内置示例数据")
    print("   安装命令: pip install datasets")
    DATASETS_AVAILABLE = False

class BBHRandomGetter:
    """BBH数据集随机获取器"""
    
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

    def display_menu(self):
        """显示任务选择菜单"""
        print("\n🧠 BBH (BIG-Bench Hard) 数据集随机获取工具")
        print("=" * 50)
        print("可用任务:")
        
        for i, task_name in enumerate(self.bbh_task_names, 1):
            description = self.task_descriptions.get(task_name, "无描述")
            print(f"  {i:2d}. {task_name}")
            print(f"      {description}")
        
        print(f"  {len(self.bbh_task_names) + 1:2d}. 随机选择任务")
        print(f"  {len(self.bbh_task_names) + 2:2d}. 显示所有任务")
        print(f"  {len(self.bbh_task_names) + 3:2d}. 退出")
        print("=" * 50)
    
    def get_user_choice(self) -> Optional[str]:
        """获取用户选择"""
        while True:
            try:
                choice = input(f"\n请选择任务 (1-{len(self.bbh_task_names) + 3}): ").strip()
                
                if not choice:
                    continue
                
                choice_num = int(choice)
                
                if choice_num == len(self.bbh_task_names) + 1:
                    # 随机选择任务
                    return random.choice(self.bbh_task_names)
                elif choice_num == len(self.bbh_task_names) + 2:
                    # 显示所有任务
                    self.show_all_tasks()
                    continue
                elif choice_num == len(self.bbh_task_names) + 3:
                    # 退出
                    return None
                elif 1 <= choice_num <= len(self.bbh_task_names):
                    # 选择特定任务
                    return self.bbh_task_names[choice_num - 1]
                else:
                    print(f"❌ 无效选择，请输入 1-{len(self.bbh_task_names) + 3}")
                    
            except ValueError:
                print("❌ 请输入有效数字")
            except KeyboardInterrupt:
                print("\n👋 再见！")
                return None
    
    def show_all_tasks(self):
        """显示所有任务的详细信息"""
        print("\n📚 所有BBH任务详细信息:")
        print("-" * 60)
        
        for i, task_name in enumerate(self.bbh_task_names, 1):
            description = self.task_descriptions.get(task_name, "无描述")
            print(f"{i:2d}. {task_name}")
            print(f"    描述: {description}")
            print()
    
    def check_huggingface_token(self):
        """检查 Hugging Face 登录状态"""
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.whoami()
            print("✅ 已登录 Hugging Face")
            return True
        except Exception:
            print("⚠️ 未登录 Hugging Face，需要登录才能访问私有数据集")
            return False
    
    def load_task_data(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """加载指定任务的数据"""
        # 首先尝试从本地JSON文件加载
        try:
            print(f"📥 正在从本地JSON文件加载任务: {task_name}")
            local_file_path = Path(__file__).parent / "BIG-Bench-Hard" / "bbh" / f"{task_name}.json"
            
            if local_file_path.exists():
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    examples = data.get('examples', [])
                    print(f"✅ 成功从本地文件加载 {len(examples)} 条数据")
                    return examples
            else:
                print(f"⚠️  本地文件不存在: {local_file_path}")
                print("🔄 尝试使用内置示例数据...")
        except Exception as e:
            print(f"⚠️  从本地文件加载失败: {e}")
            print("🔄 尝试使用内置示例数据...")
        
        # 使用内置示例数据
        return self.load_fallback_data(task_name)
    
    def load_fallback_data(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """加载内置的BBH任务示例数据"""
        print(f"📚 使用内置示例数据: {task_name}")
        
        fallback_data = {
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
            # 其他任务的内置数据...
        }
        
        if task_name in fallback_data:
            examples = fallback_data[task_name]
            print(f"✅ 成功加载 {len(examples)} 条内置示例数据")
            return examples
        else:
            print(f"❌ 任务 '{task_name}' 没有可用的内置示例数据")
            return None
    
    def get_random_prompts(self, task_data: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """获取随机prompt"""
        if not task_data:
            return []
        
        selected_count = min(count, len(task_data))
        selected_prompts = random.sample(task_data, selected_count)
        
        return selected_prompts
    
    def display_prompts(self, prompts: List[Dict[str, Any]], task_name: str):
        """显示prompt信息"""
        print(f"\n🎯 任务: {task_name}")
        print(f"📝 描述: {self.task_descriptions.get(task_name, '无描述')}")
        print(f"📊 获取数量: {len(prompts)} 条")
        print("=" * 80)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📋 Prompt {i}:")
            print(f"输入: {prompt['input']}")
            print(f"目标答案: {prompt['target']}")
            print("-" * 40)
    
    def save_prompts(self, prompts: List[Dict[str, Any]], task_name: str, filename: Optional[str] = None):
        """保存prompt到文件"""
        if not filename:
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bbh_{task_name}_{timestamp}.json"
        
        output_data = {
            "task_name": task_name,
            "task_description": self.task_descriptions.get(task_name, "无描述"),
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "count": len(prompts),
            "prompts": prompts
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"💾 已保存到文件: {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def get_prompt_count(self) -> int:
        """获取用户想要的prompt数量"""
        while True:
            try:
                count_input = input("请输入要获取的prompt数量 (默认1): ").strip()
                if not count_input:
                    return 1
                
                count = int(count_input)
                if count <= 0:
                    print("❌ 数量必须大于0")
                    continue
                elif count > 100:
                    print("⚠️  数量较大，建议不超过100")
                    confirm = input("确认继续? (y/N): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                return count
                
            except ValueError:
                print("❌ 请输入有效数字")
            except KeyboardInterrupt:
                return 0
    
    def run(self):
        """运行主程序"""
        print("🚀 启动BBH数据集随机获取工具...")
        
        while True:
            self.display_menu()
            task_name = self.get_user_choice()
            
            if task_name is None:
                break
            
            print(f"\n✅ 已选择任务: {task_name}")
            
            # 加载任务数据
            task_data = self.load_task_data(task_name)
            if not task_data:
                continue
            
            # 获取prompt数量
            count = self.get_prompt_count()
            if count == 0:
                continue
            
            # 获取随机prompt
            prompts = self.get_random_prompts(task_data, count)
            
            # 显示结果
            self.display_prompts(prompts, task_name)
            
            # 询问是否保存
            save_choice = input("\n💾 是否保存到文件? (y/N): ").strip().lower()
            if save_choice == 'y':
                filename = input("文件名 (回车使用默认): ").strip()
                if not filename:
                    filename = None
                self.save_prompts(prompts, task_name, filename)
            
            # 询问是否继续
            continue_choice = input("\n🔄 是否继续选择其他任务? (Y/n): ").strip().lower()
            if continue_choice == 'n':
                break
        
        print("\n👋 感谢使用BBH数据集随机获取工具！")

def main():
    """主函数"""
    try:
        getter = BBHRandomGetter()
        getter.run()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
