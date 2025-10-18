# BBH数据集随机获取工具

这个工具包提供了两种方式来获取BBH (BIG-Bench Hard) 数据集的随机prompt，支持交互式选择和命令行快速获取。

## 📁 文件说明

- `get_bbh_random.py` - 交互式BBH随机获取工具
- `run_bbh_random.sh` - 交互式工具运行脚本
- `get_bbh_quick.py` - 命令行快速获取工具
- `run_bbh_quick.sh` - 快速工具运行脚本
- `README_bbh_random.md` - 本说明文档

## 🚀 快速开始

### 方式一：交互式工具（推荐新手）

```bash
# 运行交互式工具
./run_bbh_random.sh

# 或者直接运行Python脚本
python3 get_bbh_random.py
```

**交互式功能：**
- 📋 显示所有可用BBH任务列表
- 🎯 选择特定任务或随机选择
- 📊 指定获取的prompt数量
- 💾 选择是否保存到文件
- 🔄 支持连续选择多个任务

### 方式二：命令行快速工具（推荐高级用户）

```bash
# 列出所有可用任务
./run_bbh_quick.sh --list

# 获取3条日期理解任务的prompt
./run_bbh_quick.sh --task date_understanding --count 3

# 只输出prompt内容（不包含答案）
./run_bbh_quick.sh --task logical_deduction_three_objects --count 1 --format prompt-only

# 保存为JSON格式
./run_bbh_quick.sh --task causal_judgment --count 5 --output results.json --format json

# 保存为CSV格式
./run_bbh_quick.sh --task geometric_shapes --count 10 --output data.csv --format csv
```

## 📚 可用任务列表

| 任务名称 | 中文描述 | 难度 |
|---------|---------|------|
| `logical_deduction_three_objects` | 逻辑推理（三个对象） | 中等 |
| `date_understanding` | 日期理解 | 简单 |
| `causal_judgment` | 因果判断 | 中等 |
| `disambiguation_qa` | 歧义问答 | 困难 |
| `geometric_shapes` | 几何形状 | 中等 |
| `logical_deduction_five_objects` | 逻辑推理（五个对象） | 困难 |
| `logical_deduction_seven_objects` | 逻辑推理（七个对象） | 困难 |
| `multistep_arithmetic_two` | 多步算术（两个步骤） | 中等 |
| `navigate` | 导航 | 中等 |
| `reasoning_about_colored_objects` | 彩色对象推理 | 中等 |
| `ruin_names` | 名字推理 | 困难 |
| `snarks` | Snarks推理 | 困难 |
| `sports_understanding` | 体育理解 | 简单 |
| `temporal_sequences` | 时间序列 | 中等 |
| `tracking_shuffled_objects_five_objects` | 跟踪洗牌对象（五个） | 困难 |
| `tracking_shuffled_objects_seven_objects` | 跟踪洗牌对象（七个） | 困难 |
| `tracking_shuffled_objects_three_objects` | 跟踪洗牌对象（三个） | 中等 |

## 🔧 命令行参数

### 快速工具参数

- `--list`: 列出所有可用任务
- `--task <任务名>`: 指定要获取的任务名称
- `--count <数量>`: 获取的prompt数量（默认: 1）
- `--format <格式>`: 输出格式
  - `text`: 完整文本格式（默认）
  - `json`: JSON格式
  - `prompt-only`: 只输出prompt内容
  - `csv`: CSV格式
- `--output <文件路径>`: 输出文件路径（默认: 标准输出）

## 📊 输出格式示例

### 文本格式 (--format text)
```
任务: date_understanding
描述: 日期理解
数量: 1 条
============================================================

Prompt 1:
输入: What is the date 3 days after January 1, 2020?
目标答案: January 4, 2020
----------------------------------------
```

### JSON格式 (--format json)
```json
{
  "task_name": "date_understanding",
  "task_description": "日期理解",
  "count": 1,
  "prompts": [
    {
      "input": "What is the date 3 days after January 1, 2020?",
      "target": "January 4, 2020"
    }
  ]
}
```

### 只输出prompt (--format prompt-only)
```
What is the date 3 days after January 1, 2020?
```

### CSV格式 (--format csv)
```csv
task_name,input,target
date_understanding,"What is the date 3 days after January 1, 2020?","January 4, 2020"
```

## 💡 使用场景

### 1. 模型测试
```bash
# 获取不同类型的推理任务进行测试
./run_bbh_quick.sh --task logical_deduction_three_objects --count 5 --format prompt-only
```

### 2. 数据集构建
```bash
# 构建混合数据集
./run_bbh_quick.sh --task date_understanding --count 10 --output date_data.json --format json
./run_bbh_quick.sh --task causal_judgment --count 10 --output causal_data.json --format json
```

### 3. 快速验证
```bash
# 快速获取一个prompt进行验证
./run_bbh_quick.sh --task geometric_shapes --count 1
```

### 4. 批量处理
```bash
# 为每个任务生成测试数据
for task in date_understanding causal_judgment geometric_shapes; do
    ./run_bbh_quick.sh --task $task --count 5 --output "${task}_test.json" --format json
done
```

## ⚠️ 注意事项

1. **依赖要求**: 需要安装 `datasets` 库
   ```bash
   pip install datasets
   ```

2. **网络连接**: 首次使用需要下载数据集，需要网络连接

3. **数据量**: 每个任务的数据量不同，某些任务可能只有少量数据

4. **随机性**: 每次运行都会获取不同的随机prompt

5. **文件权限**: 确保脚本有执行权限
   ```bash
   chmod +x run_bbh_random.sh run_bbh_quick.sh
   ```

## 🔍 故障排除

### 常见问题

1. **datasets库未安装**
   ```bash
   pip install datasets
   ```

2. **网络连接问题**
   - 确保网络连接正常
   - 可能需要配置代理

3. **任务名称错误**
   ```bash
   ./run_bbh_quick.sh --list  # 查看可用任务
   ```

4. **权限问题**
   ```bash
   chmod +x *.sh
   ```

## 📈 扩展功能

### 自定义任务
可以修改脚本中的 `bbh_task_names` 列表来添加或移除任务。

### 批量处理
可以编写shell脚本来自动化批量获取不同任务的数据。

### 数据预处理
获取的数据可以进一步处理，如过滤、格式化等。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这些工具！
