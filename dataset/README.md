# 数据集工具目录

这个目录包含了各种数据集相关的工具和脚本。

## 📁 目录结构

```
dataset/
├── README.md                    # 本说明文件
└── bbh/                         # BBH (BIG-Bench Hard) 数据集工具
    ├── README_bbh_random.md     # BBH工具详细说明
    ├── get_bbh_random.py        # 交互式BBH随机获取工具
    ├── get_bbh_quick.py         # 命令行快速获取工具
    ├── run_bbh_random.sh        # 交互式工具运行脚本
    └── run_bbh_quick.sh         # 快速工具运行脚本
```

## 🧠 BBH数据集工具

BBH (BIG-Bench Hard) 数据集工具提供了两种方式来获取BBH数据集的随机prompt：

### 快速开始

```bash
# 进入BBH工具目录
cd bbh/

# 交互式使用（推荐新手）
./run_bbh_random.sh

# 命令行快速使用（推荐高级用户）
./run_bbh_quick.sh --list                                    # 列出所有任务
./run_bbh_quick.sh --task date_understanding --count 3       # 获取3条日期理解任务
```

### 详细说明

请查看 `bbh/README_bbh_random.md` 获取完整的使用说明和示例。

## 🔧 依赖要求

- Python 3.8+
- datasets库: `pip install datasets`
- 网络连接（首次使用需要下载数据集）

## 📚 支持的数据集

### BBH (BIG-Bench Hard)
- 17个不同的推理任务
- 包括逻辑推理、日期理解、因果判断等
- 支持多种输出格式（text、json、csv、prompt-only）

## 🚀 扩展计划

未来可能会添加更多数据集工具：
- MMLU数据集工具
- HellaSwag数据集工具
- 其他基准测试数据集工具

## 🤝 贡献

欢迎提交Issue和Pull Request来添加新的数据集工具！
