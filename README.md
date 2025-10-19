# LLM DVFS 测试项目

这个项目用于测试大语言模型（LLM）在不同阶段的功耗和性能，支持传统单实例和分离式 Prefill+Decode 两种架构的对比测试。

## 🔥 快速开始 - GPU功耗监控工具

实时监控多个GPU的功耗、温度、利用率等数据，支持100ms高精度采样。

#### 基本使用

能耗测量
```bash
./quick_start.sh
```
部署vLLM
```bash
./server/traditional/start_vllm_server.sh
```
运行单批次测试
```bash
./server/traditional/run_server_test.sh
```
增加批量进行测试
```bash
./server/traditional/run_batch_test.sh
```
## 📁 项目结构

```
LLM-dvfs/
├── gpu_monitor.py                    # GPU 监控模块
├── dataset/                          # 数据集工具目录
│   ├── README.md                     # 数据集工具说明
│   └── bbh/                          # BBH数据集工具
│       ├── README_bbh_random.md      # BBH工具详细说明
│       ├── get_bbh_random.py         # 交互式BBH随机获取工具
│       ├── get_bbh_quick.py          # 命令行快速获取工具
│       ├── run_bbh_random.sh         # 交互式工具运行脚本
│       └── run_bbh_quick.sh          # 快速工具运行脚本
├── server/                           # VLLM 测试套件
│   ├── README.md                     # 测试套件说明
│   │
│   ├── traditional/                  # 传统单实例测试
│   │   ├── start_vllm_server.sh      # 启动单实例服务器
│   │   ├── test_vllm_server.py       # 单实例测试脚本
|   |   ├── simple_batch_test.py      # 批量测试脚本
│   │   ├── run_server_test.sh        # 单实例测试运行脚本
│   │   ├── run_batch_test.sh         # 批量测试运行脚本
│   │   ├── analyze_server_results.py # 单实例结果分析
│   │   ├── README_vllm_server.md     # 单实例测试说明
│   │   └── vllm_server_results/      # 单实例测试结果
│   │
│   ├── disaggregated/                # 分离式测试
│   │   ├── start_disaggregated_servers.sh    # 启动分离式服务器
│   │   ├── disagg_prefill_proxy_server.py    # 代理服务器
│   │   ├── test_disaggregated_performance.py # 分离式测试脚本
│   │   ├── run_disaggregated_test.sh         # 分离式测试运行脚本
│   │   ├── compare_performance.py            # 性能对比分析
│   │   ├── README_disaggregated.md           # 分离式测试说明
│   │   └── disaggregated_results/            # 分离式测试结果
│   │
│   └── reference/                    # 官方参考文件
│       └── disaggregated_prefill.sh          # 官方分离式示例
│
└── README.md                         # 项目说明（本文件）
```
## 🚀 快速开始

### 获得数据集的提问数据

```bash
# BBH数据集随机获取工具
cd dataset/bbh/

# 交互式使用（推荐新手）
./run_bbh_random.sh

# 命令行快速使用（推荐高级用户）
./run_bbh_quick.sh --list                                    # 列出所有任务
./run_bbh_quick.sh --task date_understanding --count 3       # 获取3条日期理解任务
```

### 方式二：传统单实例测试

```bash
# 1. 启动单实例服务器
cd server/traditional
./start_vllm_server.sh

# 2. 运行测试（新终端）
cd server/traditional
./run_server_test.sh

# 3. 分析结果
python3 analyze_server_results.py
```

### 方式二：分离式测试

```bash
# 1. 启动分离式服务器（需要 2 个 GPU）
cd server/disaggregated
./start_disaggregated_servers.sh

# 2. 运行分离式测试（新终端）
cd server/disaggregated
./run_disaggregated_test.sh

# 3. 对比分析
python3 compare_performance.py
```
## 🦙 Llama.cpp 运行指令

### RTX 4080 16GB显存优化配置
#### 详细参数说明：

**显存优化参数：**
- `-ngl 50`: 将50层加载到GPU（约占用13-15GB显存，为系统预留1-3GB）
- `-c 4096`: 上下文长度4096 tokens
- `-b 512`: 批处理大小512
- `--mlock`: 锁定内存，防止交换到磁盘
- `--no-mmap`: 禁用内存映射，减少内存碎片

#### 运行参数：

DeepSeek-R1-32B(Qwen蒸馏版本)
```bash
~/offload/llama.cpp/build/bin/llama-cli \
  -m /share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B/model.gguf \
  -ngl 56 \
  -t 192 \
  --mlock \
  --no-mmap
```

DeepSeek-R1-70B(Llama蒸馏版本)
```bash
~/offload/llama.cpp/build/bin/llama-cli \
  -m /share-data/wzk-1/model/DeepSeek-R1-Distill-Llama-70B/model.gguf \
  -ngl 36 \
  -b 1024 \
  -t 192 
```
DeepSeek-R1-Q4_K_M(深度推理4bit量化版本)
```bash
~/offload/llama.cpp/build/bin/llama-cli \
    --model /share-data/wzk-1/model/DeepSeek-R1-Q4_K_M/Deepseek_R1_Q4_K_M.gguf \
    --cache-type-k q4_0 \
    -no-cnv \
    --n-gpu-layers 12 \
    --temp 0.8 \
    --ctx-size 4096 \
    --threads 192 \
    --prompt "<｜User｜>Once upon a time, <｜Assistant｜>"
```

# the command for merge gguf tensor 
./build/bin/llama-gguf-split --merge /share-data/wzk-1/model/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf /share-data/wzk-1/model/DeepSeek-R1-Q4_K_M/Deepseek_R1_Q4_K_M.gguf