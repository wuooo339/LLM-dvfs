# LLM DVFS 测试项目

这个项目用于测试大语言模型（LLM）在不同阶段的功耗和性能，支持传统单实例和分离式 Prefill+Decode 两种架构的对比测试。

## 🔥 快速开始 - GPU功耗监控工具

### 多GPU功耗监控 (`multi_gpu_monitor.py`)

实时监控多个GPU的功耗、温度、利用率等数据，支持100ms高精度采样。

#### 基本使用

```bash
# 监控4个GPU（默认），100ms间隔
python3 multi_gpu_monitor.py

# 指定GPU和采样间隔
python3 multi_gpu_monitor.py --gpu-ids "0,1,2,3" --interval 0.1

# 保存数据到CSV文件
python3 multi_gpu_monitor.py --output gpu_power_data.csv

# 同时保存CSV和JSON格式数据
python3 multi_gpu_monitor.py --output gpu_power.csv --json gpu_power.json

# 定时监控（例如监控60秒）
python3 multi_gpu_monitor.py --duration 60 --output gpu_power_60s.csv
```

#### 参数说明

- `--gpu-ids`: 要监控的GPU ID列表，用逗号分隔（默认: "0,1,2,3"）
- `--interval`: 采样间隔秒数（默认: 0.1，即100ms）
- `--output`: CSV输出文件路径
- `--json`: JSON详细数据输出文件路径
- `--duration`: 监控持续时间（秒），不指定则持续监控直到手动停止

#### 输出数据格式

CSV文件包含以下列：
- `timestamp`: Unix时间戳
- `datetime`: 可读时间格式
- `gpu_X_power`: GPU X的功耗（W）
- `gpu_X_utilization`: GPU X的利用率（%）
- `gpu_X_temperature`: GPU X的温度（°C）
- `gpu_X_memory_used`: GPU X的已用显存（MB）
- `gpu_X_memory_total`: GPU X的总显存（MB）
- `gpu_X_graphics_clock`: GPU X的图形频率（MHz）
- `gpu_X_memory_clock`: GPU X的显存频率（MHz）

### GPU功耗数据可视化 (`plot_gpu_power.py`)

将监控数据转换为直观的图表，支持多种可视化方式。

#### 基本使用

```bash
# 生成所有图表
python3 plot_gpu_power.py gpu_power_data.csv

# 指定输出目录
python3 plot_gpu_power.py gpu_power_data.csv --output-dir ./plots

# 不显示图表，只保存文件
python3 plot_gpu_power.py gpu_power_data.csv --no-show

# 生成统计报告
python3 plot_gpu_power.py gpu_power_data.csv --report power_report.txt
```

#### 生成的图表

1. **GPU功耗变化图** (`gpu_power_consumption.png`)
   - 每个GPU的功耗时间序列
   - 包含利用率背景填充

2. **多GPU功耗对比图** (`gpu_power_comparison.png`)
   - 所有GPU功耗曲线对比
   - 显示总功耗曲线

3. **利用率和功耗关系图** (`gpu_utilization_vs_power.png`)
   - 散点图显示利用率和功耗的线性关系
   - 包含趋势线

4. **GPU温度变化图** (`gpu_temperature.png`)
   - 所有GPU的温度时间序列

#### 参数说明

- `csv_file`: 输入的CSV数据文件路径
- `--output-dir`: 图表输出目录（默认: ./plots）
- `--no-show`: 不显示图表，只保存文件
- `--report`: 生成统计报告文件

### 快速启动

```bash
# 使用交互式快速启动脚本
./quick_start.sh

# 或者直接运行测试验证工具
python3 test_gpu_monitor.py
```

### 完整使用示例

```bash
# 1. 开始监控GPU功耗（60秒）
python3 multi_gpu_monitor.py --duration 60 --output gpu_test.csv --json gpu_test.json

# 2. 生成可视化图表
python3 plot_gpu_power.py gpu_test.csv --output-dir ./test_plots --report test_report.txt

# 3. 查看结果
ls -la test_plots/
cat test_report.txt
```

### 依赖安装

```bash
# 安装Python依赖
pip install pandas matplotlib numpy

# 确保nvidia-smi可用
nvidia-smi
```

### 注意事项

1. 需要NVIDIA驱动和nvidia-smi工具
2. 确保指定的GPU ID存在且可访问
3. 高频率监控（100ms）会产生大量数据，注意磁盘空间
4. 图表生成需要matplotlib，建议在图形环境中运行

## 🏗️ 项目架构

### 传统单实例架构
```
Client Request → VLLM Server (Single Instance) → Response
```

### 分离式架构
```
Client Request → Proxy Server → Prefill Instance (GPU 0) → KV Transfer → Decode Instance (GPU 1) → Response
```

## 📁 项目结构

```
LLM-dvfs/
├── gpu_monitor.py                    # GPU 监控模块
├── server/                           # VLLM 测试套件
│   ├── README.md                     # 测试套件说明
│   │
│   ├── traditional/                  # 传统单实例测试
│   │   ├── start_vllm_server.sh      # 启动单实例服务器
│   │   ├── test_vllm_server.py       # 单实例测试脚本
│   │   ├── run_server_test.sh        # 单实例测试运行脚本
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

## 🎯 核心功能

### GPU 监控 (`gpu_monitor.py`)
- **实时监控**: GPU 功耗、频率、利用率、温度
- **数据保存**: 支持 JSON 格式保存和统计分析
- **可配置**: 采样间隔、监控 GPU 等参数
- **多 GPU**: 支持同时监控多个 GPU

### 传统单实例测试
- **模拟分离**: 通过 `max_tokens` 参数模拟 prefill 和 decode
- **功耗分析**: 详细的 GPU 功耗和性能分析
- **结果对比**: 生成性能对比图表和报告

### 分离式测试
- **真正分离**: Prefill 和 Decode 运行在不同 GPU 上
- **KV 传输**: 通过 P2P NCCL 进行高速缓存传输
- **精确指标**: TTFT、TBT、E2E Latency 测量
- **双 GPU 监控**: 同时监控两个 GPU 的实时状态

## 🚀 快速开始

### 方式一：传统单实例测试

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

## 📊 性能指标

### 传统方式指标
- **Prefill 时间**: 处理输入提示词的时间
- **Decode 时间**: 生成输出 tokens 的时间
- **总时间**: 完整的推理时间
- **功耗统计**: 平均/最大功耗、能耗

### 分离式指标
- **TTFT (Time to First Token)**: 首 token 时间
- **TBT (Time Between Tokens)**: token 间时间
- **E2E Latency**: 端到端延迟
- **KV 传输时间**: 缓存传输耗时
- **双 GPU 功耗**: 分别监控两个 GPU

## 📈 输出结果

### 传统单实例结果
- `server/traditional/vllm_server_results/prefill_results.json` - Prefill 阶段结果
- `server/traditional/vllm_server_results/decode_results.json` - Decode 阶段结果
- `server/traditional/vllm_server_results/vllm_server_comparison.png` - 性能对比图表

### 分离式结果
- `server/disaggregated/disaggregated_results/disaggregated_results.json` - 完整测试结果
- `server/disaggregated/disaggregated_results/prefill_gpu_data.json` - Prefill GPU 监控数据
- `server/disaggregated/disaggregated_results/decode_gpu_data.json` - Decode GPU 监控数据
- `server/disaggregated/disaggregated_results/disaggregated_performance_analysis.png` - 详细性能分析图表

### 对比分析结果
- `server/disaggregated/performance_comparison.png` - 传统 vs 分离式性能对比图

## 🔧 技术特点

### 功耗监控
- **采样频率**: 100ms 高精度采样
- **监控指标**: 功耗、频率、利用率、温度
- **数据源**: 直接使用 nvidia-smi 硬件传感器
- **可视化**: 实时功耗变化曲线图

### 分离式架构
- **物理分离**: Prefill 和 Decode 运行在不同 GPU
- **KV 传输**: P2P NCCL 高速缓存传输
- **代理服务**: 自动处理请求转发和结果合并
- **资源隔离**: 独立的 GPU 资源管理

### 性能分析
- **时间精度**: 微秒级时间测量
- **多维度**: 时间、功耗、频率、利用率
- **对比分析**: 传统 vs 分离式架构对比
- **可视化**: 丰富的图表和统计分析

## 📋 依赖要求

### 基础依赖
- Python 3.8+
- VLLM 0.10.2+
- PyTorch
- matplotlib
- requests
- numpy

### 分离式测试额外依赖
- Quart (代理服务器)
- aiohttp (异步 HTTP 客户端)
- 至少 2 个 GPU
- P2P 通信支持 (NVLink 或 PCIe)

### 系统要求
- nvidia-smi (GPU 监控)
- CUDA 驱动
- 足够的 GPU 内存

## ⚠️ 注意事项

### 硬件要求
- **传统测试**: 1 个 GPU，8GB+ 显存
- **分离式测试**: 2 个 GPU，每个 8GB+ 显存
- **P2P 通信**: GPU 间需要支持 P2P 通信

### 软件配置
- 确保 VLLM 版本兼容
- 检查 GPU 驱动和 CUDA 版本
- 验证模型路径和权限

### 性能优化
- 测试期间关闭其他 GPU 密集型应用
- 根据硬件调整 GPU 内存利用率
- 优化 KV 传输路径 (同 NVLink 组)

## 🔍 故障排除

### 常见问题
1. **CUDA 多进程错误**: 已通过设置 `spawn` 模式解决
2. **端口冲突**: 检查端口占用情况
3. **GPU 内存不足**: 调整 `gpu_memory_utilization` 参数
4. **KV 传输失败**: 检查 GPU 间 P2P 连接

### 调试方法
- 查看服务器启动日志
- 检查 GPU 状态: `nvidia-smi`
- 验证网络连接: `curl localhost:8000/health`
- 查看代理服务器日志

## 📚 扩展功能

### 批量测试
- 支持并发请求测试
- 批量性能分析
- 负载均衡测试

### 不同模型对比
- 支持多种模型测试
- 模型性能对比分析
- 配置参数优化

### 跨节点部署
- 支持跨机器分离式部署
- 网络带宽优化
- 分布式性能分析

## 🦙 Llama.cpp 运行指令

### RTX 4080 16GB显存优化配置

#### 基础运行指令：
```bash
~/offload/llama.cpp/build/bin/llama-cli \
  -m /share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B/model.gguf \
  -ngl 35 \
  -c 4096 \
  -b 512 \
  -t 8 \
  --mlock \
  --no-mmap
```

#### 详细参数说明：

**显存优化参数：**
- `-ngl 50`: 将50层加载到GPU（约占用13-15GB显存，为系统预留1-3GB）
- `-c 4096`: 上下文长度4096 tokens
- `-b 512`: 批处理大小512
- `--mlock`: 锁定内存，防止交换到磁盘
- `--no-mmap`: 禁用内存映射，减少内存碎片

#### 最大化显存利用版本（推荐）：
```bash
~/offload/llama.cpp/build/bin/llama-cli \
  -m /share-data/wzk-1/model/DeepSeek-R1-Distill-Qwen-32B/model.gguf \
  -ngl 50 \
  -c 16384 \
  -b 2048 \
  -t 16 \
  --mlock \
  --no-mmap
```
### 建议的测试流程：

1. **先测试保守配置**，确保模型能正常加载
2. **逐步增加`-ngl`值**（30→40→50），监控显存使用
3. **使用`nvidia-smi`监控显存占用**：
   ```bash
   watch -n 1 nvidia-smi
   ```

### 显存使用估算：
- **35层**: ~11-13GB显存  
- **40层**: ~13-15GB显存
- **50层**: ~15-18GB显存（接近极限）

### 根据您的实际显存使用情况：
- **当前使用**: 8-10GB显存
- **可用空间**: 6-8GB显存
- **建议配置**: `-ngl 50` 可以充分利用剩余显存
- **极限测试**: `-ngl 60` 可以测试显存上限

**推荐从`-ngl 50`开始测试**，充分利用您的16GB显存。

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！