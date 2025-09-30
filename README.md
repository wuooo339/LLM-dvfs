# LLM DVFS 测试项目

这个项目用于测试大语言模型（LLM）在不同阶段的功耗和性能，支持传统单实例和分离式 Prefill+Decode 两种架构的对比测试。

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

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！