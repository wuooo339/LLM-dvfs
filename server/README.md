# VLLM 测试套件

这个目录包含了 VLLM 的两种测试模式：传统单实例模式和分离式 Prefill+Decode 模式。

## 📁 目录结构

```
server/
├── traditional/              # 传统单实例测试
│   ├── start_vllm_server.sh      # 启动单实例服务器
│   ├── test_vllm_server.py       # 单实例测试脚本
│   ├── run_server_test.sh        # 单实例测试运行脚本
│   ├── analyze_server_results.py # 单实例结果分析
│   ├── README_vllm_server.md     # 单实例测试说明
│   └── vllm_server_results/      # 单实例测试结果目录
│
├── disaggregated/            # 分离式测试
│   ├── start_disaggregated_servers.sh    # 启动分离式服务器
│   ├── disagg_prefill_proxy_server.py    # 代理服务器
│   ├── test_disaggregated_performance.py # 分离式测试脚本
│   ├── run_disaggregated_test.sh         # 分离式测试运行脚本
│   ├── compare_performance.py            # 性能对比分析
│   ├── README_disaggregated.md           # 分离式测试说明
│   └── disaggregated_results/            # 分离式测试结果目录
│
└── reference/                # 官方参考文件
    └── disaggregated_prefill.sh          # 官方分离式示例
```

## 🚀 快速开始

### 传统单实例测试

```bash
# 进入传统测试目录
cd traditional

# 1. 启动单实例服务器
./start_vllm_server.sh

# 2. 运行测试（新终端）
./run_server_test.sh

# 3. 分析结果
python3 analyze_server_results.py
```

### 传统单实例多批次测试

```bash
# 进入传统测试目录
cd traditional

# 1. 启动单实例服务器
./start_vllm_server.sh

# 2. 运行测试（新终端）
./run_more_batches_test.sh

```

### 分离式测试

```bash
# 进入分离式测试目录
cd disaggregated

# 1. 启动分离式服务器（需要 2 个 GPU）
./start_disaggregated_servers.sh

# 2. 运行分离式测试（新终端）
./run_disaggregated_test.sh

# 3. 对比分析
python3 compare_performance.py
```

## 📊 测试模式对比

| 特性 | 传统单实例 | 分离式 |
|------|------------|--------|
| **GPU 需求** | 1 个 GPU | 2 个 GPU |
| **架构** | 单实例处理 | Prefill + Decode 分离 |
| **KV 传输** | 无 | P2P NCCL |
| **测试指标** | 模拟分离 | 真实分离 |
| **复杂度** | 简单 | 复杂 |
| **性能精度** | 模拟 | 真实 |

## 🔧 技术特点

### 传统单实例模式
- **模拟分离**: 通过 `max_tokens` 参数模拟 prefill 和 decode
- **简单部署**: 单服务器，易于配置
- **资源要求低**: 只需要一个 GPU
- **适合**: 初步测试和概念验证

### 分离式模式
- **真实分离**: Prefill 和 Decode 运行在不同 GPU
- **KV 传输**: 通过 P2P NCCL 进行高速缓存传输
- **精确指标**: TTFT、TBT、E2E Latency 测量
- **适合**: 生产环境和高精度测试

## 📈 输出结果

### 传统单实例结果
- `traditional/vllm_server_results/` - 所有测试结果
- 性能对比图表和分析报告

### 分离式结果
- `disaggregated/disaggregated_results/` - 所有测试结果
- 双 GPU 监控数据和性能分析
- 与传统模式的对比分析

## ⚠️ 注意事项

1. **路径依赖**: 各目录中的脚本已更新路径引用
2. **GPU 要求**: 分离式测试需要至少 2 个 GPU
3. **P2P 通信**: 分离式测试需要 GPU 间支持 P2P 通信
4. **端口管理**: 不同模式使用不同端口，避免冲突

## 🔍 故障排除

### 常见问题
1. **路径错误**: 确保在正确的目录下运行脚本
2. **GPU 不足**: 分离式测试需要 2 个 GPU
3. **端口冲突**: 检查端口占用情况
4. **依赖缺失**: 确保所有依赖已安装

### 调试方法
- 查看各目录下的 README 文件
- 检查脚本中的路径引用
- 验证 GPU 状态和端口可用性
- 查看详细的错误日志

## 📚 扩展功能

- **批量测试**: 支持多种配置的批量测试
- **性能对比**: 自动生成对比分析报告
- **结果可视化**: 丰富的图表和统计分析
- **配置优化**: 支持不同硬件配置的优化

## 🤝 贡献

欢迎为不同测试模式贡献新的功能和改进！
