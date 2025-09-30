# VLLM 分离式 Prefill+Decode 性能测试

这个项目实现了真正的 VLLM prefill 和 decode 分离，支持 TTFT、TBT、E2E Latency 测量和详细的功耗变化分析。

## 架构概述

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client        │    │   Proxy Server  │    │   Prefill       │
│   Request       │───▶│   (Port 8000)   │───▶│   Instance      │
│                 │    │                 │    │   (GPU 0, 8100) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                │               │   KV Cache      │
                                │               │   Transfer      │
                                │               │   (P2P NCCL)    │
                                │               └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Decode        │◀───│   KV Cache      │
                       │   Instance      │    │   Consumer      │
                       │   (GPU 1, 8200) │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## 文件说明

- `start_disaggregated_servers.sh` - 启动分离式服务器脚本
- `disagg_prefill_proxy_server.py` - 代理服务器实现
- `test_disaggregated_performance.py` - 分离式性能测试脚本
- `run_disaggregated_test.sh` - 完整测试运行脚本
- `compare_performance.py` - 性能对比分析脚本

## 使用步骤

### 1. 启动分离式服务器

```bash
./start_disaggregated_servers.sh
```

这将启动：
- Prefill 实例 (GPU 0, 端口 8100)
- Decode 实例 (GPU 1, 端口 8200)
- 代理服务器 (端口 8000)

### 2. 运行性能测试

```bash
./run_disaggregated_test.sh
```

### 3. 对比分析

```bash
python3 compare_performance.py
```

## 性能指标

### TTFT (Time to First Token)
- 从请求开始到第一个 token 生成的时间
- 主要包含 prefill 处理时间

### TBT (Time Between Tokens)
- 生成每个 token 之间的平均时间
- 反映 decode 阶段的效率

### E2E Latency (End-to-End Latency)
- 从请求开始到完整响应的时间
- 包含 prefill + KV 传输 + decode 的总时间

## 功耗监控

### 实时监控
- GPU 0 (Prefill): 监控 prefill 阶段的功耗变化
- GPU 1 (Decode): 监控 decode 阶段的功耗变化
- 采样频率: 100ms

### 监控指标
- 功耗 (Power Draw)
- 图形频率 (Graphics Clock)
- 显存频率 (Memory Clock)
- GPU 利用率 (GPU Utilization)
- 温度 (Temperature)

## 输出结果

测试完成后，结果保存在 `disaggregated_results/` 目录中：

- `disaggregated_results.json` - 完整测试结果
- `prefill_gpu_data.json` - Prefill GPU 监控数据
- `decode_gpu_data.json` - Decode GPU 监控数据
- `disaggregated_performance_analysis.png` - 详细性能分析图表

## 关键特性

### 1. 真正的物理分离
- Prefill 和 Decode 运行在不同的 GPU 上
- 通过 P2P NCCL 进行高速 KV 缓存传输
- 支持独立的资源优化和功耗控制

### 2. 精确的时间测量
- TTFT: 首 token 时间
- TBT: token 间时间
- E2E: 端到端延迟
- 分别测量 prefill 和 decode 阶段

### 3. 详细的功耗分析
- 实时功耗变化曲线
- 频率变化监控
- 平均/最大功耗统计
- 双 GPU 对比分析

### 4. 性能对比
- 传统单实例 vs 分离式架构
- 时间性能对比
- 功耗效率对比
- 处理效率分析

## 配置参数

### 服务器配置
- **模型**: deepseek-v2-lite
- **最大模型长度**: 512
- **GPU 内存利用率**: 80%
- **强制 Eager 模式**: 是
- **KV 传输**: P2P NCCL

### 监控配置
- **采样间隔**: 0.1 秒
- **监控 GPU**: 0 (Prefill), 1 (Decode)
- **超时时间**: 120 秒

## 注意事项

1. **硬件要求**:
   - 至少需要 2 个 GPU
   - 支持 P2P 通信 (NVLink 或 PCIe)
   - 足够的 GPU 内存

2. **软件依赖**:
   - VLLM 0.10.2+
   - Python 3.8+
   - Quart (代理服务器)
   - matplotlib (图表生成)

3. **性能优化**:
   - 将 prefill 和 decode 放在同 NVLink 组的 GPU 上
   - 控制上下文长度以优化 KV 传输
   - 根据负载调整 GPU 频率

## 故障排除

### 服务器启动失败
- 检查 GPU 可用性: `nvidia-smi`
- 确认端口未被占用: `netstat -tlnp | grep :8100`
- 检查模型路径是否正确

### KV 传输失败
- 确认 GPU 间 P2P 支持: `nvidia-smi topo -m`
- 检查 NCCL 环境变量
- 验证 GPU 拓扑连接

### 性能测试失败
- 确认所有服务器正常运行
- 检查网络连接
- 查看代理服务器日志

## 扩展功能

### 批量测试
可以修改测试脚本支持批量请求测试，评估并发性能。

### 不同模型对比
可以修改模型路径测试不同模型的性能差异。

### 跨节点部署
可以配置 `--kv-ip` 和 `--kv-port` 支持跨机器部署。
