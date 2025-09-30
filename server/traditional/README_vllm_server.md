# VLLM 服务器 Prefill/Decode 分离测试

这个项目使用 VLLM 服务器模式进行部署，并分离测试 prefill 和 decode 阶段的功耗和性能。

## 文件说明

- `start_vllm_server.sh` - VLLM 服务器启动脚本
- `test_vllm_server.py` - 客户端测试脚本
- `run_server_test.sh` - 完整测试运行脚本
- `analyze_server_results.py` - 结果分析脚本
- `gpu_monitor.py` - GPU 监控模块

## 使用步骤

### 1. 启动 VLLM 服务器

```bash
# 方法1: 使用脚本启动
./start_vllm_server.sh

# 方法2: 手动启动
vllm serve /share-data/wzk-1/model/deepseek-v2-lite \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 20 \
    --enforce-eager \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-model-len 512
```

### 2. 运行客户端测试

```bash
# 方法1: 使用完整测试脚本
./run_server_test.sh

# 方法2: 直接运行客户端测试
python3 test_vllm_server.py
```

### 3. 分析结果

```bash
python3 analyze_server_results.py
```

## 测试内容

### Prefill 阶段测试
- 设置 `max_tokens=0`，只进行 prefill 处理
- 监控 GPU 功耗、频率、利用率
- 记录处理时间和能耗

### Decode 阶段测试
- 设置 `max_tokens=16`，进行 token 生成
- 监控 GPU 功耗、频率、利用率
- 记录生成时间和能耗

## 输出结果

测试完成后，结果保存在 `vllm_server_results/` 目录中：

- `prefill_results.json` - Prefill 阶段结果
- `decode_results.json` - Decode 阶段结果
- `prefill_gpu_data.json` - Prefill GPU 监控数据
- `decode_gpu_data.json` - Decode GPU 监控数据
- `analysis_report.json` - 分析报告
- `vllm_server_comparison.png` - 对比图表

## 服务器配置参数

- **模型**: deepseek-v2-lite
- **端口**: 8000
- **CPU Offload**: 20GB
- **GPU 内存利用率**: 95%
- **最大模型长度**: 512
- **强制 Eager 模式**: 是
- **信任远程代码**: 是

## 环境变量设置

脚本会自动设置以下环境变量以避免多进程问题：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_MULTIPROC_METHOD=spawn
```

## 注意事项

1. 确保 VLLM 服务器在端口 8000 上运行
2. 测试前确保 GPU 处于空闲状态
3. 监控脚本需要 nvidia-smi 支持
4. 建议在测试期间关闭其他 GPU 密集型应用

## 故障排除

### 服务器启动失败
- 检查模型路径是否正确
- 确认 GPU 内存充足
- 检查端口 8000 是否被占用

### 客户端连接失败
- 确认服务器正在运行
- 检查防火墙设置
- 验证端口配置

### 功耗监控失败
- 确认 nvidia-smi 可用
- 检查 GPU 权限
- 验证监控脚本权限
