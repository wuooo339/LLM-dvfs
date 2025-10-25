# VLLM 单机多卡 P2P NCCL 分离式服务

基于 P2pNcclConnector 实现的 Prefill 和 Decode 分离架构，支持单机多卡部署。

## 📋 环境要求

- **GPU**: 至少 2 个 GPU
- **Python**: Python 3.8+
- **依赖包**: 
  ```bash
  pip install vllm>=0.9.2 quart msgpack zmq
  ```

## 🚀 快速开始

### 方法 1: 交互式菜单（推荐）

直接运行脚本，进入交互式菜单：

```bash
cd /home/user/vllm/LLM-dvfs/server/disaggregated
./test_single_machine.sh
```

然后根据提示选择操作：
- `1` - 启动代理服务器
- `2` - 启动 Prefill 实例
- `3` - 启动 Decode 实例
- `4` - 一键启动所有服务（后台模式）
- `5` - 检测并测试服务
- `6` - 查看配置信息
- `0` - 退出

### 方法 2: 命令行模式

在 3 个不同终端分别启动：

```bash
# 终端 1 - 启动代理服务器
./test_single_machine.sh proxy

# 终端 2 - 启动 Prefill 实例
./test_single_machine.sh prefill

# 终端 3 - 启动 Decode 实例
./test_single_machine.sh decode
```

### 方法 3: 一键启动所有服务

在后台启动所有服务：

```bash
./test_single_machine.sh all
```

## 🧪 测试服务

启动所有服务后，运行测试：

```bash
# 使用脚本测试
./test_single_machine.sh test

# 或手动测试
curl -X POST http://localhost:10001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/share-data/wzk-1/model/Qwen3-4B",
    "prompt": "Hello, world!",
    "max_tokens": 50,
    "temperature": 0
  }'
```

## 🔧 配置说明

### 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL | `/share-data/wzk-1/model/Qwen3-4B` | 模型路径 |
| PREFILL_GPU | 0 | Prefill 实例使用的 GPU |
| DECODE_GPU | 1 | Decode 实例使用的 GPU |
| PREFILL_HTTP_PORT | 20003 | Prefill HTTP 端口 |
| DECODE_HTTP_PORT | 20005 | Decode HTTP 端口 |
| HTTP_PROXY_PORT | 10001 | 代理服务器 HTTP 端口 |
| PROXY_PORT | 30001 | 代理服务器 ZMQ 端口 |

### 修改配置

通过环境变量修改配置：

```bash
# 使用不同的模型
MODEL=/path/to/your/model ./test_single_machine.sh

# 使用不同的 GPU
PREFILL_GPU=2 DECODE_GPU=3 ./test_single_machine.sh

# 使用不同的端口
PREFILL_HTTP_PORT=30003 DECODE_HTTP_PORT=30005 ./test_single_machine.sh
```

## 📊 监控和调试

### 查看日志

如果使用 `all` 模式启动，日志文件在：
- 代理服务器: `proxy.log`
- Prefill 实例: `prefill.log`
- Decode 实例: `decode.log`

### 健康检查

```bash
# 检查代理服务器
curl http://localhost:10001/health

# 查看统计信息
curl http://localhost:10001/stats | jq
```

### 查看进程

```bash
# 查看 vllm 进程
ps aux | grep vllm

# 查看代理进程
ps aux | grep disagg_prefill_proxy_server
```

## 🛑 停止服务

### 方法 1: 使用 Ctrl+C

如果在前台运行，直接按 `Ctrl+C` 停止

### 方法 2: Kill 进程

```bash
# 停止所有 vllm 进程
pkill -f "vllm serve"

# 停止代理服务器
pkill -f "disagg_prefill_proxy_server.py"
```

### 方法 3: 使用 PID

如果使用 `all` 模式启动，会显示所有进程的 PID：

```bash
kill <PID1> <PID2> <PID3>
```

## 📈 性能测试

使用 vllm bench 进行压力测试：

```bash
vllm bench serve \
  --backend vllm \
  --model /share-data/wzk-1/model/Qwen3-4B \
  --host localhost \
  --port 10001 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 100 \
  --request-rate 2
```

## ❓ 常见问题

### 1. 服务启动卡住

**原因**: 模型加载需要时间（通常 1-5 分钟）

**解决**: 耐心等待，或查看日志文件

### 2. 端口被占用

**错误**: `Address already in use`

**解决**: 修改端口配置或关闭占用端口的进程

```bash
# 查看端口占用
lsof -i :10001

# 修改端口
HTTP_PROXY_PORT=10002 ./test_single_machine.sh
```

### 3. GPU 内存不足

**错误**: `CUDA out of memory`

**解决**: 减小 `gpu-memory-utilization` 或使用更大显存的 GPU

### 4. 实例未注册到代理

**原因**: 
- 代理服务器未启动
- 端口配置不匹配
- 网络问题

**解决**: 
1. 确保先启动代理服务器
2. 检查端口配置是否一致
3. 查看代理服务器日志

## 📚 架构说明

### P2P NCCL 通信

- **Prefill 实例**: 负责 prefill 阶段，生成 KV cache
- **Decode 实例**: 负责 decode 阶段，从 Prefill 获取 KV cache
- **代理服务器**: 协调请求路由和服务发现

### 工作流程

1. 客户端请求 → 代理服务器
2. 代理选择 1 个 Prefill 实例和 1 个 Decode 实例
3. 代理将请求转发给 Prefill（max_tokens=1）
4. Prefill 生成 KV cache 并通过 P2P NCCL 发送给 Decode
5. 代理将原始请求转发给 Decode
6. Decode 从 Prefill 获取 KV cache，执行 decode
7. Decode 返回结果给代理
8. 代理返回结果给客户端

## 🔗 相关文档

- [vLLM P2P NCCL Connector 设计文档](../../docs/design/p2p_nccl_connector.md)
- [vLLM Disaggregated Prefill 文档](../../docs/features/disagg_prefill.md)
- [官方示例脚本](../../examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/)

## 📝 文件说明

- `disagg_prefill_proxy_server.py` - 代理服务器实现
- `test_single_machine.sh` - 服务管理脚本
- `README.md` - 本文档

