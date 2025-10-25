#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VLLM 分离式 Prefill+Decode 代理服务器 (P2P NCCL 版本)
基于 P2pNcclConnector 实现真正的 prefill 和 decode 分离
支持单机多卡部署和测试
"""

import os
import socket
import threading
import time
import uuid
from typing import Any, Dict

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

# 配置参数
PROXY_HTTP_PORT = int(os.environ.get("HTTP_PROXY_PORT", "10001"))  # 接收客户端请求的端口
PROXY_ZMQ_PORT = int(os.environ.get("PROXY_PORT", "30001"))   # 接收服务发现的端口
DEFAULT_PING_SECONDS = 5  # 心跳超时时间

# 静态实例配置（从环境变量获取）
PREFILL_ADDRS = os.environ.get("PREFILL_ADDRS", "")  # 格式: "host1:port1,host2:port2"
DECODE_ADDRS = os.environ.get("DECODE_ADDRS", "")    # 格式: "host1:port1,host2:port2"

# 全局变量
count = 0
prefill_instances: Dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: Dict[str, Any] = {}   # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

app = Quart(__name__)


def _remove_oldest_instances(instances: Dict[str, Any]) -> None:
    """移除超时的实例"""
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴 移除超时实例 [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    """监听服务发现注册"""
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)
                    if node is None:
                        print(f"🔵 添加 Prefill 实例 [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
                    if node is None:
                        print(f"🔵 添加 Decode 实例 [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")
            else:
                print(
                    f"⚠️  收到未知类型的消息 from {remote_address}, data: {data}"
                )


def init_static_instances():
    """从环境变量初始化静态实例配置"""
    global prefill_instances, decode_instances, prefill_cv, decode_cv
    
    # 初始化 Prefill 实例
    if PREFILL_ADDRS:
        addrs = [addr.strip() for addr in PREFILL_ADDRS.split(",") if addr.strip()]
        with prefill_cv:
            for addr in addrs:
                # 使用 HTTP 地址作为 ZMQ 地址（P2P NCCL 会直接使用 HTTP 地址）
                prefill_instances[addr] = (addr, time.time() + 999999)  # 长期有效
                print(f"🔵 添加静态 Prefill 实例: {addr}")
    
    # 初始化 Decode 实例
    if DECODE_ADDRS:
        addrs = [addr.strip() for addr in DECODE_ADDRS.split(",") if addr.strip()]
        with decode_cv:
            for addr in addrs:
                decode_instances[addr] = (addr, time.time() + 999999)  # 长期有效
                print(f"🔵 添加静态 Decode 实例: {addr}")
    
    if prefill_instances or decode_instances:
        print(f"✅ 静态配置初始化完成: {len(prefill_instances)} 个 Prefill, {len(decode_instances)} 个 Decode")


def start_service_discovery(hostname, port):
    """启动服务发现"""
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("端口不能为 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    print(f"📡 服务发现已启动: tcp://{hostname}:{port}")
    return _listener_thread


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def random_uuid() -> str:
    """生成随机 UUID"""
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    """转发请求到 prefill 或 decode 实例"""
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                "X-Request-Id": request_id,
                "Content-Type": "application/json",
            }
            print(f"    发送请求: {url}, request_id: {request_id[:50]}...")
            async with session.post(url=url, json=data, headers=headers) as response:
                print(f"    收到响应: status={response.status}")
                if response.status == 200:
                    # 流式返回
                    chunk_count = 0
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        chunk_count += 1
                        yield chunk_bytes
                    print(f"    完成，共 {chunk_count} 个数据块")
                else:
                    error_text = await response.text()
                    print(f"    ❌ 请求失败: {response.status} - {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
    except Exception as e:
        print(f"    ❌ forward_request 异常: {e}")
        raise


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """处理客户端请求"""
    try:
        original_request_data = await request.get_json()

        # 检查是否有可用的实例
        global prefill_instances, decode_instances
        if not prefill_instances:
            return {"error": "没有可用的 Prefill 实例"}, 503
        if not decode_instances:
            return {"error": "没有可用的 Decode 实例"}, 503

        # 创建 Prefill 请求（只生成 1 个 token）
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        # 轮询选择 prefill 和 decode 实例
        global count
        global prefill_cv, decode_cv
        
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"📨 请求 #{count}: [Prefill HTTP:{prefill_addr}, ZMQ:{prefill_zmq_addr}] "
            f"👉 [Decode HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        # 生成特殊的 request_id，包含 prefill 和 decode 的地址信息
        # P2pNcclConnector 会解析这个 ID 来建立 P2P 连接
        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        # 步骤 1: 发送到 Prefill 实例完成 prefill
        print(f"  → 发送 Prefill 请求到: http://{prefill_addr}{request.path}")
        prefill_success = False
        try:
            async for chunk in forward_request(
                f"http://{prefill_addr}{request.path}", prefill_request, request_id
            ):
                prefill_success = True
                # 消费响应但不返回
                pass
            print(f"  ✓ Prefill 完成")
        except Exception as e:
            print(f"  ✗ Prefill 失败: {e}")
            return {"error": f"Prefill 失败: {str(e)}"}, 500

        # 步骤 2: 发送到 Decode 实例完成 decode
        # Decode 实例会通过 request_id 从 Prefill 实例获取 KV cache
        print(f"  → 发送 Decode 请求到: http://{decode_addr}{request.path}")
        try:
            generator = forward_request(
                f"http://{decode_addr}{request.path}", original_request_data, request_id
            )
            response = await make_response(generator)
            response.timeout = None
            print(f"  ✓ Decode 完成")
            return response
        except Exception as e:
            print(f"  ✗ Decode 失败: {e}")
            return {"error": f"Decode 失败: {str(e)}"}, 500

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("❌ 代理服务器发生错误")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        return {"error": str(e)}, 500


@app.route("/health", methods=["GET"])
async def health():
    """健康检查"""
    status = {
        "status": "healthy",
        "prefill_instances": len(prefill_instances),
        "decode_instances": len(decode_instances),
        "prefill_addrs": list(prefill_instances.keys()),
        "decode_addrs": list(decode_instances.keys()),
    }
    return status


@app.route("/stats", methods=["GET"])
async def stats():
    """获取统计信息"""
    return {
        "total_requests": count,
        "prefill_instances": {
            addr: {"zmq_address": zmq_addr[0], "expires_at": zmq_addr[1]}
            for addr, zmq_addr in prefill_instances.items()
        },
        "decode_instances": {
            addr: {"zmq_address": zmq_addr[0], "expires_at": zmq_addr[1]}
            for addr, zmq_addr in decode_instances.items()
        },
    }


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 启动 VLLM P2P NCCL 分离式代理服务器 (单机多卡版本)")
    print("=" * 80)
    print(f"🌐 HTTP 服务端口: {PROXY_HTTP_PORT}")
    print(f"📡 ZMQ 服务发现端口: {PROXY_ZMQ_PORT}")
    print(f"⏱️  心跳超时: {DEFAULT_PING_SECONDS} 秒")
    print()
    print("📋 使用说明:")
    print("  1. 确保至少有 2 个 GPU 可用")
    print("  2. 启动 Prefill 实例 (kv_role=kv_producer)")
    print("  3. 启动 Decode 实例 (kv_role=kv_consumer)")
    print("  4. 实例会自动通过 ZMQ 注册到代理服务器")
    print("     或通过环境变量 PREFILL_ADDRS 和 DECODE_ADDRS 静态配置")
    print(f"  5. 客户端请求发送到: http://localhost:{PROXY_HTTP_PORT}/v1/completions")
    print()
    print("🔍 监控端点:")
    print(f"  - 健康检查: http://localhost:{PROXY_HTTP_PORT}/health")
    print(f"  - 统计信息: http://localhost:{PROXY_HTTP_PORT}/stats")
    print("=" * 80)
    print()
    
    # 初始化静态实例配置
    init_static_instances()
    print()
    
    # 启动服务发现（支持动态注册）
    t = start_service_discovery("0.0.0.0", PROXY_ZMQ_PORT)
    
    # 启动 HTTP 服务
    app.run(host="0.0.0.0", port=PROXY_HTTP_PORT)
    t.join()
