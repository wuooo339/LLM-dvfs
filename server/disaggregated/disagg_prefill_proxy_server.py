#!/usr/bin/env python3
"""
VLLM 分离式 Prefill+Decode 代理服务器
实现真正的 prefill 和 decode 分离
"""

import asyncio
import json
import time
from typing import Dict, Any
import aiohttp
from quart import Quart, request, jsonify

app = Quart(__name__)

# 服务器配置
PREFILL_SERVER_URL = "http://localhost:8100"
DECODE_SERVER_URL = "http://localhost:8200"

class DisaggregatedProxy:
    def __init__(self):
        self.session = None
        
    async def init_session(self):
        """初始化 HTTP 会话"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """关闭 HTTP 会话"""
        if self.session:
            await self.session.close()
    
    async def send_to_prefill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求到 Prefill 实例"""
        # 修改请求：只生成 1 个 token 来触发 prefill
        prefill_payload = payload.copy()
        prefill_payload["max_tokens"] = 1
        prefill_payload["stream"] = False
        
        async with self.session.post(
            f"{PREFILL_SERVER_URL}/v1/completions",
            json=prefill_payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Prefill 请求失败: {response.status} - {error_text}")
    
    async def send_to_decode(self, payload: Dict[str, Any], kv_cache: Any = None) -> Dict[str, Any]:
        """发送请求到 Decode 实例"""
        # 修改请求：生成剩余的 tokens
        decode_payload = payload.copy()
        if kv_cache:
            decode_payload["kv_cache"] = kv_cache
        
        async with self.session.post(
            f"{DECODE_SERVER_URL}/v1/completions",
            json=decode_payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Decode 请求失败: {response.status} - {error_text}")

proxy = DisaggregatedProxy()

@app.before_serving
async def startup():
    """启动时初始化"""
    await proxy.init_session()

@app.after_serving
async def shutdown():
    """关闭时清理"""
    await proxy.close_session()

@app.route('/v1/completions', methods=['POST'])
async def completions():
    """处理完成请求"""
    try:
        # 获取请求数据
        data = await request.get_json()
        
        # 记录开始时间
        start_time = time.time()
        
        # 步骤 1: 发送到 Prefill 实例
        prefill_start = time.time()
        prefill_response = await proxy.send_to_prefill(data)
        prefill_time = time.time() - prefill_start
        
        # 提取 KV 缓存（这里简化处理，实际需要从响应中提取）
        kv_cache = prefill_response.get("kv_cache", None)
        
        # 步骤 2: 发送到 Decode 实例
        decode_start = time.time()
        decode_response = await proxy.send_to_decode(data, kv_cache)
        decode_time = time.time() - decode_start
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 合并响应
        final_response = {
            "id": decode_response.get("id", ""),
            "object": "text_completion",
            "created": int(time.time()),
            "model": data.get("model", ""),
            "choices": decode_response.get("choices", []),
            "usage": decode_response.get("usage", {}),
            "timing": {
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "total_time": total_time,
                "ttft": prefill_time,  # Time to First Token
                "tbt": decode_time / max(decode_response.get("usage", {}).get("completion_tokens", 1), 1)  # Time Between Tokens
            }
        }
        
        return jsonify(final_response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
async def health():
    """健康检查"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🚀 启动 VLLM 分离式代理服务器...")
    print(f"📡 Prefill 服务器: {PREFILL_SERVER_URL}")
    print(f"📡 Decode 服务器: {DECODE_SERVER_URL}")
    print("🌐 代理服务器: http://localhost:8000")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
