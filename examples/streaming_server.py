"""
examples/streaming_server.py - OpenAI 兼容的流式 API Server

提供 /v1/chat/completions 和 /v1/completions 接口，
支持 SSE（Server-Sent Events）流式输出。

用法：
    python examples/streaming_server.py --model /path/to/Qwen3-0.6B --port 8000

测试：
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello", "max_tokens": 64, "stream": true}'
"""

import argparse
import asyncio
import json
import time
import uuid

from nano_vllm.engine.async_llm_engine import AsyncLLMEngine
from nano_vllm.sampling_params import SamplingParams

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
except ImportError:
    raise ImportError("请安装 server 依赖: pip install fastapi uvicorn sse-starlette")


app = FastAPI(title="nano-vllm API Server")
engine: AsyncLLMEngine = None


@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI Completions API 兼容接口"""
    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 64)
    temperature = body.get("temperature", 1.0)
    stream = body.get("stream", False)

    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    if stream:
        return StreamingResponse(
            _stream_response(prompt, sp),
            media_type="text/event-stream",
        )
    else:
        text = ""
        async for chunk in engine.generate_stream(prompt, sp):
            text += chunk
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        })


async def _stream_response(prompt: str, sp: SamplingParams):
    """SSE 流式响应生成器"""
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    async for chunk in engine.generate_stream(prompt, sp):
        data = {
            "id": request_id,
            "object": "text_completion",
            "choices": [{"text": chunk, "index": 0}],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return JSONResponse({
        "object": "list",
        "data": [{"id": "nano-vllm", "object": "model"}],
    })


def main():
    global engine
    parser = argparse.ArgumentParser(description="nano-vllm API Server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = AsyncLLMEngine(model=args.model, max_model_len=4096)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
