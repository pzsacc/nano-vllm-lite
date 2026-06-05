import os
import json
import uuid
from typing import List, Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# 引入我们上一节手搓的“异步离合器”引擎
from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.sampling_params import SamplingParams


# =====================================================================
# [第一层：数据契约层 Validation]
# 工业界极其看重输入校验。这里严格对齐 OpenAI 的 /v1/chat/completions 格式
# =====================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "~/huggingface/Qwen3-0.6B/"
    messages: List[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 512
    stream: bool = False  # 🌟 核心开关：决定是同步一口气返回，还是流式返回


# =====================================================================
# [第二层：全局状态与生命周期管理]
# =====================================================================
# 初始化 FastAPI 实例
app = FastAPI(title="NanoVLLM OpenAI-Compatible Server")

# 全局变量，用于在整个服务器生命周期内常驻内存
engine: AsyncLLMEngine = None
tokenizer = None


@app.on_event("startup")
async def startup_event():
    """
    【生命周期钩子】服务器启动时触发。
    整个服务器运行期间，模型只会在这里被加载一次到 GPU 显存中。
    """
    global engine, tokenizer
    print("🚀 [Server] 正在拉起 AsyncLLMEngine (初始化 GPU & 显存池)...")

    # 指向你的模型权重路径
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 初始化你的异步引擎（底层会自己去拉起那个跑 step() 的死循环线程）
    engine = AsyncLLMEngine(
        model_path,
        max_model_len=4096,
        enforce_eager=False  # 线上服务必须开启图优化，追求极致性能
    )
    print("✅ [Server] 模型加载完毕，异步网关已就绪，正在监听端口...")


# =====================================================================
# [第三层：路由与业务逻辑层 (网关枢纽)]
# 绝不包含任何张量计算，纯粹做“协议转换”和“网络分发”
# =====================================================================
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    处理客户端发来的聊天请求。
    """
    # 1. 将 OpenAI 格式的 message 转换成模型底层认识的纯文本 Prompt
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages_dict,
        tokenize=False,
        add_generation_prompt=True
    )

    # 2. 构造采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # ---------------------------------------------------------
    # 分支 A：流式生成 (Streaming) - 解决 TTFT 延迟的核心
    # ---------------------------------------------------------
    if request.stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            """
            这是一个异步生成器，像水管一样源源不断地把 engine 吐出的字推给客户端
            """
            # 调用底层引擎的异步流式接口，这里会让出主线程，FastAPI 去接待别人
            async for new_token in engine.generate_stream(prompt, sampling_params):
                # 严格按照 Server-Sent Events (SSE) 的协议规范组装 JSON
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_token}
                    }]
                }
                # SSE 格式：必须以 "data: " 开头，以 "\n\n" 结尾
                yield f"data: {json.dumps(chunk)}\n\n"

            # 生成结束时，发送 OpenAI 规定的结束标识
            yield "data: [DONE]\n\n"

        # 返回流式响应，保持 TCP 长连接不断开
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # ---------------------------------------------------------
    # 分支 B：非流式生成 (Offline/Sync) - 客户端死等完整结果
    # ---------------------------------------------------------
    else:
        full_text = ""
        # 即使是非流式请求，为了不阻塞整个服务器，我们依然走异步流式接口获取数据
        # 只是在服务器内存里把它拼接完整后，再一次性发给客户端
        async for new_token in engine.generate_stream(prompt, sampling_params):
            full_text += new_token

        # 组装 OpenAI 格式的静态完整响应
        response_json = {
            "id": request_id,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(full_text)),
                "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(full_text))
            }
        }
        return JSONResponse(content=response_json)


# =====================================================================
# [第四层：网关启动器]
# =====================================================================
if __name__ == "__main__":
    # 使用 Uvicorn 启动 ASGI 服务器
    # host="0.0.0.0" 允许局域网其他机器访问
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")