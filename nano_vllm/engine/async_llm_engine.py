"""
engine/async_llm_engine.py - 异步流式推理引擎

在同步引擎基础上提供：
1. StreamableLLMEngine: 支持逐 token 流式输出的引擎变体
2. AsyncLLMEngine: 基于 asyncio 的异步接口，适配 API Server

架构设计：
- 后台线程运行引擎循环（避免阻塞 event loop）
- 每个请求分配独立的 asyncio.Queue 接收流式 token
- 通过 loop.call_soon_threadsafe 实现线程安全的跨线程通信
"""

import asyncio
import threading
import time
from typing import AsyncGenerator

from nano_vllm.engine.llm_engine import LLMEngine
from nano_vllm.engine.sequence import Sequence
from nano_vllm.sampling_params import SamplingParams


class StreamableLLMEngine(LLMEngine):
    """支持流式输出的引擎

    重写 step 方法，每步为所有活跃序列产出新 token（而非仅输出已完成序列）。
    """

    def add_request(self, prompt, sampling_params: SamplingParams = SamplingParams()) -> int:
        """添加请求并返回序列 ID

        Args:
            prompt: 文本或 token_ids
            sampling_params: 采样参数

        Returns:
            分配的 seq_id
        """
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step_stream(self) -> list[tuple[int, str, bool]]:
        """流式推理步骤

        Returns:
            [(seq_id, new_token_text, is_finished), ...] 每个活跃序列的输出
        """
        seqs, has_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, has_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        results = []
        for seq in seqs:
            if seq.completion_token_ids:
                new_token_id = seq.completion_token_ids[-1]
                text = self.tokenizer.decode([new_token_id], skip_special_tokens=False)
                results.append((seq.seq_id, text, seq.is_finished))
        return results


class AsyncLLMEngine:
    """异步流式推理引擎

    提供 async generator 接口，适用于 FastAPI/WebSocket 等异步场景。
    后台线程驱动引擎循环，通过 asyncio.Queue 与前端通信。

    Args:
        model: 模型路径
        **kwargs: Config 参数
    """

    def __init__(self, model: str, **kwargs):
        self.engine = StreamableLLMEngine(model, **kwargs)
        self.new_requests: list[tuple] = []
        self.stream_queues: dict[int, asyncio.Queue] = {}
        self.loop = asyncio.get_event_loop()

        # 启动后台引擎线程
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def _background_loop(self):
        """后台引擎循环

        持续处理新请求并驱动推理，通过 Queue 将 token 分发给对应请求。
        """
        while True:
            # 处理新请求
            while self.new_requests:
                prompt, sp, queue = self.new_requests.pop(0)
                seq_id = self.engine.add_request(prompt, sp)
                self.stream_queues[seq_id] = queue

            # 空闲等待
            if self.engine.is_finished() and not self.new_requests:
                time.sleep(0.001)
                continue

            # 执行一步推理
            results = self.engine.step_stream()
            for seq_id, text, is_finished in results:
                queue = self.stream_queues.get(seq_id)
                if queue:
                    self.loop.call_soon_threadsafe(
                        queue.put_nowait, (text, is_finished)
                    )
                    if is_finished:
                        del self.stream_queues[seq_id]

    async def generate_stream(self, prompt, sampling_params: SamplingParams = SamplingParams()) -> AsyncGenerator[str, None]:
        """异步流式生成

        Args:
            prompt: 输入文本或 token_ids
            sampling_params: 采样参数

        Yields:
            逐 token 生成的文本片段
        """
        stream_queue: asyncio.Queue = asyncio.Queue()
        self.new_requests.append((prompt, sampling_params, stream_queue))

        while True:
            text, is_finished = await stream_queue.get()
            if text:
                yield text
            if is_finished:
                break
