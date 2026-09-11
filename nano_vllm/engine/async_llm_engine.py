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


def _utf8_truncated_len(data: bytes) -> int:
    """返回 data 尾部不完整 UTF-8 字符的字节数（0 表示完整）

    原理：UTF-8 多字节序列的首字节形如 110xxxxx/1110xxxx/11110xxx，
    续字节形如 10xxxxxx。从尾部回扫到首个非续字节即可判断序列是否完整。
    """
    if not data or data[-1] < 0x80:
        return 0
    for i in range(1, min(4, len(data)) + 1):
        byte = data[-i]
        if byte & 0xC0 != 0x80:  # 找到序列首字节
            # 完整序列长度 = 首字节高位 1 的个数
            if byte & 0xE0 == 0xC0:
                seq_len = 2
            elif byte & 0xF0 == 0xE0:
                seq_len = 3
            elif byte & 0xF8 == 0xF0:
                seq_len = 4
            else:  # 非法字节，交给下游 errors=replace 处理
                return 0
            return 0 if i == seq_len else i
    return len(data)


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
                detok = self._detokenizers.setdefault(seq.seq_id, IncrementalDetokenizer(self.tokenizer))
                text = detok.decode(new_token_id)
                results.append((seq.seq_id, text, seq.is_finished))
                if seq.is_finished:
                    self._detokenizers.pop(seq.seq_id, None)
        return results


class AsyncLLMEngine:
    """异步流式推理引擎

    提供 async generator 接口，适用于 FastAPI/WebSocket 等异步场景。
    后台线程驱动引擎循环，通过 asyncio.Queue 与前端通信。

    已知局限（见 docs/07-async-engine.md）：
    - 空闲时轮询等待（time.sleep），可改条件变量
    - new_requests 跨线程读写依赖 GIL，无显式锁

    Args:
        model: 模型路径
        **kwargs: Config 参数
    """

    def __init__(self, model: str, **kwargs):
        self.engine = StreamableLLMEngine(model, **kwargs)
        self.engine._detokenizers = {}
        self.new_requests: list[tuple] = []
        self.stream_queues: dict[int, asyncio.Queue] = {}
        self.abort_requests: set[int] = set()
        self.loop = None

        # 启动后台引擎线程
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def _background_loop(self):
        """后台引擎循环

        持续处理新请求并驱动推理，通过 Queue 将 token 分发给对应请求。
        """
        while True:
            # 处理客户端取消请求
            while self.abort_requests:
                seq_id = self.abort_requests.pop()
                self.engine.scheduler.abort(seq_id)
                queue = self.stream_queues.pop(seq_id, None)
                if queue:
                    self.loop.call_soon_threadsafe(
                        queue.put_nowait, ("", True))

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
                        queue.put_nowait, (text, is_finished))
                    if is_finished:
                        del self.stream_queues[seq_id]

    async def generate_stream(self, prompt, sampling_params: SamplingParams = SamplingParams()) -> AsyncGenerator[str, None]:
        """异步流式生成

        Args:
            prompt: 输入文本或 token_ids
            sampling_params: 采样参数

        Yields:
            逐 token 生成的文本片段（多字节 UTF-8 字符保证完整）
        """
        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        stream_queue: asyncio.Queue = asyncio.Queue()
        detokenizer = IncrementalDetokenizer(self.engine.tokenizer)
        self.new_requests.append((prompt, sampling_params, stream_queue))

        try:
            while True:
                text, is_finished = await stream_queue.get()
                if text:
                    yield text
                if is_finished:
                    break
        except asyncio.CancelledError:
            # 客户端断开/取消：通知引擎线程中止对应序列，避免空转生成
            seq_id = self._find_seq_id(stream_queue)
            if seq_id is not None:
                self.abort_requests.add(seq_id)
            raise

    def _find_seq_id(self, stream_queue: asyncio.Queue) -> int | None:
        """根据 queue 反查序列 ID（提交后即登记在 new_requests/stream_queues）"""
        for seq_id, q in self.stream_queues.items():
            if q is stream_queue:
                return seq_id
        for prompt, sp, q in self.new_requests:
            if q is stream_queue:
                # 尚未被引擎线程接收，直接从提交队列移除即可
                self.new_requests.remove((prompt, sp, q))
                return None
        return None


class IncrementalDetokenizer:
    """增量反解码器：正确处理跨 token 的多字节 UTF-8 字符

    单 token 直接 decode 会在生僻字/emoji 等场景把多字节序列截断，
    输出 U+FFFD 替换符。这里全量解码但只 flush 完整的字符增量：
    decode 本身是 O(总长度)，对典型输出长度（<4KB）开销可忽略。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self._emitted_len = 0  # 已输出的字节数（按全量解码后的字节流计）

    def decode(self, new_token_id: int) -> str:
        """追加一个 token，返回完整可显示的增量文本（可能为空）"""
        self.token_ids.append(new_token_id)
        text = self.tokenizer.decode(self.token_ids, skip_special_tokens=False)
        cur_bytes = text.encode("utf-8", errors="replace")
        delta = cur_bytes[self._emitted_len:]
        flush_len = len(delta) - _utf8_truncated_len(delta)
        self._emitted_len += flush_len
        return delta[:flush_len].decode("utf-8", errors="replace")
