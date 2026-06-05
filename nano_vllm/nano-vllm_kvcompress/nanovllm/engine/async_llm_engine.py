import asyncio
import threading
import time
from typing import AsyncGenerator

# 引入底层引擎
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


# =====================================================================
# [核心魔法]：继承并强化原生引擎 (不改动原文件，通过子类打破封锁)
# =====================================================================
class StreamableLLMEngine(LLMEngine):

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """【强化1】：重写 add_request，强制让它返回 seq_id"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

        return seq.seq_id  # 🌟 将底层的身份证号暴露给外层！

    def step_stream(self):
        """【强化2】：新增一个专门用于流式的 step 函数，绝不私吞中间 Token"""
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        outputs = []
        for seq in seqs:
            # 取出刚刚新鲜出炉的那【1个】Token
            new_token_id = seq.completion_token_ids[-1]
            # 解码成文本
            new_text = self.tokenizer.decode([new_token_id])
            # 🌟 无论是否 finished，全都抛出去！
            outputs.append((seq.seq_id, new_text, seq.is_finished))

        return outputs, num_tokens


# =====================================================================
# [异步离合器层]：负责跨线程通信与并发排队
# =====================================================================
class AsyncLLMEngine:
    def __init__(self, *args, **kwargs):
        # 🌟 使用我们强化过的高级引擎
        self.engine = StreamableLLMEngine(*args, **kwargs)

        self.new_requests_queue = []
        self.stream_queues = {}  # 直接映射: seq_id -> asyncio.Queue

        self.loop = asyncio.get_event_loop()
        self.engine_thread = threading.Thread(target=self._background_engine_loop, daemon=True)
        self.engine_thread.start()

    def _background_engine_loop(self):
        while True:
            # 1. 把外面扔进来的请求塞进 GPU，并精准绑定通讯水管
            while self.new_requests_queue:
                prompt, sampling_params, stream_queue = self.new_requests_queue.pop(0)

                # 调用强化版的 add_request，拿到真实的 seq_id
                seq_id = self.engine.add_request(prompt, sampling_params)

                # 建立精准的映射关系：内部计算号 -> 外部的异步管子
                self.stream_queues[seq_id] = stream_queue

            # 2. 空转保护
            if self.engine.is_finished():
                time.sleep(0.001)
                continue

            # 3. 🌟 调用强化版的流式步进，榨出所有实时 Token
            step_outputs, _ = self.engine.step_stream()

            # 4. 精准派送快递
            for seq_id, new_text, is_finished in step_outputs:
                if seq_id in self.stream_queues:
                    q = self.stream_queues[seq_id]
                    self.loop.call_soon_threadsafe(q.put_nowait, (new_text, is_finished))

                    if is_finished:
                        del self.stream_queues[seq_id]

    async def generate_stream(self, prompt: str, sampling_params: SamplingParams) -> AsyncGenerator[str, None]:
        # 创建一根专属流式水管
        stream_queue = asyncio.Queue()

        # 把水管直接当做包裹，扔给后台线程
        self.new_requests_queue.append((prompt, sampling_params, stream_queue))

        while True:
            # 挂起等待后台往水管里滴水
            new_text, is_finished = await stream_queue.get()

            if new_text:
                yield new_text

            if is_finished:
                break