"""
engine/llm_engine.py - 同步推理引擎

顶层编排器，协调 Scheduler 和 ModelRunner 完成完整推理流程：
1. 接收用户请求（文本或 token_ids）
2. 驱动 schedule → model_run → postprocess 循环
3. 管理 TP worker 子进程生命周期
4. 收集并返回生成结果

API 设计对标 vLLM 的 LLM 类，支持 batch offline inference。
"""

import atexit
from dataclasses import fields
from time import perf_counter

from tqdm import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nano_vllm.config import Config
from nano_vllm.sampling_params import SamplingParams
from nano_vllm.engine.sequence import Sequence
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.model_runner import ModelRunner


class LLMEngine:
    """同步 LLM 推理引擎

    提供 generate() 接口进行 batch 离线推理。
    内部驱动调度循环直到所有请求完成。

    Args:
        model: 模型路径
        **kwargs: 其他 Config 参数
    """

    def __init__(self, model: str, **kwargs):
        # 过滤有效配置字段
        valid_fields = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        self.config = Config(model=model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size

        # 启动 TP worker 子进程
        self.workers = []
        self.events = []
        if self.config.tensor_parallel_size > 1:
            ctx = mp.get_context("spawn")
            for rank in range(1, self.config.tensor_parallel_size):
                event = ctx.Event()
                self.events.append(event)
                worker = ctx.Process(
                    target=self._worker_fn,
                    args=(self.config, rank, event),
                    daemon=True,
                )
                worker.start()
                self.workers.append(worker)

        # 初始化 rank 0 的 ModelRunner
        self.model_runner = ModelRunner(self.config, rank=0, events=self.events)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.config.eos = self.tokenizer.eos_token_id

        # 初始化 scheduler
        self.scheduler = Scheduler(self.config)

        atexit.register(self.exit)

    @staticmethod
    def _worker_fn(config: Config, rank: int, event):
        """TP worker 子进程入口函数"""
        ModelRunner(config, rank=rank, events=event)

    def exit(self):
        """清理引擎资源"""
        self.model_runner.call("exit")
        del self.model_runner
        for worker in self.workers:
            worker.join(timeout=5)

    def add_request(self, prompt, sampling_params: SamplingParams = SamplingParams()):
        """添加推理请求

        Args:
            prompt: 文本字符串或 token ID 列表
            sampling_params: 采样参数
        """
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> list[tuple[int, list[int]]]:
        """执行一步推理循环

        Returns:
            本步完成的序列列表 [(seq_id, completion_token_ids), ...]
        """
        seqs, has_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, has_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        finished = []
        for seq in seqs:
            if seq.is_finished:
                finished.append((seq.seq_id, seq.completion_token_ids))
        return finished

    def is_finished(self) -> bool:
        """所有请求是否已处理完毕"""
        return self.scheduler.is_finished()

    def generate(self, prompts: list, sampling_params=None) -> list[dict]:
        """批量离线推理

        Args:
            prompts: 输入 prompt 列表（字符串或 token_ids）
            sampling_params: 采样参数（单个或列表）

        Returns:
            按请求顺序排列的结果列表，每项包含 {"text": str, "token_ids": list}
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 推理循环
        outputs = {}
        pbar = tqdm(desc="Generating", unit="tok")
        start_time = perf_counter()

        while not self.is_finished():
            finished = self.step()
            for seq_id, completion_tokens in finished:
                outputs[seq_id] = completion_tokens
            pbar.update(1)

        elapsed = perf_counter() - start_time
        total_tokens = sum(len(t) for t in outputs.values())
        pbar.set_postfix({"tok/s": f"{total_tokens / elapsed:.1f}"})
        pbar.close()

        # 按 seq_id 排序并解码
        results = []
        for seq_id in sorted(outputs.keys()):
            token_ids = outputs[seq_id]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": token_ids})
        return results
