"""
engine/scheduler.py - 混合调度器（Chunked Prefill + Continuous Batching）

实现 Sarathi 风格的混合调度策略：
1. 优先调度 Decode 序列（保证低延迟）
2. 用剩余预算执行 Chunked Prefill（避免长 prompt 阻塞 decode）
3. 支持降级为传统 Prefill-first 调度（关闭 chunked_prefill 时）

调度器维护两个队列：
- waiting: 等待 prefill 的序列（新请求或被抢占的序列）
- running: 已完成 prefill 正在 decode 的序列
"""

from collections import deque

from nano_vllm.config import Config
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.engine.block_manager import BlockManager


class Scheduler:
    """混合请求调度器

    支持两种调度模式：
    - Chunked Prefill 模式：Decode 优先 + 固定分块 Prefill
    - 传统模式：Prefill 优先，Decode 后执行

    Args:
        config: 引擎配置
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.chunk_size = config.chunk_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_caching=config.enable_prefix_caching,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """所有序列是否都已处理完毕"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新序列加入等待队列

        Args:
            seq: 新到达的序列
        """
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        """中止指定序列：从队列移除并释放其 KV cache

        用于客户端断开/取消请求的场景，避免继续生成无人消费的 token。

        Args:
            seq_id: 要中止的序列 ID

        Returns:
            是否成功中止（序列不存在时返回 False）
        """
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    seq.status = SequenceStatus.ABORTED
                    self.block_manager.deallocate(seq)
                    return True
        return False

    def schedule(self) -> tuple[list[Sequence], bool]:
        """执行一步调度，返回本步要处理的序列列表

        调度策略：
        - enable_chunked_prefill=True: Decode 优先 + Chunked Prefill
        - enable_chunked_prefill=False: Prefill 优先（传统 vLLM 风格）

        Returns:
            (scheduled_seqs, has_prefill): 被调度的序列列表和是否包含 prefill
        """
        if self.enable_chunked_prefill:
            return self._schedule_chunked()
        else:
            return self._schedule_traditional()

    def _schedule_chunked(self) -> tuple[list[Sequence], bool]:
        """Chunked Prefill 混合调度（Sarathi 风格）

        Phase 1: 优先调度所有可运行的 Decode 序列
        Phase 2: 用固定 chunk_size 预算调度 Prefill 序列
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # Phase 1: 优先调度 Decode 序列
        running_seqs = deque(self.running)
        self.running.clear()
        while running_seqs:
            seq = running_seqs.popleft()
            if len(scheduled_seqs) >= self.max_num_seqs:
                self.running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if running_seqs:
                    self.preempt(running_seqs.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1
                self.running.append(seq)

        # Phase 2: Chunked Prefill（固定大小分块）
        prefill_budget = self.chunk_size

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs and prefill_budget > 0:
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
                uncomputed_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                uncomputed_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(uncomputed_tokens, prefill_budget)
            prefill_budget -= seq.num_scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        has_prefill = any(seq.is_prefill for seq in scheduled_seqs)
        return scheduled_seqs, has_prefill

    def _schedule_traditional(self) -> tuple[list[Sequence], bool]:
        """传统调度：Prefill 优先

        先处理所有等待中的 prefill 请求，没有 prefill 时再处理 decode。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # Prefill phase
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode phase
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占序列：释放其 block 并重新加入等待队列

        当 KV cache 空间不足时，优先抢占最后加入的序列（LIFO）。

        Args:
            seq: 被抢占的序列
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """后处理：更新序列状态、注册前缀缓存、检查终止条件

        Args:
            seqs: 本步处理的序列列表
            token_ids: 每个序列本步采样得到的 token ID
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
