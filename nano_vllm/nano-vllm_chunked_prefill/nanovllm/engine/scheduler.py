from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self.chunk_size = config.chunk_size

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # TODO
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ==========================================
        # 1. 优先调度 Decode 阶段
        # ==========================================
        running_seqs = deque(self.running)		# 复制running队列 方便在遍历时修改原队列
        self.running.clear()					# 清空running队列
        while running_seqs:
            seq = running_seqs.popleft()

            # 如果待处理队列长度产过最大并发处理队列长度就将请求填充到runing队列
            if len(scheduled_seqs) >= self.max_num_seqs:
                self.running.append(seq)
                continue

            # 如果没有block分配
            while not self.block_manager.can_append(seq):
                if running_seqs:	# 若running队列不为空就把running中的最后一个请求清出到waiting
                    self.preempt(running_seqs.pop())
                else:				# 否则就清出当前请求
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1
                self.running.append(seq)

        # ==========================================
        # 2. 严格固定大小分块 (Sarathi/主流 Chunked Prefill)
        # ==========================================
        # 【核心差异】不再使用 (max_num_batched_tokens - num_batched_tokens)
        # 而是给 Prefill 赋予一个独立的、固定的预算
        prefill_budget = self.chunk_size

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs and prefill_budget > 0:
            seq = self.waiting[0]

            # 若是新请求
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)	# 是否有block
                if num_cached_blocks == -1:
                    break  # 显存不足跳出
                self.block_manager.allocate(seq, num_cached_blocks)			# 分配block
                uncomputed_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                uncomputed_tokens = seq.num_tokens - seq.num_cached_tokens

            # 严格按照剩余的 prefill_budget 预算切块
            seq.num_scheduled_tokens = min(uncomputed_tokens, prefill_budget)

            # 扣除预算，并增加当前 Batch 的总 Token 数
            prefill_budget -= seq.num_scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果当前 Chunk 走完后，序列已满，转入 RUNNING
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        has_prefill = any(seq.is_prefill for seq in scheduled_seqs)

        return scheduled_seqs, has_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # TODO
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:  # 利用seq自身cache长度是否达到nums判断是否过了prefill
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)

