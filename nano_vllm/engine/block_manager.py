"""
engine/block_manager.py - 页式 KV Cache 块管理器

实现 vLLM 风格的 Paged KV Cache 内存管理：
- 固定大小物理 Block 的分配/释放（类似操作系统页帧管理）
- 基于 xxhash 的内容寻址前缀缓存（Prefix Caching）
- 引用计数共享（多个序列共享相同前缀的 block）
- Block 淘汰策略（FIFO free list 实现近似 LRU）
"""

from collections import deque
import xxhash
import numpy as np

from nano_vllm.engine.sequence import Sequence


class Block:
    """物理 KV Cache 块

    每个 Block 存储固定数量 token 的 KV 向量，通过引用计数管理生命周期。

    Attributes:
        block_id: 全局唯一物理块 ID
        ref_count: 当前引用此块的序列数
        hash: 内容哈希值（用于前缀缓存匹配）
        token_ids: 块中存储的 token IDs（用于哈希验证）
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash_val: int, token_ids: list[int]):
        """更新块的哈希值和内容记录

        Args:
            hash_val: 基于链式哈希计算的内容指纹
            token_ids: 块中实际存储的 token IDs
        """
        self.hash = hash_val
        self.token_ids = token_ids

    def reset(self):
        """重置块状态（分配新块时调用）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """页式 KV Cache 块管理器

    管理 GPU 上所有 KV cache 物理块的分配、释放和前缀缓存。
    核心设计：
    1. free_block_ids: 空闲块 FIFO 队列
    2. hash_to_block_id: 内容哈希 → 块 ID 的映射（前缀缓存）
    3. 引用计数：支持多序列共享同一物理块

    Args:
        num_blocks: 总物理块数量
        block_size: 每块容纳的 token 数
        enable_prefix_caching: 是否启用前缀缓存
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """计算内容感知的链式哈希

        使用 xxhash64 对 token 内容进行哈希，并将前一个 block 的哈希
        作为前缀链接，确保相同内容在不同位置产生不同哈希。

        Args:
            token_ids: 当前 block 的 token 内容
            prefix: 前一个 block 的哈希值（-1 表示第一个 block）

        Returns:
            64-bit 哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲列表分配一个物理块

        如果被分配的块之前有缓存哈希记录，清除该记录。

        Returns:
            分配的物理块 ID
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """释放一个物理块回空闲列表

        Args:
            block_id: 要释放的物理块 ID
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检查序列是否可以分配所需的 block

        遍历序列的 block 内容，匹配已有的前缀缓存块。
        返回可复用的缓存块数量，如果空间不足返回 -1。

        Args:
            seq: 目标序列

        Returns:
            可复用的已缓存 block 数量，-1 表示空间不足
        """
        if not self.enable_prefix_caching:
            if len(self.free_block_ids) < seq.num_blocks:
                return -1
            return 0

        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为序列分配 block table

        前 num_cached_blocks 个块通过引用计数共享已有缓存块，
        剩余块从空闲列表新分配。

        Args:
            seq: 目标序列
            num_cached_blocks: 可复用的缓存块数量
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有 block

        递减引用计数，引用归零的块回收到空闲列表。

        Args:
            seq: 目标序列
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查序列是否可以追加一个新 token

        当序列的下一个 token 会落在新 block 的第一个位置时，
        需要有空闲 block 可供分配。

        Args:
            seq: 目标序列

        Returns:
            是否有足够空间追加
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """必要时为序列分配新的 block（decode 阶段）

        Args:
            seq: 目标序列
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """对序列新完成的 block 计算哈希并注册到缓存

        仅处理从 num_cached_tokens 到 num_cached_tokens + num_scheduled_tokens
        范围内新填满的 block。

        Args:
            seq: 目标序列
        """
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
