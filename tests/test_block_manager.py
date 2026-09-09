"""
tests/test_block_manager.py - BlockManager 单元测试

验证页式 KV Cache 管理器的核心功能：
- Block 分配与释放
- 前缀缓存匹配
- 引用计数共享
"""

import pytest
from nano_vllm.engine.block_manager import BlockManager
from nano_vllm.engine.sequence import Sequence
from nano_vllm.sampling_params import SamplingParams


class TestBlockManager:
    """BlockManager 单元测试"""

    def setup_method(self):
        """每个测试前初始化 BlockManager"""
        Sequence.block_size = 256
        self.bm = BlockManager(num_blocks=10, block_size=256, enable_prefix_caching=True)

    def test_basic_allocation(self):
        """测试基本 block 分配"""
        tokens = list(range(512))
        seq = Sequence(tokens, SamplingParams(temperature=1.0, max_tokens=10))
        num_cached = self.bm.can_allocate(seq)
        assert num_cached >= 0
        self.bm.allocate(seq, num_cached)
        assert len(seq.block_table) == seq.num_blocks
        assert len(self.bm.free_block_ids) == 10 - seq.num_blocks + num_cached

    def test_deallocation(self):
        """测试 block 释放"""
        tokens = list(range(256))
        seq = Sequence(tokens, SamplingParams(temperature=1.0, max_tokens=10))
        self.bm.allocate(seq, 0)
        self.bm.deallocate(seq)
        assert len(self.bm.free_block_ids) == 10
        assert len(seq.block_table) == 0

    def test_prefix_caching(self):
        """测试前缀缓存复用"""
        shared_prefix = list(range(256))
        seq1 = Sequence(shared_prefix + list(range(256, 512)),
                        SamplingParams(temperature=1.0, max_tokens=10))
        seq2 = Sequence(shared_prefix + list(range(512, 768)),
                        SamplingParams(temperature=1.0, max_tokens=10))

        self.bm.allocate(seq1, 0)
        # 手动注册 block hash
        seq1.num_scheduled_tokens = seq1.num_tokens
        self.bm.hash_blocks(seq1)

        # seq2 应该命中 prefix cache
        num_cached = self.bm.can_allocate(seq2)
        assert num_cached >= 1

    def test_insufficient_blocks(self):
        """测试空间不足时返回 -1"""
        bm = BlockManager(num_blocks=1, block_size=256, enable_prefix_caching=False)
        tokens = list(range(512))
        seq = Sequence(tokens, SamplingParams(temperature=1.0, max_tokens=10))
        result = bm.can_allocate(seq)
        assert result == -1
