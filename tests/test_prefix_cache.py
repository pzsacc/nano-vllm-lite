"""
tests/test_prefix_cache.py - Prefix Caching 正确性测试 (CPU)

对应 docs/02-prefix-caching.md。
这个测试防的是: 哈希链断裂导致的前缀误命中（会用错误的 KV 续算出乱码）。

注意: hash_blocks 只注册"已调度填满"的块（与引擎调度路径一致），
所以测试里先模拟 num_scheduled_tokens 推进再 hash_blocks。
"""
import pytest

from nano_vllm.engine.block_manager import BlockManager
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.sampling_params import SamplingParams


@pytest.fixture
def bm():
    Sequence.block_size = 256
    return BlockManager(num_blocks=32, block_size=256, enable_prefix_caching=True)


def _prefill_and_hash(bm, seq):
    """模拟引擎调度路径: 分配 → 填满块 → 注册哈希"""
    bm.allocate(seq, bm.can_allocate(seq))
    seq.num_scheduled_tokens = seq.num_tokens
    bm.hash_blocks(seq)


def test_hash_chain_prefix_hit(bm):
    """相同前缀的第二个请求应命中已缓存块"""
    seq1 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    _prefill_and_hash(bm, seq1)
    assert len(bm.hash_to_block_id) >= 2

    seq2 = Sequence(list(range(600)) + [99999],
                    SamplingParams(temperature=1.0, max_tokens=8))
    num_cached = bm.can_allocate(seq2)
    # 600 token = 2 满块 + 不满尾块, 应命中 2 块 = 512 token
    assert num_cached == 2, f"前缀应命中 2 块, 实际 {num_cached}"


def test_different_prefix_no_false_hit(bm):
    """不同内容的前缀绝不能命中 —— 防哈希碰撞/链断裂误命中"""
    seq1 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    _prefill_and_hash(bm, seq1)

    seq2 = Sequence([7, 7, 7] + list(range(300)),
                    SamplingParams(temperature=1.0, max_tokens=8))
    assert bm.can_allocate(seq2) == 0, "不同前缀不应命中缓存"


def test_disabled_prefix_caching():
    """关闭开关后不命中（配置路径）"""
    Sequence.block_size = 256
    bm = BlockManager(num_blocks=32, block_size=256, enable_prefix_caching=False)
    seq1 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    _prefill_and_hash(bm, seq1)

    seq2 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    assert bm.can_allocate(seq2) == 0


def test_deallocate_keeps_cached_blocks(bm):
    """请求结束后缓存块应保留（引用计数归零, 哈希索引仍可复用）"""
    seq1 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    _prefill_and_hash(bm, seq1)
    bm.deallocate(seq1)

    seq2 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    assert bm.can_allocate(seq2) == 2, "deallocate 后缓存应仍可复用"


def test_partial_block_not_cached(bm):
    """未填满的尾部块不应进入缓存（内容不完整, 复用会出错）"""
    seq1 = Sequence(list(range(600)), SamplingParams(temperature=1.0, max_tokens=8))
    _prefill_and_hash(bm, seq1)
    # 600 = 2 满块 + 88 token 尾块, 尾块无哈希
    assert len(bm.hash_to_block_id) == 2
