"""
tests/test_chunked_prefill.py - Chunked Prefill 调度逻辑测试 (CPU)

对应 docs/03-chunked-prefill.md。
验证 Sarathi 风格混合调度的核心不变量:
1. decode 序列每步都被优先调度（保证 TPOT）
2. prefill 按固定 chunk 预算分块（长 prompt 不阻塞 decode）
3. abort 后序列不再被调度（资源被释放）
"""
import pytest

from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.sampling_params import SamplingParams
from nano_vllm.config import Config


def make_config(**kw) -> Config:
    """不走 __post_init__（避免加载 HF 配置），手工构造调度器所需字段"""
    cfg = Config.__new__(Config)
    cfg.max_num_seqs = 64
    cfg.max_num_batched_tokens = 8192
    cfg.eos = -1
    cfg.kvcache_block_size = 256
    cfg.chunk_size = 256
    cfg.enable_chunked_prefill = True
    cfg.enable_prefix_caching = False
    cfg.num_kvcache_blocks = 64
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def seq_factory():
    Sequence.block_size = 256
    counter = {"id": 0}

    def _make(num_tokens: int) -> Sequence:
        sp = SamplingParams(temperature=1.0, max_tokens=4, ignore_eos=True)
        seq = Sequence(list(range(num_tokens)), sp)
        return seq
    return _make


def test_decode_priority_over_long_prefill(seq_factory):
    """decode 序列必须先于 prefill 被调度（TPOT 保障的核心）"""
    sched = Scheduler(make_config())
    long_prompt = seq_factory(1000)     # 超过 chunk_size=256, 会分块
    sched.add(long_prompt)

    # 第一步: prefill 第一个 chunk
    seqs, has_prefill = sched.schedule()
    assert has_prefill and long_prompt.num_scheduled_tokens == 256
    # 模拟这一步跑完
    long_prompt.num_cached_tokens += long_prompt.num_scheduled_tokens
    long_prompt.num_scheduled_tokens = 0
    long_prompt.is_prefill = False
    sched.running.append(long_prompt)

    # 有 decode 在跑时, 新来一个长 prefill
    new_long = seq_factory(1000)
    sched.add(new_long)
    seqs, has_prefill = sched.schedule()
    decode_first = [s for s in seqs if not s.is_prefill]
    assert long_prompt in decode_first, "decode 序列应被优先调度"


def test_prefill_chunked_by_budget(seq_factory):
    """超过 chunk_size 的 prompt 被切分, 每步不超过预算"""
    sched = Scheduler(make_config(chunk_size=256))
    seq = seq_factory(1000)
    sched.add(seq)

    steps = []
    while not sched.is_finished() and len(steps) < 10:
        seqs, has_prefill = sched.schedule()
        assert has_prefill
        scheduled_tok = sum(s.num_scheduled_tokens for s in seqs)
        assert scheduled_tok <= 256, f"chunk 预算超支: {scheduled_tok} > 256"
        for s in seqs:
            s.num_cached_tokens += s.num_scheduled_tokens
            s.num_scheduled_tokens = 0
        steps.append(scheduled_tok)
        if seq.num_cached_tokens >= seq.num_tokens:
            break
    # 1000 token / 256 预算 ≈ 4 步完成 prefill
    assert sum(steps) == 1000
    assert len(steps) <= 5, f"1000 token 应在 ~4 步内完成, 实际 {len(steps)} 步"


def test_abort_removes_from_queues(seq_factory):
    """abort 后序列不再被调度, 状态变为 ABORTED"""
    sched = Scheduler(make_config())
    seq = seq_factory(300)
    sched.add(seq)
    assert sched.abort(seq.seq_id) is True
    assert seq.status == SequenceStatus.ABORTED
    seqs, _ = sched.schedule()
    assert seq not in seqs
    assert sched.is_finished()


def test_abort_unknown_seq_returns_false(seq_factory):
    sched = Scheduler(make_config())
    assert sched.abort(99999) is False
