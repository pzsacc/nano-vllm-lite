"""
engine/sequence.py - 序列状态管理

定义推理过程中每个请求序列的完整生命周期状态，包括：
- 序列状态机（WAITING → RUNNING → FINISHED）
- Token 管理（prompt + completion tokens）
- Block table 映射（用于 Paged KV Cache）
- 序列化支持（用于 Tensor Parallel IPC 传输）
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nano_vllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列生命周期状态"""
    WAITING = auto()    # 等待 prefill 调度
    RUNNING = auto()    # 已完成 prefill，正在 decode
    FINISHED = auto()   # 生成完毕（遇到 EOS 或达到 max_tokens）


class Sequence:
    """单个推理序列的完整状态

    管理从请求到达到生成完毕的全部信息，包括 token 序列、
    KV cache block 映射、调度元数据等。

    Attributes:
        seq_id: 全局唯一序列标识符
        status: 当前生命周期状态
        token_ids: 完整 token 序列（prompt + generated）
        block_table: 物理 KV cache block ID 列表
        num_cached_tokens: 已写入 KV cache 的 token 数
        num_scheduled_tokens: 当前步骤被调度的 token 数
        is_prefill: 是否处于 prefill 阶段
    """

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        """初始化序列

        Args:
            token_ids: prompt 的 token ID 列表
            sampling_params: 采样参数
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        """返回当前总 token 数"""
        return self.num_tokens

    def __getitem__(self, key):
        """按索引访问 token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        """序列是否已完成生成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """已生成的 completion token 数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        """原始 prompt 的 token IDs"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        """已生成的 completion token IDs"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self) -> int:
        """当前序列所需的 KV cache block 数"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """最后一个 block 中的有效 token 数"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        """获取第 i 个 block 对应的 token IDs

        Args:
            i: block 索引（0-based）

        Returns:
            该 block 范围内的 token ID 列表
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """追加一个新生成的 token

        Args:
            token_id: 新生成的 token ID
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列化（用于 TP 跨进程 IPC 传输）

        prefill 阶段传输完整 token_ids，decode 阶段只传 last_token 以节省带宽
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """反序列化（TP worker 端恢复序列状态）"""
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.block_table, last_state) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
