"""
utils/context.py - 全局推理上下文管理

通过进程级全局变量传递当前推理步骤的元信息（prefill/decode 状态、
序列长度、slot 映射、block table 等），供 Attention 层在 forward 中读取。
这种设计避免了在 model.forward() 中传递大量额外参数。
"""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(slots=True)
class Context:
    """单步推理的全局上下文信息

    Attributes:
        is_prefill: 当前步骤是否包含 prefill 序列
        cu_seqlens_q: query 序列长度累积和（用于 flash_attn_varlen）
        cu_seqlens_k: key 序列长度累积和
        max_seqlen_q: 当前 batch 中最长 query 长度
        max_seqlen_k: 当前 batch 中最长 key 长度
        slot_mapping: 每个 token 在 KV cache 中的物理 slot 索引
        context_lens: decode 阶段每个序列的已缓存 token 数
        block_tables: decode 阶段每个序列的 block table（物理页映射）
    """
    is_prefill: bool = False
    cu_seqlens_q: Tensor | None = None
    cu_seqlens_k: Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Tensor | None = None
    context_lens: Tensor | None = None
    block_tables: Tensor | None = None


# 进程级全局上下文单例
_CONTEXT = Context()


def get_context() -> Context:
    """获取当前推理步骤的全局上下文"""
    return _CONTEXT


def set_context(is_prefill: bool, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q: int = 0, max_seqlen_k: int = 0,
                slot_mapping=None, context_lens=None, block_tables=None):
    """设置当前推理步骤的全局上下文

    Args:
        is_prefill: 是否为 prefill 阶段
        cu_seqlens_q: query 累积序列长度
        cu_seqlens_k: key 累积序列长度
        max_seqlen_q: 最大 query 序列长度
        max_seqlen_k: 最大 key 序列长度
        slot_mapping: KV cache 物理 slot 映射
        context_lens: 每个序列已缓存 token 数
        block_tables: 物理页块映射表
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context():
    """重置全局上下文为默认空状态"""
    global _CONTEXT
    _CONTEXT = Context()
