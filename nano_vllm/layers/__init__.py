"""
layers/__init__.py - 模型计算层组件包

提供 Transformer 模型所需的所有计算层实现，包括：
- Attention: 支持 FP8 KV Cache 的 Paged Attention
- Linear: 支持 Tensor Parallel 的线性层变体
- LayerNorm: 融合 Add+RMSNorm
- RotaryEmbedding: RoPE 位置编码
- Activation: 融合 SiLU+Gate
- Sampler: Gumbel-max 采样器
"""

from nano_vllm.layers.attention import Attention
from nano_vllm.layers.linear import (
    ReplicatedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nano_vllm.layers.layernorm import RMSNorm
from nano_vllm.layers.rotary_embedding import RotaryEmbedding, get_rope
from nano_vllm.layers.activation import SiluAndMul
from nano_vllm.layers.sampler import Sampler
from nano_vllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

__all__ = [
    "Attention", "ReplicatedLinear", "ColumnParallelLinear",
    "MergedColumnParallelLinear", "QKVParallelLinear", "RowParallelLinear",
    "RMSNorm", "RotaryEmbedding", "get_rope", "SiluAndMul", "Sampler",
    "VocabParallelEmbedding", "ParallelLMHead",
]
