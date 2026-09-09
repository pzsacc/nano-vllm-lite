"""
layers/rotary_embedding.py - RoPE 旋转位置编码

实现 Rotary Position Embedding（RoPE），通过旋转变换将位置信息
注入 Q/K 向量，使 attention score 自然编码相对位置。

特性：
- 预计算 cos/sin 缓存（避免重复计算）
- @torch.compile 优化的 forward
- lru_cache 单例模式（同配置复用同一实例）
"""

from functools import lru_cache
import torch
import torch.nn as nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对输入 tensor 应用旋转位置编码

    将 x 的最后一维分为两半，按 RoPE 公式进行旋转。

    Args:
        x: [..., head_dim] 输入（Q 或 K）
        cos: [..., head_dim/2] 余弦缓存
        sin: [..., head_dim/2] 正弦缓存

    Returns:
        旋转后的 tensor，形状不变
    """
    input_dtype = x.dtype
    x = x.float()
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码模块

    预计算从 position 0 到 max_position_embeddings 的 cos/sin 表，
    推理时通过位置索引查表实现 O(1) 的 RoPE 应用。

    Args:
        head_size: 每头维度（必须等于 rotary_dim）
        rotary_dim: 旋转维度
        max_position_embeddings: 最大支持位置
        base: RoPE 频率基数（控制频谱分布）
    """

    def __init__(self, head_size: int, rotary_dim: int,
                 max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        assert head_size == rotary_dim, "当前实现要求 head_size == rotary_dim"
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings

        # 计算逆频率: theta_i = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        # 位置序列
        t = torch.arange(max_position_embeddings).float()
        # 外积得到角度矩阵: [max_pos, rotary_dim/2]
        freqs = torch.outer(t, inv_freq)
        # 构建 cos/sin 缓存: [max_pos, 1, rotary_dim] (中间维度用于 broadcast)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache.unsqueeze(1), persistent=False)

    @torch.compile
    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        """应用 RoPE 到 query 和 key

        Args:
            positions: [num_tokens] 每个 token 的位置索引
            query: [num_tokens, num_heads, head_dim]
            key: [num_tokens, num_kv_heads, head_dim]

        Returns:
            (rotated_query, rotated_key) 元组
        """
        # 按 position 索引查 cos/sin 缓存
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, 1, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=1)
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float = 10000.0) -> RotaryEmbedding:
    """获取 RoPE 单例（相同配置复用实例）

    Args:
        head_size: 每头维度
        rotary_dim: 旋转维度
        max_position: 最大位置
        base: 频率基数

    Returns:
        RotaryEmbedding 实例
    """
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
