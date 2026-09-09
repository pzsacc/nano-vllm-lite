"""
layers/activation.py - 融合激活函数

实现 SiLU + Gate 的融合激活，用于 LLaMA/Qwen 系 MLP 的 gated 结构。
通过 @torch.compile 自动融合为单个 kernel，减少中间 tensor 分配。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """融合 SiLU + Gating 激活函数

    输入 x 的最后一维分为两半：前半部分经过 SiLU 激活后与后半部分逐元素相乘。
    公式: output = SiLU(x[:half]) * x[half:]

    典型用法：MLP 的 gate_up_proj 输出经过此层得到 MLP 中间结果。
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """融合 SiLU+Gate 前向

        Args:
            x: [..., 2 * intermediate_size] gate_up_proj 的输出

        Returns:
            [..., intermediate_size] 激活后的结果
        """
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
