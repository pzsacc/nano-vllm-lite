"""
layers/layernorm.py - RMSNorm 实现（支持融合 Add+Norm）

提供两种前向路径：
1. rms_forward: 标准 RMSNorm（无残差加法）
2. add_rms_forward: 融合 Residual Add + RMSNorm（减少一次 memory read/write）

两种路径均使用 @torch.compile 自动优化为融合 kernel。
如果 CUDA 扩展 (pz_vllm_ops) 可用，可选择使用手写 CUDA kernel 加速。
"""

import torch
import torch.nn as nn

from nano_vllm.kernels import apply_add_rmsnorm, is_cuda_kernels_available

_CUDA_ADD_RMSNORM = is_cuda_kernels_available()


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    相比 LayerNorm，RMSNorm 省略了均值减法步骤，计算更高效。
    公式: out = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        hidden_size: 归一化维度
        eps: 数值稳定性常数
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准 RMSNorm 前向

        Args:
            x: [..., hidden_size] 输入 tensor

        Returns:
            归一化后的 tensor，形状不变
        """
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(input_dtype)

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """融合 Residual Add + RMSNorm

        先执行 residual = x + residual，再对 residual 做 RMSNorm。
        融合后避免了中间 tensor 的额外显存读写。

        Args:
            x: 当前层输出
            residual: 残差连接的累积值

        Returns:
            (normalized_output, updated_residual) 元组
        """
        input_dtype = x.dtype
        residual = (x.float() + residual.float()).to(input_dtype)
        x = residual.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(input_dtype), residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """统一前向接口

        Args:
            x: 输入 tensor
            residual: 残差值（None 时使用标准 RMSNorm）

        Returns:
            residual=None: 归一化结果
            residual!=None: (归一化结果, 更新后的残差)
        """
        if residual is None:
            return self.rms_forward(x)
        if _CUDA_ADD_RMSNORM and x.is_cuda and x.dim() == 2:
            return apply_add_rmsnorm(x, residual, self.weight, self.eps)
        return self.add_rms_forward(x, residual)
