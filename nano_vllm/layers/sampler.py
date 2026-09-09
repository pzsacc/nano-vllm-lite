"""
layers/sampler.py - Token 采样器

使用 Gumbel-max trick 实现高效的 multinomial 采样。
相比 torch.multinomial，Gumbel-max 对 @torch.compile 更友好，
能被完全融合到单个 CUDA kernel 中。
"""

import torch
import torch.nn as nn


class Sampler(nn.Module):
    """基于 Gumbel-max trick 的 Token 采样器

    算法：
    1. logits /= temperature（控制随机性）
    2. probs = softmax(logits)
    3. 从 Exponential(1) 分布采样噪声
    4. token = argmax(probs / noise)（等价于 multinomial 采样）

    该实现可被 torch.compile 完全融合，避免多次 kernel launch。
    """

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """执行采样

        Args:
            logits: [batch_size, vocab_size] 模型输出 logits
            temperatures: [batch_size] 每个序列的采样温度

        Returns:
            [batch_size] 采样得到的 token IDs
        """
        logits = logits.float()
        logits = logits / temperatures.unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / noise).argmax(dim=-1)
