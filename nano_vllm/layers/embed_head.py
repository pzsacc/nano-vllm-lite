"""
layers/embed_head.py - 并行 Embedding 和 LM Head

实现 Vocabulary 维度的 Tensor Parallel 切分：
- VocabParallelEmbedding: 词表按 rank 切分，各 rank 只存部分词向量
- ParallelLMHead: 继承 Embedding 权重，支持 prefill 阶段的 last-token 优化

TP 通信：
- Embedding: 各 rank 计算局部 embedding → all_reduce 聚合
- LM Head: 各 rank 计算局部 logits → gather 到 rank 0 拼接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nano_vllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表并行 Embedding

    将完整词表按 TP size 均匀切分到各 rank，
    forward 时各 rank 只查自己负责的词表段，再 all_reduce。

    Args:
        num_embeddings: 总词表大小
        embedding_dim: 嵌入维度
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 计算当前 rank 负责的词表范围
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        """加载当前 rank 对应的词表分片"""
        param.data.copy_(loaded_weight[self.vocab_start_idx:self.vocab_end_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """词表并行 embedding 查找

        Args:
            x: [num_tokens] token IDs

        Returns:
            [num_tokens, embedding_dim] 嵌入向量
        """
        if self.tp_size == 1:
            return F.embedding(x, self.weight)

        # 将超出当前 rank 范围的 token 映射为 0（后续 mask 掉）
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        local_x = (x - self.vocab_start_idx).clamp(min=0)
        output = F.embedding(local_x, self.weight)
        output[~mask] = 0.0
        dist.all_reduce(output)
        return output


class ParallelLMHead(VocabParallelEmbedding):
    """并行 LM Head（语言模型输出头）

    与 VocabParallelEmbedding 共享权重结构，但 forward 行为不同：
    1. Prefill 阶段只取每个序列的最后一个 token 计算 logits
    2. TP 模式下通过 gather 收集所有 rank 的局部 logits 到 rank 0

    Args:
        num_embeddings: 词表大小
        embedding_dim: 隐藏维度
        bias: 是否使用偏置（通常为 False）
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        super().__init__(num_embeddings, embedding_dim)
        assert not bias, "LM Head 不支持 bias"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """计算 logits

        Prefill 阶段：只取 cu_seqlens_q 指定的最后 token
        TP 模式：rank 0 gather 所有 rank 的 logits 并拼接返回

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            rank 0: [batch_size, vocab_size] logits
            其他 rank: None
        """
        ctx = get_context()

        # Prefill: 只取每个序列最后一个 token
        if ctx.is_prefill and ctx.cu_seqlens_q is not None:
            indices = ctx.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[indices]

        # 计算局部 logits
        logits = F.linear(hidden_states, self.weight)

        if self.tp_size == 1:
            return logits

        # TP gather: 所有 rank 的 logits 收集到 rank 0
        if self.tp_rank == 0:
            gathered = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.gather(logits, gather_list=gathered, dst=0)
            return torch.cat(gathered, dim=-1)
        else:
            dist.gather(logits, dst=0)
            return None
