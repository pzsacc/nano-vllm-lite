"""
models/qwen3.py - Qwen3 模型架构实现

实现 Qwen3 Transformer Decoder 的完整推理路径：
- Qwen3Attention: GQA 注意力 + QKV 融合投影 + RoPE
- Qwen3MLP: SwiGLU 激活 + Gate/Up 融合投影
- Qwen3DecoderLayer: 标准 Pre-Norm Transformer block
- Qwen3Model: Embedding + N × DecoderLayer + Final Norm
- Qwen3ForCausalLM: 完整 Causal LM（含 LM Head + packed_modules_mapping）

设计特点：
- 所有 Linear 层支持 Tensor Parallel 自动切分
- 使用融合 Add+RMSNorm 减少显存带宽消耗
- packed_modules_mapping 支持从 HuggingFace checkpoint 直接加载融合权重
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Qwen3Config

from nano_vllm.layers.activation import SiluAndMul
from nano_vllm.layers.attention import Attention
from nano_vllm.layers.layernorm import RMSNorm
from nano_vllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nano_vllm.layers.rotary_embedding import get_rope
from nano_vllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 注意力层

    实现 GQA（Grouped Query Attention）+ RoPE + 融合 QKV 投影。
    支持可选的 QK-Norm（当 qkv_bias=False 时启用）。

    Args:
        config: Qwen3Config 模型配置
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim ** -0.5

        # 融合 QKV 投影
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=getattr(config, "attention_bias", True),
        )
        # 输出投影
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        # RoPE
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.rotary_emb = get_rope(
            self.head_dim, self.head_dim,
            config.max_position_embeddings, rope_theta,
        )

        # QK-Norm（Qwen3 无 bias 时启用）
        self.q_norm = None
        self.k_norm = None
        if not getattr(config, "attention_bias", True):
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Attention 计算核心
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads, self.scale)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """注意力层前向

        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: [num_tokens] 位置索引

        Returns:
            [num_tokens, hidden_size] 注意力输出
        """
        # QKV 投影
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape: [num_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # QK-Norm
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v)

        # Output projection
        return self.o_proj(attn_output.reshape(-1, self.num_heads * self.head_dim))


class Qwen3MLP(nn.Module):
    """Qwen3 MLP（SwiGLU 激活）

    使用融合的 gate_up_proj 将 gate 和 up 投影合并为一次矩阵乘法，
    再经过 SiLU+Gate 激活后通过 down_proj 映射回隐藏维度。

    Args:
        config: Qwen3Config 模型配置
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        assert getattr(config, "hidden_act", "silu") == "silu"

        # 融合 Gate + Up 投影
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        # Down 投影
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP 前向

        Args:
            x: [num_tokens, hidden_size]

        Returns:
            [num_tokens, hidden_size]
        """
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Transformer Decoder Layer

    标准 Pre-Norm 结构：
    residual → input_layernorm → self_attn → post_attention_layernorm → MLP

    使用融合 Add+RMSNorm 优化残差连接。

    Args:
        config: Qwen3Config 模型配置
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                residual: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder layer 前向

        Args:
            hidden_states: 当前隐藏状态
            positions: 位置索引
            residual: 残差累积值（第一层为 None）

        Returns:
            (hidden_states, residual) 元组，传递给下一层
        """
        # Pre-Attention Norm（融合 Add+RMSNorm）
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self-Attention
        hidden_states = self.self_attn(hidden_states, positions)

        # Post-Attention Norm（融合 Add+RMSNorm）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 完整 Transformer Model（不含 LM Head）

    Args:
        config: Qwen3Config 模型配置
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Model 前向

        Args:
            input_ids: [num_tokens] token IDs
            positions: [num_tokens] 位置索引

        Returns:
            [num_tokens, hidden_size] 最终隐藏状态
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, positions, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 因果语言模型（完整推理模型）

    包含 Transformer backbone + LM Head。
    定义 packed_modules_mapping 支持从标准 HuggingFace checkpoint 加载融合权重。

    Args:
        config: Qwen3Config 模型配置
    """

    # 权重融合映射：原始分散权重名 → (融合后权重名, shard_id)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 词表权重共享（如配置要求）
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """前向传播（不含 logits 计算）"""
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算 logits（独立方法，支持 CUDA Graph 分离调用）"""
        return self.lm_head(hidden_states)
