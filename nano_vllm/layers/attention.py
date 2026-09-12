"""
layers/attention.py - Paged Attention 实现

支持两种精度模式：
1. FP16/BF16: 使用 FlashAttention 库进行 prefill 和 decode
2. FP8 KV Cache: 通过 Triton kernel 实现量化写入和自定义 decode attention

核心组件：
- store_kvcache_kernel: Triton kernel，将 KV 向量量化写入 paged cache
- fp8_paged_attention_decode_kernel: Triton kernel，FP8 PagedAttention decode
- Attention module: 统一 prefill/decode 的接口层

Prefill 阶段始终使用 flash_attn_varlen_func（高效变长注意力），
Decode 阶段根据 enable_fp8_kvcache 选择 FlashAttention 或自定义 Triton kernel。
"""

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

from nano_vllm.utils.context import get_context


# ======================== Triton Kernels ========================

@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,
):
    """将单个 token 的 KV 向量写入 paged cache 对应 slot

    Grid: (num_tokens,) 每个 program 处理一个 token
    D = num_kv_heads * head_dim（单 token 的 KV 向量总维度）
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)

    if slot == -1:
        return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor, fp8_enabled: bool = False):
    """调用 Triton kernel 将 KV 写入 paged cache

    Args:
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        k_cache: paged cache (flattened as [num_blocks * block_size, num_kv_heads * head_dim])
        v_cache: 同上
        slot_mapping: [num_tokens] 每个 token 的物理 slot 索引
        fp8_enabled: 是否量化为 FP8 (当前未使用，保留接口)
    """
    num_tokens, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "store_kvcache 需要 triton (pip install triton)。"
            "无 GPU 环境请使用 tests/ 中的 CPU 单测。")
    store_kvcache_kernel[(num_tokens,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping, D=D,
    )


@triton.jit
def fp8_paged_attention_decode_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    scale: tl.constexpr,
):
    """FP8 Paged Attention Decode Kernel（Online Softmax + Masking）

    Grid: (batch_size, num_heads)
    每个 program 计算一个 (sequence, head) 的 attention 输出。

    使用 tl.where masking 替代动态 break，兼容 Triton 3.x。
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    num_heads_per_kv = num_heads // num_kv_heads
    kv_head_idx = head_idx // num_heads_per_kv

    context_len = tl.load(context_lens_ptr + seq_idx)

    # 加载 query: [head_dim]
    q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim
    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + q_offset + dim_offsets).to(tl.float32)

    # Online softmax 状态
    m_old = float("-inf")
    s = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    D = num_kv_heads * head_dim

    # 遍历所有 block（使用 constexpr 边界 + mask）
    for block_idx in range(max_num_blocks):
        block_start = block_idx * block_size
        # 跳过超出 context_len 的 block
        block_valid = block_start < context_len

        physical_block = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks + block_idx,
            mask=block_valid, other=0
        )

        # 遍历 block 内的 token（constexpr 循环 + mask）
        for token_offset in range(block_size):
            token_pos = block_start + token_offset
            valid = token_pos < context_len

            # KV cache 偏移
            slot = physical_block * block_size + token_offset
            kv_offset = slot * D + kv_head_idx * head_dim

            # 加载 FP8 KV 并反量化
            k = tl.load(k_cache_ptr + kv_offset + dim_offsets,
                        mask=valid, other=0.0).to(tl.float32)
            v = tl.load(v_cache_ptr + kv_offset + dim_offsets,
                        mask=valid, other=0.0).to(tl.float32)

            # QK^T (scaled dot product)
            score = tl.sum(q * k) * scale

            # 无效位置设为 -inf（不影响 softmax）
            score = tl.where(valid, score, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_old, score)
            correction = tl.exp(m_old - m_new)
            p = tl.exp(score - m_new)
            s = s * correction + p
            acc = acc * correction + p * v
            m_old = m_new

    # 归一化输出
    out = (acc / tl.maximum(s, 1e-10)).to(tl.float16)
    out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(out_ptr + out_offset + dim_offsets, out)


def custom_fp8_decode_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_tables: torch.Tensor, context_lens: torch.Tensor, scale: float,
    block_size: int,
) -> torch.Tensor:
    """FP8 Paged Attention Decode 的 Python 入口

    Args:
        q: [batch_size, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] (FP8)
        v_cache: 同上
        block_tables: [batch_size, max_num_blocks]
        context_lens: [batch_size]
        scale: attention scaling factor (1/sqrt(head_dim))
        block_size: KV cache block size

    Returns:
        [batch_size, num_heads, head_dim] attention 输出
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    max_num_blocks = block_tables.shape[1]

    out = torch.empty_like(q)
    grid = (batch_size, num_heads)

    fp8_paged_attention_decode_kernel[grid](
        out, q, k_cache.reshape(-1), v_cache.reshape(-1),
        block_tables, context_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        scale=scale,
    )
    return out


# ======================== Attention Module ========================

class Attention(nn.Module):
    """统一的 Attention 模块

    自动处理 KV cache 写入和 prefill/decode 分支选择。
    支持 FP8 和 FP16 两种 KV cache 精度模式。

    Args:
        num_heads: query 头数（TP 后）
        head_dim: 每头维度
        num_kv_heads: KV 头数（TP 后，用于 GQA）
        scale: attention scaling factor
    """

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int, scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = scale

        # KV cache 引用（由 ModelRunner.allocate_kv_cache 注入）
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)

    @property
    def fp8_enabled(self) -> bool:
        """当前 KV cache 是否为 FP8 格式"""
        return self.k_cache.numel() > 0 and self.k_cache.dtype == torch.float8_e4m3fn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Attention forward

        自动选择 prefill 或 decode 路径，并将 KV 写入 paged cache。

        Args:
            q: [num_tokens, num_heads, head_dim]
            k: [num_tokens, num_kv_heads, head_dim]
            v: [num_tokens, num_kv_heads, head_dim]

        Returns:
            [num_tokens, num_heads, head_dim] attention 输出
        """
        ctx = get_context()

        # 写入 KV Cache
        if self.k_cache.numel() > 0:
            store_kvcache(k, v, self.k_cache, self.v_cache,
                         ctx.slot_mapping, fp8_enabled=self.fp8_enabled)

        if ctx.is_prefill:
            return self._prefill_attention(q, k, v, ctx)
        else:
            return self._decode_attention(q, ctx)

    def _prefill_attention(self, q, k, v, ctx):
        """Prefill 路径：使用 FlashAttention varlen

        支持 prefix cache 场景（cu_seqlens_k > cu_seqlens_q 时使用 block_tables）
        FP8 KV Cache 场景下，prefix cache 命中时需要先反量化再传给 flash_attn。
        """
        if ctx.block_tables is not None:
            # Prefix cache 命中：使用 paged KV cache
            if self.fp8_enabled:
                # FP8 cache 需反量化为模型精度
                k, v = self.k_cache.to(q.dtype), self.v_cache.to(q.dtype)
            else:
                k, v = self.k_cache, self.v_cache

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            causal=True,
            softmax_scale=self.scale,
            block_table=ctx.block_tables,
        )
        return out

    def _decode_attention(self, q, ctx):
        """Decode 路径：Paged Attention

        FP8 模式使用自定义 Triton kernel，FP16 模式使用 FlashAttention。
        """
        if self.fp8_enabled:
            out = custom_fp8_decode_attention(
                q, self.k_cache, self.v_cache,
                ctx.block_tables, ctx.context_lens,
                self.scale, block_size=self.k_cache.shape[1],
            )
            return out
        else:
            # flash_attn_with_kvcache 期望:
            # q: [batch_size, 1, num_heads, head_dim]
            # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            out = flash_attn_with_kvcache(
                q.unsqueeze(1),
                self.k_cache, self.v_cache,
                cache_seqlens=ctx.context_lens,
                block_table=ctx.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            return out.squeeze(1)
