import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func
from nanovllm.utils.context import get_context


# ===================================================================
# 1. FP8 KV Cache 写入内核 (保持上一版的极致 1D 优化)
# ===================================================================
@triton.jit
def store_kvcache_kernel(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
        D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # On-the-fly FP8 极速量化
    key_fp8 = key.to(tl.float8e4nv)
    value_fp8 = value.to(tl.float8e4nv)

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key_fp8)
    tl.store(v_cache_ptr + cache_offsets, value_fp8)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ===================================================================
# 2. 手撕 FP8 PagedAttention (Decode 专用)
# ===================================================================
@triton.jit
def fp8_paged_attention_decode_kernel(
        Q, K_Cache, V_Cache, Out, scale,
        Block_Tables, Context_Lens,
        stride_q_bs, stride_q_h, stride_q_d,
        stride_k_block, stride_k_bsize, stride_k_h, stride_k_d,
        stride_v_block, stride_v_bsize, stride_v_h, stride_v_d,
        stride_out_bs, stride_out_h, stride_out_d,
        stride_bt_bs, stride_bt_b,
        num_kv_heads, num_heads_per_kv,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_heads_per_kv  # 支持 GQA

    context_len = tl.load(Context_Lens + seq_idx)
    if context_len == 0:
        return

    # 加载 FP16 的 Query
    q_offset = seq_idx * stride_q_bs + head_idx * stride_q_h + tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_offset)

    m_old = float('-inf')
    s = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # 遍历 Paged KV Block Table
    for block_idx in range(num_blocks):
        physical_block_id = tl.load(Block_Tables + seq_idx * stride_bt_bs + block_idx * stride_bt_b)
        start_token_idx = block_idx * BLOCK_SIZE
        valid_mask = (start_token_idx + tl.arange(0, BLOCK_SIZE)) < context_len

        offs_bsize = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_DIM)

        # 计算 2D 偏移指针
        k_offset = physical_block_id * stride_k_block + offs_bsize[:, None] * stride_k_bsize + kv_head_idx * stride_k_h + offs_d[None, :] * stride_k_d
        v_offset = physical_block_id * stride_v_block + offs_bsize[:, None] * stride_v_bsize + kv_head_idx * stride_v_h + offs_d[None, :] * stride_v_d

        # 从 HBM 加载 FP8 数据 (显存带宽直接减半！)
        k_fp8 = tl.load(K_Cache + k_offset, mask=valid_mask[:, None], other=0.0)
        v_fp8 = tl.load(V_Cache + v_offset, mask=valid_mask[:, None], other=0.0)

        # SRAM 内实时反量化 (零显存开销，完全利用 ALU 算力)
        k = k_fp8.to(tl.float32)
        v = v_fp8.to(tl.float32)

        # 计算 Attention Score
        qk = tl.sum(q[None, :] * k, axis=1) * scale
        qk = tl.where(valid_mask, qk, float('-inf'))

        # 在线 Softmax (Online Softmax)
        m_new = tl.maximum(m_old, tl.max(qk, 0))
        alpha = tl.exp(m_old - m_new)

        p_new = tl.exp(qk - m_new)
        s = s * alpha + tl.sum(p_new, axis=0)
        acc = acc * alpha + tl.sum(p_new[:, None] * v, axis=0)
        m_old = m_new

    acc = acc / s

    # 将结果转回 FP16 写入 HBM
    out_offset = seq_idx * stride_out_bs + head_idx * stride_out_h + tl.arange(0, HEAD_DIM)
    tl.store(Out + out_offset, acc.to(q.dtype))


def custom_fp8_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale):
    bs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]

    out = torch.empty_like(q)
    grid = (bs, num_heads)

    fp8_paged_attention_decode_kernel[grid](
        q, k_cache, v_cache, out, scale,
        block_tables, context_lens,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        num_kv_heads, num_heads // num_kv_heads,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim
    )
    return out


# ===================================================================
# 3. 模块封装
# ===================================================================
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                # 兼容 Prefix Cache 的 Hack：FA2 原生不支持 FP8 varlen
                # 所以我们在 Prefill 阶段把命中的 Prefix Cache 强转回 FP16 喂给它
                k, v = k_cache.to(q.dtype), v_cache.to(q.dtype)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # 🌟 Decode 阶段：彻底抛弃官方 API，启用我们的手撕算子！
             o = custom_fp8_decode_attention(q, k_cache, v_cache,
                                            context.block_tables, context.context_lens,
                                            self.scale)
        return o


'''# ===================================================================
# 4. 纯算子级极限 Benchmark (Triton do_bench)
# ===================================================================
if __name__ == "__main__":
    import triton.testing

    # 1. 模拟 LLM 线上真实物理配置 (以 Qwen3-0.6B / Llama 常见配置为例)
    BATCH_SIZE = 32  # 极高的并发请求数
    NUM_HEADS = 16
    NUM_KV_HEADS = 16  # 如果是 MQA/GQA，这里会小于 NUM_HEADS
    HEAD_DIM = 64
    BLOCK_SIZE = 16


    # 2. Benchmark 游标：动态增加上下文长度 (Context Length)，观察算子性能衰减曲线
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['context_len'],  # 横坐标：序列上下文长度
            x_vals=[256, 512, 1024, 2048, 4096, 8192],
            line_arg='provider',  # 不同的测试分支（比如优化前 vs 优化后）
            line_vals=['custom_fp8'],  # 目前我们只测你写的 FP8
            line_names=['FP8 Triton PagedAttention'],
            styles=[('blue', '-')],
            ylabel='Latency (ms)',  # 纵坐标：耗时
            plot_name='PagedAttention-Decode-Performance',
            args={},
        )
    )
    def benchmark_paged_attention(context_len, provider):
        # --- A. 初始化环境与伪造显存状态 ---
        q = torch.randn((BATCH_SIZE, NUM_HEADS, HEAD_DIM), dtype=torch.float16, device="cuda")
        scale = HEAD_DIM ** -0.5

        # 伪造 Paged KV Cache 显存池
        # 根据你的 stride 逻辑，形状应为: [Total_Blocks, Block_Size, KV_Heads, Head_Dim]
        num_blocks_per_seq = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        total_blocks = BATCH_SIZE * num_blocks_per_seq

        # 你的内核使用了 FP8 (float8_e4m3fn 对应 tl.float8e4nv)
        k_cache = torch.empty((total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
        v_cache = torch.empty((total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")

        # 伪造 Block Table (简单的线性映射)
        block_tables = torch.arange(0, total_blocks, dtype=torch.int32, device="cuda").view(BATCH_SIZE, num_blocks_per_seq)
        context_lens = torch.full((BATCH_SIZE,), context_len, dtype=torch.int32, device="cuda")

        # --- B. 预热与压测 ---
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'custom_fp8':
            # 测试你手写的算子
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: custom_fp8_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale),
                quantiles=quantiles
            )

        # --- C. 计算极其硬核的指标：显存带宽 (GB/s) ---
        # Decode 阶段的核心瓶颈是读 KV Cache。
        # FP8 占 1 byte。每个 Token 读 K 和 V。
        gb = (BATCH_SIZE * context_len * NUM_KV_HEADS * HEAD_DIM * 2 * 1) / (1024 ** 3)
        gbps = gb / (ms * 1e-3)  # GB/s

        return ms, max_ms, min_ms
        # 如果你想图表里直接显示带宽高低，可以改为 return gbps, max_gbps, min_gbps


    # 3. 启动台架测试
    print("🚀 开始进行微观算子级极限压测 (Triton PagedAttention + FP8)...")
    benchmark_paged_attention.run(print_data=True, show_plots=False)'''