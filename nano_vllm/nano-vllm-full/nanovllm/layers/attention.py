import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func
from nanovllm.utils.context import get_context


# ===================================================================
# 1. FP8 KV Cache 写入内核
# ===================================================================
@triton.jit
def store_kvcache_kernel(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
        D: tl.constexpr,
):
    idx = tl.program_id(0)  # 获取当前处理的 Token 索引
    slot = tl.load(slot_mapping_ptr + idx)  # 物理映射表：读取当前token被分配到哪个物理槽位
    if slot == -1: return

    # 1 查地址：计算出当前 Token 所有的维度在显存中的具体一维物理地址偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 2 读取：根据刚才算好的地址一次性把当前 Token 完整的 Key 和 Value 从HBM读到SRAM
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 3 量化：将高精度的 Key/Value 强行转换为NVIDIA 特有的 8-bit 浮点数格式（E4M3）
    key_fp8 = key.to(tl.float8e4nv)
    value_fp8 = value.to(tl.float8e4nv)

    # 4 写入：基于刚才查到的物理槽位 slot 计算出在全局 KV Cache 显存池中的目标写入地址
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key_fp8)
    tl.store(v_cache_ptr + cache_offsets, value_fp8)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    # CPU代码：准备参数并发射GPU Kernel
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ===================================================================
# 2. 手写 FP8 PagedAttention (Only Decode)
# 		因为在Decode阶段 Q只有一个token 每个线程块处理的维度为[HIDDEN_DIM]
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
    # 获取线程块在grid中的2D坐标（每个线程块处理1个头）
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_heads_per_kv  # GQA

    context_len = tl.load(Context_Lens + seq_idx)  # 显存读取当前seq请求文本长度 若为0直接退出
    if context_len == 0:
        return

    # 加载 FP16 的 Query（每个seq的每个head）
    q_offset = seq_idx * stride_q_bs + head_idx * stride_q_h + tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_offset)

    # 初始化 Online Softmax 的三个核心状态变量
    m_old = float('-inf')
    s = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # 计算当前用户的历史文本被切分成了多少个物理块

    # FlashAtten：遍历 Paged KV Block Table
    for block_idx in range(num_blocks):
        # PagedAtten查表：当前用户的逻辑块对应的物理块
        physical_block_id = tl.load(Block_Tables + seq_idx * stride_bt_bs + block_idx * stride_bt_b)
        # 生成边界掩码：最后一个block可能没被填满
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

        # FlashAtten：Online Softmax
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
    grid = (bs, num_heads)  # triton根据grid开启二维线程块阵列 每一个序列的每一个头都会独占GPU计算单元并行计算

    '''
    	关于HBM中的数据：在HBM中，所有数据都是平铺排成一列的一维连续内存。stride的定义就是在物理内存中，想把某个维度索引+1时，需要在物理内存中跨过多少个elements。例如q.stride(0), q.stride(1), q.stride(2)就是下一batch、head、hidden_dim的步长；k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),就是下一block、batch、head、hidden_dim的步长；
    '''
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
