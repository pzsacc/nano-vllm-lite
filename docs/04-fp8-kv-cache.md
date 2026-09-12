# 04 · 一半的价格，两倍的容量，六分之一的速度

> **FP8 KV Cache · 容量 61 → 123 seqs（decode 0.15×，附何时值得的决策框架）**
> 核心代码 `layers/attention.py` · 测试 `tests/test_cudagraph_parity.py`
> [战役目录](README.md) · 上一篇 [03 · 一个长请求，劫持了整个批次](03-chunked-prefill.md)

### 瓶颈现象

| 配置 | KV Block 数量 | 单 Block 大小 | 最大并发 (4096 ctx) |
|------|--------------|--------------|-------------------|
| FP16 KV | ~1,500 | 14.3 MB | ~93 seq |
| FP8 KV | **~3,000** | **7.2 MB** | **~187 seq** |

FP8 将 KV Cache 容量翻倍。

### 但有严重代价

| 配置 | low_short tok/s | high_tp tok/s | 相对 FP16 |
|------|----------------|---------------|-----------|
| Graph + FP16 | 846 | 5,266 | 1.0x |
| Graph + FP8 | **248** | **936** | **0.18x** |

**FP8 Decode Kernel 导致 ~5.6x 性能下降。**

### 根因

`flash_attn_with_kvcache` 不接受 FP8 输入。启用 FP8 后 decode 路径变为:

```python
# 自定义 Triton kernel (非优化级别的实现)
def fp8_paged_attention_decode_kernel(...):
    # 逐 block 遍历 + online softmax
    for block_idx in range(num_blocks):
        k = tl.load(k_ptr).to(tl.float32)  # FP8→FP32
        v = tl.load(v_ptr).to(tl.float32)
        scores = tl.dot(q, k.T) * scale
        # online softmax accumulation...
```

对比 `flash_attn_with_kvcache`:
- flash_attn: NVIDIA 深度优化的 CUDA kernel，利用 Tensor Core、shared memory pipeline
- 自定义 Triton: 朴素实现，无 warp-level 优化，内存访问模式未调优

### 何时值得启用

FP8 KV Cache 的价值 **不在加速，而在容量**:

```
场景: 70B 模型, 80GB GPU, 上下文 32K
- FP16 KV: 每 seq 占 ~4GB KV → 最多 ~8 并发
- FP8 KV:  每 seq 占 ~2GB KV → 最多 ~16 并发 (2x)
```

当并发受限于 KV Cache 显存（而非 compute）时，FP8 以速度换容量，使得更多请求可以并行 → 总吞吐提升。

### 实现要点

```python
# model_runner.py - 分配时选择 dtype
cache_dtype = torch.float8_e4m3fn if config.enable_fp8_kvcache else hf_config.torch_dtype
kv_cache = torch.zeros(..., dtype=cache_dtype, device="cuda")

# attention.py - store 时自动量化 (Triton kernel)
@triton.jit
def store_kvcache_kernel(k, v, k_cache, v_cache, slot_mapping, ...):
    slot = tl.load(slot_mapping + pid)
    k_val = tl.load(k + pid * D + offsets)
    tl.store(k_cache + slot * D + offsets, k_val)  # FP16→FP8 自动截断

# attention.py - decode 时直接读 FP8
def _decode_attention(self, q, ctx):
    if self.fp8_enabled:
        return custom_fp8_decode_attention(q, self.k_cache, self.v_cache, ...)
    else:
        return flash_attn_with_kvcache(q, self.k_cache, self.v_cache, ...)
```

### 已知问题

FP8 + Prefix Cache 组合存在 OOM bug:
```python
# prefill with prefix cache hit → 需要从 FP8 cache 读回
k, v = self.k_cache.to(q.dtype), self.v_cache.to(q.dtype)  # 整个 cache dequant!
```
`.to()` 分配了全量 FP16 临时 tensor → OOM。需要实现分块 dequant 或 FP8 varlen attention。

### 代价

| 代价 | 量化 |
|------|------|
| Decode 吞吐下降 | **5.6x 慢** (Triton vs flash_attn) |
| 精度损失 | FP8 E4M3 动态范围有限，长序列可能累积误差 |
| Prefix Cache 不兼容 | 当前实现 OOM |

### 结论

> FP8 KV Cache 是**大模型 + 长上下文 + 内存受限**场景的应急方案。当前 Triton decode kernel 性能远不及 flash_attn，需要更深度的 kernel 优化 (或等待 flash_attn 原生 FP8 支持) 才能实用。

---

---

**下一战**：容量到手，回头打磨算子本身——[05 · RMSNorm 的账单：每个字节都要过内存](05-cuda-kernels.md)
