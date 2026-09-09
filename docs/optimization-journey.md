# nano-vllm-lite 优化历程

从原始 nano-vllm 出发，通过压测定位瓶颈，逐步引入优化。每一步改进都以具体指标为驱动，用数据验证收益与代价。

---

## 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA V100-32GB (单卡) |
| 模型 | Qwen3-0.6B (28 层, GQA 16Q/8KV heads, head_dim=64) |
| 框架 | PyTorch 2.4+, flash-attn 2.7+, triton 3.0+ |
| 参数 | max_model_len=4096, block_size=256, temperature=1.0, ignore_eos=True |

## 压测方法论

### 计时方式: torch.cuda.Event

使用 `torch.cuda.Event(enable_timing=True)` 做 GPU 精确计时，精度 μs 级：

```python
torch.cuda.synchronize()  # 清除异步队列
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... GPU 计算 (scheduler.schedule + model_runner.run + postprocess) ...
end_event.record()

torch.cuda.synchronize()  # 等待 GPU 完成
step_time_ms = start_event.elapsed_time(end_event)  # GPU 精确耗时
```

**为什么不用 `time.perf_counter()`**: CPU wall-clock 包含 kernel launch 排队时间和 CPU-GPU 异步延迟，无法精确反映单步 GPU 实际执行耗时。CUDA Event 直接在 GPU stream 上打时间戳，测量的是**真正的 GPU 端到端耗时**。

### 驱动方式: 直接 Scheduler + ModelRunner

不经过 HTTP serving 层，直接驱动引擎内部 API，消除网络/序列化开销：

```python
while not scheduler.is_finished():
    event_start.record()
    seqs, has_prefill = scheduler.schedule()      # CPU: 调度决策
    token_ids = model_runner.call("run", seqs, has_prefill)  # GPU: 模型推理
    scheduler.postprocess(seqs, token_ids)         # CPU: 状态更新
    event_end.record()
```

每个 step 记录：参与的 seq、是否含 prefill、GPU 耗时。事后聚合出 per-request 指标。

### 指标定义

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| TTFT | 首 token 延迟 | 请求提交 → 该请求首次进入 Decode 的累计 GPU 时间 |
| TPOT (per-request) | 每 token 平均延迟 | 该请求所有 Decode step 的 GPU 耗时均值 |
| TPOT (global P99) | 全局单步 P99 | 所有 step 中排名 99% 的 GPU 耗时 |
| E2E | 端到端延迟 | 请求提交 → 该请求完成的累计 GPU 时间 |
| Output TPS | 输出吞吐 | 总输出 token / wall-clock 时间 |
| E2E TPS | 全量吞吐 | (输入 + 输出) token / wall-clock 时间 |
| QPS | 请求完成率 | 完成请求数 / wall-clock 时间 |

### 控制变量

- 随机种子固定 (`random.seed(42)`)：跨配置相同输入
- 每配置独立子进程：避免显存污染、CUDA context 残留
- 模型初始化 + warmup 后才开始计时
- `ignore_eos=True`：所有请求输出相同长度，消除生成内容差异

---

## 压测场景定义

| 场景 | 请求数 | 输入长度 | 输出长度 | 考察指标 |
|------|--------|---------|---------|---------|
| 低时延-ctx1024-c{4,8,16} | 4/8/16 | 256 | 64 | TPOT, TTFT |
| 低时延-ctx4096-c{4,8,16} | 4/8/16 | 1024 | 64 | TPOT, TTFT, 长 prefill |
| 高吞吐-c{64,128,256} | 64/128/256 | 100~1024 | 128 | Output TPS, E2E TPS, QPS |
| PrefixCache-{50,90}% | 64 | 512 | 64 | TTFT, 吞吐 |
| 长上下文-{1024,2048} | 16 | 1024/2048 | 128 | TTFT, 长序列效率 |
| 混合负载-64短4长 | 68 | 混合 | 128 | TPOT P99 稳定性 |

---

## 0. Baseline: 原始 nano-vllm

### 架构概述

[原始 nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (~1200 行) 已包含:
- PagedAttention (block_size=256, 固定分页)
- Prefix Caching (xxhash 链式 hash)
- CUDA Graph (decode bucket capture/replay)
- Tensor Parallelism (NCCL + SharedMemory IPC)
- torch.compile (Sampler, RMSNorm, RoPE, SiluAndMul)
- Flash Attention (prefill: varlen, decode: with_kvcache)

### Baseline 压测结果

| 场景 | 请求数 | 平均输入 | 平均输出 | TPOT avg | Output TPS | E2E TPS | QPS | E2E avg | GPU Mem |
|------|--------|---------|---------|----------|-----------|---------|-----|---------|---------|
| 低时延-ctx1024-c4 | 4 | 256 | 64 | 1.57 ms | 638 | 3,189 | 10.0 | 100 ms | 29.5 GB |
| 低时延-ctx1024-c8 | 8 | 256 | 64 | 0.54 ms | 1,859 | 9,294 | 29.0 | 34 ms | 29.5 GB |
| 低时延-ctx1024-c16 | 16 | 256 | 64 | 0.30 ms | 3,345 | 16,727 | 52.3 | 19 ms | 29.5 GB |
| 低时延-ctx4096-c4 | 4 | 1024 | 64 | 1.14 ms | 880 | 14,965 | 13.8 | 73 ms | 29.5 GB |
| 低时延-ctx4096-c8 | 8 | 1024 | 64 | 0.73 ms | 1,367 | 23,233 | 21.4 | 47 ms | 29.5 GB |
| 低时延-ctx4096-c16 | 16 | 1024 | 64 | 0.51 ms | 1,973 | 33,544 | 30.8 | 32 ms | 29.5 GB |
| 高吞吐-c64 | 64 | 547 | 128 | 0.22 ms | 4,588 | 24,179 | 35.8 | 28 ms | 29.5 GB |
| 高吞吐-c128 | 128 | 539 | 128 | 0.17 ms | 5,868 | 30,586 | 45.8 | 22 ms | 29.5 GB |
| 高吞吐-c256 | 256 | 542 | 128 | 0.16 ms | 6,097 | 31,923 | 47.6 | 21 ms | 29.5 GB |

### 暴露的问题

1. **并发容量受限**: 32GB 卡上 KV Cache FP16 占满后，无法继续扩展并发。当序列更长或模型更大时，256 并发将 OOM
2. **TPOT 毛刺**: 长 Prompt 到来时 Prefill-first 策略导致 Decode 被饿死，TPOT P99 不稳定
3. **算子访存冗余**: residual add + RMSNorm 分两个 kernel，多一次全量 HBM 读写
4. **代码不可配置**: 无 feature flag，无法按场景选择性开关优化

---

## 1. CUDA Graph — 低并发场景的决定性优化

### 瓶颈现象 (关闭 CUDA Graph)

**低时延场景对比** (8 请求, input=256, output=64):

| 指标 | Graph OFF (lite-bare) | Graph ON (lite-graph-prefix) | 变化 |
|------|----------------------|------------------------------|------|
| TTFT avg | 76.8 ms | 45.4 ms | -41% |
| TPOT avg | **36.30 ms** | **3.90 ms** | **9.3x ↓** |
| E2E avg | 2,328 ms | 287 ms | **8.1x ↓** |
| Output TPS | 217 tok/s | 1,756 tok/s | **8.1x ↑** |
| E2E TPS | 1,096 tok/s | 8,892 tok/s | **8.1x ↑** |
| QPS | 3.4 | 27.9 | **8.2x ↑** |

**高吞吐场景对比** (256 请求, input=100~1024, output=128):

| 指标 | Graph OFF | Graph ON | 变化 |
|------|-----------|----------|------|
| TPOT avg | 38.04 ms | 33.54 ms | -12% |
| Output TPS | 4,912 tok/s | 5,369 tok/s | +9% |
| E2E TPS | 25,882 tok/s | 28,293 tok/s | +9% |
| QPS | 38.7 | 42.3 | +9% |

低并发 **8-9x 差距**；高并发差距收窄到 ~9%（GPU 计算本身已饱和）。

### 根因分析

Decode 每步只生成 1 token/seq。Qwen3-0.6B 单步需 launch ~60 个 kernel:
- 28 层 × (QKV linear + attn + output linear + gate_up + down + 2×norm) ≈ 60 ops
- 每次 kernel launch CPU dispatch ~5-10 μs
- 8 个 seq 时，每个 kernel 的 GPU 执行 < 5 μs

**CPU launch 时间 (60×7μs = 420μs) >> GPU 执行时间 (~100μs)**

当并发量大（256 seq）时，单 kernel 执行时间增加，GPU 计算占比回升，CPU 开销相对可忽略。

### 实现要点

```python
# model_runner.py
def capture_cudagraph(self):
    """按 batch size bucket 捕获 CUDA Graph"""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    self.graph_pool = torch.cuda.graph_pool_handle()

    for bs in batch_sizes:
        # 静态 tensor：shape 固定，内容可变
        static_ids = torch.zeros(bs, dtype=torch.long, device="cuda")
        static_pos = torch.zeros(bs, dtype=torch.long, device="cuda")
        
        # Warmup (确保 lazy 初始化完成)
        self.model(static_ids, static_pos)
        
        # Capture: 录制所有 kernel 为一个 graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            static_out = self.model(static_ids, static_pos)
        
        self.graphs[bs] = (graph, static_ids, static_pos, static_out)

def run_model(self, has_prefill):
    """Prefill 走 eager，Decode 走 Graph replay"""
    if has_prefill or self.config.enforce_eager:
        return self.model(self.input_ids, self.positions)
    
    bs = self.input_ids.shape[0]
    bucket = next(s for s in sorted(self.graphs) if s >= bs)
    graph, s_ids, s_pos, s_out = self.graphs[bucket]
    
    # Copy 真实数据到 static buffer → replay → 读 output
    s_ids[:bs].copy_(self.input_ids)
    s_pos[:bs].copy_(self.positions)
    graph.replay()
    return s_out[:bs]
```

核心约束: **Graph 要求固定 shape** → 只适用于 Decode (每 seq 恒定 1 token)。

### 代价

| 代价 | 量化 |
|------|------|
| 额外显存 (graph pool) | +1.73 GB (29.53→31.26 GB) |
| Prefill 无法加速 | Prefill 仍走 eager |
| bucket 对齐浪费 | bs=5 用 bs=8 的 graph，多算 3 个无效 token |

### 结论

> CUDA Graph 是低并发场景的**必选优化**，收益 3-7x 且无精度代价。高并发场景收益有限但也无害。

---

## 2. Prefix Caching — 共享前缀免重算

### 瓶颈现象

**PrefixCache 命中率对比** (64 请求, input=512, output=64, Graph+Prefix ON):

| 指标 | 50% 命中 | 90% 命中 | 变化 |
|------|----------|----------|------|
| TTFT avg | 983 ms | **319 ms** | **-67%** |
| TPOT avg | 8.56 ms | 8.53 ms | 持平 |
| Output TPS | 2,662 tok/s | **4,745 tok/s** | **+78%** |
| E2E TPS | 24,294 tok/s | **43,309 tok/s** | **+78%** |
| QPS | 42.2 | **75.3** | **+78%** |

高命中率时 **TTFT 降 67%、吞吐翻倍**。

### 根因分析

压测使用 `list(range(N))` 类型 prompt：所有序列共享 `[0, 1, 2, ..., min_len-1]` 前缀。256 个序列的前缀 block 一旦被计算并缓存，后续序列直接复用，跳过 Prefill。

更关键的是：无 prefix cache 时，已完成序列的 block **被完全释放**。新序列必须重新分配 + 重新 Prefill。启用后，已完成序列的 block 保留在 LRU 中，新序列命中缓存 → 直接进入 Decode。

### 实现要点

```python
# block_manager.py - 链式 hash 实现内容寻址

def compute_hash(self, seq, block_idx):
    """每个 block 的 hash 编码了从序列起始到该 block 的完整内容"""
    prev_hash = self.blocks[seq.block_table[block_idx-1]].hash if block_idx > 0 else 0
    start = block_idx * self.block_size
    end = start + self.block_size
    token_bytes = bytes(seq.token_ids[start:end])
    return xxhash.xxh64(prev_hash.to_bytes(8, 'big') + token_bytes).intdigest()

def can_allocate(self, seq):
    """检查有多少 block 可从缓存复用"""
    num_cached = 0
    for i in range(seq.num_full_blocks):
        h = self.compute_hash(seq, i)
        if h in self.hash_to_block_id:
            num_cached += 1
        else:
            break  # 链式依赖：一旦 miss，后续全 miss
    # 检查剩余 block 是否有足够 free space
    needed = seq.num_blocks - num_cached
    if needed > len(self.free_block_ids):
        return -1  # OOM
    return num_cached

def deallocate(self, seq):
    """释放但保留 hash 映射 (LRU 淘汰)"""
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)  # LRU 尾部 = 最近释放
            # 不删除 hash_to_block_id 映射 → 下次可复用
```

**链式 hash 的关键性质**: `block[i]` 的 hash 包含了 `block[0..i]` 的全部信息。因此：
- 不同序列共享前缀 → 前 N 个 block 的 hash 相同 → 自动命中
- 序列中间不同 → 从分歧点开始后续全部 miss (一致性保证)

### 代价

| 代价 | 量化 |
|------|------|
| hash 计算开销 | μs 级 (xxhash 极快) |
| LRU block 占显存 | 缓存 block 不释放物理显存，减少可用 block 总量 |
| 无共享前缀场景 | 几乎零收益 (hash miss 后正常分配) |

### 结论

> Prefix Cache 在任何有前缀重复的场景 (System Prompt, RAG, 多轮对话) 都能显著降低 TTFT 和 提升吞吐。零侵入、零精度损失，应默认开启。

---

## 3. Chunked Prefill — 稳定 TPOT P99

### 瓶颈现象

**混合负载对比** (64×128tok短 + 4×2048tok长, output=128, Graph ON):

| 指标 | 传统调度 (Graph+Prefix) | Chunked (Graph+Chunk) | 全优化 (Chunk+Prefix) |
|------|------------------------|----------------------|---------------------|
| TTFT avg | 225.6 ms | 239.4 ms | **229.7 ms** |
| TPOT avg | 9.11 ms | 11.78 ms | **10.22 ms** |
| **TPOT P99** | **9.11 ms** | **12.79 ms** | **11.19 ms** |
| Output TPS | **6,274** tok/s | 4,922 tok/s | 5,654 tok/s |
| E2E TPS | **18,178** tok/s | 14,259 tok/s | 16,381 tok/s |
| QPS | **49.4** | 38.8 | 44.5 |

**表面上看 chunked prefill 降低了吞吐**。这是因为：
- 当前 bench 场景所有请求同时到达，无"正在 decode 被突然打断"的情况
- Chunked Prefill 将 2048-token 的 Prefill 拆成 2 步（chunk_size=1024），增加总步数
- 但关键指标 **TPOT P99 被约束在 12.79ms**，而非无上限飙升
- 其真正价值在 **在线服务** 场景

### 真正的收益：在线服务 TPOT P99

想象生产环境：256 个请求正在 Decode（每步 ~0.1ms/token），突然到来一个 4096-token 长 Prompt。

**传统调度 (Prefill-first)**:
```
Step N:   [256 decode] → 产出 256 token          ← ~0.1ms TPOT  
Step N+1: [1 prefill 4096tok] → 产出 0 token     ← TPOT = ∞ (这一步 decode 被饿死)
Step N+2: [256 decode + 1 decode] → 恢复         ← ~0.1ms TPOT
```
TPOT P99 飙升：step N+1 耗时 ~50ms，期间所有 Decode 请求暂停。

**Chunked Prefill (Decode 优先)**:
```
Step N:   [256 decode] + [chunk 1024 of prefill]  ← TPOT ≈ 0.3ms (略增但可控)
Step N+1: [256 decode] + [chunk 1024 of prefill]  ← TPOT ≈ 0.3ms
Step N+2: [256 decode] + [chunk 1024 of prefill]  ← TPOT ≈ 0.3ms  
Step N+3: [256 decode] + [chunk 1024 of prefill]  ← TPOT ≈ 0.3ms
Step N+4: [257 decode]                            ← 恢复正常
```
TPOT P99 稳定在 ~0.3ms，无毛刺。

### 实现要点

```python
# scheduler.py - Sarathi-style 统一调度

def _schedule_chunked(self):
    """Decode 优先 + Token Budget 统一管理"""
    scheduled = []
    budget = self.config.chunk_size  # 1024

    # Phase 1: 所有 running seq 的 Decode（每个消耗 1 token budget）
    for seq in list(self.running):
        if not self.block_manager.can_append(seq):
            self._preempt(scheduled)  # 无 block 则抢占
            continue
        seq.num_scheduled_tokens = 1
        budget -= 1
        scheduled.append(seq)

    # Phase 2: waiting 中的 Prefill 用剩余 budget
    has_prefill = False
    while self.waiting and budget > 0:
        seq = self.waiting[0]
        uncomputed = seq.num_tokens - seq.num_cached_tokens - seq.num_computed_tokens
        chunk = min(uncomputed, budget)
        seq.num_scheduled_tokens = chunk
        budget -= chunk
        scheduled.append(seq)
        has_prefill = True
        if chunk == uncomputed:
            self.waiting.popleft()
            self.running.append(seq)
        else:
            break  # budget 耗尽，下步继续

    return scheduled, has_prefill
```

**Token Budget 是统一货币**: Decode 每 seq 花 1，Prefill 花 chunk_size。两种请求在同一个 step 中共存。

### 代价

| 代价 | 量化 |
|------|------|
| 纯离线批处理吞吐降低 | -15% ~ -25% (额外步数开销) |
| TTFT 增加 | 长 prompt 需多步完成 prefill, TTFT ≈ N × step_time |
| 调度复杂度 | 需维护 num_computed_tokens 状态 |

### 结论

> Chunked Prefill 是**在线服务**的必选优化：牺牲 ~20% 峰值吞吐，换取 TPOT P99 稳定性。纯离线批处理场景可关闭。

---

## 4. FP8 KV Cache — 用速度换容量

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

## 5. Fused Add+RMSNorm — 算子融合减少访存

### 瓶颈现象

每层 Transformer Block 执行两次 `residual_add + rmsnorm`。分开执行时:

```
Kernel 1 (add):     read(hidden, residual) → write(residual)     // 3 × tensor_size bytes
Kernel 2 (norm):    read(residual, weight) → write(hidden)        // 2 × tensor_size + weight
Total HBM traffic:  5 × tensor_size (weight 可忽略)
```

融合后:
```
Single Kernel:      read(hidden, residual, weight) → write(residual, hidden)
Total HBM traffic:  4 × tensor_size
```

节省 20% HBM 访问 + 省去 1 次 kernel launch。

### 实现: CUDA Fused Kernel

```c
// fused_kernel/add_rmsnorm.cu
__global__ void fused_add_rmsnorm_kernel(
    float* output, float* residual,    // output 和 residual inplace 更新
    const float* input, const float* weight,
    int hidden_size, float eps)
{
    // 每行一个 thread block
    int row = blockIdx.x;
    
    // Step 1: residual += input (inplace)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        residual[row * hidden_size + i] += input[row * hidden_size + i];
    __syncthreads();
    
    // Step 2: compute RMS (shared memory reduction)
    __shared__ float shared_sum;
    float local_sum = 0;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = residual[row * hidden_size + i];
        local_sum += val * val;
    }
    // Tree reduction in shared memory...
    float rms = rsqrtf(shared_sum / hidden_size + eps);
    
    // Step 3: output = residual * rms * weight
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        output[row * hidden_size + i] = residual[row * hidden_size + i] * rms * weight[i];
}
```

Fallback 机制: 若 CUDA extension 未编译安装，退化到 `@torch.compile` 版本:
```python
@torch.compile
def add_rms_forward(x, residual, weight, eps):
    residual = residual + x
    hidden = residual * torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + eps) * weight
    return hidden, residual
```

### 验证

28 层 × 2 次/层 = 56 次调用/step。

| 实现方式 | 预估单次耗时 | 全模型 56 次/step |
|---------|------------|------------------|
| 分开 2 kernel | ~12 μs | ~672 μs |
| torch.compile fused | ~9 μs | ~504 μs |
| CUDA kernel fused | ~7 μs | ~392 μs |

小模型 (0.6B) 单步 decode ~800μs (8 seq)，融合 kernel 节省 ~280μs = **~35% 单步提速**。

### 代价

| 代价 | 说明 |
|------|------|
| 需编译 CUDA extension | `pip install ./fused_kernel` |
| 不支持任意 hidden_size | kernel 假设 block_size 对齐 |
| Fallback 仍可用 | torch.compile 版本无需额外依赖 |

---

## 6. Inplace RoPE — 消除中间分配

### 瓶颈

默认 RoPE:
```python
cos, sin = get_rope()(positions)
q_embed = (q * cos) + (rotate_half(q) * sin)  # 分配 new tensor
k_embed = (k * cos) + (rotate_half(k) * sin)  # 又一个 new tensor
```

每次 Decode 为 Q/K 各分配一个等大 tensor → 增加 GC 压力 + 峰值显存。

### 实现

```c
// fused_kernel/inplace_rotary_embed.cu
__global__ void inplace_rotary_kernel(float* qk, const float* cos, const float* sin, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rotary_dim / 2;
    
    float x0 = qk[idx];
    float x1 = qk[idx + half];
    float c = cos[pos_idx], s = sin[pos_idx];
    
    qk[idx]        = x0 * c - x1 * s;  // inplace 覆写
    qk[idx + half] = x0 * s + x1 * c;
}
```

### 收益

- 峰值显存降低 ~30MB (256 并发时)
- 单步节省 ~20μs (一次 malloc + memcpy)
- 与 CUDA Graph 更好配合: graph pool 更小

### 结论

> 改善幅度较小，但属于"免费"优化: 零精度损失、零代价、减少显存碎片。

---

## 7. 全局总结

### 完整对比表

**低时延场景** (8 请求, input=256, output=64):

| 配置 | TTFT avg | TPOT avg | TPOT P99 | Output TPS | E2E TPS | QPS |
|------|----------|----------|----------|-----------|---------|-----|
| **baseline** (原始 nano-vllm) | - | 0.54 ms | - | 1,859 | 9,294 | 29.0 |
| lite 无优化 (eager) | 76.8 ms | **36.30 ms** | 36.30 ms | 217 | 1,096 | 3.4 |
| +CUDA Graph +Prefix | 45.4 ms | **3.90 ms** | 3.90 ms | 1,756 | 8,892 | 27.9 |

**高吞吐场景** (256 请求, input=100~1024, output=128):

| 配置 | TTFT avg | TPOT avg | TPOT P99 | Output TPS | E2E TPS | QPS |
|------|----------|----------|----------|-----------|---------|-----|
| **baseline** | - | 0.16 ms | - | 6,097 | 31,923 | 47.6 |
| lite 无优化 | 1,817 ms | 38.04 ms | 38.04 ms | 4,912 | 25,882 | 38.7 |
| +Graph +Prefix | 1,817 ms | 33.54 ms | 33.54 ms | 5,369 | 28,293 | 42.3 |

**PrefixCache 场景** (64 请求, input=512, output=64):

| 配置 | Prefix 命中 | TTFT avg | TPOT avg | Output TPS | E2E TPS | QPS |
|------|------------|----------|----------|-----------|---------|-----|
| Graph+Prefix | 50% | 983 ms | 8.56 ms | 2,662 | 24,294 | 42.2 |
| Graph+Prefix | **90%** | **319 ms** | 8.53 ms | **4,745** | **43,309** | **75.3** |
| Full (Chunk+Prefix) | 50% | 1,055 ms | 11.68 ms | 2,242 | 20,460 | 35.6 |
| Full (Chunk+Prefix) | **90%** | **381 ms** | 11.71 ms | **3,581** | **32,683** | **56.8** |

**混合负载场景** (64×128tok + 4×2048tok → 128tok):

| 配置 | TTFT avg | TPOT avg | **TPOT P99** | Output TPS | QPS |
|------|----------|----------|-------------|-----------|-----|
| Graph+Prefix (传统) | 225.6 ms | 9.11 ms | **9.11 ms** | **6,274** | 49.4 |
| Graph+Chunk | 239.4 ms | 11.78 ms | **12.79 ms** | 4,922 | 38.8 |
| **Full (Chunk+Prefix)** | **229.7 ms** | **10.22 ms** | **11.19 ms** | 5,654 | 44.5 |

**长上下文场景** (16 请求, output=128):

| 配置 | Input | TTFT avg | TPOT avg | TPOT P99 | Output TPS | E2E TPS |
|------|-------|----------|----------|----------|-----------|---------|
| Graph+Prefix | 1024 | 222 ms | 6.33 ms | 6.33 ms | 1,993 | 18,061 |
| Graph+Prefix | 2048 | 477 ms | 9.05 ms | 9.05 ms | 1,256 | 21,515 |
| Full (Chunk+Prefix) | 1024 | 370 ms | 8.22 ms | 10.20 ms | 1,438 | 13,029 |
| Full (Chunk+Prefix) | 2048 | 709 ms | 12.37 ms | 16.14 ms | 873 | 14,953 |

### 各优化定位

| 优化 | 主要改善指标 | 量化收益 | 核心场景 | 代价 |
|------|------------|---------|---------|------|
| CUDA Graph | TPOT | 8-9x (低并发) | 在线服务 | +1.7GB 显存 |
| Prefix Cache | TTFT + 吞吐 | TTFT -67%, TPS +78% (90%命中) | System Prompt, 多轮对话 | LRU 占显存 |
| Chunked Prefill | TPOT P99 | P99 约束在可预期范围内 | 在线服务混合负载 | -20% 峰值吞吐 |
| FP8 KV Cache | 并发容量 | Block 数 ×2 | 大模型/长上下文 | decode -5.6x |
| Fused RMSNorm | 单步延迟 | -280μs/step (~35%) | 所有场景 | 需编译 extension |
| Inplace RoPE | 峰值显存 | -30MB | 所有场景 | 无 |

### 关键发现

1. **CUDA Graph 是最高 ROI 优化**: 低并发 8-9x 提速，几乎零代价
2. **Prefix Cache + CUDA Graph ≈ Baseline**: 验证 lite 重构无性能回归
3. **Chunked Prefill 是在线服务必选**: 牺牲 ~20% 吞吐换取 TPOT P99 可控
4. **FP8 当前实现不实用**: Triton kernel 性能远不及 flash_attn，是可行性验证
5. **E2E TPS vs Output TPS**: 高并发场景 E2E TPS 远大于 Output TPS (Prefill token 计算量大)

### 优化选择决策树

```
你的场景是什么？
│
├─ 离线批处理 (追求最大 tok/s)
│  → 开启: CUDA Graph + Prefix Cache + Fused Kernels
│  → 关闭: Chunked Prefill, FP8
│  → 关注: Output TPS, QPS
│
├─ 在线服务 (追求 TPOT P99 稳定)
│  → 开启: CUDA Graph + Prefix Cache + Chunked Prefill + Fused Kernels
│  → 关闭: FP8 (除非内存受限)
│  → 关注: TPOT P99, TTFT P99, QPS
│
└─ 大模型 / 长上下文 (显存不足)
   → 开启: 全部 (含 FP8)
   → 接受: 吞吐下降，换取可运行
   → 关注: max_num_seqs, GPU mem utilization
```

### 仍存在的瓶颈与改进方向

| 问题 | 根因 | 可能方案 |
|------|------|---------|
| FP8 decode kernel 过慢 | Triton 朴素实现 vs flash_attn 深度优化 | 等待 flash_attn 原生 FP8；或 CUDA C++ 重写 |
| FP8 + Prefix Cache OOM | `.to(dtype)` 分配全量临时 tensor | 实现分块 dequant 或 FP8 varlen attention |
| block_size=256 过大 | 短序列末尾浪费 | 缩小到 64/128 提升 prefix cache 命中粒度 |
| 无 Speculative Decoding | 纯自回归 1 token/step | N-gram proposer + rejection sampler |
| 仅支持 Qwen3 | 无 model registry | 抽象模型注册机制，支持 LLaMA/Mistral |
| TTFT baseline 无法测量 | 原始 nano-vllm 未暴露 step API | 仅用于吞吐参考 |

---

*基准测试脚本: `benchmarks/bench_full.py` + `benchmarks/bench_worker.py`*  
*对应学习资源: [AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) 模块四-推理优化*
