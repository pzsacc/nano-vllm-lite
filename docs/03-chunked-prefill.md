# 03 · Chunked Prefill — 稳定 TPOT P99

[上一篇](02-prefix-caching.md) | [目录](README.md) | 下一篇: [FP8 KV Cache](04-fp8-kv-cache.md)

> 核心代码: `engine/scheduler.py` · 正确性测试: `tests/test_chunked_prefill.py`

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
