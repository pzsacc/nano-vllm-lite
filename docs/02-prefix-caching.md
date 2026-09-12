# 02 · 相同的 token，凭什么算两遍？

> **Prefix Caching · 90% 命中时 TTFT -67%，吞吐 +78%**
> 核心代码 `engine/block_manager.py` · 测试 `tests/test_prefix_cache.py`
> [战役目录](README.md) · 上一篇 [01 · CPU 在喂饭，GPU 在挨饿](01-cuda-graph.md)

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

---

**下一战**：延迟稳了，但一个 2048-token 的长请求正在逼近整个批次——[03 · 一个长请求，劫持了整个批次](03-chunked-prefill.md)
