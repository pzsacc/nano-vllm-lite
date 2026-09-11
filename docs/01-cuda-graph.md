# 01 · CUDA Graph — 低并发场景的决定性优化

[目录](README.md) | 下一篇: [Prefix Caching](02-prefix-caching.md)

> 核心代码: `engine/model_runner.py` · 正确性测试: `tests/test_cudagraph_parity.py`

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
