# nano-vllm-lite 优化系列 · 总目录

> 每篇一个可验证的优化，固定结构：**现象 → Profile → 根因 → 实现 → 代价 → 教学点**。
> 所有数据为实机测量（RTX 5090 / RTX 4080 Super，Qwen3-0.6B），可复现命令见各篇。

## 阅读顺序（建议配合读码路径）

| # | 文档 | 一句话收益 | 核心文件 |
|---|------|-----------|---------|
| 00 | [架构总览](00-architecture.md) | 数据流与读码地图 | — |
| 01 | [CUDA Graph](01-cuda-graph.md) | 低并发 TPOT **36ms → 3.9ms** | `engine/model_runner.py` |
| 02 | [Prefix Caching](02-prefix-caching.md) | 90% 命中时 TTFT **-67%** | `engine/block_manager.py` |
| 03 | [Chunked Prefill](03-chunked-prefill.md) | TPOT P99 可控（混合负载） | `engine/scheduler.py` |
| 04 | [FP8 KV Cache](04-fp8-kv-cache.md) | 容量 **2×**（61→123 seqs） | `layers/attention.py` |
| 05 | [CUDA Kernels](05-cuda-kernels.md) | Add+RMSNorm 带宽 **3-4×** | `nano_vllm/kernels/` |
| 06 | [Debug 复盘](06-debugging-stories.md) | 两个真实 bug 的定位全程 | — |
| 07 | [Benchmark 方法论](07-benchmarks-5090.md) | 三层压测体系与 5090 实测 | `benchmarks/` |
| 08 | [Async 引擎](08-async-engine.md) | 流式输出的工程细节 | `engine/async_llm_engine.py` |

## 优化选择速查

```
你的场景是什么？
├─ 离线批处理（最大 tok/s）→ CUDA Graph + Prefix Cache + Fused Kernels
├─ 在线服务（TPOT P99 稳定）→ 上面全部 + Chunked Prefill
└─ 大模型/长上下文（显存不足）→ 全部 + FP8 KV（接受 decode 变慢）
```

## 一致性保障

每个优化的正确性都有测试兜底（`pytest -m gpu` 需要 GPU）：

```
CUDA Graph      →  tests/test_cudagraph_parity.py
Prefix Cache    →  tests/test_prefix_cache.py
Chunked Prefill →  tests/test_chunked_prefill.py
CUDA Kernels    →  tests/test_kernel_parity.py
Async Engine    →  tests/test_detokenizer.py
```
