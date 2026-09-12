# 技术文档总目录

> 每篇文档固定结构：**现象 → 根因分析 → 实现 → 代价 → 要点**。
> 所有数据实机测量（RTX 5090 / 4080 Super · Qwen3-0.6B），复现命令见各篇与 `benchmarks/`。

## 文档列表

| 章 | 文档 | 核心代码 | 正确性测试 |
|---|------|---------|-----------|
| [00](00-architecture.md) | 架构总览：模块分层与读码路径 | — | — |
| [01](01-cuda-graph.md) | CUDA Graph：消除 Decode 阶段的 Kernel 启动开销 | `engine/model_runner.py` | `test_cudagraph_parity.py` |
| [02](02-prefix-caching.md) | Prefix Caching：基于内容寻址的 KV Cache 复用 | `engine/block_manager.py` | `test_prefix_cache.py` |
| [03](03-chunked-prefill.md) | Chunked Prefill：混合负载下的延迟控制 | `engine/scheduler.py` | `test_chunked_prefill.py` |
| [04](04-fp8-kv-cache.md) | FP8 KV Cache：容量与吞吐的权衡分析 | `layers/attention.py` | `test_cudagraph_parity.py` |
| [05](05-cuda-kernels.md) | CUDA Kernel：算子融合与访存削减 | `nano_vllm/kernels/` | `test_kernel_parity.py` |
| [06](06-debugging-stories.md) | 正确性调试：两个隐性 Bug 的定位过程 | — | `test_cudagraph_parity.py` |
| [07](07-benchmarks-5090.md) | 性能测量：基准测试设计与实测数据 | `benchmarks/` | — |
| [08](08-async-engine.md) | 异步流式引擎：请求取消与增量解码 | `engine/async_llm_engine.py` | `test_detokenizer.py` |

## 优化效果总览

低时延场景（8 并发 · input 256 · output 64 · 4080 Super）：

| 阶段 | TPOT | Output TPS | QPS |
|---|---:|---:|---:|
| 重写版，优化全关 | 36.30 ms | 217 | 3.4 |
| + CUDA Graph（01，与 02 合测） | **3.90 ms** | 1,756 | 27.9 |
| + Prefix Cache（02，90% 命中场景） | 8.53 ms | 4,745 | **75.3** |
| 原版 nano-vllm 参照 | 0.54 ms | 1,859 | 29.0 |

> 01/02 为合并测量（二者相互独立于 decode 路径）；02 第三行是另一场景（前缀命中率不同），
> 与前两行不可直接横比——测量口径问题见 07 章。
> 高吞吐侧（256 并发）：重写版 4,912 → 全优化 5,369 tok/s；旗舰数字 **2,626 tok/s**（4080S，另一场景口径）。

## 场景决策

```
离线批处理（最大 tok/s）        → CUDA Graph + Prefix Cache + Fused Kernels
在线服务（TPOT P99 稳定）       → 以上全部 + Chunked Prefill
大模型 / 长上下文（显存受限）    → 以上全部 + FP8 KV（吞吐代价见 04）
```

## 正确性保障

```
CUDA Graph      →  tests/test_cudagraph_parity.py
Prefix Cache    →  tests/test_prefix_cache.py
Chunked Prefill →  tests/test_chunked_prefill.py
CUDA Kernels    →  tests/test_kernel_parity.py
Async Engine    →  tests/test_detokenizer.py
```
