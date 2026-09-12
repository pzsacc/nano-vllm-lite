# 优化系列 · 从 36ms 到 3.9ms 的八场战役

> 这是一个"重写 → 崩溃 → 夺回"的真实故事：我们把 nano-vllm 重写了一遍，
> 第一次跑分全线崩溃——**TPOT 36.30ms，而原版 baseline 是 0.54ms**。
> 接下来的 8 章，是每一场夺回战：现象 → Profile → 根因 → 实现 → 代价 → 教学点。
> 所有数字实机测量（RTX 5090 / 4080 Super · Qwen3-0.6B），命令均可复现。

## 战役列表

| 章 | 战役 | 敌人 | 战果 |
|---|------|------|------|
| [00](00-architecture.md) | **先看地图** | 迷路 | 3900 行怎么转 + 3 天读码计划 |
| [01](01-cuda-graph.md) | **CPU 在喂饭，GPU 在挨饿** | 60 次 kernel launch | TPOT 36.30 → 3.90ms（9.3×） |
| [02](02-prefix-caching.md) | **相同的 token，凭什么算两遍？** | 重复 prefill | TTFT -67%，吞吐 +78% |
| [03](03-chunked-prefill.md) | **一个长请求，劫持了整个批次** | TPOT 无界尖刺 | P99 有界化，代价 ≈0 |
| [04](04-fp8-kv-cache.md) | **一半的价格，两倍的容量，六分之一的速度** | KV 显存 | 容量 2×（附决策框架） |
| [05](05-cuda-kernels.md) | **RMSNorm 的账单** | 冗余访存 | 算子带宽 3-4× |
| [06](06-debugging-stories.md) ⭐ | **指标全绿，输出全错** | 两个真实 bug | 一套定位方法论 |
| [07](07-benchmarks-5090.md) | **数字从哪来** | 测量误差 | 三层压测体系 |
| [08](08-async-engine.md) | **token 的最后一公里** | 断连 / 乱码 | 取消传播 + UTF-8 完整性 |

⭐ = 最适合作为第一篇阅读，也是本系列技术含量最高的一章。

## 战绩板

低时延场景（8 并发 · input 256 · output 64 · 4080 Super）：

| 阶段 | TPOT | Output TPS | QPS |
|---|---:|---:|---:|
| 起点：重写版，优化全关 | 36.30 ms | 217 | 3.4 |
| + CUDA Graph（01，与 02 合测） | **3.90 ms** | 1,756 | 27.9 |
| + Prefix Cache（02，90% 命中场景） | 8.53 ms | 4,745 | **75.3** |
| 原版 baseline 参照 | 0.54 ms | 1,859 | 29.0 |

> 01/02 合并测量（二者互相独立于 decode 路径）；02 第三行是另一场景（前缀命中率不同），
> 不可与前两行直接横比——这本身就是 07 章要讲的"怎么读数字"。
> 高吞吐侧（256 并发）：重写版 4,912 → 全优化 5,369 tok/s；旗舰数字 **2,626 tok/s**（4080S，另一场景口径）。

## 决策速查

```
你的场景是什么？
├─ 离线批处理（最大 tok/s）→ CUDA Graph + Prefix Cache + Fused Kernels
├─ 在线服务（TPOT P99 稳定）→ 上面全部 + Chunked Prefill
└─ 大模型/长上下文（显存不足）→ 全部 + FP8 KV（接受 decode 变慢，见 04）
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
