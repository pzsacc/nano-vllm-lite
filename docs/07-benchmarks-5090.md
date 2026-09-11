# 07 · Benchmark 方法论与 5090 实测

[上一篇](06-debugging-stories.md) | [目录](README.md) | 下一篇: [Async 引擎](08-async-engine.md)

> 本篇回答：数字从哪来、怎么测才可信、如何在你的机器上复现。
> 核心代码: `benchmarks/`（统一入口 `python -m benchmarks.bench {1|2|3}`）

## 三层压测体系

```
L1 服务层压测      bench.py 1    "多快"     TTFT/TPOT/P99/TPS/QPS
  ↓ 数字反常?
L2 显存分析        bench.py 2    "显存去哪了"  组件占比 / KV 容量探针
  ↓ 排除容量因素?
L3 算子剖析        bench.py 3    "慢在哪一行"  torch.profiler / 带宽对拍 / nsys
  ↓ 定位到 kernel
L0 正确性回归      pytest tests/ "对不对"     优化后必跑, 防止"快速生成垃圾"
```

**测不准的三件事**（踩过的坑）：

1. **用 wall-clock 而非 datetime**：`time.perf_counter()` 单调时钟；直接驱动
   Scheduler+ModelRunner 绕过 Python 门面层，减少一层噪声
2. **TPOT 必须分位数看**：avg 会掩盖长 prefill 对 decode 的阻塞（见 03 篇），
   P99 才是在线服务要守的指标；`mixed` 场景是唯一能暴露它的
3. **每配置独立子进程**：CUDA Graph 池和 NCCL 状态会跨实例污染，靠子进程
   隔离才能拿到干净数据

## 5090 实测（2026-09-09，部分 09-11 复测）

环境：RTX 5090 32GB（SM120）· torch 2.13+cu130 · triton 3.7 · flash-attn 2.8.3（源码重编，CCCL 补丁）
模型：Qwen3-0.6B（28 层，GQA 16Q/8KV，head_dim=128，hidden=1024）

### 1. Fused Add+RMSNorm（`bench.py 3 --tool bandwidth`）

| tokens | eager GB/s | torch.compile GB/s | CUDA kernel GB/s | CUDA/eager |
|---:|---:|---:|---:|---:|
| 1024 | 408 | 1046 | 1458 | 3.57× |
| 4096 | 361 | 1837 | 1198 | 3.32× |
| 8192 | 232 | 1136 | 960 | 4.13× |
| 16384 | 223 | 1138 | 979 | 4.39× |

vs eager 3.3-4.4×；vs torch.compile 大致持平 → **融合本身是主要收益，手写与
compile 互有胜负**（手写的价值在可控 + 可作为更大融合的基座）。
正确性：residual 与参考实现逐位一致（见 `test_kernel_parity.py`）。

### 2. FP8 KV 容量探针（`bench.py 2 --mode kv_capacity`）

| | KV 池 | 每 token | 满长度最大并发 |
|---|---|---:|---:|
| FP16 | 27.0 GiB | 114688 B | **61 seqs** |
| FP8 | 27.1 GiB | 57344 B | **123 seqs** |

容量 **2.0×**（2026-09-11 复测 2.02×）。池字节由 utilization 预算决定，
dtype 减半 → 单价减半 → 容量翻倍。

### 3. 端到端吞吐 A/B（64 seqs, in≤512 / out 256）

| 配置 | 吞吐 | 相对 FP16 |
|---|---:|---:|
| FP16 KV（chunked on） | 9119-9370 tok/s | 1.0× |
| FP16 KV（chunked off） | 9098 tok/s | ~1.0×（±2%） |
| FP8 KV（自研 Triton decode） | 1372 tok/s | **0.15×** |

结论：0.6B 上 KV 容量不是瓶颈，自研 FP8 decode kernel 的速度损失远超容量
收益 → **"FP8 的价值在容量不在速度"**，容量收益只在 KV 成为瓶颈时兑现
（大模型/长上下文）。改进方向见 README Roadmap（FlashInfer 已实测可跑通
FP8 decode 且快于 FP16）。

## 在你的机器上复现

```bash
# L1: 全场景压测 (~5 min)
python -m benchmarks.bench 1 --model /path/to/Qwen3-0.6B
# 只跑低时延 + 混合
python -m benchmarks.bench 1 --model /path/to/Qwen3-0.6B --scenarios low_latency mixed
# L2: 显存足迹 + 容量探针
python -m benchmarks.bench 2 --model /path/to/Qwen3-0.6B
# L3: 算子对拍 + trace
python -m benchmarks.bench 3 --tool bandwidth
python -m benchmarks.bench 3 --tool profiler --model /path/to/Qwen3-0.6B
```

⚠️ 5090 环境注意：flash-attn 需源码重编（加
`-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`，`--no-deps` 防 pip 升级 torch），
详见仓库 setup.py 注释。

## 历史数据（4080 Super 时代）

完整的多场景对比表（低时延/高吞吐/PrefixCache/长上下文 × 多配置）最初
在 4080 Super 上采集，数字已在各优化篇章内随文给出。两代 GPU 的相对
结论一致：优化排序不变，绝对值随显存带宽放大。

[上一篇](06-debugging-stories.md) | [目录](README.md) | [下一篇](08-async-engine.md)
