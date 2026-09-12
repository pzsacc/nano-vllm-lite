<p align="center">
  <img src="docs/assets/logo.svg" width="180" alt="nano-vllm-lite">
</p>

<h1 align="center">nano-vllm-lite</h1>

<p align="center">
  <b>从零手写的 LLM 推理引擎：每个优化都有实测数据、踩坑复盘与正确性测试</b><br>
  A from-scratch LLM inference engine — every optimization backed by real benchmarks, war stories, and parity tests.
</p>

<p align="center">
  <a href="README.en.md">English</a> · <a href="docs/README.md">优化系列文档</a> · <a href="#-优化系列">优化列表</a> · <a href="README.en.md#-optimization-series">Docs</a>
</p>

## 这是什么

一个 ~3900 行的 LLM 推理引擎（Python + CUDA + Triton），实现企业级推理优化栈。
它不是 vLLM 的精简复制，而是**每个优化都可以追问"为什么快、快多少、错在哪"的教学项目**：

- **有数据**：每个优化附实测 benchmark（RTX 5090 / 4080 Super），附可复现命令
- **有深度**：[docs/06](docs/06-debugging-stories.md) 复盘两个真实 bug——所有指标正常但输出是错的，如何定位
- **有兜底**：23 个测试，每个优化都有 parity 断言，防止"性能提升，正确性崩塌"

兼容 vLLM 风格 API：`LLM(model).generate(prompts, SamplingParams(...))`

## 核心指标

| 场景 | 指标 | 数值 |
|---|---|---|
| 低时延（8 并发） | TPOT | **3.9ms**（eager 36.3ms，**9.3×**） |
| 高吞吐（256 并发） | 输出吞吐 | **2626 tok/s**（单卡 4080S） |
| Prefix 90% 命中 | TTFT | **-67%**，吞吐 +78% |
| FP8 KV | 容量 | **2×**（61→123 并发序列） |

> 实测环境与复现命令见 [docs/07](docs/07-benchmarks-5090.md)

## 核心代码地图（3 天读完）

```
入口    llm.py ───────────────────────────────────── ~50 行
          │
调度 ★  engine/scheduler.py ──────────────────────── 260 行   decode优先 + chunked预算
          │
执行 ★  engine/model_runner.py ───────────────────── 450 行   CUDA Graph / 输入准备 / 采样
          │
算子 ★  layers/attention.py ──────────────────────── 310 行   PagedAttn / FP8 / Triton
          │        layers/linear.py · layernorm.py · rotary_embedding.py · sampler.py
          │
内核    nano_vllm/kernels/*.cu ───────────────────── ~200 行   fused rmsnorm / inplace rope
          │
基础    engine/block_manager.py ──────────────────── 270 行 ★ prefix caching
        engine/sequence.py · utils/context.py · engine/async_llm_engine.py
```

3 天读码计划与全局设计模式 → [docs/00-architecture.md](docs/00-architecture.md)

## 快速开始

```bash
pip install -e .            # 含 CUDA kernel 编译 (无 toolkit 时自动降级 torch.compile)
```

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(model="/path/to/Qwen3-0.6B")
out = llm.generate(["介绍一下 CUDA Graph"], SamplingParams(temperature=0.6, max_tokens=128))
print(out[0]["text"])
```

流式 API Server（OpenAI 兼容 SSE）：

```bash
python examples/streaming_server.py --model /path/to/Qwen3-0.6B --port 8000
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 64, "stream": true}'
```

## ⚡ 优化系列

| 章 | 战役 | 收益 |
|---|------|------|
| 01 | [CPU 在喂饭，GPU 在挨饿](docs/01-cuda-graph.md) · CUDA Graph | TPOT 36.30 → 3.90ms（**9.3×**） |
| 02 | [相同的 token，凭什么算两遍？](docs/02-prefix-caching.md) · Prefix Caching | 命中 90% 时 TTFT **-67%** |
| 03 | [一个长请求，劫持了整个批次](docs/03-chunked-prefill.md) · Chunked Prefill | 混合负载 TPOT P99 有界化 |
| 04 | [一半的价格，两倍的容量，六分之一的速度](docs/04-fp8-kv-cache.md) · FP8 KV | 容量 **2×**（附决策框架） |
| 05 | [RMSNorm 的账单](docs/05-cuda-kernels.md) · CUDA Kernels | 算子带宽 **3-4×** |
| 06 ⭐ | [指标全绿，输出全错](docs/06-debugging-stories.md) · Debug 复盘 | 两个真实 bug 的定位方法论 |
| 07 | [数字从哪来](docs/07-benchmarks-5090.md) · Benchmark | 三层压测体系 |
| 08 | [token 的最后一公里](docs/08-async-engine.md) · Async 引擎 | 流式输出 3 个工程细节 |

## Profiling 路径（L0→L3）

优化不是玄学，是一条可复现的验证链：

```
L0 对不对   pytest tests/                正确性断言 (23 个, GPU 用例 -m gpu 分流)
L1 多快     bench.py 1   场景压测          TTFT/TPOT/TPS/QPS
L2 显存     bench.py 2   组件占比/容量     KV 探针 (FP8 vs FP16)
L3 算子     bench.py 3   profiler/带宽     定位到具体 kernel
```

```bash
python -m benchmarks.bench 1 --model /path/to/model --scenarios mixed   # 暴露 P99
python -m benchmarks.bench 2 --model /path/to/model                     # KV 容量探针
python -m benchmarks.bench 3 --tool bandwidth                           # 算子对拍
```

## 测试

```bash
pytest tests/ -m "not gpu"   # CPU 可跑 (调度/缓存/解码逻辑)
pytest tests/                # 全量 (含 CUDA Graph parity / kernel parity)
```

每个测试文件头部标注"防的是哪个 bug"——测试即文档。

## Roadmap

- [ ] FP8 decode 换 FlashInfer（已在本机验证 FP8 KV decode 跑通且快于 FP16）
- [ ] ZMQ 进程化引擎（对照 vLLM V1 EngineCore 架构）
- [ ] Speculative Decoding（N-gram proposer）
- [ ] 模型注册机制（LLaMA / Mistral）

## 致谢

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — 本项目的起点与 baseline
- [vLLM](https://github.com/vllm-project/vllm) — 架构参照

## License

MIT
