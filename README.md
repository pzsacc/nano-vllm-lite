<p align="center">
  <img src="docs/assets/logo.svg" width="180" alt="nano-vllm-lite">
</p>

<h1 align="center">nano-vllm-lite</h1>

<p align="center">
  <b>轻量级 LLM 推理引擎：约 3900 行实现核心优化栈，每项优化附实测数据与正确性测试</b>
</p>

<p align="center">
  <a href="README.en.md">English</a> · <a href="docs/README.md">技术文档</a> · <a href="#性能指标">性能指标</a>
</p>

## 项目简介

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 重写的轻量级 LLM 推理引擎，约 3900 行 Python / CUDA / Triton，覆盖 PagedAttention、Prefix Caching、Chunked Prefill、FP8 KV Cache、CUDA Graph、张量并行、自研 CUDA Kernel 与异步流式引擎。

与同类学习项目的差异：每项优化均附**实机测量数据**（RTX 5090 / 4080 Super）、**完整的根因分析**与**正确性测试**，所有结论可在本地复现。

API 兼容 vLLM 风格：`LLM(model).generate(prompts, SamplingParams(...))`

## 性能指标

| 优化项 | 指标 | 数值 | 明细 |
|---|---|---|---|
| CUDA Graph | Decode TPOT（8 并发） | 36.30 → **3.90 ms**（9.3×） | [docs/01](docs/01-cuda-graph.md) |
| Prefix Caching | TTFT（90% 命中） | **-67%**，吞吐 +78% | [docs/02](docs/02-prefix-caching.md) |
| Chunked Prefill | 混合负载 TPOT P99 | 无界尖刺 → 有界可控 | [docs/03](docs/03-chunked-prefill.md) |
| FP8 KV Cache | 并发容量（4096 上下文） | 61 → **123 seqs**（2×） | [docs/04](docs/04-fp8-kv-cache.md) |
| Fused Add+RMSNorm | 算子带宽 | **3-4×** vs eager | [docs/05](docs/05-cuda-kernels.md) |
| 整机吞吐 | 256 并发输出吞吐（4080S） | **2626 tok/s** | [docs/07](docs/07-benchmarks-5090.md) |

测试环境：RTX 5090 32GB（torch 2.13+cu130）/ RTX 4080 Super 32GB，Qwen3-0.6B（BF16）。
复现命令见 [docs/07](docs/07-benchmarks-5090.md)。

## 快速开始

```bash
pip install -e .
# 无 CUDA toolkit 时自动跳过 kernel 编译，运行时降级为 torch.compile 实现
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

## 架构与读码路径

```
入口    llm.py ───────────────────────────────────── ~50 行
          │
调度 ★  engine/scheduler.py ──────────────────────── 260 行   decode 优先 + chunked 预算
          │
执行 ★  engine/model_runner.py ───────────────────── 450 行   CUDA Graph / 输入准备 / 采样
          │
算子 ★  layers/attention.py ──────────────────────── 310 行   PagedAttention / FP8 / Triton
          │        layers/linear.py · layernorm.py · rotary_embedding.py · sampler.py
          │
内核    nano_vllm/kernels/*.cu ───────────────────── ~200 行   fused rmsnorm / inplace rope
          │
基础    engine/block_manager.py ──────────────────── 270 行 ★ prefix caching
        engine/sequence.py · utils/context.py · engine/async_llm_engine.py
```

3 天读码顺序与全局设计模式见 [docs/00-architecture.md](docs/00-architecture.md)。

## 技术文档

| 章 | 文档 | 内容 |
|---|------|------|
| 00 | [架构总览：模块分层与读码路径](docs/00-architecture.md) | 数据流图、模块依赖、读码顺序 |
| 01 | [CUDA Graph：消除 Decode 阶段的 Kernel 启动开销](docs/01-cuda-graph.md) | 静态捕获/回放、显存代价 |
| 02 | [Prefix Caching：基于内容寻址的 KV Cache 复用](docs/02-prefix-caching.md) | 链式哈希、引用计数共享 |
| 03 | [Chunked Prefill：混合负载下的延迟控制](docs/03-chunked-prefill.md) | decode 优先调度、预算分块 |
| 04 | [FP8 KV Cache：容量与吞吐的权衡分析](docs/04-fp8-kv-cache.md) | Triton 量化写入、适用场景决策 |
| 05 | [CUDA Kernel：算子融合与访存削减](docs/05-cuda-kernels.md) | 融合 Add+RMSNorm、Inplace RoPE |
| 06 | [正确性调试：两个隐性 Bug 的定位过程](docs/06-debugging-stories.md) | 指标正常但输出错误的根因分析 |
| 07 | [性能测量：基准测试设计与实测数据](docs/07-benchmarks-5090.md) | 三层压测体系、测量方法 |
| 08 | [异步流式引擎：请求取消与增量解码](docs/08-async-engine.md) | 后台线程引擎、UTF-8 完整性 |

## 性能分析方法

```
L0 正确性    pytest tests/                断言优化后行为一致
L1 服务压测  bench.py 1                   TTFT / TPOT / P99 / TPS / QPS
L2 显存分析  bench.py 2                   组件占比 / KV 容量探针
L3 算子剖析  bench.py 3                   torch.profiler / 带宽对拍 / nsys
```

```bash
python -m benchmarks.bench 1 --model /path/to/model --scenarios mixed
python -m benchmarks.bench 2 --model /path/to/model
python -m benchmarks.bench 3 --tool bandwidth
```

分层定位逻辑与结果解读见 [docs/07](docs/07-benchmarks-5090.md)。

## 测试

```bash
pytest tests/ -m "not gpu"   # CPU 可运行：调度 / 缓存 / 解码逻辑
pytest tests/                # 全量：含 CUDA Graph parity / kernel parity
```

每个测试文件头部说明其防护的回归点。

## Roadmap

- FP8 decode 接入 FlashInfer（本地已验证 FP8 KV decode 可运行且快于 FP16）
- ZMQ 进程化引擎（参照 vLLM V1 EngineCore 架构）
- Speculative Decoding（N-gram proposer）
- 模型注册机制（LLaMA / Mistral）

## 致谢

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — 本项目的起点与 baseline
- [vLLM](https://github.com/vllm-project/vllm) — 架构参照

## License

MIT
