<p align="center">
  <img src="docs/assets/logo.svg" width="180" alt="nano-vllm-lite">
</p>

<h1 align="center">nano-vllm-lite</h1>

<p align="center">
  <b>A lightweight LLM inference engine in ~3900 lines — every optimization measured and covered by parity tests</b>
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="docs/README.md">Documentation</a> · <a href="#benchmarks">Benchmarks</a>
</p>

## Overview

A lightweight LLM inference engine rewritten from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): ~3900 lines of Python / CUDA / Triton covering PagedAttention, Prefix Caching, Chunked Prefill, FP8 KV Cache, CUDA Graph, tensor parallelism, hand-written CUDA kernels, and an async streaming engine.

What sets this project apart: every optimization ships with **real measurement data** (RTX 5090 / 4080 Super), **root-cause analysis**, and **correctness tests** — all reproducible on your own machine.

vLLM-style API: `LLM(model).generate(prompts, SamplingParams(...))`

## Benchmarks

| Optimization | Metric | Value | Details |
|---|---|---|---|
| CUDA Graph | Decode TPOT (8 concurrent) | 36.30 → **3.90 ms** (9.3×) | [docs/01](docs/01-cuda-graph.md) |
| Prefix Caching | TTFT (90% hit rate) | **-67%**, throughput +78% | [docs/02](docs/02-prefix-caching.md) |
| Chunked Prefill | Mixed-load TPOT P99 | unbounded spikes → bounded | [docs/03](docs/03-chunked-prefill.md) |
| FP8 KV Cache | Capacity (4096 ctx) | 61 → **123 seqs** (2×) | [docs/04](docs/04-fp8-kv-cache.md) |
| Fused Add+RMSNorm | Op bandwidth | **3-4×** vs eager | [docs/05](docs/05-cuda-kernels.md) |
| End-to-end | Output throughput (256 conc., 4080S) | **2626 tok/s** | [docs/07](docs/07-benchmarks-5090.md) |

Environment: RTX 5090 32GB (torch 2.13+cu130) / RTX 4080 Super 32GB, Qwen3-0.6B (BF16).
Reproduction commands in [docs/07](docs/07-benchmarks-5090.md).

## Quick Start

```bash
pip install -e .
# Skips CUDA kernel compilation without a toolkit; falls back to torch.compile at runtime
```

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(model="/path/to/Qwen3-0.6B")
out = llm.generate(["Explain CUDA Graph"], SamplingParams(temperature=0.6, max_tokens=128))
print(out[0]["text"])
```

Streaming API server (OpenAI-compatible SSE):

```bash
python examples/streaming_server.py --model /path/to/Qwen3-0.6B --port 8000
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 64, "stream": true}'
```

## Architecture & Reading Path

```
entry   llm.py ───────────────────────────────────── ~50 lines
          │
sched ★ engine/scheduler.py ──────────────────────── 260 lines   decode-first + chunked budget
          │
exec  ★ engine/model_runner.py ───────────────────── 450 lines   CUDA Graph / inputs / sampling
          │
ops   ★ layers/attention.py ──────────────────────── 310 lines   PagedAttention / FP8 / Triton
          │        layers/linear.py · layernorm.py · rotary_embedding.py · sampler.py
          │
kernels nano_vllm/kernels/*.cu ───────────────────── ~200 lines   fused rmsnorm / inplace rope
          │
core    engine/block_manager.py ──────────────────── 270 lines ★ prefix caching
        engine/sequence.py · utils/context.py · engine/async_llm_engine.py
```

The 3-day reading plan and global design patterns: [docs/00-architecture.md](docs/00-architecture.md).

## Documentation

| Ch | Document | Content |
|---|------|------|
| 00 | [Architecture: modules and reading path](docs/00-architecture.md) | Data flow, dependencies, reading order |
| 01 | [CUDA Graph: eliminating decode kernel launch overhead](docs/01-cuda-graph.md) | Static capture/replay, memory cost |
| 02 | [Prefix Caching: content-addressed KV reuse](docs/02-prefix-caching.md) | Chained hashing, ref-count sharing |
| 03 | [Chunked Prefill: latency control under mixed load](docs/03-chunked-prefill.md) | Decode-first scheduling, budgeted chunks |
| 04 | [FP8 KV Cache: capacity vs throughput trade-off](docs/04-fp8-kv-cache.md) | Triton quantized store, when to enable |
| 05 | [CUDA Kernel: operator fusion and memory traffic](docs/05-cuda-kernels.md) | Fused Add+RMSNorm, Inplace RoPE |
| 06 | [Debugging: locating two silent bugs](docs/06-debugging-stories.md) | Root-cause analysis with green metrics |
| 07 | [Performance measurement: design and data](docs/07-benchmarks-5090.md) | 3-level benchmarking methodology |
| 08 | [Async engine: cancellation and incremental decoding](docs/08-async-engine.md) | Background-thread engine, UTF-8 handling |

## Profiling

```
L0 Correctness  pytest tests/                parity assertions
L1 Service      bench.py 1                   TTFT / TPOT / P99 / TPS / QPS
L2 Memory       bench.py 2                   footprint / KV capacity probe
L3 Kernel       bench.py 3                   torch.profiler / bandwidth / nsys
```

```bash
python -m benchmarks.bench 1 --model /path/to/model --scenarios mixed
python -m benchmarks.bench 2 --model /path/to/model
python -m benchmarks.bench 3 --tool bandwidth
```

Methodology and result interpretation: [docs/07](docs/07-benchmarks-5090.md).

## Tests

```bash
pytest tests/ -m "not gpu"   # CPU-only: scheduling / caching / decoding logic
pytest tests/                # full: CUDA Graph parity / kernel parity
```

Each test file header documents the regression it guards against.

## Roadmap

- FlashInfer-backed FP8 decode (verified locally: runs and outperforms FP16)
- ZMQ-based multi-process engine (mirroring vLLM V1 EngineCore)
- Speculative Decoding (N-gram proposer)
- Model registry (LLaMA / Mistral)

## Acknowledgements

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — the starting point and baseline of this project
- [vLLM](https://github.com/vllm-project/vllm) — architecture reference

## License

MIT
