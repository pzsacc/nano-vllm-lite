<p align="center">
  <img src="docs/assets/logo.svg" width="180" alt="nano-vllm-lite">
</p>

<h1 align="center">nano-vllm-lite</h1>

<p align="center">
  <b>A from-scratch LLM inference engine — every optimization backed by real benchmarks, war stories, and parity tests.</b><br>
  从零手写的 LLM 推理引擎：每个优化都有实测数据、踩坑复盘与正确性测试。
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="docs/README.md">Optimization Series</a> · <a href="#-optimization-series">Series</a>
</p>

## What is this

A ~3900-line LLM inference engine (Python + CUDA + Triton) implementing an
enterprise-grade optimization stack. This is not a stripped-down vLLM clone —
it is a **teaching project where every optimization answers "why is it faster,
how much faster, and how do we know it's still correct"**:

- **Measured**: every optimization ships with real benchmarks (RTX 5090 / 4080 Super) and reproducible commands
- **Deep**: [docs/06](docs/06-debugging-stories.md) walks through two real bugs where *all metrics looked perfect but the output was garbage* — and how they were hunted down
- **Guarded**: 23 tests; every optimization has a parity assertion, so performance gains never silently break correctness

vLLM-style API: `LLM(model).generate(prompts, SamplingParams(...))`

## Headline Numbers

| Scenario | Metric | Value |
|---|---|---|
| Low latency (8 concurrent) | TPOT | **3.9ms** (eager: 36.3ms, **9.3×**) |
| High throughput (256 concurrent) | Output throughput | **2626 tok/s** (single 4080S) |
| Prefix cache @ 90% hit | TTFT | **-67%**, throughput +78% |
| FP8 KV cache | Capacity | **2×** (61→123 concurrent sequences) |

> Environment and repro commands: [docs/07](docs/07-benchmarks-5090.md)

## Code Map (readable in 3 days)

```
entry   llm.py ───────────────────────────────────── ~50 lines
          │
sched ★ engine/scheduler.py ──────────────────────── 260 lines   decode-first + chunked budget
          │
exec  ★ engine/model_runner.py ───────────────────── 450 lines   CUDA Graph / inputs / sampling
          │
ops   ★ layers/attention.py ──────────────────────── 310 lines   PagedAttn / FP8 / Triton
          │        layers/linear.py · layernorm.py · rotary_embedding.py · sampler.py
          │
kernels nano_vllm/kernels/*.cu ───────────────────── ~200 lines   fused rmsnorm / inplace rope
          │
core    engine/block_manager.py ──────────────────── 270 lines ★ prefix caching
        engine/sequence.py · utils/context.py · engine/async_llm_engine.py
```

3-day reading plan and global design patterns → [docs/00-architecture.md](docs/00-architecture.md)

## Quick Start

```bash
pip install -e .            # compiles CUDA kernels (falls back to torch.compile without nvcc)
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

## ⚡ Optimization Series

| # | Optimization | Gain | Doc |
|---|------|------|------|
| 01 | CUDA Graph | Low-concurrency TPOT 36→3.9ms (9.3×) | [01-cuda-graph](docs/01-cuda-graph.md) |
| 02 | Prefix Caching | TTFT -67% @ 90% hit rate | [02-prefix-caching](docs/02-prefix-caching.md) |
| 03 | Chunked Prefill | Bounded TPOT P99 under mixed load | [03-chunked-prefill](docs/03-chunked-prefill.md) |
| 04 | FP8 KV Cache | 2× capacity (with a "when it's worth it" framework) | [04-fp8-kv-cache](docs/04-fp8-kv-cache.md) |
| 05 | CUDA Kernels | Add+RMSNorm bandwidth 3-4× | [05-cuda-kernels](docs/05-cuda-kernels.md) |
| 06 | **Debug War Stories** | Methodology for "metrics fine, output garbage" | [06-debugging-stories](docs/06-debugging-stories.md) ⭐ |
| 07 | Benchmark Methodology | 3-level measurement (service/memory/kernel) | [07-benchmarks-5090](docs/07-benchmarks-5090.md) |
| 08 | Async Engine | 3 engineering details of token streaming | [08-async-engine](docs/08-async-engine.md) |

## Profiling Path (L0→L3)

Optimization is not guesswork — it's a reproducible verification chain:

```
L0 Correct?   pytest tests/                parity assertions (23; GPU ones tagged -m gpu)
L1 How fast?  bench.py 1   service-level   TTFT/TPOT/TPS/QPS
L2 Memory     bench.py 2   footprint/KV    capacity probe (FP8 vs FP16)
L3 Kernel     bench.py 3   profiler/bw     pin down the exact kernel
```

```bash
python -m benchmarks.bench 1 --model /path/to/model --scenarios mixed   # exposes P99
python -m benchmarks.bench 2 --model /path/to/model                     # KV capacity probe
python -m benchmarks.bench 3 --tool bandwidth                           # kernel shootout
```

## Tests

```bash
pytest tests/ -m "not gpu"   # CPU-only (scheduling / caching / detokenization)
pytest tests/                # full (CUDA Graph parity / kernel parity)
```

Every test file header names **the bug it guards against** — tests as documentation.

## Roadmap

- [ ] FlashInfer-backed FP8 decode (verified locally: FP8 KV decode runs and beats FP16)
- [ ] ZMQ-based multi-process engine (mirroring vLLM V1 EngineCore)
- [ ] Speculative Decoding (N-gram proposer)
- [ ] Model registry (LLaMA / Mistral)

## Acknowledgements

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — the starting point and baseline of this project
- [vLLM](https://github.com/vllm-project/vllm) — architecture reference

## License

MIT
