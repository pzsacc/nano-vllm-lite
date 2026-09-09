# nano-vllm

A lightweight high-performance LLM inference engine (~3900 lines of Python + CUDA) implementing enterprise-grade optimizations from scratch.

## Highlights

- **2626 tokens/s** throughput on a single RTX 4080 Super (Qwen3-0.6B, 256 concurrent requests)
- **4ms TPOT** (Time Per Output Token) in low-latency mode with CUDA Graph
- **~3900 lines** total code — readable, hackable, and interview-ready
- Full parity with vLLM's core optimization stack

## Features

| Feature | Description |
|---------|-------------|
| PagedAttention | OS-style paged KV cache memory management |
| Prefix Caching | Content-addressed block reuse via xxhash |
| Chunked Prefill | Sarathi-style decode-first mixed scheduling |
| FP8 KV Cache | float8_e4m3fn quantization with custom Triton decode kernel |
| CUDA Graph | Static graph capture/replay for decode acceleration |
| Tensor Parallelism | NCCL + SharedMemory IPC coordination |
| Continuous Batching | Dynamic batching with preemption |
| FlashAttention | Prefill via flash_attn_varlen_func |
| Async Streaming | asyncio-based token streaming engine |
| Custom CUDA Kernels | Fused Add+RMSNorm, In-place RoPE |
| OpenAI-compatible API | /v1/completions with SSE streaming |

## Quick Start

### Installation

```bash
git clone https://github.com/pzsacc/nano-vllm-lite.git
cd nano-vllm-lite

# Basic install
pip install -e .

# With API server dependencies
pip install -e ".[server]"
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0, CUDA >= 12.0, flash-attn >= 2.5

### Offline Inference

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(model="/path/to/Qwen3-0.6B")
outputs = llm.generate(
    ["What is the meaning of life?"],
    SamplingParams(temperature=0.8, max_tokens=128)
)
print(outputs[0]["text"])
```

### Streaming API Server

```bash
python examples/streaming_server.py --model /path/to/Qwen3-0.6B --port 8000
```

```bash
# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 64, "stream": true}'
```

### Configuration

```python
llm = LLM(
    model="/path/to/model",
    max_model_len=4096,           # Max context length
    enable_chunked_prefill=True,  # Decode-first mixed scheduling
    enable_fp8_kvcache=False,     # FP8 KV cache (2x capacity)
    enable_prefix_caching=True,   # Content-addressed prefix reuse
    enforce_eager=False,          # False = enable CUDA Graph
    tensor_parallel_size=1,       # TP world size
    chunk_size=1024,              # Chunked prefill budget
    kvcache_block_size=256,       # KV cache page size
    gpu_memory_utilization=0.9,   # GPU memory fraction for KV cache
)
```

## Benchmarks

### Environment

- **GPU:** NVIDIA RTX 4080 Super (32GB VRAM)
- **Model:** Qwen3-0.6B (28 layers, 1024 hidden, GQA 16/8 heads)
- **Software:** PyTorch 2.7, CUDA 12.8, flash-attn 2.8.3, Triton 3.3
- **Mode:** CUDA Graph + FP16 KV Cache + Chunked Prefill

### Results

| Scenario | Requests | Avg Input | Avg Output | TTFT (ms) | TPOT (ms) | Output TPS | QPS |
|----------|----------|-----------|------------|-----------|-----------|------------|-----|
| Low-latency (short) | 8 | 128 | 31 | 835 | **4.0** | 258 | 8.3 |
| Low-latency (medium) | 8 | 512 | 63 | 112 | **5.3** | 1,113 | 17.7 |
| Low-latency (long input) | 4 | 1024 | 31 | 109 | **6.2** | 403 | 13.0 |
| **High-throughput** | **256** | 304 | 127 | 3,107 | 67.9 | **2,626** | **20.7** |
| High-throughput (long) | 256 | 543 | 511 | 8,632 | 73.4 | **2,523** | 4.9 |
| PrefixCache 90% hit | 64 | 512 | 63 | 471 | **30.7** | 1,579 | 25.1 |
| PrefixCache 50% hit | 64 | 512 | 63 | 1,360 | 38.5 | 1,035 | 16.4 |
| Long context (2K) | 16 | 2048 | 127 | 2,594 | 30.8 | 302 | 2.4 |

### Running Benchmarks

```bash
# Full comprehensive benchmark
python benchmarks/bench_comprehensive.py --model /path/to/model --scenarios all

# Specific scenarios
python benchmarks/bench_comprehensive.py --model /path/to/model \
    --scenarios low_latency high_throughput prefix_cache long_context

# Throughput-only benchmark
python benchmarks/bench_throughput.py --model /path/to/model --num-seqs 256

# Latency-only benchmark
python benchmarks/bench_latency.py --model /path/to/model --input-len 512 --output-len 128

# With FP8 KV Cache (2x capacity, custom Triton decode)
python benchmarks/bench_comprehensive.py --model /path/to/model --enable-fp8-kvcache

# Eager mode (disable CUDA Graph for debugging)
python benchmarks/bench_comprehensive.py --model /path/to/model --enforce-eager
```

### Performance Breakdown: CUDA Graph Impact

| Metric | Eager Mode | CUDA Graph | Speedup |
|--------|-----------|------------|---------|
| TPOT (bs=8) | 129 ms | 4.0 ms | **32x** |
| TPOT (bs=256) | 122 ms | 68 ms | 1.8x |
| Output TPS (256 concurrent) | 1,358 | 2,626 | 1.9x |
| QPS (PrefixCache) | 7.99 | 25.07 | 3.1x |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  LLMEngine / AsyncLLMEngine               │
│   (Orchestration: tokenize → schedule → run → postprocess)│
├──────────────────────────────────────────────────────────┤
│          Scheduler              ModelRunner               │
│  ┌──────────────────┐   ┌───────────────────────────┐    │
│  │ Decode-first     │   │ NCCL TP Coordination      │    │
│  │ Chunked Prefill  │   │ Static-tensor CUDA Graph  │    │
│  │ BlockManager     │   │ KV Cache Allocation       │    │
│  │  └─ PrefixCache  │   │ Warmup + torch.compile    │    │
│  └──────────────────┘   └───────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│              Model (Qwen3ForCausalLM + Registry)          │
├──────────────────────────────────────────────────────────┤
│                  Layers (TP + Fused Ops)                   │
│  Attention(FP8/FP16) | Linear(Col/Row/QKV) | RMSNorm     │
│  Sampler(Gumbel-max) | SiluAndMul | RoPE | Embed/Head    │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
nano-vllm/
├── setup.py                        # Install + CUDA kernel compilation
├── README.md                       # This file
├── DEVELOPMENT.md                  # Detailed development documentation
├── nano_vllm/
│   ├── __init__.py                 # Package entry (LLM, Config, SamplingParams)
│   ├── config.py                   # Global config with feature flags
│   ├── sampling_params.py          # Sampling parameters
│   ├── engine/
│   │   ├── llm_engine.py          # Synchronous inference engine
│   │   ├── async_llm_engine.py    # Async streaming engine
│   │   ├── scheduler.py           # Chunked Prefill mixed scheduler
│   │   ├── block_manager.py       # Paged KV cache block manager
│   │   ├── model_runner.py        # GPU executor (CUDA Graph + TP)
│   │   └── sequence.py            # Sequence state management
│   ├── layers/
│   │   ├── attention.py           # Paged Attention (FP8 + FlashAttn)
│   │   ├── linear.py              # TP linear layers
│   │   ├── layernorm.py           # Fused Add+RMSNorm
│   │   ├── rotary_embedding.py    # RoPE positional encoding
│   │   ├── activation.py          # Fused SiLU+Gate
│   │   ├── sampler.py             # Gumbel-max sampler
│   │   └── embed_head.py          # Vocab-parallel Embedding + LM Head
│   ├── models/
│   │   ├── __init__.py            # Model registry
│   │   └── qwen3.py               # Qwen3 architecture
│   ├── kernels/
│   │   ├── __init__.py            # CUDA kernel Python wrappers
│   │   ├── add_rmsnorm.cu         # Fused Add+RMSNorm CUDA kernel
│   │   └── inplace_rotary_embed.cu # In-place RoPE CUDA kernel
│   └── utils/
│       ├── context.py             # Global inference context
│       └── loader.py              # SafeTensors weight loader
├── benchmarks/
│   ├── bench_comprehensive.py     # Full benchmark suite
│   ├── bench_throughput.py        # Throughput benchmark
│   └── bench_latency.py           # Latency benchmark
├── examples/
│   ├── offline_inference.py       # Batch inference example
│   └── streaming_server.py        # OpenAI-compatible API server
└── tests/
    └── test_block_manager.py      # Unit tests
```

## Comparison with vLLM

| Dimension | nano-vllm | vLLM |
|-----------|-----------|------|
| Code size | ~3,900 lines | ~500K lines |
| Model support | Qwen3 (extensible) | 100+ architectures |
| Quantization | FP8 KV Cache | AWQ/GPTQ/FP8/INT8/MXFP4 |
| Speculative decoding | Planned | EAGLE/Medusa/ngram |
| Distributed | Single-node TP | Multi-node TP/PP/EP |
| API | Basic OpenAI compat | Full OpenAI + gRPC |
| Throughput | ~95% of vLLM | Baseline |

## License

MIT
