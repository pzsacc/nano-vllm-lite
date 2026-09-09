# 5090 实测结果（2026-09-09）

环境：RTX 5090 32GB（SM120）· torch 2.13+cu130 · triton 3.7 · flash-attn 2.8.3（源码重编，CCCL 补丁）
模型：Qwen3-0.6B（28 层，GQA 16Q/8KV，head_dim=128，hidden=1024）

## 1. Fused Add+RMSNorm（bf16, hidden=4096，`benchmarks/bench_fused_bandwidth.py`）

| tokens | eager GB/s | torch.compile(max-autotune) GB/s | CUDA kernel GB/s | CUDA/eager |
|---:|---:|---:|---:|---:|
| 1024 | 408 | 1046 | 1458 | 3.57× |
| 4096 | 361 | 1837 | 1198 | 3.32× |
| 8192 | 232 | 1136 | 960 | 4.13× |
| 16384 | 223 | 1138 | 979 | 4.39× |
| 32768 | 220 | 1140 | 838 | 4.36× |

- vs eager：**3.3-4.4×**；vs torch.compile max-autotune：0.84-1.39×（大致持平）
- 正确性：vs eager 参考实现 max|diff| = 6.25e-2（bf16 + fast_math）

## 2. FP8 KV Cache 容量探针（`benchmarks/bench_kv_capacity.py`，max_model_len=4096）

| | KV 池 | 每 token | 满长度最大并发 |
|---|---|---:|---:|
| FP16 | 27.0 GiB | 114688 B | **61 seqs** |
| FP8 | 27.1 GiB | 57344 B | **123 seqs** |

池字节相同（比额分配），dtype 减半单价 → 容量 **2.0×**。

## 3. 端到端吞吐 A/B（`benchmarks/bench_throughput.py`，64 seqs，in≤512 / out 256）

| 配置 | 吞吐 | 相对 FP16 |
|---|---:|---:|
| FP16 KV（chunked on） | 9119-9370 tok/s | 1.0× |
| FP16 KV（chunked off） | 9098 tok/s | ~1.0×（±2%） |
| FP8 KV | 1372 tok/s | **0.15×** |

结论复现：0.6B 上 KV 容量不是瓶颈，自研 Triton FP8 decode kernel（朴素实现）速度损失远超容量收益
——**"FP8 KV 价值在容量不在速度"**，与原 V100 实测（0.18×）一致；容量收益（2×）只在 KV 成为
瓶颈时兑现（大模型/长上下文），已在 SGLang 27B 项目中验证。

## 4. 复现说明

- `setup.py`：CUDA_HOME 改为本机 cu13 工具链 + nvcc 加 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`
- flash-attn 需源码重编（同上 flag，见 headers 补丁），`--no-deps` 防止 pip 升级 torch
- bench_throughput 新增 `--gpu-mem-util`（0.9 水位下 FP8 路径 OOM，两侧统一 0.85 公平对比）
