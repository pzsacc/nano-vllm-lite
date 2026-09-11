"""
benchmarks/level2_memory.py - L2 显存分析

回答"显存去哪了、还能塞多少"的问题。两个模式：

- memory_footprint: 组件显存占比 (权重 / CUDA Graph / KV 池 / 激活)
  教学点: gpu_memory_utilization 的水位计算 —— 引擎启动时先跑一次
  空 forward 统计"非 KV"占用, 剩余才分给 KV 池
- kv_capacity: KV 容量探针 (FP16 vs FP8)
  教学点: 容量 2× 的推导 —— 池字节相同, dtype 减半 → token 单价减半
  (与 docs/04-fp8-kv-cache.md 的"价值在容量不在速度"呼应)
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nano_vllm import LLM
from nano_vllm.config import Config


def gib(nbytes: int) -> float:
    return nbytes / 2**30


def memory_footprint(model_path: str, max_model_len: int, fp8: bool):
    """组件显存占比分析"""
    print("=" * 64)
    print("L2 显存足迹分析")
    print("=" * 64)

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    engine = LLM(model_path, max_model_len=max_model_len, enable_fp8_kvcache=fp8)
    weights = torch.cuda.memory_allocated() - before

    mr = engine.model_runner
    hf = engine.config.hf_config
    n_blocks = engine.config.num_kvcache_blocks
    block_size = engine.config.kvcache_block_size
    kv_per_token = hf.num_hidden_layers * 2 * hf.num_key_value_heads * hf.head_dim
    kv_bytes_per_token = kv_per_token * (1 if fp8 else 2)
    kv_pool = n_blocks * block_size * kv_bytes_per_token

    # CUDA Graph 显存 = 捕获后峰值 - 稳态 (含 graph pool 与 static buffer)
    peak = torch.cuda.max_memory_allocated()
    graph_overhead = max(0, peak - weights - kv_pool)

    rows = [
        ("模型权重", weights),
        ("KV Cache 池", kv_pool),
        ("CUDA Graph + 激活", graph_overhead),
    ]
    total = sum(sz for _, sz in rows)
    print(f"\n{'组件':<20}{'大小':>10}{'占比':>8}")
    print("-" * 40)
    for name, sz in rows:
        print(f"{name:<20}{gib(sz):>9.2f}G{sz / total:>7.1%}")
    print("-" * 40)
    print(f"{'合计':<20}{gib(total):>9.2f}G")

    print(f"\nKV 参数: block_size={block_size}, blocks={n_blocks}, "
          f"每 token {kv_bytes_per_token}B ({'FP8' if fp8 else 'FP16'})")
    max_concurrent = n_blocks * block_size // max_model_len
    print(f"满 {max_model_len} 上下文最大并发: {max_concurrent} seqs")

    print(f"\n水位: gpu_memory_utilization={engine.config.gpu_memory_utilization}")
    free, total_gpu = torch.cuda.mem_get_info()
    print(f"  当前已用 {gib(total):.1f}G / GPU 总量 {gib(total_gpu):.1f}G")

    if not fp8:
        print("\n[提示] 用 --fp8 再跑一次, 对比 KV 池容量变化 (预期 2×)")


def kv_capacity(model_path: str, max_model_len: int):
    """FP16 vs FP8 容量探针 (不跑请求, 只读配置; 每种 dtype 独立子进程)"""
    print("=" * 64)
    print("L2 KV 容量探针: FP16 vs FP8")
    print("=" * 64)
    results = {}
    for label, fp8 in [("FP16", False), ("FP8", True)]:
        # 引擎的 NCCL process group 不可重复初始化, 每次用子进程隔离
        code = (
            "import sys, json, torch;"
            f"sys.path.insert(0, {os.getcwd()!r});"
            "from nano_vllm import LLM;"
            f"e = LLM({model_path!r}, max_model_len={max_model_len}, "
            f"enable_fp8_kvcache={fp8});"
            "c, hf = e.config, e.config.hf_config;"
            "print(json.dumps(dict("
            "blocks=c.num_kvcache_blocks,"
            "block_size=c.kvcache_block_size,"
            "layers=hf.num_hidden_layers,"
            "kv_heads=hf.num_key_value_heads,"
            "head_dim=hf.head_dim)))"
        )
        import subprocess
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        info = json.loads(out.stdout.strip().splitlines()[-1])
        kv_per_token = info["layers"] * 2 * info["kv_heads"] * info["head_dim"]
        bytes_per_token = kv_per_token * (1 if fp8 else 2)
        pool = info["blocks"] * info["block_size"] * bytes_per_token
        seqs = info["blocks"] * info["block_size"] // max_model_len
        results[label] = dict(bytes_per_token=bytes_per_token,
                              pool_gib=gib(pool), max_seqs=seqs)
        print(f"\n[{label}] blocks={info['blocks']} "
              f"pool={gib(pool):.1f}GiB 每 token {bytes_per_token}B "
              f"满长度并发 {seqs} seqs")

    r16, r8 = results["FP16"], results["FP8"]
    print(f"\n容量倍率: {r8['max_seqs'] / r16['max_seqs']:.2f}× "
          f"(预期 2.0×, 偏差来自 block 取整)")
    print("教学点: 池字节由 utilization 预算决定, dtype 减半 → 单价减半 → 容量翻倍;")
    print("        但 decode 速度代价要结合 L3 才能看到 (见 docs/04)。")


def main():
    parser = argparse.ArgumentParser(description="L2 显存分析")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--mode", choices=["memory_footprint", "kv_capacity", "all"],
                        default="all")
    parser.add_argument("--fp8", action="store_true", help="footprint 模式用 FP8 KV")
    args = parser.parse_args()

    if args.mode in ("memory_footprint", "all"):
        memory_footprint(args.model, args.max_model_len, args.fp8)
    if args.mode in ("kv_capacity", "all"):
        kv_capacity(args.model, args.max_model_len)


if __name__ == "__main__":
    main()
