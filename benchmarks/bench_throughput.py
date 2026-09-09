"""
benchmarks/bench_throughput.py - 吞吐量基准测试

测试场景：256 个并发序列，随机输入长度 100-1024 tokens，
每个序列最大生成 1024 tokens。报告总吞吐量（tokens/second）。

用法：
    python -m benchmarks.bench_throughput --model /path/to/model [--options]
"""

import argparse
import random
import time

from nano_vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="nano-vllm 吞吐量基准测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--num-seqs", type=int, default=256, help="并发序列数")
    parser.add_argument("--max-input-len", type=int, default=1024, help="最大输入长度")
    parser.add_argument("--max-output-len", type=int, default=1024, help="最大输出长度")
    parser.add_argument("--max-model-len", type=int, default=4096, help="模型最大上下文长度")
    parser.add_argument("--enable-chunked-prefill", action="store_true", default=True)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-fp8-kvcache", action="store_true", default=True)
    parser.add_argument("--disable-fp8-kvcache", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = parser.parse_args()

    enable_chunked = not args.disable_chunked_prefill
    enable_fp8 = not args.disable_fp8_kvcache

    print(f"=== nano-vllm Throughput Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Sequences: {args.num_seqs}")
    print(f"Input length: 100-{args.max_input_len}")
    print(f"Max output: {args.max_output_len}")
    print(f"Chunked Prefill: {enable_chunked}")
    print(f"FP8 KV Cache: {enable_fp8}")
    print(f"CUDA Graph: {not args.enforce_eager}")
    print()

    # 构造随机输入
    prompts = [
        list(range(random.randint(100, args.max_input_len)))
        for _ in range(args.num_seqs)
    ]
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=args.max_output_len,
        ignore_eos=True,
    )

    # 初始化引擎
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=enable_chunked,
        enable_fp8_kvcache=enable_fp8,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    # 运行推理
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    # 统计结果
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    total_input_tokens = sum(len(p) for p in prompts)
    throughput = total_output_tokens / elapsed

    print(f"\n=== Results ===")
    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Avg latency per token: {elapsed / total_output_tokens * 1000:.2f} ms")


if __name__ == "__main__":
    main()
