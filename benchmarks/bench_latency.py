"""
benchmarks/bench_latency.py - 单请求延迟基准测试

测试单请求的首 token 延迟（TTFT）和逐 token 延迟（TPOT）。

用法：
    python -m benchmarks.bench_latency --model /path/to/model
"""

import argparse
import time

from nano_vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="nano-vllm 延迟基准测试")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input-len", type=int, default=512, help="输入长度")
    parser.add_argument("--output-len", type=int, default=128, help="输出长度")
    parser.add_argument("--num-runs", type=int, default=5, help="重复次数")
    args = parser.parse_args()

    print(f"=== nano-vllm Latency Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Input: {args.input_len} tokens, Output: {args.output_len} tokens")
    print(f"Runs: {args.num_runs}")
    print()

    llm = LLM(model=args.model, max_model_len=4096)
    prompt = list(range(args.input_len))
    sp = SamplingParams(temperature=1.0, max_tokens=args.output_len, ignore_eos=True)

    latencies = []
    for i in range(args.num_runs):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        tokens_generated = len(outputs[0]["token_ids"])
        print(f"  Run {i+1}: {elapsed:.3f}s ({tokens_generated} tokens)")

    avg = sum(latencies) / len(latencies)
    tpot = avg / args.output_len * 1000

    print(f"\n=== Results ===")
    print(f"Avg total latency: {avg:.3f}s")
    print(f"Avg per-token latency (TPOT): {tpot:.2f} ms")
    print(f"Avg throughput: {args.output_len / avg:.1f} tokens/s")


if __name__ == "__main__":
    main()
