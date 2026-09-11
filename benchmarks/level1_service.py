"""
benchmarks/level1_service.py - L1 服务层压测（黑盒）

回答"多快"的问题：把引擎当黑盒，测量端到端服务指标。

场景与教学动机：
- low_latency:      少量短请求, 关注 TTFT/TPOT (在线服务核心指标)
- high_throughput:  256 并发, 关注 Output TPS (离线批处理核心指标)
- prefix_cache:     90%/50% 前缀命中, 验证缓存收益 (多轮对话/system prompt)
- long_context:     2K 输入, 关注长上下文下的退化曲线
- mixed:            长短请求混合, 唯一能暴露 TPOT P99 尾延迟的场景

计时方法论: 直接驱动 Scheduler + ModelRunner, time.perf_counter 记录
每个请求的首 token 与逐 token 时间戳（比外层 HTTP 压测少一层噪声,
与 docs/07-benchmarks.md 的方法论章节一一对应）。
"""
import os
import sys
import time
import random
import json
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nano_vllm.config import Config
from nano_vllm.sampling_params import SamplingParams
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.model_runner import ModelRunner


@dataclass
class ScenarioResult:
    """单场景压测结果"""
    scenario: str = ""
    num_requests: int = 0
    avg_input_len: float = 0
    avg_output_len: float = 0
    prefix_hit_ratio: float = 0
    ttft_avg_ms: float = 0
    ttft_p99_ms: float = 0
    tpot_avg_ms: float = 0
    tpot_p99_ms: float = 0
    e2e_avg_ms: float = 0
    output_tps: float = 0
    qps: float = 0
    peak_mem_gb: float = 0


def percentile(arr, p):
    if not arr:
        return 0.0
    arr = sorted(arr)
    idx = min(int(len(arr) * p / 100), len(arr) - 1)
    return arr[idx]


class ServiceBenchmark:
    """L1 服务层压测器：直驱引擎循环, 逐请求记录时间戳"""

    def __init__(self, model_path: str, max_model_len: int = 4096,
                 enable_chunked_prefill: bool = True,
                 enable_fp8_kvcache: bool = False,
                 enforce_eager: bool = False):
        self.config = Config(
            model=model_path,
            max_model_len=max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_fp8_kvcache=enable_fp8_kvcache,
            enforce_eager=enforce_eager,
        )
        Sequence.block_size = self.config.kvcache_block_size
        print(f"[Init] model={model_path} len={max_model_len} "
              f"chunked={enable_chunked_prefill} fp8={enable_fp8_kvcache} eager={enforce_eager}")
        self.model_runner = ModelRunner(self.config, rank=0, events=[])
        self.config.eos = -1  # 压测禁用 EOS, 输出长度可控

    def run(self, name: str, num_requests: int, input_lens: list[int],
            output_len: int, prefix_ratio: float = 0.0,
            request_rate: float = 0.0) -> ScenarioResult:
        """运行单个场景

        Args:
            request_rate: >0 时按泊松到达提交请求（模拟在线流量）, 0=全部同时提交
        """
        print(f"\n[Scenario] {name}: {num_requests} reqs, "
              f"in~{sum(input_lens) // len(input_lens)}, out={output_len}, "
              f"prefix={prefix_ratio:.0%}")

        scheduler = Scheduler(self.config)
        shared_len = int(input_lens[0] * prefix_ratio) if prefix_ratio > 0 else 0
        shared_prefix = list(range(shared_len)) if shared_len else []

        # 生成请求
        seqs = []
        for i in range(num_requests):
            body = [random.randint(100, 30000)
                    for _ in range(input_lens[i % len(input_lens)] - shared_len)]
            sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
            seq = Sequence(shared_prefix + body if shared_prefix else body, sp)
            seqs.append(seq)

        # 提交（支持泊松到达模拟在线流量）
        metrics = defaultdict(lambda: {"submit": None, "first": None,
                                       "end": None, "n_tok": 0, "tok_times": []})
        t_origin = time.perf_counter()
        for idx, seq in enumerate(seqs):
            if request_rate > 0:
                # 指数分布间隔 = 泊松过程
                delay = random.expovariate(request_rate) * idx
            else:
                delay = 0.0
            while time.perf_counter() - t_origin < delay:
                time.sleep(0.0005)
            metrics[seq.seq_id]["submit"] = time.perf_counter()
            scheduler.add(seq)

        # 引擎循环 + 记时
        n_steps = 0
        while not scheduler.is_finished():
            step_start = time.perf_counter()
            scheduled, has_prefill = scheduler.schedule()
            token_ids = self.model_runner.call("run", scheduled, has_prefill)
            scheduler.postprocess(scheduled, token_ids)
            step_end = time.perf_counter()

            for seq in scheduled:
                m = metrics[seq.seq_id]
                if seq.completion_token_ids:
                    if m["first"] is None:
                        m["first"] = step_end
                    m["n_tok"] += 1
                    m["tok_times"].append(step_end - step_start)
                if seq.is_finished:
                    m["end"] = step_end
            n_steps += 1

        # 汇总
        ttfts, tpots, e2es = [], [], []
        total_out = 0
        for seq in seqs:
            m = metrics[seq.seq_id]
            if m["first"]:
                ttfts.append((m["first"] - m["submit"]) * 1000)
            if m["tok_times"]:
                tpots.append(sum(m["tok_times"]) / len(m["tok_times"]) * 1000)
            if m["end"]:
                e2es.append((m["end"] - m["submit"]) * 1000)
            total_out += m["n_tok"]

        span = max(m["end"] for m in metrics.values() if m["end"]) - t_origin
        result = ScenarioResult(
            scenario=name,
            num_requests=num_requests,
            avg_input_len=sum(input_lens) / len(input_lens),
            avg_output_len=total_out / num_requests,
            prefix_hit_ratio=prefix_ratio,
            ttft_avg_ms=sum(ttfts) / len(ttfts) if ttfts else 0,
            ttft_p99_ms=percentile(ttfts, 99),
            tpot_avg_ms=sum(tpots) / len(tpots) if tpots else 0,
            tpot_p99_ms=percentile(tpots, 99),
            e2e_avg_ms=sum(e2es) / len(e2es) if e2es else 0,
            output_tps=total_out / span,
            qps=num_requests / span,
            peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
        )
        torch.cuda.reset_peak_memory_stats()
        return result


def print_table(results: list[ScenarioResult]):
    header = (f"{'场景':<22}{'请求':>5}{'输入':>6}{'输出':>6}"
              f"{'TTFT avg':>10}{'TTFT P99':>10}{'TPOT avg':>10}{'TPOT P99':>10}"
              f"{'TPS':>8}{'QPS':>7}")
    print("\n" + "=" * len(header))
    print("L1 服务层压测汇总")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.scenario:<22}{r.num_requests:>5}{r.avg_input_len:>6.0f}"
              f"{r.avg_output_len:>6.0f}{r.ttft_avg_ms:>10.1f}{r.ttft_p99_ms:>10.1f}"
              f"{r.tpot_avg_ms:>10.2f}{r.tpot_p99_ms:>10.2f}"
              f"{r.output_tps:>8.0f}{r.qps:>7.1f}")
    print("=" * len(header))


SCENARIOS = {
    "low_latency": [
        dict(name="低时延-短输入", num_requests=8, input_lens=[128] * 8, output_len=32),
        dict(name="低时延-中输入", num_requests=8, input_lens=[512] * 8, output_len=64),
        dict(name="低时延-长输入", num_requests=4, input_lens=[1024] * 4, output_len=32),
    ],
    "high_throughput": [
        dict(name="高吞吐-256并发", num_requests=256,
             input_lens=[random.randint(100, 512) for _ in range(256)], output_len=128),
        dict(name="高吞吐-256并发长输出", num_requests=256,
             input_lens=[random.randint(100, 1024) for _ in range(256)], output_len=512),
    ],
    "prefix_cache": [
        dict(name="PrefixCache-90%", num_requests=64, input_lens=[512] * 64,
             output_len=64, prefix_ratio=0.9),
        dict(name="PrefixCache-50%", num_requests=64, input_lens=[512] * 64,
             output_len=64, prefix_ratio=0.5),
    ],
    "long_context": [
        dict(name="长上下文-2K", num_requests=16, input_lens=[2048] * 16, output_len=128),
    ],
    "mixed": [
        # 教学点: 长短混合是唯一暴露 TPOT P99 的场景 (长 prefill 阻塞 decode)
        dict(name="混合-64短+4长", num_requests=68,
             input_lens=[128] * 64 + [2048] * 4, output_len=128),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="L1 服务层压测")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                        choices=["all", "low_latency", "high_throughput",
                                 "prefix_cache", "long_context", "mixed"])
    parser.add_argument("--enable-fp8-kvcache", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=str, default="l1_results.json")
    args = parser.parse_args()

    bench = ServiceBenchmark(
        args.model, args.max_model_len,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        enable_fp8_kvcache=args.enable_fp8_kvcache,
        enforce_eager=args.enforce_eager,
    )

    names = list(SCENARIOS) if "all" in args.scenarios else args.scenarios
    results = []
    for name in names:
        for sc in SCENARIOS[name]:
            results.append(bench.run(**sc))
    print_table(results)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
