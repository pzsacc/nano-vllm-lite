"""
benchmarks/bench_comprehensive.py - 全面性能压测

对标企业级推理基准测试指标，覆盖：
- 低时延场景：短输入短输出，关注 TTFT 和 TPOT
- 高吞吐场景：大 batch 长序列，关注 tokens/s 和 QPS
- 功能性场景：大上下文 + PrefixCache，验证正确性

输出指标对标 Image 中表格：
| 平均输入 | 平均输出 | PrefixCache命中率 | 请求数 | 总并发数 |
| 请求频率(req/s) | TTFT平均(ms) | TPOT卡阈(ms) | TPOT平均(ms) |
| E2E时间(ms) | 输出token/s(tps) | 单卡输出tps | QPS |

用法:
    python benchmarks/bench_comprehensive.py --model /path/to/Qwen3-0.6B
"""

import argparse
import sys
import os
import time
import random
import json
from dataclasses import dataclass, field, asdict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nano_vllm.config import Config
from nano_vllm.sampling_params import SamplingParams
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.model_runner import ModelRunner


@dataclass
class BenchmarkResult:
    """单场景压测结果"""
    scenario: str = ""
    avg_input_len: float = 0
    avg_output_len: float = 0
    prefix_cache_hit_rate: float = 0
    num_requests: int = 0
    concurrency: int = 0
    request_rate_rps: float = 0
    ttft_avg_ms: float = 0
    ttft_p99_ms: float = 0
    tpot_avg_ms: float = 0
    tpot_p99_ms: float = 0
    e2e_avg_ms: float = 0
    e2e_total_s: float = 0
    output_tps: float = 0
    total_tps: float = 0
    qps: float = 0
    gpu_mem_used_gb: float = 0


class ComprehensiveBenchmark:
    """全面性能压测器

    驱动引擎执行多种场景的推理，记录每个请求的细粒度时间戳。
    """

    def __init__(self, model_path: str, max_model_len: int = 4096,
                 enable_chunked_prefill: bool = True,
                 enable_fp8_kvcache: bool = True,
                 enforce_eager: bool = False):
        """初始化压测引擎"""
        self.config = Config(
            model=model_path,
            max_model_len=max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_fp8_kvcache=enable_fp8_kvcache,
            enforce_eager=enforce_eager,
        )
        Sequence.block_size = self.config.kvcache_block_size
        print(f"[Init] Loading model: {model_path}")
        print(f"[Init] Config: max_model_len={max_model_len}, chunked_prefill={enable_chunked_prefill}, "
              f"fp8_kv={enable_fp8_kvcache}, eager={enforce_eager}")

        self.model_runner = ModelRunner(self.config, rank=0, events=[])
        self.config.eos = -1  # 压测时禁用 EOS 停止

        print(f"[Init] KV Cache blocks: {self.config.num_kvcache_blocks}")
        print(f"[Init] GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[Init] Ready.\n")

    def run_scenario(self, scenario_name: str, num_requests: int,
                     input_lens: list[int], output_len: int,
                     prefix_ratio: float = 0.0) -> BenchmarkResult:
        """运行单个压测场景

        Args:
            scenario_name: 场景名称
            num_requests: 请求总数
            input_lens: 每个请求的输入长度列表
            output_len: 每个请求的目标输出长度
            prefix_ratio: 共享前缀比例（0=无前缀缓存）

        Returns:
            BenchmarkResult 压测结果
        """
        print(f"{'='*60}")
        print(f"[Scenario] {scenario_name}")
        print(f"  Requests: {num_requests}, Avg input: {sum(input_lens)//len(input_lens)}, "
              f"Output: {output_len}, Prefix ratio: {prefix_ratio:.0%}")
        print(f"{'='*60}")

        scheduler = Scheduler(self.config)

        # 生成请求（支持共享前缀）
        shared_prefix_len = int(input_lens[0] * prefix_ratio) if prefix_ratio > 0 else 0
        shared_prefix = list(range(shared_prefix_len)) if shared_prefix_len > 0 else []

        seqs = []
        for i in range(num_requests):
            if shared_prefix:
                suffix = [random.randint(100, 30000) for _ in range(input_lens[i % len(input_lens)] - shared_prefix_len)]
                token_ids = shared_prefix + suffix
            else:
                token_ids = [random.randint(100, 30000) for _ in range(input_lens[i % len(input_lens)])]

            sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
            seq = Sequence(token_ids, sp)
            seqs.append(seq)
            scheduler.add(seq)

        # 运行推理并记录时间
        request_metrics = defaultdict(lambda: {
            "first_token_time": None,
            "tokens_generated": 0,
            "start_time": None,
            "end_time": None,
            "token_times": [],
        })

        # 标记所有请求开始时间
        global_start = time.perf_counter()
        for seq in seqs:
            request_metrics[seq.seq_id]["start_time"] = global_start

        step_count = 0
        total_prefill_tokens = 0
        total_decode_tokens = 0

        while not scheduler.is_finished():
            step_start = time.perf_counter()
            scheduled_seqs, has_prefill = scheduler.schedule()
            token_ids = self.model_runner.call("run", scheduled_seqs, has_prefill)
            scheduler.postprocess(scheduled_seqs, token_ids)
            step_end = time.perf_counter()

            # 记录指标
            for seq in scheduled_seqs:
                sid = seq.seq_id
                metrics = request_metrics[sid]

                if has_prefill and metrics["first_token_time"] is None:
                    if seq.status == SequenceStatus.RUNNING or seq.is_finished:
                        metrics["first_token_time"] = step_end
                        total_prefill_tokens += seq.num_prompt_tokens

                if not has_prefill or seq.status in (SequenceStatus.RUNNING, SequenceStatus.FINISHED):
                    if not seq.is_prefill:
                        metrics["tokens_generated"] += 1
                        metrics["token_times"].append(step_end - step_start)
                        total_decode_tokens += 1

                if seq.is_finished:
                    metrics["end_time"] = step_end

            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.perf_counter() - global_start
                print(f"  Step {step_count}: {total_decode_tokens} tokens decoded, "
                      f"{elapsed:.1f}s elapsed")

        global_end = time.perf_counter()
        total_time = global_end - global_start

        # 计算统计指标
        ttfts = []
        tpots = []
        e2es = []
        total_output_tokens = 0

        for sid, metrics in request_metrics.items():
            if metrics["first_token_time"] and metrics["start_time"]:
                ttft = (metrics["first_token_time"] - metrics["start_time"]) * 1000
                ttfts.append(ttft)

            if metrics["token_times"]:
                avg_tpot = sum(metrics["token_times"]) / len(metrics["token_times"]) * 1000
                tpots.append(avg_tpot)

            if metrics["end_time"] and metrics["start_time"]:
                e2e = (metrics["end_time"] - metrics["start_time"]) * 1000
                e2es.append(e2e)

            total_output_tokens += metrics["tokens_generated"]

        # 构建结果
        result = BenchmarkResult(
            scenario=scenario_name,
            avg_input_len=sum(input_lens) / len(input_lens),
            avg_output_len=total_output_tokens / num_requests if num_requests > 0 else 0,
            prefix_cache_hit_rate=prefix_ratio,
            num_requests=num_requests,
            concurrency=num_requests,
            request_rate_rps=num_requests / total_time if total_time > 0 else 0,
            ttft_avg_ms=sum(ttfts) / len(ttfts) if ttfts else 0,
            ttft_p99_ms=sorted(ttfts)[int(len(ttfts) * 0.99)] if ttfts else 0,
            tpot_avg_ms=sum(tpots) / len(tpots) if tpots else 0,
            tpot_p99_ms=sorted(tpots)[int(len(tpots) * 0.99)] if tpots else 0,
            e2e_avg_ms=sum(e2es) / len(e2es) if e2es else 0,
            e2e_total_s=total_time,
            output_tps=total_output_tokens / total_time if total_time > 0 else 0,
            total_tps=(total_prefill_tokens + total_output_tokens) / total_time if total_time > 0 else 0,
            qps=num_requests / total_time if total_time > 0 else 0,
            gpu_mem_used_gb=torch.cuda.memory_allocated() / 1e9,
        )

        self._print_result(result)
        return result

    def _print_result(self, r: BenchmarkResult):
        """打印压测结果"""
        print(f"\n{'─'*60}")
        print(f"  场景: {r.scenario}")
        print(f"  请求数: {r.num_requests} | 并发数: {r.concurrency}")
        print(f"  平均输入: {r.avg_input_len:.0f} tokens | 平均输出: {r.avg_output_len:.0f} tokens")
        print(f"  PrefixCache命中率: {r.prefix_cache_hit_rate:.0%}")
        print(f"{'─'*60}")
        print(f"  TTFT 平均: {r.ttft_avg_ms:.2f} ms | P99: {r.ttft_p99_ms:.2f} ms")
        print(f"  TPOT 平均: {r.tpot_avg_ms:.2f} ms | P99: {r.tpot_p99_ms:.2f} ms")
        print(f"  E2E  平均: {r.e2e_avg_ms:.2f} ms | 总时间: {r.e2e_total_s:.2f} s")
        print(f"{'─'*60}")
        print(f"  输出 TPS:  {r.output_tps:.2f} tokens/s")
        print(f"  总 TPS:    {r.total_tps:.2f} tokens/s (含 prefill)")
        print(f"  QPS:       {r.qps:.4f} req/s")
        print(f"  GPU 显存:  {r.gpu_mem_used_gb:.2f} GB")
        print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="nano-vllm 全面性能压测")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enable-chunked-prefill", action="store_true", default=True)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-fp8-kvcache", action="store_true", default=True)
    parser.add_argument("--disable-fp8-kvcache", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                        choices=["all", "low_latency", "high_throughput", "prefix_cache", "long_context"])
    args = parser.parse_args()

    bench = ComprehensiveBenchmark(
        model_path=args.model,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        enable_fp8_kvcache=not args.disable_fp8_kvcache,
        enforce_eager=args.enforce_eager,
    )

    results = []
    scenarios = args.scenarios
    if "all" in scenarios:
        scenarios = ["low_latency", "high_throughput", "prefix_cache", "long_context"]

    # ==================== 低时延场景 ====================
    if "low_latency" in scenarios:
        # 场景1: 少量短请求
        results.append(bench.run_scenario(
            scenario_name="低时延-短输入短输出",
            num_requests=8,
            input_lens=[128] * 8,
            output_len=32,
        ))

        # 场景2: 中等输入
        results.append(bench.run_scenario(
            scenario_name="低时延-中等输入",
            num_requests=8,
            input_lens=[512] * 8,
            output_len=64,
        ))

        # 场景3: 长输入短输出 (类似单轮对话)
        results.append(bench.run_scenario(
            scenario_name="低时延-长输入短输出",
            num_requests=4,
            input_lens=[1024] * 4,
            output_len=32,
        ))

    # ==================== 高吞吐场景 ====================
    if "high_throughput" in scenarios:
        # 场景4: 大batch
        results.append(bench.run_scenario(
            scenario_name="高吞吐-256并发",
            num_requests=256,
            input_lens=[random.randint(100, 512) for _ in range(256)],
            output_len=128,
        ))

        # 场景5: 超大batch + 长输出
        results.append(bench.run_scenario(
            scenario_name="高吞吐-256并发长输出",
            num_requests=256,
            input_lens=[random.randint(100, 1024) for _ in range(256)],
            output_len=512,
        ))

    # ==================== PrefixCache 场景 ====================
    if "prefix_cache" in scenarios:
        # 场景6: 高前缀命中率
        results.append(bench.run_scenario(
            scenario_name="PrefixCache-90%命中",
            num_requests=64,
            input_lens=[512] * 64,
            output_len=64,
            prefix_ratio=0.9,
        ))

        # 场景7: 50%前缀命中
        results.append(bench.run_scenario(
            scenario_name="PrefixCache-50%命中",
            num_requests=64,
            input_lens=[512] * 64,
            output_len=64,
            prefix_ratio=0.5,
        ))

    # ==================== 长上下文场景 ====================
    if "long_context" in scenarios:
        # 场景8: 2K 上下文
        results.append(bench.run_scenario(
            scenario_name="长上下文-2K输入",
            num_requests=16,
            input_lens=[2048] * 16,
            output_len=128,
        ))

    # ==================== 汇总报告 ====================
    print("\n" + "=" * 100)
    print(f"{'全面压测汇总报告':^100}")
    print("=" * 100)
    header = f"{'场景':<24}{'请求数':>6}{'输入':>6}{'输出':>6}{'TTFT(ms)':>10}{'TPOT(ms)':>10}{'E2E(ms)':>10}{'输出TPS':>10}{'QPS':>8}"
    print(header)
    print("-" * 100)
    for r in results:
        row = (f"{r.scenario:<24}{r.num_requests:>6}{r.avg_input_len:>6.0f}{r.avg_output_len:>6.0f}"
               f"{r.ttft_avg_ms:>10.2f}{r.tpot_avg_ms:>10.2f}{r.e2e_avg_ms:>10.2f}"
               f"{r.output_tps:>10.2f}{r.qps:>8.4f}")
        print(row)
    print("=" * 100)

    # 保存 JSON 结果
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
