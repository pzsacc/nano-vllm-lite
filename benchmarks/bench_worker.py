"""
bench_worker.py - 完整指标压测 worker

每个配置独立子进程运行，直接驱动 Scheduler + ModelRunner 获取细粒度指标:
- TTFT (首 token 延迟): 每请求从提交到首 token 产出
- TPOT (per-token 延迟): 每请求 decode 阶段平均/P50/P99
- E2E (端到端延迟): 每请求从提交到完成
- 输出 TPS: 总输出 token / 总时间
- E2E TPS: 总处理 token (input+output) / 总时间
- QPS: 完成请求数 / 总时间
- 请求频率: 提交请求数 / 总时间
- GPU 显存使用 / KV blocks 数量

输出 JSON 到 stdout。
"""
import sys
import os
import time
import json
import argparse
import random
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/root/autodl-tmp/projects/nano-vllm-lite")

random.seed(42)
np.random.seed(42)


def percentile(arr, p):
    if not arr:
        return 0
    s = sorted(arr)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


# ============================================================
# 场景定义
# ============================================================

def build_scenarios():
    scenarios = []

    # --- 低时延 ---
    for ctx_len in [1024, 4096]:
        for conc in [4, 8, 16]:
            scenarios.append({
                "name": f"低时延-ctx{ctx_len}-c{conc}",
                "category": "低时延",
                "num_seqs": conc,
                "input_len": ctx_len // 4,
                "output_len": 64,
                "prefix_rate": 0.0,
            })

    # --- 高吞吐 ---
    for conc in [64, 128, 256]:
        scenarios.append({
            "name": f"高吞吐-c{conc}",
            "category": "高吞吐",
            "num_seqs": conc,
            "input_len": -1,
            "output_len": 128,
            "prefix_rate": 0.0,
        })

    # --- PrefixCache ---
    for hit_rate in [0.5, 0.9]:
        scenarios.append({
            "name": f"PrefixCache-{int(hit_rate*100)}%",
            "category": "PrefixCache",
            "num_seqs": 64,
            "input_len": 512,
            "output_len": 64,
            "prefix_rate": hit_rate,
        })

    # --- 长上下文 ---
    for inp in [1024, 2048]:
        scenarios.append({
            "name": f"长上下文-{inp}",
            "category": "长上下文",
            "num_seqs": 16,
            "input_len": inp,
            "output_len": 128,
            "prefix_rate": 0.0,
        })

    # --- 混合负载 ---
    scenarios.append({
        "name": "混合负载-64短4长",
        "category": "混合负载",
        "num_seqs": -1,
        "input_len": -1,
        "output_len": 128,
        "prefix_rate": 0.0,
    })

    return scenarios


def gen_prompts(scenario):
    if scenario["num_seqs"] == -1:
        return [list(range(128)) for _ in range(64)] + [list(range(2048)) for _ in range(4)]

    num = scenario["num_seqs"]
    inp_len = scenario["input_len"]
    prefix_rate = scenario["prefix_rate"]

    if inp_len == -1:
        return [[random.randint(100, 30000) for _ in range(random.randint(100, 1024))]
                for _ in range(num)]

    if prefix_rate > 0:
        shared_len = int(inp_len * prefix_rate)
        shared = list(range(shared_len))
        return [shared + [random.randint(100, 30000) for _ in range(inp_len - shared_len)]
                for _ in range(num)]

    return [[random.randint(100, 30000) for _ in range(inp_len)] for _ in range(num)]


# ============================================================
# 核心: step-level 压测
# ============================================================

def run_benchmark_lite(model_path, scenarios, enable_chunked, enable_fp8,
                       enforce_eager, enable_prefix):
    import torch
    from nano_vllm.config import Config
    from nano_vllm.sampling_params import SamplingParams
    from nano_vllm.engine.sequence import Sequence, SequenceStatus
    from nano_vllm.engine.scheduler import Scheduler
    from nano_vllm.engine.model_runner import ModelRunner

    config = Config(
        model=model_path, max_model_len=4096,
        enable_chunked_prefill=enable_chunked,
        enable_fp8_kvcache=enable_fp8,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix,
    )
    Sequence.block_size = config.kvcache_block_size
    model_runner = ModelRunner(config, rank=0, events=[])
    config.eos = -1

    num_kv_blocks = config.num_kvcache_blocks
    gpu_mem = torch.cuda.memory_allocated() / 1e9

    all_results = []
    for sc in scenarios:
        random.seed(42)
        try:
            r = _run_scenario(config, model_runner, sc)
            r["gpu_mem_gb"] = round(gpu_mem, 2)
            r["num_kv_blocks"] = num_kv_blocks
            all_results.append(r)
        except Exception as e:
            all_results.append({"name": sc["name"], "category": sc["category"],
                                "error": str(e)[:200]})
    return all_results


def _run_scenario(config, model_runner, scenario):
    """使用 torch.cuda.Event 进行 GPU 精确计时"""
    import torch
    from nano_vllm.sampling_params import SamplingParams
    from nano_vllm.engine.sequence import Sequence
    from nano_vllm.engine.scheduler import Scheduler

    prompts = gen_prompts(scenario)
    num_seqs = len(prompts)
    output_len = scenario["output_len"]

    scheduler = Scheduler(config)
    seqs = []
    for tids in prompts:
        sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
        seq = Sequence(tids, sp)
        seqs.append(seq)
        scheduler.add(seq)

    # per-request tracking
    req = {}
    for seq in seqs:
        req[seq.seq_id] = {"input_len": len(seq.token_ids), "ttft_ms": None,
                           "step_times_ms": [], "end_ms": None, "output_n": 0}

    # Warmup: 确保 CUDA context / torch.compile 缓存已就绪
    # (已在 run_benchmark_lite 的 model init 中完成)

    # ---- 使用 CUDA Event 精确计时 ----
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    t0_cpu = time.perf_counter()  # CPU 参考时钟 (用于 wall-clock)

    steps = 0
    step_events = []  # [(step_start_event, step_end_event), ...]

    while not scheduler.is_finished():
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        ev_start.record()
        sched_seqs, has_prefill = scheduler.schedule()
        tids = model_runner.call("run", sched_seqs, has_prefill)
        scheduler.postprocess(sched_seqs, tids)
        ev_end.record()

        step_events.append((ev_start, ev_end, sched_seqs, has_prefill))
        steps += 1

    # 同步等待所有 GPU 操作完成
    torch.cuda.synchronize()
    total_wall_time = time.perf_counter() - t0_cpu

    # ---- 从 CUDA Events 提取精确时间 ----
    cumulative_ms = 0.0
    for ev_start, ev_end, sched_seqs, has_prefill in step_events:
        step_ms = ev_start.elapsed_time(ev_end)  # GPU 精确 ms
        cumulative_ms += step_ms

        for seq in sched_seqs:
            r = req[seq.seq_id]
            if not seq.is_prefill:
                if r["ttft_ms"] is None:
                    r["ttft_ms"] = cumulative_ms
                r["step_times_ms"].append(step_ms)
                r["output_n"] += 1
            if seq.is_finished and r["end_ms"] is None:
                r["end_ms"] = cumulative_ms

    # ---- 聚合指标 ----
    ttfts, tpots, tpot_all, e2es = [], [], [], []
    total_in = total_out = 0
    for r in req.values():
        total_in += r["input_len"]
        total_out += r["output_n"]
        if r["ttft_ms"] is not None:
            ttfts.append(r["ttft_ms"])
        if r["step_times_ms"]:
            tpots.append(float(np.mean(r["step_times_ms"])))
            tpot_all.extend(r["step_times_ms"])
        if r["end_ms"] is not None:
            e2es.append(r["end_ms"])

    total_gpu_ms = cumulative_ms  # GPU 总执行时间

    return {
        "name": scenario["name"],
        "category": scenario["category"],
        "num_seqs": num_seqs,
        "avg_input_len": round(total_in / num_seqs, 1),
        "avg_output_len": round(total_out / num_seqs, 1),
        "prefix_rate": scenario["prefix_rate"],
        # 时间
        "total_gpu_time_ms": round(total_gpu_ms, 2),
        "total_wall_time_s": round(total_wall_time, 3),
        # 请求频率
        "request_rate_rps": round(num_seqs / total_wall_time, 2),
        "qps": round(num_seqs / total_wall_time, 2),
        # TTFT (GPU 精确)
        "ttft_avg_ms": round(float(np.mean(ttfts)), 2) if ttfts else 0,
        "ttft_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_p99_ms": round(percentile(ttfts, 99), 2),
        # TPOT (GPU 精确, per-request 均值)
        "tpot_avg_ms": round(float(np.mean(tpots)), 2) if tpots else 0,
        "tpot_p50_ms": round(percentile(tpots, 50), 2),
        "tpot_p99_ms": round(percentile(tpots, 99), 2),
        # TPOT 全局 (所有 step 的分布)
        "tpot_global_avg_ms": round(float(np.mean(tpot_all)), 2) if tpot_all else 0,
        "tpot_global_p99_ms": round(percentile(tpot_all, 99), 2),
        # E2E
        "e2e_avg_ms": round(float(np.mean(e2es)), 2) if e2es else 0,
        "e2e_p99_ms": round(percentile(e2es, 99), 2),
        "e2e_total_s": round(total_wall_time, 3),
        # Throughput (基于 wall-clock，因为是实际可观测指标)
        "output_tps": round(total_out / total_wall_time, 1),
        "e2e_tps": round((total_in + total_out) / total_wall_time, 1),
        # Steps
        "total_steps": steps,
    }


# ============================================================
# Baseline
# ============================================================

def run_benchmark_baseline(model_path, scenarios):
    sys.path.insert(0, "/root/autodl-tmp/projects/nano-vllm-main")
    import torch
    from nanovllm import LLM, SamplingParams

    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
    llm.generate([[1, 2, 3]], SamplingParams(temperature=1.0, max_tokens=4, ignore_eos=True))
    gpu_mem = torch.cuda.memory_allocated() / 1e9

    all_results = []
    for sc in scenarios:
        random.seed(42)
        prompts = gen_prompts(sc)
        num_seqs = len(prompts)
        sp = SamplingParams(temperature=1.0, max_tokens=sc["output_len"], ignore_eos=True)
        try:
            start = time.perf_counter()
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            elapsed = time.perf_counter() - start
            total_out = sum(len(o["token_ids"]) for o in outputs)
            total_in = sum(len(p) for p in prompts)
            all_results.append({
                "name": sc["name"], "category": sc["category"],
                "num_seqs": num_seqs,
                "avg_input_len": round(total_in / num_seqs, 1),
                "avg_output_len": round(total_out / num_seqs, 1),
                "prefix_rate": sc["prefix_rate"],
                "request_rate_rps": round(num_seqs / elapsed, 2),
                "qps": round(num_seqs / elapsed, 2),
                "ttft_avg_ms": 0, "ttft_p50_ms": 0, "ttft_p99_ms": 0,
                "tpot_avg_ms": round(elapsed / total_out * 1000, 3) if total_out else 0,
                "tpot_p50_ms": 0, "tpot_p99_ms": 0,
                "e2e_avg_ms": round(elapsed * 1000 / num_seqs, 2),
                "e2e_p99_ms": 0, "e2e_total_s": round(elapsed, 3),
                "output_tps": round(total_out / elapsed, 1),
                "e2e_tps": round((total_in + total_out) / elapsed, 1),
                "gpu_mem_gb": round(gpu_mem, 2), "num_kv_blocks": 0, "total_steps": 0,
            })
        except Exception as e:
            all_results.append({"name": sc["name"], "category": sc["category"],
                                "error": str(e)[:200]})
    return all_results


# ============================================================
CONFIG_MAP = {
    "baseline": {"type": "baseline"},
    "lite-bare": {"chunked": False, "fp8": False, "eager": True, "prefix": False},
    "lite-graph": {"chunked": False, "fp8": False, "eager": False, "prefix": False},
    "lite-graph-prefix": {"chunked": False, "fp8": False, "eager": False, "prefix": True},
    "lite-graph-chunk": {"chunked": True, "fp8": False, "eager": False, "prefix": False},
    "lite-graph-fp8": {"chunked": False, "fp8": True, "eager": False, "prefix": False},
    "lite-full-nofp8": {"chunked": True, "fp8": False, "eager": False, "prefix": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenarios", default="all")
    args = parser.parse_args()

    all_scenarios = build_scenarios()
    if args.scenarios != "all":
        cats = args.scenarios.split(",")
        all_scenarios = [s for s in all_scenarios if s["category"] in cats]

    cfg = CONFIG_MAP.get(args.config)
    if not cfg:
        print(json.dumps({"error": f"Unknown config: {args.config}"}))
        sys.exit(1)

    if cfg.get("type") == "baseline":
        results = run_benchmark_baseline(args.model, all_scenarios)
    else:
        results = run_benchmark_lite(
            args.model, all_scenarios,
            enable_chunked=cfg["chunked"], enable_fp8=cfg["fp8"],
            enforce_eager=cfg["eager"], enable_prefix=cfg["prefix"],
        )

    print(json.dumps({"config": args.config, "scenarios": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
