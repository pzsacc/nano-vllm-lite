"""
bench_full.py - 完整对比压测框架

每个配置独立子进程运行，避免显存污染。
输出统一 JSON，包含所有配置 × 所有场景的数据。

用法:
    python bench_full.py --model /root/autodl-tmp/models/Qwen3-0.6B
"""
import subprocess
import sys
import json
import os
import argparse

MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"

CONFIGS = [
    {"name": "baseline", "desc": "原始 nano-vllm (CUDA Graph + Prefix Cache)"},
    {"name": "lite-bare", "desc": "nano-vllm-lite 无优化 (eager, no fp8, no chunk, no prefix)"},
    {"name": "lite-graph", "desc": "+CUDA Graph"},
    {"name": "lite-graph-prefix", "desc": "+CUDA Graph +Prefix Cache"},
    {"name": "lite-graph-chunk", "desc": "+CUDA Graph +Chunked Prefill"},
    {"name": "lite-graph-fp8", "desc": "+CUDA Graph +FP8 KV (no prefix cache)"},
    {"name": "lite-full-nofp8", "desc": "全优化 (无FP8)"},
    {"name": "lite-full", "desc": "全优化 (FP8, 无 prefix cache 以避免 OOM)"},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--configs", nargs="+", default=["all"])
    args = parser.parse_args()

    if "all" in args.configs:
        configs_to_run = CONFIGS
    else:
        configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]

    all_results = {}
    for cfg in configs_to_run:
        print(f"\n{'='*70}")
        print(f"Running: {cfg['name']} ({cfg['desc']})")
        print(f"{'='*70}")

        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "bench_worker.py"),
             "--config", cfg["name"], "--model", args.model],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-500:]}")
            all_results[cfg["name"]] = {"error": result.stderr[-200:]}
        else:
            try:
                data = json.loads(result.stdout)
                all_results[cfg["name"]] = data
                for sc in data.get("scenarios", []):
                    print(f"  [{sc['name']}] {sc['output_tps']:.1f} tok/s | "
                          f"TPOT={sc['avg_tpot_ms']:.2f}ms | mem={sc['gpu_mem_gb']:.2f}GB")
            except json.JSONDecodeError:
                print(f"  Parse error. stdout: {result.stdout[-200:]}")
                all_results[cfg["name"]] = {"error": "json parse failed"}

    # 保存完整结果
    outfile = "bench_full_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\n全部结果保存到: {outfile}")

    # 打印对比表
    print_comparison(all_results)


def print_comparison(results):
    scenarios = ["low_latency_short", "low_latency_medium", "high_throughput",
                 "high_throughput_long", "long_context", "mixed_load"]
    print(f"\n{'='*120}")
    print(f"{'对比总表':^120}")
    print(f"{'='*120}")

    header = f"{'配置':<28}"
    for s in scenarios:
        header += f"{'  ' + s[:12]:<14}"
    print(header)
    print("-" * 120)

    for cfg_name, data in results.items():
        if "error" in data:
            print(f"{cfg_name:<28} ERROR")
            continue
        row = f"{cfg_name:<28}"
        for s in scenarios:
            found = next((sc for sc in data.get("scenarios", []) if sc["name"] == s), None)
            if found:
                row += f"  {found['output_tps']:>6.0f}t/s   "
            else:
                row += f"  {'N/A':>6}     "
        print(row)
    print("=" * 120)


if __name__ == "__main__":
    main()
