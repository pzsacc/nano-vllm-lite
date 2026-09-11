"""
benchmarks/bench.py - 唯一压测入口

三层定位（详见 profiling/README.md 引导页）:
  L1 服务层   bench.py 1 --scenario ...      黑盒压测: TTFT/TPOT/TPS/QPS
  L2 显存分析 bench.py 2 --mode ...          显存去哪了 / KV 容量探针
  L3 算子剖析 bench.py 3 --tool ...          torch.profiler / 带宽对拍 / nsys 引导

使用漏斗: L1 发现异常 → L2 排除显存因素 → L3 定位到 kernel
"""
import sys
import os
import subprocess

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("1", "2", "3"):
        print(__doc__)
        print("用法: python -m benchmarks.bench {1|2|3} [原有参数...]")
        sys.exit(1)

    level = sys.argv[1]
    rest = sys.argv[2:]
    scripts = {"1": "level1_service.py", "2": "level2_memory.py", "3": "level3_profile.py"}
    cmd = [sys.executable, os.path.join(BENCH_DIR, scripts[level])] + rest
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
