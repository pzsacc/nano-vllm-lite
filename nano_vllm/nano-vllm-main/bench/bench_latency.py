import os
import time
import numpy as np
from nanovllm import LLM, SamplingParams

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 模拟真实流式场景，限制 batch size 观察 chunk 效果
    llm = LLM(path, enforce_eager=True, max_model_len=4096, max_num_batched_tokens=512)

    # 1. 构建背景稳态流量 (模拟正在流式输出的老用户)
    num_bg_tasks = 16
    for _ in range(num_bg_tasks):
        llm.add_request(
            prompt=[0] * 10,
            sampling_params=SamplingParams(temperature=0.01, ignore_eos=True, max_tokens=1000)
        )

    # 预热并进入稳定的 Decode 阶段
    for _ in range(10):
        llm.step()

    # 2. 注入破坏性长文本请求 (模拟突发的高并发长文输入)
    print("Injecting long prefill request (3000 tokens)...")
    llm.add_request(
        prompt=[0] * 3000,
        sampling_params=SamplingParams(temperature=0.01, max_tokens=10)
    )

    # 3. 高精度捕获后续 Step 的微观耗时
    step_latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        llm.step()
        step_latencies.append((time.perf_counter() - t0) * 1000)

    # 数据剥离分析
    print("\n[Micro-Scheduling Jitter Report]")
    print(f"Avg Step Latency: {np.mean(step_latencies):.2f} ms")
    print(f"Max Step Latency: {np.max(step_latencies):.2f} ms (The Spike)")
    print(f"P99 Latency:      {np.percentile(step_latencies, 99):.2f} ms")

if __name__ == "__main__":
    main()