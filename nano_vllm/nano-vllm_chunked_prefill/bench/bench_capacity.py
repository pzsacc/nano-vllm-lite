import os
import torch
from nanovllm import LLM, SamplingParams


def main():
    print(f"\n{'=' * 50}")
    print("🚀 启动大模型单机显存极限容量压测 (Capacity Benchmark)")
    print(f"{'=' * 50}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager 确保按真实物理轨迹分配显存，屏蔽 CUDA Graph 预分配的干扰
    llm = LLM(path, enforce_eager=True, max_model_len=4096)

    # 工业标准负载：固定输入 512, 输出 512 (总计 1024 Token 的上下文)
    prompt_len = 512
    max_output_tokens = 512
    dummy_prompt = [1] * prompt_len

    # ignore_eos=True 强迫模型跑满 512 步，逼出最大的 KV Cache 物理占用
    # temperature 设置为极小值以绕过 nano-vllm 的 assert 限制
    sp = SamplingParams(temperature=0.01, ignore_eos=True, max_tokens=max_output_tokens)

    success_count = 0
    step_count = 0
    max_running_capacity = 0

    print("\n-> 开始阶梯式洪水注入，正在监控 Scheduler 泄洪闸...")

    try:
        while True:
            # 每次注入 4 个请求（步子迈小一点，测出的临界点更精准）
            for _ in range(4):
                llm.add_request(dummy_prompt, sp)
                success_count += 1

            # 强制引擎推进行程，逼迫底层的 Block Manager 划拨物理显存块
            llm.step()
            step_count += 1

            # 🌟 核心透视逻辑：直接读取底层调度器的队列状态
            # (注意：如果你的 nano-vllm 调度器变量名不同，请修改为对应的属性，通常是 .running 和 .waiting)
            current_running = len(llm.scheduler.running)
            current_waiting = len(llm.scheduler.waiting)

            if current_running > max_running_capacity:
                max_running_capacity = current_running

            if step_count % 10 == 0:
                print(f"[监控] GPU 运行中(Running): {current_running:4d} | CPU 积压等待(Waiting): {current_waiting:4d}")

            # 🌟 破局判定：如果 CPU 积压的等待请求超过了 30 个，
            # 且 GPU 运行队列不再增长，说明 GPU 显存的物理 Block 已经被彻底榨干了！
            if current_waiting > 30 and current_running == max_running_capacity:
                print("\n" + "!" * 50)
                print("💥 [极限达成] Block Manager 彻底饱和，调度器已锁死泄洪闸！")
                print(f"🏆 最终单卡最高并发驻留量 (Max Capacity): {max_running_capacity} 序列")
                print("!" * 50 + "\n")
                break

    except Exception as e:
        # 捕捉真正的物理显存 OOM 或其他异常
        print("\n" + "!" * 50)
        print(f"❌ 触发异常中断 (可能是底层物理 OOM): {e}")
        print(f"🏆 崩溃前最高并发驻留量 (Max Capacity): {max_running_capacity} 序列")
        print("!" * 50 + "\n")


if __name__ == "__main__":
    main()