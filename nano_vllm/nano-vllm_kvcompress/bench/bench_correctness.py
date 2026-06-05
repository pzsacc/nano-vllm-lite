import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入你的自研引擎 (根据你的实际 API 调整)
from nanovllm import LLM


def benchmark_logits_correctness():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    device = "cuda"

    print("🚀 启动 Logits 精度零发散校验 (Correctness Benchmark)...")

    # ==========================================
    # 1. 初始化基线模型 (HuggingFace 原生 FP16)
    # ==========================================
    print("-> 加载 HuggingFace Baseline (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    ).eval()

    # ==========================================
    # 2. 初始化自研引擎 (带 Triton FP8 KV Cache 量化)
    # ==========================================
    print("-> 加载 Nano-vLLM Engine (FP8 KV Cache)...")
    # 此处可以根据你的引擎配置开启 FP8 KV 量化标志
    llm = LLM(model_path, enforce_eager=True, max_model_len=4096)

    # 测试用例：覆盖短文本与触发 Chunked Prefill 的长文本
    test_prompts = [
        "The capital of France is",  # 短文本预热
        "def quicksort(arr):\n    if len(arr) <= 1:",  # 代码生成
        "A system architect is designing a high-throughput" * 50  # 长文本 (触发 KV Cache 存取与分块)
    ]

    for idx, text in enumerate(test_prompts):
        print(f"\n[测试用例 {idx + 1}] 文本长度: {len(tokenizer.encode(text))} tokens")
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        with torch.no_grad():
            # ----------------------------------
            # 步骤 A: 获取 HF 原生 Logits (Ground Truth)
            # ----------------------------------
            hf_outputs = hf_model(input_ids)
            # 取出最后一个 Token 的预测概率分布 (Shape: [Vocab_Size])
            hf_logits = hf_outputs.logits[0, -1, :]

            # ----------------------------------
            # 步骤 B: 获取自研引擎 Logits
            # ----------------------------------
            # ⚠️ 注意：这里需要替换为你引擎实际获取 Logits 的方式！
            # 如果你的 llm.step() 不返回 logits，你可能需要在 engine 内部写一个 forward_test 接口
            # 假设你的引擎可以这样获取 Logits：
            # custom_logits = llm.forward_to_logits(input_ids)[0, -1, :]

            # TODO: 替换为你的真实 API ⬇️
            custom_logits = hf_logits.clone()  # 占位符，请删除此行并换成你的引擎调用
            # TODO: 替换为你的真实 API ⬆️

            # ----------------------------------
            # 步骤 C: 精度校验计算
            # ----------------------------------
            # 确保数据类型一致，转为 float32 进行高精度比对
            hf_logits = hf_logits.to(torch.float32)
            custom_logits = custom_logits.to(torch.float32)

            # 1. 绝对对齐校验 (Exact Match - Argmax)
            hf_pred_token = torch.argmax(hf_logits).item()
            custom_pred_token = torch.argmax(custom_logits).item()
            exact_match = (hf_pred_token == custom_pred_token)

            # 2. 误差统计 (Mean Absolute Error & Max Absolute Error)
            diff = torch.abs(hf_logits - custom_logits)
            mae = torch.mean(diff).item()
            max_err = torch.max(diff).item()

            # 3. 概率分布相似度 (Cosine Similarity)
            cos_sim = F.cosine_similarity(hf_logits.unsqueeze(0), custom_logits.unsqueeze(0)).item()

            print(f"  ├─ 预测 Token 对齐 (Exact Match): {'✅ Pass' if exact_match else '❌ Fail'} (HF: {hf_pred_token}, Custom: {custom_pred_token})")
            print(f"  ├─ 最大绝对误差 (Max Error)   : {max_err:.6e}")
            print(f"  ├─ 平均绝对误差 (MAE)         : {mae:.6e}")
            print(f"  └─ 概率分布相似度 (Cos Sim)   : {cos_sim:.6f}")

            # 校验 FP8 的物理底噪 (通常 > 0.999 就算完美，绝对误差在 1e-3 量级)
            assert exact_match, "严重错误：预测的 Top-1 Token 发生偏离！"
            assert cos_sim > 0.999, "严重错误：输出概率分布发生坍塌！"

    print("\n🎉 全部精度校验通过！引擎输出分布达到工业级对齐标准。")


if __name__ == "__main__":
    benchmark_logits_correctness()