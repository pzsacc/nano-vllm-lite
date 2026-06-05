import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入你的自研引擎 (根据你的实际 API 调整)
from nanovllm import LLM, SamplingParams


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
            # 步骤 B: 获取自研引擎 Logits (穿透式探针获取)
            # ----------------------------------
            # 方案：绕过高层的 Scheduler 和 Step 封装，直接调用底层 PyTorch Model 的 forward。

            # 1. 取出你引擎内部真正的 PyTorch 模型实体
            # ⚠️ 这里需要根据你的 model_runner 结构微调，通常是 llm.model_runner.model
            nano_model = llm.model_runner.model

            # 2. 构造适合引擎底层的输入格式 (通常只需要展开的 1D token_ids 和位置编码)
            # 对于纯粹的 Logits 校验，我们可以模拟一个最基础的 Prefill 前向传播
            # 提取 1D 的输入向量
            flattened_input_ids = input_ids.view(-1)
            seq_len = flattened_input_ids.size(0)

            # 构造输入位置编码 (0, 1, 2, ..., seq_len-1)
            positions = torch.arange(0, seq_len, dtype=torch.long, device=device)

            # 3. 执行单次的前向传播
            # ⚠️ 请确保这里的传参和你 Qwen3_0.6B 模型类的 forward 方法对齐
            # 通常至少需要 (input_ids, positions)，因为是测纯模型精度，我们临时构造一个假的 KV Cache 指针(None 或空张量)
            # 以绕过 PagedAttention 的复杂调度，纯测模型权重的计算一致性。

            try:
                # 尝试最基础的穿透调用 (假设你的模型前向传播兼容原生 HF 或基础形态)
                custom_outputs = nano_model(input_ids=flattened_input_ids.unsqueeze(0), positions=positions.unsqueeze(0))
                # 引擎输出的 Logits shape 可能是 [1, seq_len, vocab_size] 或者 [seq_len, vocab_size]
                if custom_outputs.dim() == 3:
                    custom_logits = custom_outputs[0, -1, :]
                else:
                    custom_logits = custom_outputs[-1, :]

            except Exception as e:
                print(f"\n⚠️ 底层穿透失败，原因: {e}")
                print("-> 引擎模型前向传播拦截到了缺少 KV Cache 或参数不匹配。")
                print("-> 正在启动备用方案：通过 Scheduler 注入探针...")

                # ==== 备用方案：如果底层模型强制要求调度器状态 ====
                # 我们向引擎塞入这个请求，跑刚好 1 个 Step（做完 Prefill）
                llm.add_request(prompt=flattened_input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))

                # 使用 PyTorch 的 Hook 机制（探针），强行截获模型最后一层输出的 Logits！
                captured_logits = []

                def capture_hook(module, args, output):
                    # 截获线性层的输出 (通常是 lm_head 或 output layer)
                    captured_logits.append(output.clone().detach())

                # ⚠️ 需要替换为你引擎里最后一层线性层的名字，比如 nano_model.lm_head
                hook_handle = nano_model.lm_head.register_forward_hook(capture_hook)

                # 跑一步，触发 Hook
                llm.step()

                # 拿掉探针
                hook_handle.remove()

                # 提取抓到的 Logits (取最后一个 token 的分布)
                # captured_logits[0] 的形状通常是 [Total_Tokens, Vocab_Size]
                custom_logits = captured_logits[0][-1, :]

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