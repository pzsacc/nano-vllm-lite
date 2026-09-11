# 06 · Debug 复盘 — 两个真实 Bug 的完整定位过程

[上一篇](05-cuda-kernels.md) | [目录](README.md) | 下一篇: [Benchmark 方法论](07-benchmarks-5090.md)

> 这两个 bug 都是在接入 Fused Add+RMSNorm kernel 时暴露的。现象极具迷惑性：
> **所有性能指标完全正常，唯独输出是错的**。定位过程比修复本身更有价值。

## Bug 1: 输出"慢一拍" — context_len 差一

### 现象

```
Prompt:   "用一句话说明什么是 Prefix Caching"
实际输出:  'PrefixPrefix C Cachingaching 是 是指前在前缓缀缓存存前...'
预期输出:  'Prefix Caching 是一种缓存策略...'
```

关键观察：**内容是对的，但每个片段恰好重复两次**。TPOT 4.0ms，吞吐正常，
没有任何报错。

### 定位过程

1. **排除采样**：把 temperature 调到 0.01 仍复现 → 不是随机性问题
2. **A/B 隔离**：原版 nano-vllm 同 prompt 正常 → 问题在 lite 的重构
3. **关键一步：对比逐层 logits**。eager 模式下两条路径逐位一致，
   说明模型前向本身没错 → 问题在"喂数据"层
4. **插桩看每步 token**：

```
[step 2] sampled: [14335]
[step 3] sampled: [14335]   ← 上一步的重复!
[step 4] sampled: [356]     ← 又是上一步的
```

每步都在"重新预测上一步已生成的 token"→ 模型每步都**看不到刚生成的
那个 token**。

### 根因

```python
# engine/model_runner.py — prepare_decode
context_lens.append(seq.num_tokens - 1)   # ❌ 少算 1
context_lens.append(seq.num_tokens)       # ✅ 修复后
```

`context_lens` 告诉 attention "这个序列已缓存了多少 token"。本步新写的
KV 也已经在 cache 里（`store_kvcache` 先于 attention 执行），所以必须
**包含当前 token 自己**。少 1 的效果 = 模型每步戴着"看不见最后一个字"
的眼镜做因果注意力 → 预测永远滞后一拍。

### 教学点

- **PagedAttention 的 `cache_seqlens` 语义是"含当前 token"**。这个 off-by-one
  在 vLLM 里也出现过同类 issue。
- 启发式：**"输出内容对但节奏错位" = 输入构造层 bug；"输出乱码" = 计算层 bug**。
- 回归测试：`test_cudagraph_parity.py::test_graph_generation_quality` 的
  重复检测断言（4-gram 连续重复 3 次 fail）。

## Bug 2: CUDA Graph replay 空转 — kernel launch 用错 stream

### 现象

接入手写 Fused Add+RMSNorm 后：
- eager 模式：输出正确 ✅
- CUDA Graph 模式：输出又退化了 ❌
- 且单独隔离测试 kernel + graph，**手工捕获的 graph "看起来是对的"**

### 定位过程（本 bug 的弯路值得细读）

1. **直觉陷阱**：capture 后马上读输出，是对的 → 误以为 capture 成功。
   实际上那是 **capture 时同步执行的结果**，不是 replay 的结果！
2. **判别实验**（本篇最核心的方法论）：

```python
# capture 后修改输入，再 replay：
x.copy_(new_data)
g.replay()
# 若 replay 有效 → 输出应随 new_data 变化
# 若 replay 空转 → 输出停在 capture 时的旧值
```

3. 结果：输出纹丝不动 → **graph 是空的**，replay 只是心理安慰
4. 找"谁没被录进去"：pytorch 提示 `The CUDA Graph is empty`，且逐层
   对比发现分歧从第一个 RMSNorm 开始 → kernel launch 没进 capture

### 根因

```c
// ❌ 默认 stream（legacy default stream）不被 graph capture 录制
add_rmsnorm_kernel<<<blocks, threads, shared_mem>>>(...);

// ✅ 当前 torch stream（capture 时 torch 会切换到 side stream）
auto stream = at::cuda::getCurrentCUDAStream();
add_rmsnorm_kernel<<<blocks, threads, shared_mem, stream>>>(...);
```

CUDA Graph capture 的原理是劫持 **当前 stream** 上的所有 launch。
手写扩展如果用三箭头语法不带 stream 参数，kernel 跑在 legacy stream
上，capture 完全看不到它——**不报错、不警告、replay 时安静地跳过**。

### 教学点

- **写 CUDA 扩展必须在 launch 时传 `at::cuda::getCurrentCUDAStream()`**，
  否则与 torch 生态（capture / stream 并行）全部不兼容。这是 torch 扩展
  开发的第一戒律，但教材很少强调。
- **"capture 后立刻读输出"是假验证**。判定 replay 有效性唯一方法是
  "改输入 → replay → 看输出是否变"。
- 事件顺序：本 bug 是接第 5 篇的 kernel 时暴露的——**每次接入新算子后
  必须跑 graph 模式的 parity 测试**，这正是 `test_cudagraph_parity.py`
  存在的意义。

---

## 本篇速记

| | Bug 1 | Bug 2 |
|---|---|---|
| 层 | 输入构造（decode 准备） | CUDA 扩展（launch 语义） |
| 现象 | 内容对、节奏错位重复 | graph 模式输出退化，eager 正常 |
| 定位钥匙 | 逐层 logits 对比 + 逐步 token 插桩 | "改输入再 replay"判别实验 |
| 根因 | context_len 少算当前 token | kernel launch 未指定 torch stream |
| 防回归 | graph parity 重复检测 | graph parity + kernel parity |

[上一篇](05-cuda-kernels.md) | [目录](README.md) | [下一篇](07-benchmarks-5090.md)
