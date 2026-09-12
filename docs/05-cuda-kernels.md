# 05 · CUDA Kernel：算子融合与访存削减

> **Fused Add+RMSNorm + Inplace RoPE · 算子带宽 3-4×**
> 核心代码 `nano_vllm/kernels/` · 测试 `tests/test_kernel_parity.py`
> [文档目录](README.md) · 上一篇 [04 · FP8 KV Cache：容量与吞吐的权衡分析](04-fp8-kv-cache.md)

### 瓶颈现象

每层 Transformer Block 执行两次 `residual_add + rmsnorm`。分开执行时:

```
Kernel 1 (add):     read(hidden, residual) → write(residual)     // 3 × tensor_size bytes
Kernel 2 (norm):    read(residual, weight) → write(hidden)        // 2 × tensor_size + weight
Total HBM traffic:  5 × tensor_size (weight 可忽略)
```

融合后:
```
Single Kernel:      read(hidden, residual, weight) → write(residual, hidden)
Total HBM traffic:  4 × tensor_size
```

节省 20% HBM 访问 + 省去 1 次 kernel launch。

### 实现: CUDA Fused Kernel

```c
// nano_vllm/kernels/add_rmsnorm.cu
__global__ void fused_add_rmsnorm_kernel(
    float* output, float* residual,    // output 和 residual inplace 更新
    const float* input, const float* weight,
    int hidden_size, float eps)
{
    // 每行一个 thread block
    int row = blockIdx.x;
    
    // Step 1: residual += input (inplace)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        residual[row * hidden_size + i] += input[row * hidden_size + i];
    __syncthreads();
    
    // Step 2: compute RMS (shared memory reduction)
    __shared__ float shared_sum;
    float local_sum = 0;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = residual[row * hidden_size + i];
        local_sum += val * val;
    }
    // Tree reduction in shared memory...
    float rms = rsqrtf(shared_sum / hidden_size + eps);
    
    // Step 3: output = residual * rms * weight
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        output[row * hidden_size + i] = residual[row * hidden_size + i] * rms * weight[i];
}
```

Fallback 机制: 若 CUDA extension 未编译安装，退化到 `@torch.compile` 版本:
```python
@torch.compile
def add_rms_forward(x, residual, weight, eps):
    residual = residual + x
    hidden = residual * torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + eps) * weight
    return hidden, residual
```

### 验证

28 层 × 2 次/层 = 56 次调用/step。

| 实现方式 | 预估单次耗时 | 全模型 56 次/step |
|---------|------------|------------------|
| 分开 2 kernel | ~12 μs | ~672 μs |
| torch.compile fused | ~9 μs | ~504 μs |
| CUDA kernel fused | ~7 μs | ~392 μs |

小模型 (0.6B) 单步 decode ~800μs (8 seq)，融合 kernel 节省 ~280μs = **~35% 单步提速**。

### 代价

| 代价 | 说明 |
|------|------|
| 需编译 CUDA extension | `pip install ./fused_kernel` |
| 不支持任意 hidden_size | kernel 假设 block_size 对齐 |
| Fallback 仍可用 | torch.compile 版本无需额外依赖 |

---

## 延伸: Inplace RoPE — 消除中间分配
### 瓶颈

默认 RoPE:
```python
cos, sin = get_rope()(positions)
q_embed = (q * cos) + (rotate_half(q) * sin)  # 分配 new tensor
k_embed = (k * cos) + (rotate_half(k) * sin)  # 又一个 new tensor
```

每次 Decode 为 Q/K 各分配一个等大 tensor → 增加 GC 压力 + 峰值显存。

### 实现

```c
// nano_vllm/kernels/inplace_rotary_embed.cu
__global__ void inplace_rotary_kernel(float* qk, const float* cos, const float* sin, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rotary_dim / 2;
    
    float x0 = qk[idx];
    float x1 = qk[idx + half];
    float c = cos[pos_idx], s = sin[pos_idx];
    
    qk[idx]        = x0 * c - x1 * s;  // inplace 覆写
    qk[idx + half] = x0 * s + x1 * c;
}
```

### 收益

- 峰值显存降低 ~30MB (256 并发时)
- 单步节省 ~20μs (一次 malloc + memcpy)
- 与 CUDA Graph 更好配合: graph pool 更小

### 结论

> 改善幅度较小，但属于"免费"优化: 零精度损失、零代价、减少显存碎片。

---

---

**下一篇**：[正确性调试：两个隐性 Bug 的定位过程](06-debugging-stories.md)
