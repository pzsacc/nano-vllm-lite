#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename scalar_t>
__global__ void rope_v3_ultimate_kernel(
    scalar_t* __restrict__ x,
    const float* __restrict__ cos_sin_cache,
    const int* __restrict__ pos_ids,
    const int head_size)
{
    // 🚀 终极架构：2D Grid 映射
    // blockIdx.x  -> 映射到 Token ID
    // blockIdx.y  -> 映射到 Head ID
    // threadIdx.x -> 映射到具体的维度对 d_idx (0 到 head_size/2 - 1)

    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int d_idx = threadIdx.x;

    // gridDim.y 就是这个张量的总头数 (q_heads 或 k_heads)
    const int num_heads = gridDim.y;
    const int half_head_size = blockDim.x;

    // 1. 获取绝对位置
    const int pos = pos_ids[token_idx];

    // 2. 极致纯净的坐标推导！(没有任何 %, / 等昂贵的算术指令)
    const int base_idx = token_idx * (num_heads * head_size) + head_idx * head_size;
    const int idx1 = base_idx + d_idx;
    const int idx2 = base_idx + half_head_size + d_idx;

    // 3. 从 L1/L2 Cache 极速加载 cos/sin (硬件会自动广播给同一 Token 的所有 Head)
    const float cos = cos_sin_cache[pos * head_size + d_idx];
    const float sin = cos_sin_cache[pos * head_size + half_head_size + d_idx];

    // 4. 读取 -> 乘加 -> 回写
    float x1 = static_cast<float>(x[idx1]);
    float x2 = static_cast<float>(x[idx2]);

    x[idx1] = static_cast<scalar_t>(x1 * cos - x2 * sin);
    x[idx2] = static_cast<scalar_t>(x2 * cos + x1 * sin);
}

void apply_fused_rope_inplace(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& pos_ids,
    torch::Tensor& cos_sin_cache)
{
    const int num_tokens = q.size(0);
    const int num_q_heads = q.size(1);
    const int num_k_heads = k.size(1);
    const int head_size = q.size(2);

    // 🚀 Block 大小恰好等于半个 Head (对于 128 维度就是 64 线程)
    // 现代 GPU 支持每个 SM 并发执行数十个 Block，64 线程完全能够打满 Occupancy
    dim3 block(head_size / 2);

    // 🚀 Grid 大小：X轴为 Token 数，Y轴为 Head 数
    dim3 grid_q(num_tokens, num_q_heads);
    dim3 grid_k(num_tokens, num_k_heads);

    // 将 Q 和 K 拆分为两次 Kernel Launch，逻辑更加纯粹，网格对齐更完美
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "rope_v3_q", ([&] {
        rope_v3_ultimate_kernel<scalar_t><<<grid_q, block>>>(
            q.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<float>(),
            pos_ids.data_ptr<int>(),
            head_size
        );
    }));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, k.scalar_type(), "rope_v3_k", ([&] {
        rope_v3_ultimate_kernel<scalar_t><<<grid_k, block>>>(
            k.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<float>(),
            pos_ids.data_ptr<int>(),
            head_size
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_fused_rope_inplace", &apply_fused_rope_inplace, "Fused RoPE Inplace Kernel V3");
}