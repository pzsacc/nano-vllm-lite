/*
 * kernels/inplace_rotary_embed.cu - In-place RoPE 旋转位置编码 CUDA Kernel
 *
 * 功能：原地对 Q/K tensor 应用旋转位置编码，零额外内存分配
 * 优势：相比 PyTorch 实现，避免中间 tensor 分配和多次 kernel launch
 *
 * 并行策略：
 * - Grid: (num_tokens, num_heads) — 每个 block 处理一个 (token, head) 对
 * - Block: (head_dim / 2) — 每个线程处理一对维度的旋转
 *
 * 支持 Q 和 K 头数不同（GQA），通过两次独立 kernel launch 处理
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rope_v3_ultimate_kernel(
    scalar_t* __restrict__ x,               // [num_tokens, num_heads, head_dim]
    const int* __restrict__ pos_ids,        // [num_tokens]
    const scalar_t* __restrict__ cos_sin_cache, // [max_pos, head_dim]
    const int num_heads,
    const int head_size
) {
    // block 索引：token 和 head
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    // thread 索引：维度对中的位置
    const int d_idx = threadIdx.x;
    const int half_head = head_size / 2;

    // 获取当前 token 的位置
    const int pos = pos_ids[token_idx];

    // 计算 x 中的偏移
    const int base_idx = token_idx * (num_heads * head_size) + head_idx * head_size;
    const int idx1 = base_idx + d_idx;
    const int idx2 = base_idx + half_head + d_idx;

    // 查找 cos/sin 缓存
    const float cos_val = (float)cos_sin_cache[pos * head_size + d_idx];
    const float sin_val = (float)cos_sin_cache[pos * head_size + half_head + d_idx];

    // 读取原始值
    const float x1 = (float)x[idx1];
    const float x2 = (float)x[idx2];

    // 应用旋转
    x[idx1] = (scalar_t)(x1 * cos_val - x2 * sin_val);
    x[idx2] = (scalar_t)(x2 * cos_val + x1 * sin_val);
}

/*
 * C++ wrapper: 分别对 Q 和 K launch kernel
 */
void apply_fused_rope_inplace(
    torch::Tensor q,            // [num_tokens, num_q_heads, head_dim]
    torch::Tensor k,            // [num_tokens, num_k_heads, head_dim]
    torch::Tensor pos_ids,      // [num_tokens]
    torch::Tensor cos_sin_cache // [max_pos, head_dim]
) {
    const int num_tokens = q.size(0);
    const int num_q_heads = q.size(1);
    const int num_k_heads = k.size(1);
    const int head_size = q.size(2);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "apply_fused_rope_inplace", ([&] {
            // Launch for Q
            dim3 grid_q(num_tokens, num_q_heads);
            dim3 block(head_size / 2);
            auto stream = at::cuda::getCurrentCUDAStream();
            rope_v3_ultimate_kernel<scalar_t><<<grid_q, block, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                pos_ids.data_ptr<int>(),
                cos_sin_cache.data_ptr<scalar_t>(),
                num_q_heads, head_size
            );

            // Launch for K
            dim3 grid_k(num_tokens, num_k_heads);
            rope_v3_ultimate_kernel<scalar_t><<<grid_k, block, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                pos_ids.data_ptr<int>(),
                cos_sin_cache.data_ptr<scalar_t>(),
                num_k_heads, head_size
            );
        })
    );
}

// PyBind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_fused_rope_inplace", &apply_fused_rope_inplace,
          "In-place Rotary Position Embedding (CUDA)");
}
