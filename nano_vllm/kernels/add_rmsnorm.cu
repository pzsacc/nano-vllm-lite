/*
 * kernels/add_rmsnorm.cu - 融合 Residual Add + RMSNorm CUDA Kernel
 *
 * 功能：单个 kernel 完成 residual = x + residual, out = RMSNorm(residual)
 * 优势：相比分开执行，减少一次全局内存读写（节省 ~30% 带宽）
 *
 * 并行策略：每个 block 处理一行（一个 token），block 内线程协作计算方差
 * 使用 shared memory tree reduction 高效归约 sum_of_squares
 *
 * 支持 FP16 和 BF16 输入/输出
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_rmsnorm_kernel(
    scalar_t* __restrict__ out,         // [N, D] 输出
    scalar_t* __restrict__ residual,    // [N, D] 残差（原地更新）
    const scalar_t* __restrict__ x,     // [N, D] 输入
    const scalar_t* __restrict__ weight,// [D] 归一化权重
    const int hidden_size,              // D
    const float eps                     // 数值稳定常数
) {
    // 每个 block 处理一行
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t* x_row = x + row * hidden_size;
    scalar_t* res_row = residual + row * hidden_size;
    scalar_t* out_row = out + row * hidden_size;

    // Phase 1: residual += x，同时累加 sum_of_squares
    extern __shared__ float shared_mem[];
    float local_sum_sq = 0.0f;

    for (int col = tid; col < hidden_size; col += blockDim.x) {
        float val = (float)x_row[col] + (float)res_row[col];
        res_row[col] = (scalar_t)val;
        local_sum_sq += val * val;
    }

    // Phase 2: Tree reduction 计算总 sum_of_squares
    shared_mem[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Phase 3: 计算 rstd = rsqrt(mean(x^2) + eps)
    __shared__ float rstd;
    if (tid == 0) {
        rstd = rsqrtf(shared_mem[0] / (float)hidden_size + eps);
    }
    __syncthreads();

    // Phase 4: 归一化输出 = residual * rstd * weight
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        float val = (float)res_row[col];
        out_row[col] = (scalar_t)(val * rstd * (float)weight[col]);
    }
}

/*
 * C++ wrapper: 分配输出 tensor 并 launch kernel
 */
std::tuple<torch::Tensor, torch::Tensor> add_rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    float eps
) {
    const int N = x.size(0);
    const int D = x.size(1);
    auto out = torch::empty_like(x);

    const int threads = 1024;
    const int blocks = N;
    const int shared_mem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "add_rmsnorm_forward", ([&] {
            add_rmsnorm_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                out.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                D, eps
            );
        })
    );

    return std::make_tuple(out, residual);
}

// PyBind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_rmsnorm_forward, "Fused Add + RMSNorm (CUDA)");
}
