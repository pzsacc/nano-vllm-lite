#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple> // 引入 tuple 头文件

// ==========================================================
// 1. CUDA Kernel (运行在 GPU 上的代码)
// ==========================================================
template <typename scalar_t>
__global__ void add_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ residual, // 注意：这里既是输入也是输出 (in-place add)
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int N,           // 总行数 (Batch * SeqLen)
    int hidden_size, // 列数 (Hidden Dimension)
    float eps)
{
    // 经典的 "一行一个 Block" 调度策略
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    // 定位到当前行的起始指针
    const scalar_t* x_row = x + row * hidden_size;
    scalar_t* res_row = residual + row * hidden_size;
    scalar_t* out_row = out + row * hidden_size;

    // --- 第一阶段：Add 计算 & 局部平方和 ---
    float local_sum_sq = 0.0f;

    // 一个 Block 可能只有 1024 个线程，但 hidden_size 可能是 4096
    // 所以让每个线程用 for 循环跳跃处理 (Grid-Stride Loop 在 Block 内部的体现)
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        // 读取并转为 float32 防止精度溢出
        float x_val = static_cast<float>(x_row[col]);
        float r_val = static_cast<float>(res_row[col]);

        float x_r = x_val + r_val;

        // Add 的结果写回 residual (In-place)
        res_row[col] = static_cast<scalar_t>(x_r);

        // 累加局部平方和
        local_sum_sq += x_r * x_r;
    }

    // --- 第二阶段：Block 内部规约 (Reduction) 求全局平方和 ---
    // 这里用共享内存来做 Block 级别的求和 (简化版，工业界通常用 Warp Shuffle)
    extern __shared__ float shared_sum[];
    shared_sum[tid] = local_sum_sq;
    __syncthreads(); // 等待所有线程把局部和写进共享内存

    // 树状规约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // 由 0 号线程计算出 RMSNorm 的逆标准差 (rstd)
    __shared__ float rstd;
    if (tid == 0) {
        float var = shared_sum[0] / hidden_size;
        rstd = rsqrtf(var + eps);
    }
    __syncthreads(); // 确保所有线程都看到了 rstd

    // --- 第三阶段：RMSNorm 并乘上 Weight ---
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        float x_r = static_cast<float>(res_row[col]); // 刚才 Add 写回的值
        float w_val = static_cast<float>(weight[col]);

        float out_val = x_r * rstd * w_val;

        // 写回最终输出
        out_row[col] = static_cast<scalar_t>(out_val);
    }
}

// ==========================================================
// 2. C++ Wrapper (运行在 CPU 上，负责调度)
// ==========================================================
// 🚨 修改点 1：返回值类型改为 std::tuple<torch::Tensor, torch::Tensor>
std::tuple<torch::Tensor, torch::Tensor> add_rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    float eps)
{
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int N = x.size(0);
    int hidden_size = x.size(1);
    auto out = torch::empty_like(x);

    int threads = 1024;
    int blocks = N;
    size_t shared_mem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    x.scalar_type(), "add_rmsnorm_forward", ([&] {
        add_rmsnorm_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N,
            hidden_size,
            eps
        );
    }));

    // 修改点 2：将两个 Tensor 打包成 tuple 返回
    return std::make_tuple(out, residual);
}

// ==========================================================
// 3. PyBind11 绑定 (暴露给 Python)
// ==========================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_rmsnorm_forward, "Fused Add + RMSNorm forward (CUDA)");
}