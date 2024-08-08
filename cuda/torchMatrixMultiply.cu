#include <torch/torch.h>
#include "util.hpp"

__global__ void matrixMulKernel(float *m, float *n, float *p, int size);
__global__ void matrixMulKernelRow(float *m, float *n, float *p, int size);
__global__ void matrixMulKernelCol(float *m, float *n, float *p, int size);

using KernelFunc = void (*)(float *, float *, float *, int);

torch::Tensor cuda_matrixMultiply(const torch::Tensor &a, const torch::Tensor &b, KernelFunc kernel)
{
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == torch::kFloat);
    TORCH_CHECK(b.dtype() == torch::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == torch::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == torch::DeviceType::CUDA);

    torch::Tensor a_contiguous{a.contiguous()};
    torch::Tensor b_contiguous{b.contiguous()};
    torch::Tensor result{torch::empty(a_contiguous.sizes(), a_contiguous.options())};

    float *a_ptr = a_contiguous.data_ptr<float>();
    float *b_ptr = b_contiguous.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();

    // Assumes square matrices and we cast to int for simplicity
    // and compatibility with our existing kernel code. In practice,
    // we would need to handle non-square matrices and use an unsigned long
    // to match PyTorch's tensor sizes.
    int dim{static_cast<int>(a.sizes()[0])};

    dim3 blockSize(16, 16);
    dim3 gridSize((dim + blockSize.x - 1) / blockSize.x, (dim + blockSize.y - 1) / blockSize.y);

    kernel<<<gridSize, blockSize>>>(a_ptr, b_ptr, result_ptr, dim);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    return result;
}

torch::Tensor matrixMultiply(const torch::Tensor &a, const torch::Tensor &b, const std::optional<std::string> &kernel_type)
{
    if (kernel_type.has_value())
    {
        if (kernel_type == "row")
        {
            return cuda_matrixMultiply(a, b, matrixMulKernelRow);
        }
        else if (kernel_type == "col")
        {
            return cuda_matrixMultiply(a, b, matrixMulKernelCol);
        }
        else
        {
            throw std::invalid_argument("Invalid kernel type");
        }
    }
    else
    {
        return cuda_matrixMultiply(a, b, matrixMulKernel);
    }
}

TORCH_LIBRARY(myextension, m)
{
    m.def("mymatrixmultiply(Tensor a, Tensor b, str? kernel_type = None) -> Tensor");
}

TORCH_LIBRARY_IMPL(myextension, CUDA, m)
{
    m.impl("mymatrixmultiply", TORCH_FN(matrixMultiply));
}