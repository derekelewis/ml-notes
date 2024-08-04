#include <cuda_runtime.h>
#include <iostream>
#include <array>
#include "util.hpp"

__global__ void matrixVectorMulKernel(float *b, float *c, float *a, int size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size)
    {
        float aValue = 0;
        for (int col = 0; col < size; ++col)
        {
            aValue += b[row * size + col] * c[col];
        }
        a[row] = aValue;
    }
}

int main()
{
    const int MATRIX_SIZE{16};
    const int VECTOR_SIZE{4};

    std::array<float, MATRIX_SIZE> b_h;
    std::array<float, VECTOR_SIZE> c_h;
    std::array<float, VECTOR_SIZE> a_h{};

    initializeMatrix(b_h);
    initializeMatrix(c_h);

    std::cout << "b:" << '\n';
    printMatrix(b_h.data(), 4, 4);
    std::cout << "c:" << '\n';
    printMatrix(c_h.data(), 4, 1);

    float *b_d;
    float *c_d;
    float *a_d;

    cudaMalloc((void **)&b_d, MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&c_d, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void **)&a_d, VECTOR_SIZE * sizeof(float));

    cudaMemcpy(b_d, b_h.data(), MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_d, a_h.data(), VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "matrixVectorMulKernel b*c: " << '\n';
    matrixVectorMulKernel<<<1, 4>>>(b_d, c_d, a_d, 4);
    cudaMemcpy(a_h.data(), a_d, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(a_h.data(), 4, 1);

    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(a_d);

    b_d = nullptr;
    c_d = nullptr;
    a_d = nullptr;

    return 0;
}
