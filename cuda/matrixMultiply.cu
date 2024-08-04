#include <cuda_runtime.h>
#include <iostream>
#include <array>
#include "util.hpp"

__global__ void matrixMulKernel(float *m, float *n, float *p, int size)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size && j < size)
    {
        float pValue = 0;
        for (int k = 0; k < size; ++k)
        {
            pValue += m[i * size + k] * n[k * size + j];
        }
        p[i * size + j] = pValue;
    }
}

__global__ void matrixMulKernelRow(float *m, float *n, float *p, int size)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size)
    {
        for (int col = 0; col < size; ++col)
        {
            float pValue = 0;
            for (int i = 0; i < size; ++i)
            {
                pValue += m[row * size + i] * n[i * size + col];
            }
            p[row * size + col] = pValue;
        }
    }
}

__global__ void matrixMulKernelCol(float *m, float *n, float *p, int size)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < size)
    {
        for (int row = 0; row < size; ++row)
        {
            float pValue = 0;
            for (int i = 0; i < size; ++i)
            {
                pValue += m[row * size + i] * n[i * size + col];
            }
            p[row * size + col] = pValue;
        }
    }
}

int main()
{
    const int SIZE{16};

    std::array<float, SIZE> m_h;
    std::array<float, SIZE> n_h;
    std::array<float, SIZE> p_h{};

    initializeMatrix(m_h);
    initializeMatrix(n_h);

    std::cout << "m:" << '\n';
    printMatrix(m_h.data(), 4, 4);
    std::cout << "n:" << '\n';
    printMatrix(n_h.data(), 4, 4);

    float *m_d;
    float *n_d;
    float *p_d;

    cudaMalloc((void **)&m_d, SIZE * sizeof(float));
    cudaMalloc((void **)&n_d, SIZE * sizeof(float));
    cudaMalloc((void **)&p_d, SIZE * sizeof(float));

    cudaMemcpy(m_d, m_h.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(n_d, n_h.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p_h.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize(2, 2);
    dim3 blockSize(2, 2);
    std::cout << "matrixMulKernel m@n: " << '\n';
    matrixMulKernel<<<gridSize, blockSize>>>(m_d, n_d, p_d, 4);
    cudaMemcpy(p_h.data(), p_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(p_h.data(), 4, 4);

    // change to thread count equal to matrix dimension for inefficient implementations
    gridSize = 1;
    blockSize = 4;

    std::cout << "matrixMulKernelRow m@n: " << '\n';
    // thread 0:
    // p_0,0 p[0]
    // p_0,1 p[1]
    // ..
    // thread 1:
    // p_1,0 p[4]
    // p_1,1 p[5]
    // ..
    matrixMulKernelRow<<<gridSize, blockSize>>>(m_d, n_d, p_d, 4);
    cudaMemcpy(p_h.data(), p_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(p_h.data(), 4, 4);

    // thread 0:
    // p_0,0 p[0]
    // p_1,0 p[4]
    // ..
    // thread 1:
    // p_0,1 p[1]
    // p_1,1 p[5]
    // ..
    std::cout << "matrixMulKernel m@n: " << '\n';
    cudaMemcpy(p_h.data(), p_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(p_h.data(), 4, 4);

    cudaFree(m_d);
    cudaFree(n_d);
    cudaFree(p_d);

    m_d = nullptr;
    n_d = nullptr;
    p_d = nullptr;

    return 0;
}
