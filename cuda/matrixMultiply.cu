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

    checkCudaError(cudaMalloc((void **)&m_d, SIZE * sizeof(float)), "Failed to allocate m_d");
    checkCudaError(cudaMalloc((void **)&n_d, SIZE * sizeof(float)), "Failed to allocate n_d");
    checkCudaError(cudaMalloc((void **)&p_d, SIZE * sizeof(float)), "Failed to allocate p_d");

    checkCudaError(cudaMemcpy(m_d, m_h.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy m_h to m_d");
    checkCudaError(cudaMemcpy(n_d, n_h.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy n_h to n_d");
    checkCudaError(cudaMemcpy(p_d, p_h.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy p_h to p_d");

    dim3 gridSize(2, 2);
    dim3 blockSize(2, 2);
    std::cout << "matrixMulKernel m@n: " << '\n';
    matrixMulKernel<<<gridSize, blockSize>>>(m_d, n_d, p_d, 4);
    checkCudaError(cudaGetLastError(), "matrixMulKernel launch failed");
    checkCudaError(cudaMemcpy(p_h.data(), p_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy p_d to p_h");
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
    checkCudaError(cudaGetLastError(), "matrixMulKernelRow launch failed");
    checkCudaError(cudaMemcpy(p_h.data(), p_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy p_d to p_h");
    printMatrix(p_h.data(), 4, 4);

    std::cout << "matrixMulKernelCol m@n: " << '\n';
    // thread 0:
    // p_0,0 p[0]
    // p_1,0 p[4]
    // ..
    // thread 1:
    // p_0,1 p[1]
    // p_1,1 p[5]
    // ..
    matrixMulKernelCol<<<gridSize, blockSize>>>(m_d, n_d, p_d, 4);
    checkCudaError(cudaGetLastError(), "matrixMulKernelCol failed to launch");
    checkCudaError(cudaMemcpy(p_h.data(), p_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy p_d to p_h");
    printMatrix(p_h.data(), 4, 4);

    checkCudaError(cudaFree(m_d), "Failed to free m_d");
    checkCudaError(cudaFree(n_d), "Failed to free n_d");
    checkCudaError(cudaFree(p_d), "Failed to free p_d");

    m_d = nullptr;
    n_d = nullptr;
    p_d = nullptr;

    return 0;
}
