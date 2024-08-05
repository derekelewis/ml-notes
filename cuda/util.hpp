inline void checkCudaError(cudaError_t err, const std::string &msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
void printMatrix(const T *mat, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << mat[i * n + j] << ' ';
        }
        std::cout << std::endl;
    }
}

template <typename T, std::size_t N>
void initializeMatrix(std::array<T, N> &mat)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        mat[i] = static_cast<float>(i);
    }
}