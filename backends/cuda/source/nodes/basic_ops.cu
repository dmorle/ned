#include <cuned/cunodes.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace nn;
using namespace cuda;

constexpr int bsz = 32;

template<typename INP1, typename INP2, typename OUT>
__global__ void add_pointwise(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_pointwise(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_pointwise(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_pointwise(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void add_scalar(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_scalar(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_scalar(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_scalar(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[0];
}

#include "./basic_ops_out.cu"
