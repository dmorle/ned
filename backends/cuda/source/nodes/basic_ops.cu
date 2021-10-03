#include <cuned/cunodes.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace nn;
using namespace cuda;

constexpr int bsz = 32;

template<typename T>
__device__ T sum_vec(const T* vec, size_t sz)
{
    size_t offset = sz / 2;
    if (sz % 2)
        return sum_vec(vec, offset) + sum_vec(vec + offset, offset) + vec[sz - 1];
    return sum_vec(vec, offset) + sum_vec(vec + offset, offset);
}

template<typename INP1, typename INP2, typename OUT>
__global__ void add_pointwise_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_pointwise_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_pointwise_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_pointwise_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void add_scalar_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_scalar_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_scalar_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_scalar_forward(const INP1* a, const INP2* b, OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void add_pointwise_backward(INP1* a, INP2* b, const OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        a[i] += dst[i];
        b[i] += dst[i];
    }
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_pointwise_backward(INP1* a, INP2* b, const OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        a[i] += dst[i];
        b[i] -= dst[i];
    }
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_pointwise_backward(INP1* a, INP2* b, const OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        a[i] += b[i] * dst[i];
        b[i] += a[i] * dst[i];
    }
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_pointwise_backward(INP1* a, INP2* b, const OUT* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        a[i] += dst[i] / b[i];
        b[i] -= dst[i] * a[i] / (b[i] * b[i]);
    }
}

void AddScalar::backward(RunId id)
{
    throw GraphError("Not Implemented");
}

void SubScalar::backward(RunId id)
{
    throw GraphError("Not Implemented");
}

void MulScalar::backward(RunId id)
{
    throw GraphError("Not Implemented");
}

void DivScalar::backward(RunId id)
{
    throw GraphError("Not Implemented");
}

#include "./basic_ops_out.cu"
