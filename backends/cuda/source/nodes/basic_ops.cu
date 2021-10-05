#include <cuned/cunodes.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace nn;
using namespace cuda;

BinOpSame::BinOpSame(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp1, core::Edge* inp2, core::Edge* out)
{
    assert(out->opaque);

    sz = 1;
    for (auto& e : inp1->dsc.dims)
        sz *= e;
    if (!inp1->opaque)
    {
        inp1->opaque = new Edge();
        // TODO: check to make sure it was allocated
        cudaMalloc(&((Edge*)inp1->opaque)->forward_data, sz * core::dtype_size(inp1->dsc.dty));
        cudaMalloc(&((Edge*)inp1->opaque)->backward_data, sz * core::dtype_size(inp1->dsc.dty));
    }
    if (!inp2->opaque)
    {
        inp2->opaque = new Edge();
        // TODO: check to make sure it was allocated
        cudaMalloc(&((Edge*)inp2->opaque)->forward_data, sz * core::dtype_size(inp2->dsc.dty));
        cudaMalloc(&((Edge*)inp2->opaque)->backward_data, sz * core::dtype_size(inp2->dsc.dty));
    }
    this->inp1 = (Edge*)inp1->opaque;
    this->inp2 = (Edge*)inp2->opaque;
    this->out = (Edge*)out->opaque;
    inp1_dty = inp1->dsc.dty;
    inp2_dty = inp2->dsc.dty;
    out_dty = out->dsc.dty;
}

BinOpScalar::BinOpScalar(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp, core::Edge* val, core::Edge* out)
{
    assert(out->opaque);

    sz = 1;
    for (auto& e : inp->dsc.dims)
        sz *= e;
    if (!inp->opaque)
    {
        inp->opaque = new Edge();
        // TODO: check to make sure it was allocated
        cudaMalloc(&((Edge*)inp->opaque)->forward_data, sz * core::dtype_size(inp->dsc.dty));
        cudaMalloc(&((Edge*)inp->opaque)->backward_data, sz * core::dtype_size(inp->dsc.dty));
    }
    if (!val->opaque)
    {
        val->opaque = new Edge();
        // TODO: check to make sure it was allocated
        cudaMalloc(&((Edge*)val->opaque)->forward_data, core::dtype_size(val->dsc.dty));
        cudaMalloc(&((Edge*)val->opaque)->backward_data, core::dtype_size(val->dsc.dty));
    }
    this->inp = (Edge*)inp->opaque;
    this->val = (Edge*)val->opaque;
    this->out = (Edge*)out->opaque;
    inp_dty = inp->dsc.dty;
    val_dty = val->dsc.dty;
    out_dty = out->dsc.dty;
}

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
