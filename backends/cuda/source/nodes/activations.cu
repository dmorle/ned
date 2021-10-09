#include <cuned/cunodes.h>

#include <cuda_runtime.h>

using namespace nn;
using namespace cuda;

ActivationFn::ActivationFn(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp, core::Edge* out)
{
    assert(out->opaque);
    assert(cargs["fw"]->ty == lang::ObjType::FWIDTH);
    assert(cargs["shape"]->ty == lang::ObjType::ARRAY);
    for (auto& e : std::static_pointer_cast<lang::ObjArray>(cargs["shape"])->data.elems)
    {
        assert(e->ty == lang::ObjType::INT);
        assert(std::static_pointer_cast<lang::ObjInt>(e)->data.val >= 0);
    }

    dty = std::static_pointer_cast<lang::ObjFWidth>(cargs["fw"])->data.dty;
    sz = 1;
    for (auto& e : std::static_pointer_cast<lang::ObjArray>(cargs["shape"])->data.elems)
        sz *= std::static_pointer_cast<lang::ObjInt>(e)->data.val;

    if (!inp->opaque)
    {
        inp->opaque = new Edge();
        ((Edge*)inp->opaque)->sz = sz * core::dtype_size(inp->dsc.dty);
        // TODO: check to make sure it was allocated
        cudaMalloc(&((Edge*)inp->opaque)->forward_data, ((Edge*)inp->opaque)->sz);
        cudaMalloc(&((Edge*)inp->opaque)->backward_data, ((Edge*)inp->opaque)->sz);
    }

    this->inp = (Edge*)inp->opaque;
    this->out = (Edge*)out->opaque;
}

constexpr int bsz = 32;

template<typename T>
__global__ void sigmoid_forward(const T* a, T* dst, size_t sz);

template<>
__global__ void sigmoid_forward<float>(const float* a, float* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = (tanhf(0.5 * a[i]) + 1) / 2;
}

template<>
__global__ void sigmoid_forward<double>(const double* a, double* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = (tanh(0.5 * a[i]) + 1) / 2;
}

template<typename T>
__global__ void tanh_forward(const T* a, T* dst, size_t sz);

template<>
__global__ void tanh_forward<float>(const float* a, float* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = tanhf(a[i]);
}

template<>
__global__ void tanh_forward<double>(const double* a, double* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = tanh(a[i]);
}

template<typename T>
__global__ void relu_forward(const T* a, T* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = max(a[i], (T)0);
}

void Sigmoid::forward(RunId id)
{
    inp->forward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        sigmoid_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)out->forward_data, sz);
        break;
    case core::tensor_dty::F64:
        sigmoid_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)out->forward_data, sz);
        break;
    }
    out->forward_id = id;
}

void Tanh::forward(RunId id)
{
    inp->forward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        tanh_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)out->forward_data, sz);
        break;
    case core::tensor_dty::F64:
        tanh_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)out->forward_data, sz);
        break;
    }
    out->forward_id = id;
}

void ReLU::forward(RunId id)
{
    inp->forward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        relu_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)out->forward_data, sz);
        break;
    case core::tensor_dty::F64:
        relu_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)out->forward_data, sz);
        break;
    }
    out->forward_id = id;
}

void Sigmoid::backward(RunId id)
{
    throw GraphError("Not Implemented");
}

void Tanh::backward(RunId id)
{
    throw GraphError("Not Implemented");
}

void ReLU::backward(RunId id)
{
    throw GraphError("Not Implemented");
}
