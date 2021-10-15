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

template<typename T> __device__ T tanh_g(T);
template<> __device__ float tanh_g<float>(float x) { return tanhf(x); }
template<> __device__ double tanh_g<double>(double x) { return tanh(x); }

template<typename T> __device__ T ln_g(T);
template<> __device__ float ln_g<float>(float x) { return logf(x); }
template<> __device__ double ln_g<double>(double x) { return log(x); }

template<typename T> __device__ T sigmoid_g(T x) { return (tanh_g(0.5 * x) + 1) / 2; }

template<typename T>
__global__ void sigmoid_forward(const T* a, T* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = sigmoid_g(a[i]);
}

template<typename T>
__global__ void sigmoid_backward(const T* inp_forward, T* inp, const T* grad, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        dst[i] = sigmoid_g(inp_forward[i]);
        dst[i] = grad[i] * dst[i] * (1 - dst[i]);
    }
}

template<typename T>
__global__ void tanh_forward(const T* a, T* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = tanh_g(a[i]);
}

template<typename T>
__global__ void tanh_backward(const T* inp_forward, T* inp, const T* grad, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
    {
        inp[i] = tanh_g(inp_forward[i]);
        inp[i] = grad[i] * (1 - inp[i] * inp[i]);
    }
}

template<typename T>
__global__ void relu_forward(const T* a, T* dst, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = max(a[i], (T)0);
}

template<typename T>
__global__ void relu_backward(const T* inp_forward, T* inp, const T* grad, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        inp[i] = grad[i] * (inp_forward[i] > (T)0);
}

template<typename T>
__global__ void ln_forward(const T* inp, T* out, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        out[i] = ln_g(inp[i]);
}

template<typename T>
__global__ void ln_backward(const T* inp_forward, T* inp, const T* grad, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        inp[i] += grad[i] / inp_forward[i];
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

void NaturalLog::forward(RunId id)
{
    inp->forward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        ln_forward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)out->forward_data, sz);
        break;
    case core::tensor_dty::F64:
        ln_forward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)out->forward_data, sz);
        break;
    }
    out->forward_id = id;
}

void Sigmoid::backward(RunId id)
{
    out->backward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        sigmoid_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)inp->backward_data, (float*)out->backward_data, sz);
        break;
    case core::tensor_dty::F64:
        sigmoid_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)inp->backward_data, (double*)out->backward_data, sz);
        break;
    }
    inp->backward_id = id;
}

void Tanh::backward(RunId id)
{
    out->backward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        tanh_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)inp->backward_data, (float*)out->backward_data, sz);
        break;
    case core::tensor_dty::F64:
        tanh_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)inp->backward_data, (double*)out->backward_data, sz);
        break;
    }
    inp->backward_id = id;
}

void ReLU::backward(RunId id)
{
    out->backward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        relu_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)inp->backward_data, (float*)out->backward_data, sz);
        break;
    case core::tensor_dty::F64:
        relu_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)inp->backward_data, (double*)out->backward_data, sz);
        break;
    }
    inp->backward_id = id;
}

void NaturalLog::backward(RunId id)
{
    out->backward(id);
    switch (dty)
    {
    case core::tensor_dty::F32:
        ln_backward<<<(sz + bsz - 1) / bsz, bsz>>>((float*)inp->forward_data, (float*)inp->backward_data, (float*)out->backward_data, sz);
        break;
    case core::tensor_dty::F64:
        ln_backward<<<(sz + bsz - 1) / bsz, bsz>>>((double*)inp->forward_data, (double*)inp->backward_data, (double*)out->backward_data, sz);
        break;
    }
    inp->backward_id = id;
}
