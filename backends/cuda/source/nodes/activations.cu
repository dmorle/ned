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
        // TODO: check to make sure it was allocated
        cudaMalloc(&((Edge*)inp->opaque)->forward_data, sz * core::dtype_size(inp->dsc.dty));
        cudaMalloc(&((Edge*)inp->opaque)->backward_data, sz * core::dtype_size(inp->dsc.dty));
    }

    this->inp = (Edge*)inp->opaque;
    this->out = (Edge*)out->opaque;
}

constexpr int bsz = 32;

template<typename T>
__global__ void sigmoid_forward(const T* a, T* dst, size_t sz);

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

void Sigmoid::forward(RunId id)
{
    throw GraphError("Not Implemented");
}

void Tanh::forward(RunId id)
{
    throw GraphError("Not Implemented");
}

void ReLU::forward(RunId id)
{
    throw GraphError("Not Implemented");
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
