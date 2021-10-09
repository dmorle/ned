#include <cuned/cunodes.h>

using namespace nn;
using namespace cuda;

MatMul::MatMul(std::map<std::string, std::shared_ptr<lang::Obj>>& cargs, core::Edge* inp1, core::Edge* inp2, core::Edge* out)
{
    assert(out->opaque);
    assert(cargs["ofw"]->ty == lang::ObjType::FWIDTH);
    assert(cargs["ifw"]->ty == lang::ObjType::FWIDTH);
    assert(cargs["m"]->ty == lang::ObjType::INT);
    assert(cargs["s"]->ty == lang::ObjType::INT);
    assert(cargs["n"]->ty == lang::ObjType::INT);
    
    inp_dty = std::static_pointer_cast<lang::ObjFWidth>(cargs["ifw"])->data.dty;
    out_dty = std::static_pointer_cast<lang::ObjFWidth>(cargs["ofw"])->data.dty;

    if (inp_dty != core::tensor_dty::F32 || out_dty != core::tensor_dty::F32)
        throw GraphError("Not implemented");  // TODO: implement other float widths

    m = std::static_pointer_cast<lang::ObjInt>(cargs["m"])->data.val;
    s = std::static_pointer_cast<lang::ObjInt>(cargs["s"])->data.val;
    n = std::static_pointer_cast<lang::ObjInt>(cargs["n"])->data.val;
    if (!inp1->opaque)
    {
        inp1->opaque = new Edge{};
        ((Edge*)inp1->opaque)->sz = m * s * core::dtype_size(inp_dty);
        cudaMalloc(&((Edge*)inp1->opaque)->forward_data, ((Edge*)inp1->opaque)->sz);
        cudaMalloc(&((Edge*)inp1->opaque)->backward_data, ((Edge*)inp1->opaque)->sz);
    }
    if (!inp2->opaque)
    {
        inp2->opaque = new Edge{};
        ((Edge*)inp2->opaque)->sz = s * n * fwidth_size(inp_dty);
        cudaMalloc(&((Edge*)inp2->opaque)->forward_data, ((Edge*)inp2->opaque)->sz);
        cudaMalloc(&((Edge*)inp2->opaque)->backward_data, ((Edge*)inp2->opaque)->sz);
    }
    this->inp1 = (Edge*)inp1->opaque;
    this->inp2 = (Edge*)inp2->opaque;
    this->out = (Edge*)out->opaque;
}

constexpr int bsz = 32;

template<typename T>
__global__ void matmul(const T* A, const T* B, T* C, size_t m, size_t s, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = idx / n;
    size_t j = idx % n;
    if (i < m)
    {
        C[idx] = 0;
        for (size_t k = 0; k < s; k++)
            C[idx] += A[i * s + k] * B[k * n + j];
    }
}

template<typename T>
__global__ void matmul_back_a(T* A, const T* B, const T* C, size_t m, size_t s, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = idx / s;
    size_t j = idx % s;
    if (i < m)
    {
        for (size_t k = 0; k < n; k++)
            A[idx] += C[i * n + k] * B[j * n + k];
    }
}

template<typename T>
__global__ void matmul_back_b(const T* A, T* B, const T* C, size_t m, size_t s, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = idx / n;
    size_t j = idx % n;
    if (i < s)
    {
        for (size_t k = 0; k < m; k++)
            B[idx] += A[k * s + i] * C[k * n + j];
    }
}

void MatMul::forward(RunId id)
{
    inp1->forward(id);
    inp2->forward(id);
    matmul<<<(m * n + bsz - 1) / bsz, bsz>>>((float*)inp1->forward_data, (float*)inp2->forward_data, (float*)out->forward_data, m, s, n);
    out->forward_id = id;
}

void MatMul::backward(RunId id)
{
    out->backward(id);
    matmul_back_a<<<(m * s + bsz - 1) / bsz, bsz>>>((float*)inp1->backward_data, (float*)inp2->forward_data, (float*)out->backward_data, m, s, n);
    matmul_back_b<<<(s * n + bsz - 1) / bsz, bsz>>>((float*)inp1->forward_data, (float*)inp2->backward_data, (float*)out->backward_data, m, s, n);
    inp1->backward_id = id;
    inp2->backward_id = id;
}
