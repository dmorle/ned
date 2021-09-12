#include <cuned/cunodes.h>

#include <cuda_runtime.h>

using namespace nn;
using namespace cuda;

constexpr int bsz = 32;

// Compile time recursion = manual recursion
#define dispatch_binop(binop, dty1, dty2, dty3, arg1, arg2, arg3, arg4)                                                                            \
    switch (dty1)                                                                                                                                  \
    {                                                                                                                                              \
    case core::tensor_dty::F16:                                                                                                                    \
        switch (dty2)                                                                                                                              \
        {                                                                                                                                          \
        case core::tensor_dty::F16:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F16, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F16, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F16, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        case core::tensor_dty::F32:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F32, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F32, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F32, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        case core::tensor_dty::F64:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F64, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F64, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F16, core::tensor_dty::F64, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        }                                                                                                                                          \
        break;                                                                                                                                     \
    case core::tensor_dty::F32:                                                                                                                    \
        switch (dty2)                                                                                                                              \
        {                                                                                                                                          \
        case core::tensor_dty::F16:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F16, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F16, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F16, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        case core::tensor_dty::F32:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F32, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F32, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F32, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        case core::tensor_dty::F64:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F64, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F64, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F32, core::tensor_dty::F64, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        }                                                                                                                                          \
        break;                                                                                                                                     \
    case core::tensor_dty::F64:                                                                                                                    \
        switch (dty2)                                                                                                                              \
        {                                                                                                                                          \
        case core::tensor_dty::F16:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F16, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F16, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F16, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        case core::tensor_dty::F32:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F32, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F32, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F32, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        case core::tensor_dty::F64:                                                                                                                \
            switch (dty3)                                                                                                                          \
            {                                                                                                                                      \
            case core::tensor_dty::F16:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F64, core::tensor_dty::F16><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F32:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F64, core::tensor_dty::F32><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            case core::tensor_dty::F64:                                                                                                            \
                binop<core::tensor_dty::F64, core::tensor_dty::F64, core::tensor_dty::F64><<<(sz + bsz - 1) / bsz, bsz>>>(arg1, arg2, arg3, arg4); \
                break;                                                                                                                             \
            }                                                                                                                                      \
            break;                                                                                                                                 \
        }                                                                                                                                          \
        break;                                                                                                                                     \
    }

template<typename INP1, typename INP2, typename OUT>
__global__ void add_pointwise(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_pointwise(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_pointwise(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[i];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_pointwise(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[i];
}

void AddSame::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    dispatch_binop(add_pointwise, inp1_dty, inp2_dty, out_dty, out->data, data1, data2, sz);
    out->id = id;
}

void SubSame::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    dispatch_binop(sub_pointwise, inp1_dty, inp2_dty, out_dty, out->data, data1, data2, sz);
    out->id = id;
}

void MulSame::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    dispatch_binop(mul_pointwise, inp1_dty, inp2_dty, out_dty, out->data, data1, data2, sz);
    out->id = id;
}

void DivSame::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    dispatch_binop(div_pointwise, inp1_dty, inp2_dty, out_dty, out->data, data1, data2, sz);
    out->id = id;
}

template<typename INP1, typename INP2, typename OUT>
__global__ void add_scalar(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void sub_scalar(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void mul_scalar(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[0];
}

template<typename INP1, typename INP2, typename OUT>
__global__ void div_scalar(OUT* dst, const INP1* a, const INP2* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[0];
}

void AddScalar::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    dispatch_binop(add_scalar, inp_dty, val_dty, out_dty, out->data, data, scalar, sz);
    out->id = id;
}

void SubScalar::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    dispatch_binop(sub_scalar, inp_dty, val_dty, out_dty, out->data, data, scalar, sz);
    out->id = id;
}

void MulScalar::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    dispatch_binop(mul_scalar, inp_dty, val_dty, out_dty, out->data, data, scalar, sz);
    out->id = id;
}

void DivScalar::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    dispatch_binop(div_scalar, inp_dty, val_dty, out_dty, out->data, data, scalar, sz);
    out->id = id;
}
