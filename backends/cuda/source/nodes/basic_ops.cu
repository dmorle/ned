#include <cuned/cunodes.h>

#include <cuda_runtime.h>

using namespace nn;
using namespace cuda;

constexpr int bsz = 32;

template<typename T>
__global__ void add_pointwise(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[i];
}

template<typename T>
__global__ void sub_pointwise(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[i];
}

template<typename T>
__global__ void mul_pointwise(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[i];
}

template<typename T>
__global__ void div_pointwise(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[i];
}

template<typename T>
void AddSame<T>::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    add_pointwise<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data1, data2, sz);
    out->id = id;
}

template<typename T>
void SubSame<T>::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    sub_pointwise<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data1, data2, sz);
    out->id = id;
}

template<typename T>
void MulSame<T>::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    mul_pointwise<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data1, data2, sz);
    out->id = id;
}

template<typename T>
void DivSame<T>::eval(RunId id)
{
    void* data1 = inp1->get_data(id);
    void* data2 = inp2->get_data(id);
    div_pointwise<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data1, data2, sz);
    out->id = id;
}

template<typename T>
__global__ void add_scalar(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b[0];
}

template<typename T>
__global__ void sub_scalar(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b[0];
}

template<typename T>
__global__ void mul_scalar(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b[0];
}

template<typename T>
__global__ void div_scalar(T* dst, const T* a, const T* b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b[0];
}

template<typename T>
void AddScalar<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    add_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, scalar, sz);
    out->id = id;
}

template<typename T>
void SubScalar<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    sub_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, scalar, sz);
    out->id = id;
}

template<typename T>
void MulScalar<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    mul_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, scalar, sz);
    out->id = id;
}

template<typename T>
void DivScalar<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    void* scalar = val->get_data(id);
    div_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, scalar, sz);
    out->id = id;
}

template<typename T>
__global__ void add_const(T* dst, const T* a, T b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] + b;
}

template<typename T>
__global__ void sub_const(T* dst, const T* a, T b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] - b;
}

template<typename T>
__global__ void mul_const(T* dst, const T* a, T b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] * b;
}

template<typename T>
__global__ void div_const(T* dst, const T* a, T b, size_t sz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz)
        dst[i] = a[i] / b;
}

template<typename T>
void AddConst<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    add_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, val, sz);
    out->id = id;
}

template<typename T>
void SubConst<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    sub_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, val, sz);
    out->id = id;
}

template<typename T>
void MulConst<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    mul_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, val, sz);
    out->id = id;
}

template<typename T>
void DivConst<T>::eval(RunId id)
{
    void* data = inp->get_data(id);
    div_scalar<T><<<(sz + bsz - 1) / bsz, bsz>>>(out->data, data, val, sz);
    out->id = id;
}
