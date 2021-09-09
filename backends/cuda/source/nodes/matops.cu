#include <cuned/cunodes.h>

using namespace nn;
using namespace cuda;

constexpr int bsz = 32;

template<typename T>
__global__ void matmul(T* dst, const T* A, const T* B)
{
}


