#ifndef NN_TENSOR_H
#define NN_TENSOR_H

#include <vector>

namespace nn
{
    enum class tensor_dty
    {
        F16,
        F32,
        F64,
        I16,
        I32,
        I64,
        U16,
        U32,
        U64
    };

    struct tensor_dsc
    {
        tensor_dty dty = tensor_dty::F32;
        int32_t rk = -1;  // rk = -1 => placeholder tensor size (only valid during graph construction)
        std::vector<uint32_t> dims = {};
    };
}

#endif
