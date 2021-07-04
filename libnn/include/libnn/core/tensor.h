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
        uint32_t rk = 0;
        std::vector<uint32_t> dims = {};
    };
}

#endif
