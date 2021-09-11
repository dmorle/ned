#ifndef NED_TENSOR_H
#define NED_TENSOR_H

#include <vector>

namespace nn
{
    namespace core
    {
        enum class tensor_dty
        {
            F16,
            F32,
            F64
        };

        size_t dtype_size(tensor_dty dty)
        {
            switch (dty)
            {
            case tensor_dty::F16:
                return 2;
            case tensor_dty::F32:
                return 4;
            case tensor_dty::F64:
                return 8;
            }
            return -1;
        }

        struct tensor_dsc
        {
            tensor_dty dty = tensor_dty::F32;
            int32_t rk = -1;  // rk = -1 => placeholder tensor size (only valid during graph construction)
            std::vector<uint32_t> dims = {};
        };
    }
}

#endif
