#ifndef NED_TENSOR_H
#define NED_TENSOR_H

#include <vector>
#include <string>

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

        size_t dtype_size(tensor_dty dty);
        bool dtype_str(tensor_dty dty, std::string& str);

        struct tensor_dsc
        {
            tensor_dty dty = tensor_dty::F32;
            int32_t rk = -1;  // rk = -1 => placeholder tensor size (only valid during graph construction)
            std::vector<uint32_t> dims = {};
        };
    }
}

#endif
