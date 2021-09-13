#include <ned/core/tensor.h>

namespace nn
{
    namespace core
    {
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
    }
}
