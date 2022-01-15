#include <ned/core/config.h>

namespace nn
{
    namespace core
    {
        size_t fty_size(EdgeFty fty)
        {
            switch (fty)
            {
            case EdgeFty::F16:
                return 2;
            case EdgeFty::F32:
                return 4;
            case EdgeFty::F64:
                return 8;
            }
            return -1;
        }

        bool fty_str(EdgeFty fty, std::string& str)
        {
            switch (fty)
            {
            case EdgeFty::F16:
                str = "f16";
                return false;
            case EdgeFty::F32:
                str = "f32";
                return false;
            case EdgeFty::F64:
                str = "f64";
                return false;;
            }
            return true;
        }
    }
}
