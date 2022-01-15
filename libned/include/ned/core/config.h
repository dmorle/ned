#ifndef NED_CONFIG_H
#define NED_CONFIG_H

#include <vector>
#include <string>
#include <memory>

namespace nn
{
    namespace core
    {
        enum class EdgeFty
        {
            F16,
            F32,
            F64
        };

        struct EdgeInfo
        {
            EdgeFty fty = EdgeFty::F32;
            std::vector<size_t> dims;
        };

        size_t fty_size(EdgeFty fty);
        bool fty_str(EdgeFty fty, std::string& str);

        enum class ConfigType
        {
            FTY,
            BOOL,
            INT,
            FLOAT,
            STRING,
            LIST
        };

        struct Config
        {
            ConfigType ty;
        };

        struct FtyConfig :
            public Config
        {
            EdgeFty val;
        };

        struct BoolConfig :
            public Config
        {
            bool val;
        };

        struct IntConfig :
            public Config
        {
            int64_t val;
        };

        struct FloatConfig :
            public Config
        {
            double val;
        };

        struct StringConfig :
            public Config
        {
            std::string val;
        };

        struct ListConfig :
            public Config
        {
            std::vector<std::unique_ptr<Config>> val;
        };
    }
}

#endif
