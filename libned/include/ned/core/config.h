#ifndef NED_CONFIG_H
#define NED_CONFIG_H

#include <vector>
#include <string>

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

        bool operator==(const EdgeInfo& ei1, const EdgeInfo& ei2);

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
            Config(ConfigType ty) : ty(ty) {}
            virtual ~Config() {}
        };

        struct BoolConfig :
            public Config
        {
            bool val;
            BoolConfig(bool val) : Config(ConfigType::BOOL), val(val) {}
        };

        struct FtyConfig :
            public Config
        {
            EdgeFty val;
            FtyConfig(EdgeFty val) : Config(ConfigType::FTY), val(val) {}
        };

        struct IntConfig :
            public Config
        {
            int64_t val;
            IntConfig(int64_t val) : Config(ConfigType::INT), val(val) {}
        };

        struct FloatConfig :
            public Config
        {
            double val;
            FloatConfig(double val) : Config(ConfigType::FLOAT), val(val) {}
        };

        struct StringConfig :
            public Config
        {
            std::string val;
            StringConfig(const std::string& val) : Config(ConfigType::STRING), val(val) {}
        };

        struct ListConfig :
            public Config
        {
            std::vector<Config*> val;
            ListConfig(const std::vector<Config*>& val) : Config(ConfigType::LIST), val(val) {}
            ~ListConfig() { for (Config* cfg : val) delete cfg; }
        };
    }
}

#endif
