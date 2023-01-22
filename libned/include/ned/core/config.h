#ifndef NED_CORE_CONFIG_H
#define NED_CORE_CONFIG_H

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

        bool operator==(const EdgeInfo& ei1, const EdgeInfo& ei2);

        size_t fty_size(EdgeFty fty);
        bool fty_str(EdgeFty fty, std::string& str);

        struct Config
        {
            static Config make_null();
            static Config make_bool(bool val);
            static Config make_fty(EdgeFty val);
            static Config make_int(int64_t val);
            static Config make_float(double val);
            static Config make_str(const std::string& val);
            static Config make_list(const std::vector<Config>& val);

            enum class Tag
            {
                INVALID,
                NUL,
                BOOL,
                FTY,
                INT,
                FLOAT,
                STRING,
                LIST
            }
            ty = Tag::INVALID;

            union
            {
                bool                val_bool;
                EdgeFty             val_fty;
                int64_t             val_int;
                double              val_float;
                std::string         val_str;
                std::vector<Config> val_list;
            };

#pragma warning( push )
#pragma warning( disable : 26495 )
            Config() {}
#pragma warning ( pop )
            Config(const Config& val);
            Config(Config&& val) noexcept;
            Config& operator=(const Config& val);
            Config& operator=(Config&& val) noexcept;
            ~Config();
        };

        bool operator==(const Config& lhs, const Config& rhs);
    }
}

#endif
