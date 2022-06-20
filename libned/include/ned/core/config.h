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

        struct ConfigVal
        {
            static ConfigVal make_bool(bool val);
            static ConfigVal make_fty(EdgeFty val);
            static ConfigVal make_int(int64_t val);
            static ConfigVal make_float(double val);
            static ConfigVal make_str(const std::string& val);
            static ConfigVal make_list(const std::vector<ConfigVal>& val);

            enum class Type
            {
                INVALID,
                FTY,
                BOOL,
                INT,
                FLOAT,
                STRING,
                LIST
            }
            ty = Type::INVALID;

            union
            {
                bool                   val_bool;
                EdgeFty                val_fty;
                int64_t                val_int;
                double                 val_float;
                std::string            val_str;
                std::vector<ConfigVal> val_list;
            };

            ConfigVal() {}
            ConfigVal(const ConfigVal& val);
            ConfigVal(ConfigVal&& val) noexcept;
            ConfigVal& operator=(const ConfigVal& val);
            ConfigVal& operator=(ConfigVal&& val) noexcept;
            ~ConfigVal();

        private:
            void do_move(ConfigVal&& val) noexcept;
            void do_copy(const ConfigVal& val);
        };

        struct ConfigType
        {
            static ConfigType make_bool();
            static ConfigType make_fty();
            static ConfigType make_int();
            static ConfigType make_float();
            static ConfigType make_str();
            static ConfigType make_array(const ConfigType& val);
            static ConfigType make_tuple(const std::vector<ConfigType>& val);

            enum class Type
            {
                INVALID,
                BOOL,
                FTY,
                INT,
                FLOAT,
                STRING,
                ARRAY,
                TUPLE
            }
            ty = Type::INVALID;

            union
            {
                std::unique_ptr<ConfigType> type_arr;
                std::vector<ConfigType>     type_tuple;
            };

            ConfigType() {}
            ConfigType(const ConfigType& val);
            ConfigType(ConfigType&& val) noexcept;
            ConfigType& operator=(const ConfigType& val);
            ConfigType& operator=(ConfigType&& val) noexcept;
            ~ConfigType();

        private:
            void do_move(ConfigType&& val) noexcept;
            void do_copy(const ConfigType& val);
        };
    }
}

#endif
