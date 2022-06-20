#include <ned/core/config.h>

namespace nn
{
    namespace core
    {
        bool operator==(const EdgeInfo& ei1, const EdgeInfo& ei2)
        {
            return ei1.fty == ei2.fty && ei1.dims == ei2.dims;
        }

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

        ConfigVal ConfigVal::make_bool(bool val)
        {
            ConfigVal ret;
            new (&ret.val_bool) decltype(ret.val_bool)();
            ret.ty = Type::BOOL;
            ret.val_bool = val;
            return ret;
        }

        ConfigVal ConfigVal::make_fty(EdgeFty val)
        {
            ConfigVal ret;
            new (&ret.val_fty) decltype(ret.val_fty)();
            ret.ty = Type::FTY;
            ret.val_fty = val;
            return ret;
        }

        ConfigVal ConfigVal::make_int(int64_t val)
        {
            ConfigVal ret;
            new (&ret.val_int) decltype(ret.val_int)();
            ret.ty = Type::INT;
            ret.val_int = val;
            return ret;
        }

        ConfigVal ConfigVal::make_float(double val)
        {
            ConfigVal ret;
            new (&ret.val_float) decltype(ret.val_float)();
            ret.ty = Type::FLOAT;
            ret.val_float = val;
            return ret;
        }

        ConfigVal ConfigVal::make_str(const std::string& val)
        {
            ConfigVal ret;
            new (&ret.val_str) decltype(ret.val_str)();
            ret.ty = Type::FLOAT;
            ret.val_str = val;
            return ret;
        }

        ConfigVal ConfigVal::make_list(const std::vector<ConfigVal>& val)
        {
            ConfigVal ret;
            new (&ret.val_list) decltype(ret.val_list)();
            ret.ty = Type::FLOAT;
            ret.val_list = val;
            return ret;
        }

        ConfigVal::ConfigVal(const ConfigVal& val)
        {
            do_copy(val);
        }

        ConfigVal::ConfigVal(ConfigVal&& val) noexcept
        {
            do_move(std::move(val));
        }

        ConfigVal& ConfigVal::operator=(const ConfigVal& val)
        {
            if (&val == this)
                return *this;
            this->~ConfigVal();
            do_copy(val);
            return *this;
        }

        ConfigVal& ConfigVal::operator=(ConfigVal&& val) noexcept
        {
            if (&val == this)
                return *this;
            this->~ConfigVal();
            do_move(std::move(val));
            return *this;
        }

        ConfigVal::~ConfigVal()
        {
            switch (ty)
            {
            case Type::INVALID:
                break;
            case Type::BOOL:
                val_bool.~decltype(val_bool)();
                break;
            case Type::FTY:
                val_fty.~decltype(val_fty)();
                break;
            case Type::INT:
                val_int.~decltype(val_int)();
                break;
            case Type::FLOAT:
                val_float.~decltype(val_float)();
                break;
            case Type::STRING:
                val_str.~decltype(val_str)();
                break;
            case Type::LIST:
                val_list.~decltype(val_list)();
                break;
            }
        }

        void ConfigVal::do_copy(const ConfigVal& val)
        {
            ty = val.ty;

            switch (ty)
            {
            case Type::INVALID:
                break;
            case Type::BOOL:
                new (&val_bool) decltype(val_bool)(val.val_bool);
                break;
            case Type::FTY:
                new (&val_fty) decltype(val_fty)(val.val_fty);
                break;
            case Type::INT:
                new (&val_int) decltype(val_int)(val.val_int);
                break;
            case Type::FLOAT:
                new (&val_float) decltype(val_float)(val.val_float);
                break;
            case Type::STRING:
                new (&val_str) decltype(val_str)(val.val_str);
                break;
            case Type::LIST:
                new (&val_list) decltype(val_list)(val.val_list);
                break;
            }
        }

        void ConfigVal::do_move(ConfigVal&& val) noexcept
        {
            ty = val.ty;
            val.ty = Type::INVALID;

            switch (ty)
            {
            case Type::INVALID:
                break;
            case Type::BOOL:
                new (&val_bool) decltype(val_bool)(std::move(val.val_bool));
                break;
            case Type::FTY:
                new (&val_fty) decltype(val_fty)(std::move(val.val_fty));
                break;
            case Type::INT:
                new (&val_int) decltype(val_int)(std::move(val.val_int));
                break;
            case Type::FLOAT:
                new (&val_float) decltype(val_float)(std::move(val.val_float));
                break;
            case Type::STRING:
                new (&val_str) decltype(val_str)(std::move(val.val_str));
                break;
            case Type::LIST:
                new (&val_list) decltype(val_list)(std::move(val.val_list));
                break;
            }
        }

        ConfigType ConfigType::make_bool()
        {
            ConfigType ret;
            ret.ty = Type::BOOL;
            return ret;
        }

        ConfigType ConfigType::make_fty()
        {
            ConfigType ret;
            ret.ty = Type::FTY;
            return ret;
        }

        ConfigType ConfigType::make_int()
        {
            ConfigType ret;
            ret.ty = Type::INT;
            return ret;
        }

        ConfigType ConfigType::make_float()
        {
            ConfigType ret;
            ret.ty = Type::FLOAT;
            return ret;
        }

        ConfigType ConfigType::make_str()
        {
            ConfigType ret;
            ret.ty = Type::STRING;
            return ret;
        }

        ConfigType ConfigType::make_array(const ConfigType& val)
        {
            ConfigType ret;
            new (&ret.type_arr) decltype(ret.type_arr)();
            ret.ty = Type::ARRAY;
            *ret.type_arr = val;
            return ret;
        }

        ConfigType ConfigType::make_tuple(const std::vector<ConfigType>& val)
        {
            ConfigType ret;
            new (&ret.type_tuple) decltype(ret.type_tuple)();
            ret.ty = Type::TUPLE;
            ret.type_tuple = val;
            return ret;
        }

        ConfigType::ConfigType(const ConfigType& val)
        {
            do_copy(val);
        }

        ConfigType::ConfigType(ConfigType&& val) noexcept
        {
            do_move(std::move(val));
        }

        ConfigType& ConfigType::operator=(const ConfigType& val)
        {
            if (&val == this)
                return *this;
            this->~ConfigType();
            do_copy(val);
            return *this;
        }

        ConfigType& ConfigType::operator=(ConfigType&& val) noexcept
        {
            if (&val == this)
                return *this;
            this->~ConfigType();
            do_move(std::move(val));
            return *this;
        }

        ConfigType::~ConfigType()
        {
            switch (ty)
            {
            case Type::INVALID:
            case Type::BOOL:
            case Type::FTY:
            case Type::INT:
            case Type::FLOAT:
            case Type::STRING:
                break;
            case Type::ARRAY:
                type_arr.~decltype(type_arr)();
                break;
            case Type::TUPLE:
                type_tuple.~decltype(type_tuple)();
                break;
            }
        }

        void ConfigType::do_copy(const ConfigType& val)
        {
            ty = val.ty;

            switch (ty)
            {
            case Type::INVALID:
            case Type::BOOL:
            case Type::FTY:
            case Type::INT:
            case Type::FLOAT:
            case Type::STRING:
                break;
            case Type::ARRAY:
                new (&type_arr) decltype(type_arr)(std::make_unique<ConfigType>(*val.type_arr));
                break;
            case Type::TUPLE:
                new (&type_tuple) decltype(type_tuple)(val.type_tuple);
                break;
            }
        }

        void ConfigType::do_move(ConfigType&& val) noexcept
        {
            ty = val.ty;
            val.ty = Type::INVALID;

            switch (ty)
            {
            case Type::INVALID:
            case Type::BOOL:
            case Type::FTY:
            case Type::INT:
            case Type::FLOAT:
            case Type::STRING:
                break;
            case Type::ARRAY:
                new (&type_arr) decltype(type_arr)(std::move(val.type_arr));
                break;
            case Type::TUPLE:
                new (&type_tuple) decltype(type_tuple)(std::move(val.type_tuple));
                break;
            }
        }
    }
}
