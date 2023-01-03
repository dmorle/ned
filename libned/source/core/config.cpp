#include <ned/core/config.h>

#include <cassert>

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

        ConfigVal ConfigVal::make_null()
        {
            ConfigVal ret;
            ret.ty = Tag::NUL;
            return ret;
        }

        ConfigVal ConfigVal::make_bool(bool val)
        {
            ConfigVal ret;
            new (&ret.val_bool) decltype(ret.val_bool)();
            ret.ty = Tag::BOOL;
            ret.val_bool = val;
            return ret;
        }

        ConfigVal ConfigVal::make_fty(EdgeFty val)
        {
            ConfigVal ret;
            new (&ret.val_fty) decltype(ret.val_fty)();
            ret.ty = Tag::FTY;
            ret.val_fty = val;
            return ret;
        }

        ConfigVal ConfigVal::make_int(int64_t val)
        {
            ConfigVal ret;
            new (&ret.val_int) decltype(ret.val_int)();
            ret.ty = Tag::INT;
            ret.val_int = val;
            return ret;
        }

        ConfigVal ConfigVal::make_float(double val)
        {
            ConfigVal ret;
            new (&ret.val_float) decltype(ret.val_float)();
            ret.ty = Tag::FLOAT;
            ret.val_float = val;
            return ret;
        }

        ConfigVal ConfigVal::make_str(const std::string& val)
        {
            ConfigVal ret;
            new (&ret.val_str) decltype(ret.val_str)();
            ret.ty = Tag::STRING;
            ret.val_str = val;
            return ret;
        }

        ConfigVal ConfigVal::make_list(const std::vector<ConfigVal>& val)
        {
            ConfigVal ret;
            new (&ret.val_list) decltype(ret.val_list)();
            ret.ty = Tag::LIST;
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
            case Tag::INVALID:
            case Tag::NUL:
                break;
            case Tag::BOOL:
                val_bool.~decltype(val_bool)();
                break;
            case Tag::FTY:
                val_fty.~decltype(val_fty)();
                break;
            case Tag::INT:
                val_int.~decltype(val_int)();
                break;
            case Tag::FLOAT:
                val_float.~decltype(val_float)();
                break;
            case Tag::STRING:
                val_str.~decltype(val_str)();
                break;
            case Tag::LIST:
                val_list.~decltype(val_list)();
                break;
            }
        }

        void ConfigVal::do_copy(const ConfigVal& val)
        {
            ty = val.ty;

            switch (ty)
            {
            case Tag::INVALID:
            case Tag::NUL:
                break;
            case Tag::BOOL:
                new (&val_bool) decltype(val_bool)(val.val_bool);
                break;
            case Tag::FTY:
                new (&val_fty) decltype(val_fty)(val.val_fty);
                break;
            case Tag::INT:
                new (&val_int) decltype(val_int)(val.val_int);
                break;
            case Tag::FLOAT:
                new (&val_float) decltype(val_float)(val.val_float);
                break;
            case Tag::STRING:
                new (&val_str) decltype(val_str)(val.val_str);
                break;
            case Tag::LIST:
                new (&val_list) decltype(val_list)(val.val_list);
                break;
            }
        }

        void ConfigVal::do_move(ConfigVal&& val) noexcept
        {
            ty = val.ty;
            val.ty = Tag::INVALID;

            switch (ty)
            {
            case Tag::INVALID:
            case Tag::NUL:
                break;
            case Tag::BOOL:
                new (&val_bool) decltype(val_bool)(std::move(val.val_bool));
                break;
            case Tag::FTY:
                new (&val_fty) decltype(val_fty)(std::move(val.val_fty));
                break;
            case Tag::INT:
                new (&val_int) decltype(val_int)(std::move(val.val_int));
                break;
            case Tag::FLOAT:
                new (&val_float) decltype(val_float)(std::move(val.val_float));
                break;
            case Tag::STRING:
                new (&val_str) decltype(val_str)(std::move(val.val_str));
                break;
            case Tag::LIST:
                new (&val_list) decltype(val_list)(std::move(val.val_list));
                break;
            }
        }

        ConfigType ConfigType::make_null()
        {
            ConfigType ret;
            ret.ty = Tag::NUL;
            return ret;
        }

        ConfigType ConfigType::make_bool()
        {
            ConfigType ret;
            ret.ty = Tag::BOOL;
            return ret;
        }

        ConfigType ConfigType::make_fty()
        {
            ConfigType ret;
            ret.ty = Tag::FTY;
            return ret;
        }

        ConfigType ConfigType::make_int()
        {
            ConfigType ret;
            ret.ty = Tag::INT;
            return ret;
        }

        ConfigType ConfigType::make_float()
        {
            ConfigType ret;
            ret.ty = Tag::FLOAT;
            return ret;
        }

        ConfigType ConfigType::make_str()
        {
            ConfigType ret;
            ret.ty = Tag::STR;
            return ret;
        }

        ConfigType ConfigType::make_arr(const ConfigType& val)
        {
            ConfigType ret;
            new (&ret.type_arr) decltype(ret.type_arr)();
            ret.ty = Tag::ARR;
            ret.type_arr = std::make_unique<ConfigType>(val);
            return ret;
        }

        ConfigType ConfigType::make_agg(const std::vector<ConfigType>& val)
        {
            ConfigType ret;
            new (&ret.type_agg) decltype(ret.type_agg)();
            ret.ty = Tag::AGG;
            ret.type_agg = val;
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
            case Tag::INVALID:
            case Tag::NUL:
            case Tag::BOOL:
            case Tag::FTY:
            case Tag::INT:
            case Tag::FLOAT:
            case Tag::STR:
                break;
            case Tag::ARR:
                type_arr.~decltype(type_arr)();
                break;
            case Tag::AGG:
                type_agg.~decltype(type_agg)();
                break;
            }
        }

        void ConfigType::do_copy(const ConfigType& val)
        {
            ty = val.ty;

            switch (ty)
            {
            case Tag::INVALID:
            case Tag::NUL:
            case Tag::BOOL:
            case Tag::FTY:
            case Tag::INT:
            case Tag::FLOAT:
            case Tag::STR:
                break;
            case Tag::ARR:
                new (&type_arr) decltype(type_arr)(std::make_unique<ConfigType>(*val.type_arr));
                break;
            case Tag::AGG:
                new (&type_agg) decltype(type_agg)(val.type_agg);
                break;
            }
        }

        void ConfigType::do_move(ConfigType&& val) noexcept
        {
            ty = val.ty;
            val.ty = Tag::INVALID;

            switch (ty)
            {
            case Tag::INVALID:
            case Tag::NUL:
            case Tag::BOOL:
            case Tag::FTY:
            case Tag::INT:
            case Tag::FLOAT:
            case Tag::STR:
                break;
            case Tag::ARR:
                new (&type_arr) decltype(type_arr)(std::move(val.type_arr));
                break;
            case Tag::AGG:
                new (&type_agg) decltype(type_agg)(std::move(val.type_agg));
                break;
            }
        }

#pragma warning(push)
#pragma warning(disable: 5232)  // warning: == is recursive (I already know, it was designed that way...)
        bool operator==(const ConfigType& lhs, const ConfigType& rhs)
        {
            if (lhs.ty != rhs.ty)
                return false;

            switch (lhs.ty)
            {
            case ConfigType::Tag::INVALID:
                return false;
            case ConfigType::Tag::NUL:
            case ConfigType::Tag::BOOL:
            case ConfigType::Tag::FTY:
            case ConfigType::Tag::INT:
            case ConfigType::Tag::FLOAT:
            case ConfigType::Tag::STR:
                // All non-recursive data types
                return true;
            case ConfigType::Tag::ARR:
                return *lhs.type_arr == *rhs.type_arr;
            case ConfigType::Tag::AGG:
                if (lhs.type_agg.size() != rhs.type_agg.size())
                    return false;
                for (size_t i = 0; i < lhs.type_agg.size(); i++)
                    if (lhs.type_agg[i] != rhs.type_agg[i])
                        return false;
                return true;
            }
            assert(false);
            return false;
        }
#pragma warning(pop)
    }
}
