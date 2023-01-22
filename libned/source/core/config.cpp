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

        Config Config::make_null()
        {
            Config ret;
            ret.ty = Tag::NUL;
            return ret;
        }

        Config Config::make_bool(bool val)
        {
            Config ret;
            new (&ret.val_bool) decltype(ret.val_bool)(val);
            ret.ty = Tag::BOOL;
            return ret;
        }

        Config Config::make_fty(EdgeFty val)
        {
            Config ret;
            new (&ret.val_fty) decltype(ret.val_fty)(val);
            ret.ty = Tag::FTY;
            return ret;
        }

        Config Config::make_int(int64_t val)
        {
            Config ret;
            new (&ret.val_int) decltype(ret.val_int)(val);
            ret.ty = Tag::INT;
            return ret;
        }

        Config Config::make_float(double val)
        {
            Config ret;
            new (&ret.val_float) decltype(ret.val_float)(val);
            ret.ty = Tag::FLOAT;
            return ret;
        }

        Config Config::make_str(const std::string& val)
        {
            Config ret;
            new (&ret.val_str) decltype(ret.val_str)(val);
            ret.ty = Tag::STRING;
            return ret;
        }

        Config Config::make_list(const std::vector<Config>& val)
        {
            Config ret;
            new (&ret.val_list) decltype(ret.val_list)(val);
            ret.ty = Tag::LIST;
            return ret;
        }

        Config::Config(const Config& val)
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

        Config::Config(Config&& val) noexcept
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

        Config& Config::operator=(const Config& val)
        {
            if (&val == this)
                return *this;
            this->~Config();
            new (this) Config(val);
            return *this;
        }

        Config& Config::operator=(Config&& val) noexcept
        {
            if (&val == this)
                return *this;
            this->~Config();
            new (this) Config(std::move(val));
            return *this;
        }

        Config::~Config()
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

#pragma warning(push)
#pragma warning(disable: 5232)  // warning: == is recursive (I already know, it was designed that way...)
        bool operator==(const Config& lhs, const Config& rhs)
        {
            if (lhs.ty != rhs.ty)
                return false;

            switch (lhs.ty)
            {
            case Config::Tag::INVALID:
                return false;
            case Config::Tag::NUL:
                return true;
            case Config::Tag::BOOL:
                return lhs.val_bool == rhs.val_bool;
            case Config::Tag::FTY:
                return lhs.val_fty == rhs.val_fty;
            case Config::Tag::INT:
                return lhs.val_int == rhs.val_int;
            case Config::Tag::FLOAT:
                return lhs.val_float == rhs.val_float;
            case Config::Tag::STRING:
                return lhs.val_str == rhs.val_str;
            case Config::Tag::LIST:
                if (lhs.val_list.size() != rhs.val_list.size())
                    return false;
                for (size_t i = 0; i < lhs.val_list.size(); i++)
                    if (lhs.val_list[i] != rhs.val_list[i])
                        return false;
                return true;
            }
            assert(false);
            return false;
        }
#pragma warning(pop)
    }
}
