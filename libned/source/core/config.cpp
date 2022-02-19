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

        Config* BoolConfig::clone() const
        {
            return new BoolConfig(val);
        }

        Config* FtyConfig::clone() const
        {
            return new FtyConfig(val);
        }

        Config* IntConfig::clone() const
        {
            return new IntConfig(val);
        }

        Config* FloatConfig::clone() const
        {
            return new FloatConfig(val);
        }

        Config* StringConfig::clone() const
        {
            return new StringConfig(val);
        }

        ListConfig::~ListConfig()
        {
            for (Config* cfg : val)
                delete cfg;
        }

        Config* ListConfig::clone() const
        {
            std::vector<Config*> cfg{ val.size() };
            for (const auto& e : val)
                cfg.push_back(e->clone());
            return new ListConfig(cfg);
        }
    }
}
