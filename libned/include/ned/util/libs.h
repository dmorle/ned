#ifndef NED_UTIL_LIBS_H
#define NED_UTIL_LIBS_H

#include <string>
#include <functional>

namespace nn
{
    namespace util
    {
        struct Library;

        bool lib_new(Library*& lib, const std::string& name);
        bool lib_del(Library* lib);

        bool lib_load_symbol(Library* lib, const std::string& name, void*& data);

        template<typename Ret, typename... Args>
        bool lib_load_symbol(Library* lib, const std::string& name, std::function<Ret(Args...)>& proc)
        {
            // This should compile into literally nothing.  Its just to satisfy typing
            void* data;
            if (lib_load_symbol(lib, name, data))
                return true;
            proc = reinterpret_cast<Ret(*)(Args...)>(data);
            return false;
        }
    }
}

#endif
