#include <ned/core/setup.h>
#include <ned/util/libs.h>

#include <vector>
#include <memory>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;

namespace nn
{
    namespace core
    {
        struct InitFnInfo
        {
            std::string fn_name;
            std::function<InitGenResult(const std::vector<std::unique_ptr<Config>>&, void*&)> init_gen;
        };

        struct InitLibInfo
        {
            std::string name;
            util::Library* lib;
            std::vector<InitFnInfo> fns;
        };

        std::vector<InitLibInfo> init_libs;

        void load_inits(std::vector<std::string> search_paths)
        {
            for (const auto& pth : search_paths)
                for (const auto& dir : fs::directory_iterator(pth))
                {
                    if (!dir.is_regular_file() || dir.path().extension().string() != "init")
                        continue;

                    // TODO: Error handling
                    InitLibInfo lib_info{ .name = dir.path().string() };
                    util::lib_new(lib_info.lib, lib_info.name);
                    std::function<bool(std::vector<InitFnInfo>&)> proc;
                    util::lib_load_symbol(lib_info.lib, "init_info", proc);
                    proc(lib_info.fns);
                    init_libs.push_back(lib_info);
                }
        }
    }
}
