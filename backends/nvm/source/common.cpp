#include <nvm/common.h>
#include <ned/util/libs.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace nvm
{

    NodeImplMap::NodeImplMap(std::map<std::string, std::vector<NodeImpl>>& node_impls) :
        node_impls(node_impls) {}

    void NodeImplMap::insert(const std::string& name, const NodeImpl& impl)
    {
        node_impls[name].push_back(impl);
    }

    llvm::Type* get_fptype(llvm::LLVMContext& ctx, nn::core::EdgeFty fty)
    {
        switch (fty)
        {
        case nn::core::EdgeFty::F16:
            return llvm::Type::getHalfTy(ctx);
        case nn::core::EdgeFty::F32:
            return llvm::Type::getFloatTy(ctx);
        case nn::core::EdgeFty::F64:
            return llvm::Type::getDoubleTy(ctx);
        }
        assert(false);
        return nullptr;
    }

    void find_plugins(std::map<std::string, std::vector<NodeImpl>>& node_map, std::vector<nn::util::Library*>& libs)
    {
        using namespace nn::util;
        NodeImplMap impl_map(node_map);

        for (const auto& entry : fs::directory_iterator("./extensions"))
        {
            if (!entry.is_regular_file())
                continue;
            const auto& pth = entry.path();
            if (pth.extension() != L".dll")
                continue;
            // ned/util/* doesn't use ned/errors.h to register errors
            // so I don't need to worry about clearing out the error buffer if any of the functions fail
            Library* lib;
            if (lib_new(lib, pth.string()))
                continue;
            std::function<bool(error_fn_t, NodeImplMap&)> init_fn;
            if (lib_load_symbol(lib, NVM_ENTRY_FN_STR, init_fn))
                goto on_error;
            if (init_fn(nn::error::graph, impl_map))
                goto on_error;
            // I need to keep the libraries loaded util the end of compilation since the function pointers
            // pulled from the libraries will be called throughout compilation, and freeing the library now
            // would invalidate those pointers
            libs.push_back(lib);
            continue;

        on_error:
            lib_del(lib);  // nothing I can do if this fails
        }
    }

}
