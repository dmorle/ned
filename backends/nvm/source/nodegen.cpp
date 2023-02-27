#include <nvm/nodegen.h>
#include <ned/util/libs.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace nvm
{

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

}
