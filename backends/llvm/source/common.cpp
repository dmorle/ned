#include <nedvm/common.h>

namespace nn
{
    namespace nedvm
    {
        llvm::Type* get_fptype(llvm::LLVMContext& ctx, core::tensor_dty dty)
        {
            llvm::Type* elem_ty = nullptr;
            switch (dty)
            {
            case core::tensor_dty::F16:
                elem_ty = llvm::Type::getHalfTy(ctx);
                break;
            case core::tensor_dty::F32:
                elem_ty = llvm::Type::getFloatTy(ctx);
                break;
            case core::tensor_dty::F64:
                elem_ty = llvm::Type::getDoubleTy(ctx);
                break;
            }
            if (!elem_ty)
                throw GraphGenError("Invalid tensor type");
        }
    }
}
