#ifndef NEDVM_VMNODES_H
#define NEDVM_VMNODES_H

#include <nedvm/common.h>
#include <nedvm/graphgen.h>

namespace nn
{
    namespace nedvm
    {
        void generate_node_forward_func(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode);
        void generate_node_backward_func(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode);

        void generate_add_same_forward(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode);
        void generate_add_same_backward(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode);
    }
}

#endif
