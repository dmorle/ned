#include <nedvm/vmnodes.h>

#include <iostream>

#define FNV_PRIME 0x00000100000001B3ULL
#define FNV_OFFSET_BASIS 0XCBF29CE484222325ULL

constexpr size_t hash(const char* s)
{
    size_t h = FNV_OFFSET_BASIS;
    for (const char* c = s; *c; c++)
        h = (h * FNV_PRIME) ^ *c;
    return h;
}

constexpr size_t hash(const std::string& s)
{
    return hash(s.c_str());
}

namespace nn
{
    namespace nedvm
    {
        void generate_node_forward_func(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode)
        {
            switch (hash(pnode->name))
            {
            case hash("add_same_intr"):
                if (pnode->name != "add_same_intr") throw GraphGenError("Unrecognized graph intrinsic name: " + pnode->name);
                generate_add_same_forward(ctx, func, pbuilder, pnode);
                break;
            default:
                throw GraphGenError("Unrecognized graph intrinsic name: " + pnode->name);
            }
            if (llvm::verifyFunction(*func, &llvm::errs()))
                throw GraphGenError("Code generation error");
        }

        void generate_node_backward_func(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode)
        {
            switch (hash(pnode->name))
            {
            case hash("add_same_intr"):
                if (pnode->name != "add_same_intr") throw GraphGenError("Unrecognized graph intrinsic name: " + pnode->name);
                generate_add_same_backward(ctx, func, pbuilder, pnode);
                break;
            default:
                throw GraphGenError("Unrecognized graph intrinsic name: " + pnode->name);
            }
            if (llvm::verifyFunction(*func, &llvm::errs()))
                throw GraphGenError("Code generation error");
        }
    }
}
