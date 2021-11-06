#ifndef NEDVM_COMMON_H
#define NEDVM_COMMON_H

// The remains of my failed attempt at linking llvm release binaries to a debug build
#ifdef _DEBUG
#define DEBUG_WAS_DEFINED
#undef _DEBUG
#endif

#define LLVM_WARNINGS 4267 4244 4624
#pragma warning( disable : LLVM_WARNINGS )

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ADT/ArrayRef.h>

#pragma warning( default : LLVM_WARNINGS )

#ifdef DEBUG_WAS_DEFINED
#define _DEBUG
#undef DEBUG_WAS_DEFINED
#endif

#include <ned/core/graph.h>

namespace nn
{
    namespace nedvm
    {
        class GraphGenError :
            public std::exception
        {
        public:
            std::string errmsg;

            GraphGenError(const std::string& errmsg);

            const char* what() const override;
        };

        llvm::Type* get_fptype(llvm::LLVMContext& ctx, core::tensor_dty);
    }
}

#endif
