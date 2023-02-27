#ifndef NVM_NODEGEN_H
#define NVM_NODEGEN_H

#pragma warning(push, 0)

#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/BitCode/BitcodeWriter.h>

#pragma warning(pop)

#include <ned/errors.h>
#include <ned/util/libs.h>
#include <ned/core/graph.h>
#include <ned/core/config.h>
#include <ned/core/reduce.h>

#include <map>
#include <concepts>
#include <type_traits>

namespace nvm
{

    using Builder = llvm::IRBuilder<>;

    struct NodeCtx
    {
        nn::core::MdNodeRef node;
        const nn::core::MdGraph* graph;
    };

    class GraphCompiler;
    struct CompCtx
    {
        GraphCompiler* comp;
        llvm::Module* mod;
        Builder* builder;
        llvm::Function* func;
    };

    llvm::Type* get_fptype(llvm::LLVMContext& ctx, nn::core::EdgeFty fty);
    llvm::Value* get_fpval(llvm::LLVMContext& ctx, nn::core::EdgeFty fty, double val);

}

#endif
