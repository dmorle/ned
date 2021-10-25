#ifndef NEDVM_GRAPHGEN_H
#define NEDVM_GRAPHGEN_H

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>

#include <ned/core/graph.h>

namespace nn
{
    namespace nedvm
    {
        struct EdgeData
        {
            llvm::Value* forward_val = nullptr;
            llvm::Value* backward_val = nullptr;
        };

        class GraphCompiler
        {
            llvm::LLVMContext& ctx;
            llvm::IRBuilder<>& builder;
            llvm::Module mod;
            const core::Graph* pgraph;

            void initEdgeOpaque(const core::Edge* pedge);
            void delEdgeOpaque(const core::Edge* pedge);

        public:
            GraphCompiler(const core::Graph* pgraph);
            ~GraphCompiler();

            void generate_forward();
            void generate_backward();
        };
    }
}

#endif
