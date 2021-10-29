#ifndef NEDVM_GRAPHGEN_H
#define NEDVM_GRAPHGEN_H

#include <nedvm/common.h>
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

        struct EdgeData
        {
            llvm::Value* forward_val = nullptr;
            llvm::Value* backward_val = nullptr;
        };

        class GraphCompiler
        {
            llvm::LLVMContext* pctx;
            llvm::IRBuilder<>* pbuilder;
            llvm::Module* pmod;
            const core::Graph* pgraph;

            void init_edge_opaque(const core::Edge* pedge);
            void del_edge_opaque(const core::Edge* pedge);

        public:
            GraphCompiler(const core::Graph* pgraph);
            GraphCompiler(const GraphCompiler&) = delete;
            GraphCompiler(GraphCompiler&&) = delete;
            GraphCompiler& operator=(const GraphCompiler&) = delete;
            GraphCompiler& operator=(GraphCompiler&&) = delete;
            ~GraphCompiler();

            void generate_forward();
            void generate_backward();
        };
    }
}

#endif
