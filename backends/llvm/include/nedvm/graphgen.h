#ifndef NEDVM_GRAPHGEN_H
#define NEDVM_GRAPHGEN_H

#include <nedvm/common.h>
#include <ned/core/graph.h>

namespace nn
{
    namespace nedvm
    {
        struct EdgeData  // POD
        {
            llvm::GlobalVariable* forward_val = nullptr;
            llvm::GlobalVariable* backward_val = nullptr;
        };

        struct NodeData  // POD
        {
            llvm::Function* forward_func = nullptr;
            llvm::Function* backward_func = nullptr;
        };

        using Builder = llvm::IRBuilder<>;

        class GraphCompiler
        {
            llvm::LLVMContext* pctx;
            llvm::Module* pmod;
            Builder* pbuilder;
            const core::Graph* pgraph;

            void init_edge_opaque(const core::Edge* pedge);
            void init_node_opaque(const core::Node* pnode);

            void del_edge_opaque(const core::Edge* pedge);
            void del_node_opaque(const core::Node* pnode);

            void generate_forward_edge(const core::Edge* pnode);
            void generate_backward_edge(const core::Edge* pedge);

            void generate_forward_node(const core::Node* pnode);
            void generate_backward_node(const core::Node* pnode);

        public:
            GraphCompiler(const core::Graph* pgraph);
            GraphCompiler(const GraphCompiler&) = delete;
            GraphCompiler(GraphCompiler&&) = delete;
            GraphCompiler& operator=(const GraphCompiler&) = delete;
            GraphCompiler& operator=(GraphCompiler&&) = delete;
            ~GraphCompiler();

            void generate_forward();
            void generate_backward();

            void print();
        };
    }
}

#endif
