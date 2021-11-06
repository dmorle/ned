#include <nedvm/graphgen.h>
#include <nedvm/vmnodes.h>

#include <iostream>
#include <cassert>

namespace nn
{
    namespace nedvm
    {
        GraphGenError::GraphGenError(const std::string& errmsg)
        {
            this->errmsg = errmsg;
        }

        const char* GraphGenError::what() const
        {
            return errmsg.c_str();
        }

        llvm::ArrayType* get_edge_type(llvm::LLVMContext& ctx, const core::tensor_dsc& dsc)
        {
            uint64_t sz = 1;
            for (auto dim : dsc.dims)
                sz *= dim;
            return llvm::ArrayType::get(get_fptype(ctx, dsc.dty), sz);
        }

        llvm::GlobalVariable* init_edge(llvm::LLVMContext& ctx, llvm::GlobalVariable* pval, llvm::Type* edge_ty)
        {
            pval->setConstant(false);
            pval->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
            pval->setInitializer(llvm::ConstantAggregateZero::get(edge_ty));
            return pval;
        }

        GraphCompiler::GraphCompiler(const core::Graph* pgraph)
        {
            pctx = new llvm::LLVMContext();
            pbuilder = new llvm::IRBuilder<>(*pctx);
            pmod = new llvm::Module("mod", *pctx);

            this->pgraph = pgraph;
            // Allocating and initializing the opaque pointers for each of the edges and nodes
            for (const core::Edge* pout : pgraph->outputs)
                init_edge_opaque(pout);
        }

        GraphCompiler::~GraphCompiler()
        {
            delete pctx;
            delete pbuilder;
            // For some reason, delete pmod; results in a read access violation...

            // Deleting the opaque pointers for each of the edges and nodes
            for (const core::Edge* pedge : pgraph->outputs)
                del_edge_opaque(pedge);
        }

        void GraphCompiler::generate_forward_node(const core::Node* pnode)
        {
            static size_t count = 1;
            NodeData* data = (NodeData*)pnode->opaque;
            assert(data);
            if (data->forward_func)
                return;  // The forward method has already been generated for this node
            llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(pmod->getContext()), false);
            llvm::Function* func = llvm::Function::Create(func_ty, llvm::GlobalValue::LinkageTypes::InternalLinkage, "forward_node_" + std::to_string(count), pmod);

            data->forward_func = func;

            for (const auto pedge : pnode->inputs)
                generate_forward_edge(pedge);

            // Need to ensure that the edges have been allocated before generating the function
            generate_node_forward_func(pmod->getContext(), func, pbuilder, pnode);
        }

        void GraphCompiler::generate_backward_node(const core::Node* pnode)
        {
            static size_t count = 1;
            NodeData* data = (NodeData*)pnode->opaque;
            assert(data);
            if (data->backward_func)
                return;  // The backward method has already been generated for this node
            llvm::FunctionType* func_ty = llvm::FunctionType::get(llvm::Type::getVoidTy(pmod->getContext()), false);
            llvm::Function* func = llvm::Function::Create(func_ty, llvm::GlobalValue::LinkageTypes::InternalLinkage, "backward_node_" + std::to_string(count), pmod);

            data->backward_func = func;

            for (const auto pedge : pnode->inputs)
                generate_backward_edge(pedge);

            // Need to ensure that the edges have been allocated before generating the function
            generate_node_backward_func(pmod->getContext(), func, pbuilder, pnode);
        }

        void GraphCompiler::generate_forward_edge(const core::Edge* pedge)
        {
            static size_t count = 1;
            EdgeData* data = (EdgeData*)pedge->opaque;
            assert(data);
            if (data->forward_val)
                return;
            std::string edge_name = std::string("forward_edge_") + std::to_string(count++);
            auto edge_ty = get_edge_type(pmod->getContext(), pedge->dsc);
            pmod->getOrInsertGlobal(edge_name, edge_ty);
            data->forward_val = init_edge(pmod->getContext(), pmod->getGlobalVariable(edge_name), edge_ty);

            if (pedge->input)
                generate_forward_node(pedge->input);
        }

        void GraphCompiler::generate_backward_edge(const core::Edge* pedge)
        {
            static size_t count = 1;
            EdgeData* data = (EdgeData*)pedge->opaque;
            assert(data);
            if (data->backward_val)
                return;
            std::string edge_name = std::string("backward_edge_") + std::to_string(count++);
            auto edge_ty = get_edge_type(pmod->getContext(), pedge->dsc);
            pmod->getOrInsertGlobal(edge_name, edge_ty);
            data->backward_val = init_edge(pmod->getContext(), pmod->getGlobalVariable(edge_name), edge_ty);

            if (pedge->input)
                generate_backward_node(pedge->input);
        }

        void GraphCompiler::init_edge_opaque(const core::Edge* pedge)
        {
            // Checking if the edge has already been visited
            if (pedge->opaque)
                return;

            // Initializing the opaque pointer
            pedge->opaque = new EdgeData();

            // As long as the edge isn't an input edge, dfs through the input node
            if (pedge->input)
                init_node_opaque(pedge->input);
        }

        void GraphCompiler::init_node_opaque(const core::Node* pnode)
        {
            if (pnode->opaque)
                return;
            pnode->opaque = new NodeData();

            for (const core::Edge* pedge : pnode->inputs)
                init_edge_opaque(pedge);
        }

        void GraphCompiler::del_edge_opaque(const core::Edge* pedge)
        {
            // Checking if the edge has already been visited
            if (!pedge->opaque)
                return;
            delete (EdgeData*)pedge->opaque;
            pedge->opaque = nullptr;  // I have to assign it to null to prevent a double delete

            if (pedge->input)
                del_node_opaque(pedge->input);
        }

        void GraphCompiler::del_node_opaque(const core::Node* pnode)
        {
            if (!pnode->opaque)
                return;
            delete (NodeData*)pnode->opaque;
            pnode->opaque = nullptr;  // I have to assign it to null to prevent a double delete

            for (const core::Edge* pedge : pnode->inputs)
                del_edge_opaque(pedge);
        }

        void GraphCompiler::generate_forward()
        {
            for (const core::Edge* pedge : pgraph->outputs)
                generate_forward_edge(pedge);

            // TODO: Generate the top level forward method
        }

        void GraphCompiler::generate_backward()
        {
            for (const core::Edge* pedge : pgraph->outputs)
                generate_backward_edge(pedge);

            // TODO: Generate the top level backward method
        }

        void GraphCompiler::print()
        {
            pmod->print(llvm::errs(), nullptr);
        }
    }
}
