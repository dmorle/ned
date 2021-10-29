#include <nedvm/graphgen.h>

#include <iostream>

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

        llvm::GlobalVariable* create_edge_variable(llvm::Module* pmod, const core::tensor_dsc& dsc)
        {
            static size_t edge_count = 1;

            llvm::Type* elem_ty = nullptr;
            llvm::Constant* const_val = nullptr;
            switch (dsc.dty)
            {
            case core::tensor_dty::F16:
                elem_ty = llvm::Type::getHalfTy(pmod->getContext());
                break;
            case core::tensor_dty::F32:
                elem_ty = llvm::Type::getFloatTy(pmod->getContext());
                break;
            case core::tensor_dty::F64:
                elem_ty = llvm::Type::getDoubleTy(pmod->getContext());
                break;
            }
            if (!elem_ty)
                throw GraphGenError("Invalid tensor type");

            uint64_t sz = 1;
            for (auto dim : dsc.dims)
                sz *= dim;
            llvm::ArrayType* arr_ty = llvm::ArrayType::get(elem_ty, sz);
            std::string edge_name = std::string("edge_") + std::to_string(edge_count);
            edge_count++;
            pmod->getOrInsertGlobal(edge_name, arr_ty);
            llvm::GlobalVariable* pval = pmod->getGlobalVariable(edge_name);
            pval->setConstant(false);
            pval->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
            pval->setInitializer(llvm::ConstantAggregateZero::get(arr_ty));
            return pval;
        }

        GraphCompiler::GraphCompiler(const core::Graph* pgraph)
        {
            pctx = new llvm::LLVMContext();
            pmod = new llvm::Module("mod", *pctx);
            pbuilder = new llvm::IRBuilder<>(*pctx);

            this->pgraph = pgraph;
            for (const core::Edge* pout : pgraph->outputs)
                init_edge_opaque(pout);

            pmod->print(llvm::errs(), nullptr);
        }

        GraphCompiler::~GraphCompiler()
        {
            delete pctx;
            // For some reason, delete pmod; results in a read access violation...
            delete pbuilder;

            for (const core::Edge* pout : pgraph->outputs)
                if (pout->input)  // Not an input edge, and the input node hasn't been translated yet
                    del_edge_opaque(pout);
        }

        void GraphCompiler::init_edge_opaque(const core::Edge* pout)
        {
            // Checking if the edge has already been visited
            if (pout->opaque)
                return;

            // As long as the edge isn't an input edge, dfs through the input of the node
            if (pout->input)
                for (const core::Edge* pedge : pout->input->inputs)
                    init_edge_opaque(pedge);
            
            EdgeData* edge_data = new EdgeData();
            edge_data->forward_val = create_edge_variable(pmod, pout->dsc);
            edge_data->backward_val = create_edge_variable(pmod, pout->dsc);
            pout->opaque = edge_data;
        }

        void GraphCompiler::del_edge_opaque(const core::Edge* pedge)
        {
            if (!pedge->opaque)
                return;

            for (const core::Edge* pedge : pedge->input->inputs)
                del_edge_opaque(pedge);

            delete (EdgeData*)pedge->opaque;
            pedge->opaque = nullptr;
        }
    }
}
