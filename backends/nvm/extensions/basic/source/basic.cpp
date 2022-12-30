#include <ned/core/config.h>
#include <basic.h>

#include <iostream>
#include <map>

nvm::error_fn_t error = nullptr;

template<class Impl>
void register_node_impl(nvm::NodeImplMap& nodes)
{ nodes.insert(Impl::name, { &Impl::match, &Impl::compile }); }

NVM_INIT(error_fn, nodes)
{
	error = error_fn;
	register_node_impl<AddImpl>(nodes);
	return false;
}

const std::vector<std::pair<std::string, nn::core::ConfigType>> AddImpl::cargs = {
	{ "fp", nn::core::ConfigType::make_fty() },
	{ "shape", nn::core::ConfigType::make_arr(nn::core::ConfigType::make_int()) }
};

bool AddImpl::compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& comp_ctx)
{
    // TODO: handle edge views
    auto& node = node_ctx.graph->get(node_ctx.node);
    assert(node.inps.size() == 2);
    assert(node.outs.size() == 1);
    const auto& lhs = comp_ctx.comp->inp_edge(node, 0);
    const auto& rhs = comp_ctx.comp->inp_edge(node, 1);
    const auto& out = comp_ctx.comp->out_edge(node, 0);
    
    llvm::GlobalVariable* lvec = lhs.var;
    llvm::GlobalVariable* rvec = rhs.var;
    llvm::GlobalVariable* ovec = out.var;

    size_t sz = 1;
    for (auto& e : node.configs.at("shape").val.val_list)
    {
        assert(e.ty == nn::core::ConfigVal::Tag::INT);
        sz *= e.val_int;
    }

    auto& ctx = comp_ctx.mod->getContext();
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", comp_ctx.func);
    llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx, "loop", comp_ctx.func);
    llvm::BasicBlock* end = llvm::BasicBlock::Create(ctx, "end", comp_ctx.func);

    llvm::Type* int_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Value* zero_val = llvm::ConstantInt::get(int_ty, 0);
    llvm::Value* end_val = llvm::ConstantInt::get(int_ty, sz);
    llvm::Value* step_val = llvm::ConstantInt::get(int_ty, 1);

    comp_ctx.builder->SetInsertPoint(entry);
    comp_ctx.builder->CreateBr(loop);

    comp_ctx.builder->SetInsertPoint(loop);
    llvm::PHINode* idx_val = comp_ctx.builder->CreatePHI(int_ty, 2, "idx");

    // start doing the vector addition
    llvm::Type* fp_type = nvm::get_fptype(ctx, node.configs.at("fp").val.val_fty);
    llvm::Value* lptr_val = comp_ctx.builder->CreateGEP(fp_type, lvec, { idx_val }, "lptr");
    llvm::Value* rptr_val = comp_ctx.builder->CreateGEP(fp_type, rvec, { idx_val }, "rptr");
    llvm::Value* optr_val = comp_ctx.builder->CreateGEP(fp_type, ovec, { idx_val }, "optr");

    llvm::Value* lval_val = comp_ctx.builder->CreateLoad(fp_type, lptr_val, "lval");
    llvm::Value* rval_val = comp_ctx.builder->CreateLoad(fp_type, rptr_val, "rval");
    llvm::Value* oval_val = comp_ctx.builder->CreateFAdd(lval_val, rval_val, "oval");
    comp_ctx.builder->CreateStore(oval_val, optr_val);
    // finished the vector addition

    llvm::Value* next_val = comp_ctx.builder->CreateAdd(idx_val, step_val, "next");
    llvm::Value* done_val = comp_ctx.builder->CreateICmpEQ(next_val, end_val, "done");
    comp_ctx.builder->CreateCondBr(done_val, end, loop);

    idx_val->addIncoming(zero_val, entry);
    idx_val->addIncoming(next_val, loop);

    comp_ctx.builder->SetInsertPoint(end);
    comp_ctx.builder->CreateRetVoid();
    return false;
}
