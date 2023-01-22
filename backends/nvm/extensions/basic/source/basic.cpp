#include <ned/core/config.h>
#include <basic.h>

#include <iostream>
#include <map>

nvm::error_fn_t error = nullptr;

template<class Impl>
void register_node_impl(nvm::NodeImplMap& nodes)
{ nodes.insert(Impl::name, { &Impl::match, &Impl::compile }); }

#define add_node(name) register_node_impl<name>(nodes)

NVM_INIT(error_fn, nodes)
{
    error = error_fn;
    add_node(       AddImpl );
    add_node(       MulImpl );
    add_node(  ConstValImpl );
    add_node( TransposeImpl );
    add_node(    MatmulImpl );
    return false;
}

const std::vector<std::pair<std::string, nn::core::ConfigType>> AddImpl::cargs = {
    { "fp", nn::core::ConfigType::make_fty() },
    { "shape", nn::core::ConfigType::make_arr(nn::core::ConfigType::make_int()) }
};

const std::vector<std::pair<std::string, nn::core::ConfigType>> MulImpl::cargs = {
    { "fp", nn::core::ConfigType::make_fty() },
    { "shape", nn::core::ConfigType::make_arr(nn::core::ConfigType::make_int()) }
};

const std::vector<std::pair<std::string, nn::core::ConfigType>> ConstValImpl::cargs = {
    { "val", nn::core::ConfigType::make_float() },
    { "fp", nn::core::ConfigType::make_fty() },
    { "shape", nn::core::ConfigType::make_arr(nn::core::ConfigType::make_int()) }
};

const std::vector<std::pair<std::string, nn::core::ConfigType>> TransposeImpl::cargs = {
    { "fp", nn::core::ConfigType::make_fty() },
    {  "M", nn::core::ConfigType::make_int() },
    {  "N", nn::core::ConfigType::make_int() }
};

const std::vector<std::pair<std::string, nn::core::ConfigType>> MatmulImpl::cargs = {
    { "fp", nn::core::ConfigType::make_fty() },
    {  "M", nn::core::ConfigType::make_int() },
    {  "K", nn::core::ConfigType::make_int() },
    {  "N", nn::core::ConfigType::make_int() }
};

bool AddImpl::compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx)
{
    // TODO: handle edge views
    auto& node = node_ctx.graph->get(node_ctx.node);
    assert(node.inps.size() == 2);
    assert(node.outs.size() == 1);
    auto& ctx = llvm_ctx.mod->getContext();
    auto& builder = *llvm_ctx.builder;
    llvm::GlobalVariable* lvec = llvm_ctx.comp->inp_edge(node, 0).var;
    llvm::GlobalVariable* rvec = llvm_ctx.comp->inp_edge(node, 1).var;
    llvm::GlobalVariable* ovec = llvm_ctx.comp->out_edge(node, 0).var;

    size_t sz = 1;
    for (auto& e : node.configs.at("shape").val.val_list)
    {
        assert(e.ty == nn::core::ConfigVal::Tag::INT);
        sz *= e.val_int;
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", llvm_ctx.func);
    llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx, "loop", llvm_ctx.func);
    llvm::BasicBlock* end = llvm::BasicBlock::Create(ctx, "end", llvm_ctx.func);

    llvm::Type* int_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Value* zero_val = llvm::ConstantInt::get(int_ty, 0);
    llvm::Value* end_val = llvm::ConstantInt::get(int_ty, sz);
    llvm::Value* step_val = llvm::ConstantInt::get(int_ty, 1);

    builder.SetInsertPoint(entry);
    builder.CreateBr(loop);

    builder.SetInsertPoint(loop);
    llvm::PHINode* idx_val = builder.CreatePHI(int_ty, 2, "idx");

    // start doing the vector addition
    llvm::Type* fp_type = nvm::get_fptype(ctx, node.configs.at("fp").val.val_fty);
    llvm::Value* lptr_val = builder.CreateGEP(fp_type, lvec, { idx_val }, "lptr");
    llvm::Value* rptr_val = builder.CreateGEP(fp_type, rvec, { idx_val }, "rptr");
    llvm::Value* optr_val = builder.CreateGEP(fp_type, ovec, { idx_val }, "optr");

    llvm::Value* lval_val = builder.CreateLoad(fp_type, lptr_val, "lval");
    llvm::Value* rval_val = builder.CreateLoad(fp_type, rptr_val, "rval");
    llvm::Value* oval_val = builder.CreateFAdd(lval_val, rval_val, "oval");
    builder.CreateStore(oval_val, optr_val);
    // finished the vector addition

    llvm::Value* next_val = builder.CreateAdd(idx_val, step_val, "next");
    llvm::Value* done_val = builder.CreateICmpEQ(next_val, end_val, "done");
    builder.CreateCondBr(done_val, end, loop);

    builder.SetInsertPoint(end);
    builder.CreateRetVoid();

    idx_val->addIncoming(zero_val, entry);
    idx_val->addIncoming(next_val, loop);
    return false;
}

bool MulImpl::compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx)
{
    // TODO: handle edge views
    auto& node = node_ctx.graph->get(node_ctx.node);
    assert(node.inps.size() == 2);
    assert(node.outs.size() == 1);
    auto& ctx = llvm_ctx.mod->getContext();
    auto& builder = *llvm_ctx.builder;
    llvm::GlobalVariable* lvec = llvm_ctx.comp->inp_edge(node, 0).var;
    llvm::GlobalVariable* rvec = llvm_ctx.comp->inp_edge(node, 1).var;
    llvm::GlobalVariable* ovec = llvm_ctx.comp->out_edge(node, 0).var;

    size_t sz = 1;
    for (auto& e : node.configs.at("shape").val.val_list)
    {
        assert(e.ty == nn::core::ConfigVal::Tag::INT);
        sz *= e.val_int;
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", llvm_ctx.func);
    llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx, "loop", llvm_ctx.func);
    llvm::BasicBlock* end = llvm::BasicBlock::Create(ctx, "end", llvm_ctx.func);

    llvm::Type* int_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Value* zero_val = llvm::ConstantInt::get(int_ty, 0);
    llvm::Value* end_val = llvm::ConstantInt::get(int_ty, sz);
    llvm::Value* step_val = llvm::ConstantInt::get(int_ty, 1);

    builder.SetInsertPoint(entry);
    builder.CreateBr(loop);

    builder.SetInsertPoint(loop);
    llvm::PHINode* idx_val = builder.CreatePHI(int_ty, 2, "idx");

    // start doing the vector addition
    llvm::Type* fp_type = nvm::get_fptype(ctx, node.configs.at("fp").val.val_fty);
    llvm::Value* lptr_val = builder.CreateGEP(fp_type, lvec, { idx_val }, "lptr");
    llvm::Value* rptr_val = builder.CreateGEP(fp_type, rvec, { idx_val }, "rptr");
    llvm::Value* optr_val = builder.CreateGEP(fp_type, ovec, { idx_val }, "optr");

    llvm::Value* lval_val = builder.CreateLoad(fp_type, lptr_val, "lval");
    llvm::Value* rval_val = builder.CreateLoad(fp_type, rptr_val, "rval");
    llvm::Value* oval_val = builder.CreateFMul(lval_val, rval_val, "oval");
    builder.CreateStore(oval_val, optr_val);
    // finished the vector addition

    llvm::Value* next_val = builder.CreateAdd(idx_val, step_val, "next");
    llvm::Value* done_val = builder.CreateICmpEQ(next_val, end_val, "done");
    builder.CreateCondBr(done_val, end, loop);

    builder.SetInsertPoint(end);
    builder.CreateRetVoid();

    idx_val->addIncoming(zero_val, entry);
    idx_val->addIncoming(next_val, loop);
    return false;
}

bool ConstValImpl::compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx)
{
    // TODO: handle edge views
    auto& node = node_ctx.graph->get(node_ctx.node);
    assert(node.inps.size() == 0);
    assert(node.outs.size() == 1);
    auto& ctx = llvm_ctx.mod->getContext();
    auto& builder = *llvm_ctx.builder;
    double const_val = node.configs.at("val").val.val_float;
    nn::core::EdgeFty fp = node.configs.at("fp").val.val_fty;
    
    size_t sz = 1;
    for (auto& e : node.configs.at("shape").val.val_list)
    {
        assert(e.ty == nn::core::ConfigVal::Tag::INT);
        sz *= e.val_int;
    }

    llvm::Value* out_val = llvm_ctx.comp->out_edge(node, 0).var;

    llvm::Type* fp_ty = nvm::get_fptype(ctx, fp);
    llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Type* i64_ty = llvm::Type::getInt64Ty(ctx);
    llvm::Type* f64_ty = llvm::Type::getDoubleTy(ctx);
    llvm::Value* zero_val = llvm::ConstantInt::get(i32_ty, 0);
    llvm::Value* step_val = llvm::ConstantInt::get(i32_ty, 1);
    llvm::Value* end_val = llvm::ConstantInt::get(i32_ty, sz);

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", llvm_ctx.func);
    llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx, "loop", llvm_ctx.func);
    llvm::BasicBlock* end = llvm::BasicBlock::Create(ctx, "end", llvm_ctx.func);

    builder.SetInsertPoint(entry);
    llvm::Value* i64_val = llvm::ConstantInt::get(i64_ty, *(int64_t*)&const_val);
    llvm::Value* f64_val = builder.CreateBitCast(i64_val, f64_ty);
    llvm::Value* fp_val = builder.CreateFPCast(f64_val, fp_ty);
    builder.CreateBr(loop);

    builder.SetInsertPoint(loop);
    llvm::PHINode* idx = builder.CreatePHI(i32_ty, 2, "idx");

    llvm::Value* ptr = builder.CreateGEP(fp_ty, out_val, { idx }, "ptr");
    builder.CreateStore(fp_val, ptr);

    llvm::Value* nidx = builder.CreateAdd(idx, step_val, "nidx");
    llvm::Value* cond = builder.CreateICmpEQ(nidx, end_val, "cond");
    builder.CreateCondBr(cond, end, loop);

    builder.SetInsertPoint(end);
    builder.CreateRetVoid();

    idx->addIncoming(zero_val, entry);
    idx->addIncoming(nidx, loop);

    return false;
}

bool TransposeImpl::compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx)
{
    // TODO: handle edge views
    auto& node = node_ctx.graph->get(node_ctx.node);
    assert(node.inps.size() == 1);
    assert(node.outs.size() == 1);
    auto& ctx = llvm_ctx.mod->getContext();
    auto& builder = *llvm_ctx.builder;
    nn::core::EdgeFty fp = node.configs.at("fp").val.val_fty;
    int64_t m = node.configs.at("M").val.val_int;
    int64_t n = node.configs.at("N").val.val_int;
        
    llvm::Value* inp_val = llvm_ctx.comp->inp_edge(node, 0).var;
    llvm::Value* out_val = llvm_ctx.comp->out_edge(node, 0).var;

    llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Value* zero_val = llvm::ConstantInt::get(i32_ty, 0);
    llvm::Value* step_val = llvm::ConstantInt::get(i32_ty, 1);

    llvm::Type* fp_ty = nvm::get_fptype(ctx, fp);
    llvm::Value* m_val = llvm::ConstantInt::get(i32_ty, m);
    llvm::Value* n_val = llvm::ConstantInt::get(i32_ty, n);

    if (m * n == 0)
    {
        // Nothing to transpose
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", llvm_ctx.func);
        builder.SetInsertPoint(entry);
        builder.CreateRetVoid();
        return false;
    }

    llvm::BasicBlock* entry    = llvm::BasicBlock::Create(ctx, "entry",    llvm_ctx.func);
    llvm::BasicBlock* row_loop = llvm::BasicBlock::Create(ctx, "row_loop", llvm_ctx.func);
    llvm::BasicBlock* col_loop = llvm::BasicBlock::Create(ctx, "col_loop", llvm_ctx.func);
    llvm::BasicBlock* row_end  = llvm::BasicBlock::Create(ctx, "row_end",  llvm_ctx.func);
    llvm::BasicBlock* end      = llvm::BasicBlock::Create(ctx, "end",      llvm_ctx.func);

    builder.SetInsertPoint(entry);
    builder.CreateBr(row_loop);

    builder.SetInsertPoint(row_loop);
    llvm::PHINode* row_idx_val = builder.CreatePHI(i32_ty, 2, "row_idx");
    builder.CreateBr(col_loop);

    builder.SetInsertPoint(col_loop);
    llvm::PHINode* col_idx_val = builder.CreatePHI(i32_ty, 2, "col_idx");

    llvm::Value* inp_row_off_val = builder.CreateMul(row_idx_val, n_val, "inp_row_off");
    llvm::Value* inp_off_val     = builder.CreateAdd(inp_row_off_val, col_idx_val, "inp_off");
    llvm::Value* out_row_off_val = builder.CreateMul(col_idx_val, m_val, "out_row_off");
    llvm::Value* out_off_val     = builder.CreateAdd(out_row_off_val, row_idx_val, "out_off");

    llvm::Value* inp_ptr_val = builder.CreateGEP(fp_ty, inp_val, { inp_off_val }, "inp_ptr");
    llvm::Value* out_ptr_val = builder.CreateGEP(fp_ty, out_val, { out_off_val }, "out_ptr");
    llvm::Value* tmp_val = builder.CreateLoad(fp_ty, inp_ptr_val, "tmp");
    builder.CreateStore(tmp_val, out_ptr_val);

    llvm::Value* col_nidx_val = builder.CreateAdd(col_idx_val, step_val, "col_nidx");
    llvm::Value* col_cond_val = builder.CreateICmpEQ(col_nidx_val, n_val, "col_cond");
    builder.CreateCondBr(col_cond_val, row_end, col_loop);

    builder.SetInsertPoint(row_end);
    llvm::Value* row_nidx_val = builder.CreateAdd(row_idx_val, step_val, "row_nidx");
    llvm::Value* row_cond_val = builder.CreateICmpEQ(row_nidx_val, m_val, "row_cond");
    builder.CreateCondBr(row_cond_val, end, row_loop);

    builder.SetInsertPoint(end);
    builder.CreateRetVoid();

    row_idx_val->addIncoming(zero_val, entry);
    row_idx_val->addIncoming(row_nidx_val, row_end);
    col_idx_val->addIncoming(zero_val, row_loop);
    col_idx_val->addIncoming(col_nidx_val, col_loop);
    
    return false;
}

bool MatmulImpl::compile(const nvm::NodeCtx& node_ctx, nvm::CompCtx& llvm_ctx)
{
    // TODO: handle edge views
    auto& node = node_ctx.graph->get(node_ctx.node);
    assert(node.inps.size() == 2);
    assert(node.outs.size() == 1);
    auto& ctx = llvm_ctx.mod->getContext();
    auto& builder = *llvm_ctx.builder;
    nn::core::EdgeFty fp = node.configs.at("fp").val.val_fty;
    int64_t m = node.configs.at("M").val.val_int;
    int64_t k = node.configs.at("K").val.val_int;
    int64_t n = node.configs.at("N").val.val_int;

    llvm::Value* lhs_val = llvm_ctx.comp->inp_edge(node, 0).var;
    llvm::Value* rhs_val = llvm_ctx.comp->inp_edge(node, 1).var;
    llvm::Value* out_val = llvm_ctx.comp->out_edge(node, 0).var;

    llvm::Type* i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Value* zero_val = llvm::ConstantInt::get(i32_ty, 0);
    llvm::Value* step_val = llvm::ConstantInt::get(i32_ty, 1);

    llvm::Type* fp_ty = nvm::get_fptype(ctx, fp);
    llvm::Value* m_val = llvm::ConstantInt::get(i32_ty, m);
    llvm::Value* k_val = llvm::ConstantInt::get(i32_ty, k);
    llvm::Value* n_val = llvm::ConstantInt::get(i32_ty, n);

    if (m * n == 0)
    {
        // No output to write to
        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", llvm_ctx.func);
        builder.SetInsertPoint(entry);
        builder.CreateRetVoid();
        return false;
    }

    if (k == 0)
    {
        // The base case of the add reduce component is a matrix of all zeros
        llvm::Value* mn_val = llvm::ConstantInt::get(i32_ty, m * n);

        llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", llvm_ctx.func);
        llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx, "loop", llvm_ctx.func);
        llvm::BasicBlock* end = llvm::BasicBlock::Create(ctx, "end", llvm_ctx.func);

        builder.SetInsertPoint(entry);
        llvm::Value* fp_zero = builder.CreateCast(llvm::Instruction::CastOps::UIToFP, zero_val, fp_ty, "fp_zero");
        builder.CreateBr(loop);

        builder.SetInsertPoint(loop);
        llvm::PHINode* idx = builder.CreatePHI(i32_ty, 2, "idx");

        llvm::Value* ptr = builder.CreateGEP(fp_ty, out_val, { idx }, "ptr");
        builder.CreateStore(fp_zero, ptr);

        llvm::Value* nidx = builder.CreateAdd(idx, step_val, "nidx");
        llvm::Value* cond = builder.CreateICmpEQ(nidx, mn_val, "cond");
        builder.CreateCondBr(cond, end, loop);

        builder.SetInsertPoint(end);
        builder.CreateRetVoid();

        idx->addIncoming(zero_val, entry);
        idx->addIncoming(nidx, loop);

        return false;
    }

    // General case

    llvm::BasicBlock* entry    = llvm::BasicBlock::Create(ctx, "entry",    llvm_ctx.func);
    llvm::BasicBlock* row_loop = llvm::BasicBlock::Create(ctx, "row_loop", llvm_ctx.func);
    llvm::BasicBlock* col_loop = llvm::BasicBlock::Create(ctx, "col_loop", llvm_ctx.func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx, "sum_loop", llvm_ctx.func);
    llvm::BasicBlock* sum_end  = llvm::BasicBlock::Create(ctx, "sum_end",  llvm_ctx.func);
    llvm::BasicBlock* col_end  = llvm::BasicBlock::Create(ctx, "row_end",  llvm_ctx.func);
    llvm::BasicBlock* row_end  = llvm::BasicBlock::Create(ctx, "row_end",  llvm_ctx.func);
    llvm::BasicBlock* end      = llvm::BasicBlock::Create(ctx, "end",      llvm_ctx.func);

    // Function entrance and exit
    builder.SetInsertPoint(entry);
    llvm::Value* fp_zero = builder.CreateCast(llvm::Instruction::CastOps::UIToFP, zero_val, fp_ty, "fp_zero");
    builder.CreateBr(row_loop);

    builder.SetInsertPoint(end);
    builder.CreateRetVoid();

    // Looping over the rows of lhs - columns of rhs
    builder.SetInsertPoint(row_loop);
    llvm::PHINode* row_idx = builder.CreatePHI(i32_ty, 2, "row_idx");
    builder.CreateBr(col_loop);

    builder.SetInsertPoint(row_end);
    llvm::Value* row_nidx = builder.CreateAdd(row_idx, step_val, "row_nidx");
    llvm::Value* row_cond = builder.CreateICmpEQ(row_nidx, m_val, "row_cond");
    builder.CreateCondBr(row_cond, end, row_loop);

    row_idx->addIncoming(zero_val, entry);
    row_idx->addIncoming(row_nidx, row_end);

    // Looping over the columns of rhs - rows of lhs
    builder.SetInsertPoint(col_loop);
    llvm::PHINode* col_idx = builder.CreatePHI(i32_ty, 2, "col_idx");
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(col_end);
    llvm::Value* col_nidx = builder.CreateAdd(col_idx, step_val, "col_nidx");
    llvm::Value* col_cond = builder.CreateICmpEQ(col_nidx, n_val, "col_cond");
    builder.CreateCondBr(col_cond, row_end, col_loop);

    col_idx->addIncoming(zero_val, row_loop);
    col_idx->addIncoming(col_nidx, col_end);

    // sum reduce over k of lhs[i, k] * rhs[k, j]
    builder.SetInsertPoint(sum_loop);
    llvm::PHINode* sum_idx = builder.CreatePHI(i32_ty, 2, "sum_idx");
    llvm::PHINode* acc = builder.CreatePHI(fp_ty, 2, "acc");

    llvm::Value* lhs_row_off = builder.CreateMul(row_idx, k_val, "lhs_row_off");
    llvm::Value* lhs_off = builder.CreateAdd(lhs_row_off, sum_idx, "lhs_off");
    llvm::Value* lhs_ptr = builder.CreateGEP(fp_ty, lhs_val, { lhs_off }, "lhs_ptr");
    llvm::Value* lhs = builder.CreateLoad(fp_ty, lhs_ptr, "lhs_val");

    llvm::Value* rhs_row_off = builder.CreateMul(sum_idx, n_val, "rhs_row_off");
    llvm::Value* rhs_off = builder.CreateAdd(rhs_row_off, col_idx, "rhs_off");
    llvm::Value* rhs_ptr = builder.CreateGEP(fp_ty, rhs_val, { rhs_off }, "rhs_ptr");
    llvm::Value* rhs = builder.CreateLoad(fp_ty, rhs_ptr, "rhs_val");

    llvm::Value* prod = builder.CreateFMul(lhs, rhs, "prod");
    llvm::Value* nacc = builder.CreateFAdd(acc, prod, "nacc");;

    llvm::Value* sum_nidx = builder.CreateAdd(sum_idx, step_val, "sum_nidx");
    llvm::Value* sum_cond = builder.CreateICmpEQ(sum_nidx, k_val, "sum_cond");
    builder.CreateCondBr(sum_cond, sum_end, sum_loop);

    builder.SetInsertPoint(sum_end);
    llvm::Value* out_row_off = builder.CreateMul(row_idx, n_val, "out_row_off");
    llvm::Value* out_off = builder.CreateAdd(out_row_off, col_idx, "out_off");
    llvm::Value* out_ptr = builder.CreateGEP(fp_ty, out_val, { out_off }, "out_ptr");
    builder.CreateStore(nacc, out_ptr);
    builder.CreateBr(col_end);

    sum_idx->addIncoming(zero_val, col_loop);
    sum_idx->addIncoming(sum_nidx, sum_loop);
    acc->addIncoming(fp_zero, col_loop);
    acc->addIncoming(nacc, sum_loop);

    return false;
}
