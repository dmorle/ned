#include <nedvm/common.h>
#include <nedvm/vmnodes.h>
#include <nedvm/graphgen.h>

#include <ned/lang/obj.h>

namespace nn
{
    namespace nedvm
    {
        void generate_add_same_forward(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode)
        {
            assert(pnode->inputs.size() == 2);
            assert(((EdgeData*)pnode->inputs[0]->opaque)->forward_val);
            assert(((EdgeData*)pnode->inputs[1]->opaque)->forward_val);
            assert(pnode->outputs.size() == 1);
            assert(((EdgeData*)pnode->outputs[0]->opaque)->forward_val);
            llvm::GlobalVariable* lvec = ((EdgeData*)pnode->inputs[0]->opaque)->forward_val;
            llvm::GlobalVariable* rvec = ((EdgeData*)pnode->inputs[1]->opaque)->forward_val;
            llvm::GlobalVariable* ovec = ((EdgeData*)pnode->outputs[0]->opaque)->forward_val;

            assert(pnode->cargs.at("ofw")->ty == lang::ObjType::FWIDTH);
            assert(pnode->cargs.at("lfw")->ty == lang::ObjType::FWIDTH);
            assert(pnode->cargs.at("rfw")->ty == lang::ObjType::FWIDTH);
            core::tensor_dty lfw = std::static_pointer_cast<lang::ObjFWidth>(pnode->cargs.at("lfw"))->data.dty;
            core::tensor_dty rfw = std::static_pointer_cast<lang::ObjFWidth>(pnode->cargs.at("rfw"))->data.dty;
            core::tensor_dty ofw = std::static_pointer_cast<lang::ObjFWidth>(pnode->cargs.at("ofw"))->data.dty;
            llvm::Type* lty = get_fptype(ctx, lfw);
            llvm::Type* rty = get_fptype(ctx, rfw);
            llvm::Type* oty = get_fptype(ctx, ofw);

            size_t sz = 1;
            for (auto& e : pnode->cargs.at("shape")->iter())
            {
                assert(e->ty == lang::ObjType::INT);
                sz *= std::static_pointer_cast<lang::ObjInt>(e)->data.val;
            }

            llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", func);
            llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx, "loop", func);
            llvm::BasicBlock* end = llvm::BasicBlock::Create(ctx, "end", func);
            llvm::IntegerType* int_ty = llvm::IntegerType::get(ctx, (unsigned int)(std::log2((double)sz) + 2));
            llvm::Value* start_val = llvm::ConstantInt::get(int_ty, 0);
            llvm::Value* end_val = llvm::ConstantInt::get(int_ty, sz);
            llvm::Value* step_val = llvm::ConstantInt::get(int_ty, 1);

            pbuilder->SetInsertPoint(entry);
            pbuilder->CreateBr(loop);

            pbuilder->SetInsertPoint(loop);
            llvm::PHINode* idx_val = pbuilder->CreatePHI(int_ty, 2, "idx");

            // start doing the vector addition
            llvm::Value* lptr_val = pbuilder->CreateGEP(lty, lvec, idx_val, "lptr");
            llvm::Value* rptr_val = pbuilder->CreateGEP(rty, rvec, idx_val, "rptr");
            llvm::Value* optr_val = pbuilder->CreateGEP(oty, ovec, idx_val, "optr");
            llvm::Value* lval_val = pbuilder->CreateLoad(lptr_val, "lval");
            llvm::Value* rval_val = pbuilder->CreateLoad(rptr_val, "rval");
            if (lfw != ofw)
                lval_val = pbuilder->CreateFPCast(lval_val, oty, "lcast");
            if (rfw != ofw)
                rval_val = pbuilder->CreateFPCast(rval_val, oty, "rcast");
            llvm::Value* oval_val = pbuilder->CreateFAdd(lval_val, rval_val, "oval");
            pbuilder->CreateStore(oval_val, optr_val);
            // finished the vector addition

            llvm::Value* next_val = pbuilder->CreateAdd(idx_val, step_val, "next");
            llvm::Value* done_val = pbuilder->CreateICmpEQ(next_val, end_val, "done");
            pbuilder->CreateCondBr(done_val, end, loop);

            idx_val->addIncoming(start_val, entry);
            idx_val->addIncoming(next_val, loop);

            pbuilder->SetInsertPoint(end);
            pbuilder->CreateRetVoid();
        }

        void generate_add_same_backward(llvm::LLVMContext& ctx, llvm::Function* func, Builder* pbuilder, const core::Node* pnode)
        {
            throw GraphGenError("The backward method for add_same_intr has not been implmented");
        }
    }
}
