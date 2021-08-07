#include <libnn/frontend/eval.h>
#include <libnn/frontend/obj.h>
#include <libnn/frontend/ast.h>

#include <cassert>

namespace nn
{
    namespace impl
    {
        // Definitions for eval.h

        void AstExpr::append_cargs(EvalCtx& ctx, std::vector<Obj*>& cargs) const
        {
            cargs.push_back(eval(ctx));
        }

        void AstPack::append_cargs(EvalCtx& ctx, std::vector<Obj*>& cargs)
        {
            std::vector<Obj*> eval_result = eval(ctx)->iter(ctx);
            cargs.insert(
                cargs.end(),
                std::make_move_iterator(eval_result.begin()),
                std::make_move_iterator(eval_result.end())
            );
        }

        GenerationError::GenerationError(const std::string& errmsg) :
            errmsg(errmsg)
        {}

        void GenerationError::tb_touch(AstBlock* pblock)
        {
            traceback.push_back({ pblock->file_name, pblock->line_num, pblock->col_num });
        }

        void GenerationError::tb_touch(const std::string& file_name, uint32_t line_num, uint32_t col_num)
        {
            traceback.push_back({ file_name, line_num, col_num });
        }

        EvalCtx::EvalCtx()
        {
            model_params = {};
            pgraph = new Graph();
            pscope = nullptr;

            state = EvalState::CALL;
        }

        // AST node evaluation

        static std::unordered_map<std::string, int> varcounts = {};

        Obj* AstBool::eval(EvalCtx& ctx) const
        {
            ObjBool* nobj = new ObjBool();
            nobj->data.val = val;
            nobj->init = true;
            return nobj;
        }

        Obj* AstInt::eval(EvalCtx& ctx) const
        {
            ObjInt* nobj = new ObjInt();
            nobj->data.val = val;
            nobj->init = true;
            return nobj;
        }

        Obj* AstFloat::eval(EvalCtx& ctx) const
        {
            ObjFloat* nobj = new ObjFloat();
            nobj->data.val = val;
            nobj->init = true;
            return nobj;
        }

        Obj* AstStr::eval(EvalCtx& ctx) const
        {
            ObjStr* nobj = new ObjStr();
            nobj->data.val = val;
            nobj->init = true;
            return nobj;
        }

        Obj* AstIdn::eval(EvalCtx& ctx) const
        {
            if (ctx.contains(idn))
                return ctx[idn];
            throw GenerationError("Unable to find a variable with name " + idn);
        }

        Obj* AstTuple::eval(EvalCtx& ctx) const
        {
            ObjTuple* nobj = new ObjTuple();
            for (auto e : this->elems)
                nobj->data.elems.push_back(e->eval(ctx));
            nobj->init = true;
            return nobj;
        }

        Obj* AstCall::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstCargs::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstIdx::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstDot::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstNeg::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstPack::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstAdd::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstSub::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstMul::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstDiv::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstEq::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstNe::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstGe::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstLe::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstGt::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstLt::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstAnd::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstOr::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstIAdd::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstISub::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstIMul::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstIDiv::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstAssign::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstDecl::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstSeq::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstIf::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstWhile::eval(EvalCtx& ctx) const
        {

        }

        Obj* AstFor::eval(EvalCtx& ctx) const
        {

        }

        void AstDef::eval(EvalCtx& ctx) const
        {

        }

        void AstIntr::eval(EvalCtx& ctx) const
        {

        }

        void AstFn::eval(EvalCtx& ctx) const
        {

        }

        void AstModImp::eval(EvalCtx& ctx) const
        {

        }

        EvalCtx* AstModule::eval(const std::string& entry_point, std::vector<Obj*>& cargs)
        {
            EvalCtx* pctx = new EvalCtx();

            // Building EvalCtx
            for (auto e : imps)
                e.eval(*pctx);

            for (auto e : fns)
                e.eval(*pctx);
            for (auto e : intrs)
                e.eval(*pctx);
            for (auto e : defs)
                e.eval(*pctx);

            // Finding the entry point
            decltype(pctx->defs)::iterator it = pctx->defs.find(entry_point);
            if (it == pctx->defs.end())
                throw GenerationError("Unable to find a valid entry point with name " + entry_point);
            Obj* entry_def = it->second;
            
            // applying cargs to the entry def
            entry_def = entry_def->cargs(cargs);

            // Generating the arguments for the entry point
            const auto& vargs = static_cast<ObjDef*>(entry_def)->data.def.vargs;
            std::vector<Obj*> args;
            for (const AstDecl& e : vargs)
            {
                if (e.type_name != "tensor")
                    throw GenerationError("'def' must have only tensor types for varargs");
                // building the arguments one by one
                args.push_back(e.eval(*pctx));
            }

            // running model generation
            entry_def->call(*pctx, args);

            return pctx;
        }
    }
}
